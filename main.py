from __future__ import annotations

import argparse
import gc
import platform
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None

from data.loaders import ConversationExample, subsample_examples
from data.multiturn_jailbreak import load_multiturn_jailbreak
from data.sleeper_builder import build_sleeper_dataset
from data.splits import apply_split_labels, deterministic_split, save_splits
from data.wildjailbreak import load_wildjailbreak
from eval.causal_interventions import causal_intervention_rows
from eval.conjunction_tests import conjunction_control_rows
from eval.generate import extract_hidden_trajectories, generate_responses
from eval.judge import assert_judge_access, judge_predictions, load_judge, sanity_check_judge
from eval.layer_sweep import layer_summary_rows
from eval.mechanistic_taxonomy import mechanistic_taxonomy_rows
from eval.metrics import compute_metrics
from eval.null_controls import null_control_rows
from eval.significance import significance_rows
from eval.transfer import transfer_rows
from models.baselines import ActivationProbe, DenseLinearProbe, mean_difference_vector, top_mean_difference_dims
from models.baselines import TorchMLPProbe
from models.subject import SubjectModel, require_full_experiment_device
from train.train_fuser import train_fuser
from train.train_lora import train_lora
from train.train_sae import train_saes
from utils.config_utils import load_config
from utils.dependencies import assert_runtime_dependencies
from utils.io import make_run_dir, stable_hash, write_json, write_jsonl, write_yaml
from utils.logging_utils import setup_logging
from utils.seed import set_seed


METHODS = [
    "dense_probe",
    "mlp_probe",
    "turn_concat_mlp",
    "mean_diff",
    "repe",
    "activation_probe",
    "trajectory_only",
    "linear_sae_only",
    "linear_sae_trajectory",
    "bilinear_sae_only",
    "bilinear_sae_trajectory",
    "full_fused",
]


def _targets(examples: List[ConversationExample]) -> torch.Tensor:
    return torch.tensor([ex.target for ex in examples], dtype=torch.long)


def _split_and_save(run_dir: Path, name: str, examples: List[ConversationExample], ratios, seed: int, stratify_key: str | None) -> List[ConversationExample]:
    splits = deterministic_split(examples, ratios, seed, stratify_key=stratify_key)
    examples = apply_split_labels(examples, splits)
    save_splits(run_dir, name, examples, splits)
    return examples


def build_all_datasets(cfg: Dict[str, Any], run_dir: Path, seed: int, logger):
    logger.info("Loading WildJailbreak.")
    wild = _split_and_save(run_dir, "wildjailbreak", load_wildjailbreak(), (0.70, 0.10, 0.20), seed, "mode")
    logger.info("Loading multi-turn jailbreak dataset.")
    multi, note = load_multiturn_jailbreak(cfg)
    multi = _split_and_save(run_dir, "multiturn_jailbreak", multi, (0.60, 0.20, 0.20), seed, "family")
    benign = [ex.turns[-1]["content"] for ex in wild if ex.source_is_benign][:200]
    harmful = [ex.turns[-1]["content"] for ex in wild if ex.target == 1][:200]
    sleeper = _split_and_save(run_dir, "constructed_sleeper", build_sleeper_dataset(cfg, seed, benign, harmful), (0.60, 0.20, 0.20), seed, "family")
    summary = [
        {"group": "ordinary_harmful_benign", "source": "allenai/wildjailbreak", "n": len(wild), "note": "Gated WildJailbreak; train/eval configs adapted by data_type."},
        {"group": "ordinary_multiturn_jailbreak", "source": cfg.get("multiturn_dataset", "ScaleAI/mhj"), "n": len(multi), "note": note},
        {"group": "sleeper_distributed_trigger", "source": "constructed_sleeper", "n": len(sleeper), "note": "Programmatic sleeper-style distributed triggers; separated from public data."},
    ]
    return {"wild": wild, "multi": multi, "sleeper": sleeper}, summary


def _select_eval_examples(datasets: Dict[str, List[ConversationExample]], cfg: Dict[str, Any]) -> Dict[str, List[ConversationExample]]:
    train = [ex for group in datasets.values() for ex in group if ex.split == "train"]
    val = [ex for group in datasets.values() for ex in group if ex.split == "val"]
    test_groups = {
        "ordinary_harmful_benign": [ex for ex in datasets["wild"] if ex.split == "test"],
        "ordinary_multiturn_jailbreak": [ex for ex in datasets["multi"] if ex.split == "test"],
        "sleeper_distributed_trigger": [ex for ex in datasets["sleeper"] if ex.split == "test" and not ex.is_ood],
        "sleeper_ood": [ex for ex in datasets["sleeper"] if ex.is_ood],
    }
    return {
        "train": subsample_examples(train, int(cfg["train_subsample"])),
        "val": subsample_examples(val, int(cfg["val_subsample"])),
        **{k: subsample_examples(v, int(cfg["ood_subsample"] if k == "sleeper_ood" else cfg["test_subsample"])) for k, v in test_groups.items()},
    }


def _rank_features(acts_bad: np.ndarray, acts_clean: np.ndarray, device: torch.device) -> torch.Tensor:
    if acts_bad.size == 0 or acts_clean.size == 0:
        width = acts_bad.shape[1] if acts_bad.ndim == 2 else acts_clean.shape[1]
        return torch.arange(min(64, width), device=device)
    x = np.vstack([acts_bad, acts_clean])
    y = np.concatenate([np.ones(len(acts_bad)), np.zeros(len(acts_clean))])
    if len(np.unique(y)) < 2:
        return torch.arange(min(64, x.shape[1]), device=device)
    if LogisticRegression is None:
        diff = np.abs(acts_bad.mean(axis=0) - acts_clean.mean(axis=0))
        return torch.tensor(np.argsort(diff)[::-1].copy()[:64], device=device)
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=2000)
    clf.fit(x, y)
    return torch.tensor(np.argsort(np.abs(clf.coef_[0]))[::-1].copy()[:64], device=device)


@torch.inference_mode()
def score_methods(h_seq: torch.Tensor, models: Dict[str, Any], y: torch.Tensor) -> Dict[str, np.ndarray]:
    h_seq = h_seq.float()
    h_last = h_seq[:, -1, :]
    out: Dict[str, np.ndarray] = {}
    out["dense_probe"] = models["dense_probe"].score(h_last.cpu().float().numpy()).scores
    out["mlp_probe"] = models["mlp_probe"].score(h_last.cpu().float().numpy()).scores
    out["turn_concat_mlp"] = models["turn_concat_mlp"].score(h_seq.reshape(h_seq.size(0), -1).cpu().float().numpy()).scores
    vec = models["mean_vec"].to(h_last.device)
    raw = (h_last.float() @ vec.float()).detach().cpu().numpy()
    out["mean_diff"] = raw
    out["repe"] = raw
    out["activation_probe"] = models["activation_probe"].score(h_last.cpu().float().numpy()).scores
    threat, _ = models["trajectory"](h_seq)
    traj = threat[:, -1].float()
    a1 = models["linear_sae"].get_sparse_acts(h_last)
    a2 = models["bilinear_sae"].get_sparse_acts(h_last)
    s1 = a1[:, models["bad_feats"]["linear"]].sum(-1).float()
    s2 = a2[:, models["bad_feats"]["bilinear"]].sum(-1).float()
    out["trajectory_only"] = traj.detach().cpu().numpy()
    out["linear_sae_only"] = s1.detach().cpu().numpy()
    out["bilinear_sae_only"] = s2.detach().cpu().numpy()
    out["linear_sae_trajectory"] = (s1 + traj).detach().cpu().numpy()
    out["bilinear_sae_trajectory"] = (s2 + traj).detach().cpu().numpy()
    out["full_fused"] = models["fuser"](s1, s2, traj).detach().cpu().numpy()
    return out


def _threshold_label(t: float) -> str:
    return "inf" if np.isinf(t) else f"{t:.2f}"


def _guarded_responses(base_responses: List[str], scores: np.ndarray, threshold: float) -> tuple[List[str], List[bool]]:
    if np.isinf(threshold):
        return list(base_responses), [False for _ in base_responses]
    flags = scores > threshold
    responses = [
        "I cannot help with harmful, unauthorized, or operationally unsafe instructions. I can offer safe alternatives."
        if flag else resp
        for resp, flag in zip(base_responses, flags)
    ]
    return responses, flags.tolist()


def _defended_perplexity(subject, cfg: Dict[str, Any]) -> float:
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    except Exception:
        return 0.0
    texts = []
    for item in ds:
        text = str(item.get("text", "")).strip()
        if len(text) > 50:
            texts.append(text)
        if len(texts) >= int(cfg["data"]["wikitext_batches"]) * int(cfg["eval"]["batch_size"]):
            break
    if not texts:
        return 0.0
    tok = subject.tokenizer
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    total_loss = 0.0
    total_toks = 0
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.inference_mode():
        for i in range(0, len(texts), int(cfg["eval"]["batch_size"])):
            bx = enc["input_ids"][i:i + int(cfg["eval"]["batch_size"])].to(subject.device)
            bm = enc["attention_mask"][i:i + int(cfg["eval"]["batch_size"])].to(subject.device)
            logits = subject.model(input_ids=bx, attention_mask=bm).logits
            loss = ce(logits[:, :-1].contiguous().view(-1, logits.size(-1)), bx[:, 1:].contiguous().view(-1))
            loss = loss.view(bx.size(0), -1)
            total_loss += (loss * bm[:, 1:]).sum().item()
            total_toks += bm[:, 1:].sum().item()
    return float(np.exp(total_loss / max(total_toks, 1)))


def _feature_rows(seed: int, linear_sae, bilinear_sae, h_seq: torch.Tensor, y: torch.Tensor, bad_feats: Dict[str, torch.Tensor], sae_stats: Dict[str, float]) -> List[Dict[str, Any]]:
    rows = []
    for name, sae, feats in [("linear", linear_sae, bad_feats["linear"]), ("bilinear", bilinear_sae, bad_feats["bilinear"])]:
        with torch.inference_mode():
            for turn in range(3):
                acts = sae.get_sparse_acts(h_seq[:, turn, :])[:, feats[:20]]
                for rank in range(acts.size(1)):
                    bad = acts[y.bool(), rank].float()
                    clean = acts[~y.bool(), rank].float()
                    sigma = ((bad.std(unbiased=False) + clean.std(unbiased=False)) / 2).clamp(min=1e-8)
                    rows.append({
                        "seed": seed,
                        "sae": name,
                        "turn": turn + 1,
                        "feature_rank": rank,
                        "activation": float(acts[:, rank].float().mean().cpu()),
                        "d_prime": float(((bad.mean() - clean.mean()) / sigma).cpu()) if len(bad) and len(clean) else 0.0,
                        "selectivity": float((bad.mean() / (clean.mean() + 1e-8)).cpu()) if len(bad) and len(clean) else 0.0,
                        "decoder_norm": float(sae.dec.weight[:, feats[rank]].norm().cpu()) if rank < len(feats) else 0.0,
                        "reconstruction_mse": sae_stats.get(f"{name}_mse", 0.0),
                        "pair_synergy": 0.0,
                    })
    return rows


def run_seed(seed: int, cfg: Dict[str, Any], run_dir: Path, datasets: Dict[str, List[ConversationExample]], judge, logger):
    set_seed(seed)
    selected = _select_eval_examples(datasets, cfg)
    device = require_full_experiment_device(bool(cfg.get("require_gpu", True)))
    subject = SubjectModel(cfg, device)
    lora_train = selected["train"][: int(cfg.get("train_subsample", len(selected["train"])))]
    train_lora(subject, lora_train, cfg, seed, run_dir / "checkpoints", logger)
    intercept_layer = int(cfg["intercept_layers"][0])

    train_h, _ = extract_hidden_trajectories(subject, selected["train"], cfg, intercept_layer)
    val_h, _ = extract_hidden_trajectories(subject, selected["val"], cfg, intercept_layer)
    train_h = train_h.to(device).float()
    val_h = val_h.to(device).float()
    train_y = _targets(selected["train"]).to(device)
    val_y = _targets(selected["val"]).to(device)

    linear_sae, bilinear_sae, sae_stats = train_saes(train_h, cfg, subject.hidden_size, run_dir / "checkpoints", seed, logger)
    h_bad = train_h[train_y == 1, -1, :]
    h_clean = train_h[train_y == 0, -1, :]
    with torch.inference_mode():
        bad_feats = {
            "linear": _rank_features(linear_sae.get_sparse_acts(h_bad).cpu().numpy(), linear_sae.get_sparse_acts(h_clean).cpu().numpy(), device),
            "bilinear": _rank_features(bilinear_sae.get_sparse_acts(h_bad).cpu().numpy(), bilinear_sae.get_sparse_acts(h_clean).cpu().numpy(), device),
        }
    trajectory, fuser, fuser_stats = train_fuser(train_h, train_y, linear_sae, bilinear_sae, bad_feats, cfg, subject.hidden_size, run_dir / "checkpoints", seed, logger)

    dense = DenseLinearProbe()
    dense.fit(train_h[:, -1, :].cpu().float().numpy(), train_y.cpu().numpy())
    activation = ActivationProbe()
    activation.fit(train_h[:, -1, :].cpu().float().numpy(), train_y.cpu().numpy())
    baseline_cfg = cfg.get("baselines", {})
    mlp = TorchMLPProbe(
        subject.hidden_size,
        int(baseline_cfg.get("mlp_hidden_dim", 128)),
        int(baseline_cfg.get("mlp_steps", 200)),
        float(baseline_cfg.get("mlp_lr", 1e-3)),
        seed,
    )
    mlp.fit(train_h[:, -1, :].cpu().float().numpy(), train_y.cpu().numpy())
    turn_mlp = TorchMLPProbe(
        subject.hidden_size * train_h.size(1),
        int(baseline_cfg.get("mlp_hidden_dim", 128)),
        int(baseline_cfg.get("mlp_steps", 200)),
        float(baseline_cfg.get("mlp_lr", 1e-3)),
        seed + 17,
    )
    turn_mlp.fit(train_h.reshape(train_h.size(0), -1).cpu().float().numpy(), train_y.cpu().numpy())
    mean_vec = mean_difference_vector(h_bad, h_clean)
    top_dims = top_mean_difference_dims(h_bad, h_clean)
    model_bundle = {
        "dense_probe": dense,
        "mlp_probe": mlp,
        "turn_concat_mlp": turn_mlp,
        "activation_probe": activation,
        "mean_vec": mean_vec,
        "top_dims": top_dims,
        "linear_sae": linear_sae,
        "bilinear_sae": bilinear_sae,
        "trajectory": trajectory,
        "fuser": fuser,
        "bad_feats": bad_feats,
    }

    metric_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    feature_rows = _feature_rows(seed, linear_sae, bilinear_sae, train_h, train_y, bad_feats, sae_stats)
    locality_rows = []
    taxonomy_rows: List[Dict[str, Any]] = []
    causal_rows: List[Dict[str, Any]] = []
    layer_rows = layer_summary_rows(train_h, train_y, seed, intercept_layer, "train")
    ppx = _defended_perplexity(subject, cfg)

    for group, examples in selected.items():
        if group in {"train", "val"} or not examples:
            continue
        eval_h, _ = extract_hidden_trajectories(subject, examples, cfg, intercept_layer)
        eval_h = eval_h.to(device).float()
        eval_y = _targets(examples).to(device)
        scores = score_methods(eval_h, model_bundle, eval_y)
        taxonomy_rows.extend(mechanistic_taxonomy_rows(examples, scores, seed, group))
        causal_rows.extend(causal_intervention_rows(examples, eval_h, linear_sae, bilinear_sae, bad_feats, seed, group))
        layer_rows.extend(layer_summary_rows(eval_h, eval_y, seed, intercept_layer, group))
        base_responses, tok_sec = generate_responses(subject, examples, cfg)
        processor, judge_model = judge
        base_preds = judge_predictions(processor, judge_model, [ex.prompt_text for ex in examples], base_responses, int(cfg["max_seq_len"]))
        base_rows = [{"target": ex.target, "judge_pred": pred, "score": pred, "family_holdout": ex.family_holdout, "is_ood": ex.is_ood} for ex, pred in zip(examples, base_preds)]
        base_asr = compute_metrics(base_rows).get("asr", 0.0)
        for method in METHODS:
            for threshold in cfg["eval"]["thresholds"]:
                responses, intervened = _guarded_responses(base_responses, scores[method], float(threshold))
                preds = judge_predictions(processor, judge_model, [ex.prompt_text for ex in examples], responses, int(cfg["max_seq_len"]))
                rows = []
                for ex, pred, score, flag, response in zip(examples, preds, scores[method], intervened, responses):
                    row = {
                        "seed": seed,
                        "group": ex.group if group != "sleeper_ood" else "sleeper_distributed_trigger",
                        "eval_slice": group,
                        "method": method,
                        "threshold": float(threshold),
                        "threshold_label": _threshold_label(float(threshold)),
                        "example_id": ex.id,
                        "target": ex.target,
                        "judge_pred": pred,
                        "score": float(score),
                        "intervened": bool(flag),
                        "family": ex.family,
                        "mode": ex.mode,
                        "family_holdout": ex.family_holdout,
                        "is_ood": ex.is_ood,
                        "response": response,
                    }
                    rows.append(row)
                    raw_rows.append(row)
                metrics = compute_metrics(rows, base_asr=base_asr)
                metrics.update({"seed": seed, "group": rows[0]["group"], "eval_slice": group, "method": method, "threshold": float(threshold), "threshold_label": _threshold_label(float(threshold)), "defended_perplexity": ppx, "tokens_sec": tok_sec, **sae_stats, **fuser_stats})
                metrics["in_family_asr_reduction"] = metrics["asr_reduction"] if group == "sleeper_distributed_trigger" else 0.0
                metric_rows.append(metrics)
                clean_flags = [flag for ex, flag in zip(examples, intervened) if ex.target == 0]
                attack_flags = [flag for ex, flag in zip(examples, intervened) if ex.target == 1]
                locality_rows.extend([
                    {"seed": seed, "method": method, "prompt_type": "benign", "intervention_rate": float(np.mean(clean_flags)) if clean_flags else 0.0},
                    {"seed": seed, "method": method, "prompt_type": "harmful", "intervention_rate": float(np.mean(attack_flags)) if attack_flags else 0.0},
                    {"seed": seed, "method": method, "prompt_type": "matched_controls", "intervention_rate": float(np.mean(intervened)) if intervened else 0.0},
                ])

    write_jsonl(run_dir / "raw_metrics" / f"raw_generations_seed_{seed}.jsonl", raw_rows)
    write_jsonl(run_dir / "human_eval_samples" / f"samples_seed_{seed}.jsonl", raw_rows[: int(cfg["eval"].get("human_eval_samples", 96))])
    del subject
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    conjunction_rows = conjunction_control_rows(taxonomy_rows)
    null_rows = null_control_rows(taxonomy_rows, seed)
    transfer_summary_rows = transfer_rows(taxonomy_rows)
    return metric_rows, feature_rows, locality_rows, taxonomy_rows, conjunction_rows, causal_rows, null_rows, transfer_summary_rows, layer_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert_runtime_dependencies()
    from plots.appendix_figures import make_appendix_figures
    from plots.latex_tables import export_tables
    from plots.main_figures import make_main_figures
    from plots.mechanistic_figures import make_mechanistic_claim_figures

    run_dir = make_run_dir(cfg.get("output_root", "outputs"))
    logger = setup_logging(run_dir / "logs")
    write_yaml(run_dir / "config_used.yaml", cfg)
    write_json(run_dir / "raw_metrics" / "config_hash.json", {"config_hash": stable_hash(cfg), "python": platform.python_version(), "torch": torch.__version__})

    device = require_full_experiment_device(bool(cfg.get("require_gpu", True)))
    assert_judge_access(cfg["judge_model"])
    datasets, dataset_summary = build_all_datasets(cfg, run_dir, int(cfg["eval"]["seeds"][0]), logger)
    judge = load_judge(cfg["judge_model"], device)
    write_json(run_dir / "raw_metrics" / "judge_sanity.json", sanity_check_judge(*judge))

    all_metrics: List[Dict[str, Any]] = []
    all_features: List[Dict[str, Any]] = []
    all_locality: List[Dict[str, Any]] = []
    all_taxonomy: List[Dict[str, Any]] = []
    all_conjunction: List[Dict[str, Any]] = []
    all_causal: List[Dict[str, Any]] = []
    all_null: List[Dict[str, Any]] = []
    all_transfer: List[Dict[str, Any]] = []
    all_layers: List[Dict[str, Any]] = []
    for seed in cfg["eval"]["seeds"]:
        logger.info("Starting seed %s", seed)
        rows, feats, locality, taxonomy, conjunction, causal, nulls, transfer, layers = run_seed(int(seed), cfg, run_dir, datasets, judge, logger)
        all_metrics.extend(rows)
        all_features.extend(feats)
        all_locality.extend(locality)
        all_taxonomy.extend(taxonomy)
        all_conjunction.extend(conjunction)
        all_causal.extend(causal)
        all_null.extend(nulls)
        all_transfer.extend(transfer)
        all_layers.extend(layers)

    metrics_df = pd.DataFrame(all_metrics)
    feature_df = pd.DataFrame(all_features)
    locality_df = pd.DataFrame(all_locality)
    taxonomy_df = pd.DataFrame(all_taxonomy)
    conjunction_df = pd.DataFrame(all_conjunction)
    causal_df = pd.DataFrame(all_causal)
    null_df = pd.DataFrame(all_null)
    transfer_df = pd.DataFrame(all_transfer)
    layer_df = pd.DataFrame(all_layers)
    sig_df = pd.DataFrame(significance_rows(all_metrics))
    dataset_df = pd.DataFrame(dataset_summary)
    config_df = pd.DataFrame([{"key": k, "value": str(v)} for k, v in cfg.items()])
    metrics_df.to_json(run_dir / "raw_metrics" / "metrics.json", orient="records", indent=2)
    metrics_df.to_csv(run_dir / "raw_metrics" / "metrics.csv", index=False)
    feature_df.to_csv(run_dir / "raw_metrics" / "feature_metrics.csv", index=False)
    locality_df.to_csv(run_dir / "raw_metrics" / "intervention_locality.csv", index=False)
    taxonomy_df.to_csv(run_dir / "raw_metrics" / "mechanistic_taxonomy.csv", index=False)
    conjunction_df.to_csv(run_dir / "raw_metrics" / "conjunction_controls.csv", index=False)
    causal_df.to_csv(run_dir / "raw_metrics" / "causal_interventions.csv", index=False)
    null_df.to_csv(run_dir / "raw_metrics" / "null_controls.csv", index=False)
    transfer_df.to_csv(run_dir / "raw_metrics" / "transfer_results.csv", index=False)
    layer_df.to_csv(run_dir / "raw_metrics" / "layer_summary.csv", index=False)
    export_tables(metrics_df, sig_df, dataset_df, config_df, run_dir / "tables")
    make_main_figures(metrics_df, feature_df, run_dir / "figures")
    make_appendix_figures(metrics_df, feature_df, locality_df, run_dir / "figures")
    make_mechanistic_claim_figures(taxonomy_df, conjunction_df, causal_df, run_dir / "figures")
    logger.info("Run complete: %s", run_dir)


if __name__ == "__main__":
    main()
