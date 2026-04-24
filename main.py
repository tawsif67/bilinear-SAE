from __future__ import annotations

import argparse
import gc
import platform
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None

from data.attack_fingerprints import build_attack_fingerprint_pairs
from data.external_sleeper import load_external_sleeper_benchmark
from data.external_sources import load_external_source_pools
from data.loaders import ConversationExample, normalize_text, subsample_examples
from data.multiturn_jailbreak import load_multiturn_jailbreak
from data.real_attack_corpus import load_real_attack_corpus
from data.sleeper_builder import build_sleeper_dataset
from data.splits import apply_split_labels, deterministic_group_split, deterministic_split, save_splits
from data.wildjailbreak import load_wildjailbreak
from eval.causal_interventions import causal_intervention_rows
from eval.conjunction_tests import conjunction_control_rows
from eval.generate import extract_hidden_trajectories, generate_responses
from eval.judge import assert_judge_access, judge_predictions, load_judge, sanity_check_judge
from eval.layer_sweep import layer_summary_rows
from eval.mechanistic_taxonomy import mechanistic_taxonomy_rows
from eval.metrics import compute_metrics, safe_auprc, safe_auroc
from eval.null_controls import null_control_rows
from eval.significance import significance_rows
from eval.strong_judge import export_strong_judge_requests, run_configured_strong_adjudication, strong_judge_status
from eval.transfer import transfer_rows
from models.baselines import ActivationProbe, DenseLinearProbe, mean_difference_vector, top_mean_difference_dims
from models.baselines import TorchMLPProbe
from models.arf import (
    arf_diagnostic_rows,
    arf_significance_rows,
    extract_attack_residuals,
    extract_deployable_attack_features,
    extract_global_control_baseline,
    FingerprintClassifier,
    fingerprint_metric_rows,
    fingerprint_metric_rows_with_scores,
    lexical_baseline_rows,
    sparse_features_for_residuals,
)
from models.ctcr import build_ctcr_residual_dataset, ctcr_analysis_rows, ctcr_scores_for_examples
from models.subject import SubjectModel, require_full_experiment_device
from train.train_ctcr import train_ctcr_sae
from train.train_arf import train_arf_sae_classifier
from train.train_fuser import train_fuser
from train.train_lora import train_lora
from train.train_sae import train_saes
from utils.config_utils import load_config
from utils.dependencies import assert_runtime_dependencies
from utils.hf_auth import configure_hf_auth
from utils.gpu_memory import clear_cuda_cache, cuda_memory_snapshot, is_cuda_oom, log_cuda_memory, tune_batch_sizes_for_memory
from utils.io import append_jsonl, make_run_dir, stable_hash, write_json, write_jsonl, write_yaml
from utils.logging_utils import setup_logging
from utils.seed import set_seed


DEPLOYMENT_METHODS = [
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

ANALYSIS_ONLY_METHODS = ["ctcr_residual_bilinear"]

METHOD_MANIFEST = [
    {"method": method, "category": "deployable_detector", "test_time_forward_passes": 1, "uses_matched_controls_at_test": False}
    for method in DEPLOYMENT_METHODS
] + [
    {
        "method": "ctcr_residual_bilinear",
        "category": "mechanistic_diagnostic_only",
        "test_time_forward_passes": 4,
        "uses_matched_controls_at_test": True,
        "note": "CTCR requires pre-specified A/B/0 matched controls and is not treated as a deployment detector.",
    }
]


def _last_user_text(ex: ConversationExample) -> str:
    for turn in reversed(ex.turns):
        if turn.get("role") == "user":
            text = normalize_text(turn.get("content", ""))
            if text:
                return text
    return normalize_text(ex.prompt_text)


def _base_prompt_record(ex: ConversationExample) -> Dict[str, Any]:
    meta = ex.metadata or {}
    return {
        "text": _last_user_text(ex),
        "base_source": ex.source,
        "base_example_id": ex.id,
        "base_group": ex.group,
        "base_family": ex.family,
        "base_mode": ex.mode,
        "base_target": ex.target,
        "base_data_type": meta.get("data_type", ""),
        "base_tactics": meta.get("tactics", ""),
        "base_row_id": meta.get("row_id", ""),
        "base_source_split": meta.get("source_split", ""),
        "base_goal_id": meta.get("goal_id", ""),
    }


def _real_corpus_summary(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    topics = Counter(str(row.get("base_topic", "")) for row in rows)
    sources = Counter(str(row.get("base_original_source", "")) for row in rows)
    return {
        "dataset_id": cfg.get("real_attack_corpus", {}).get("dataset_id", "mvrcii/safety-harmful"),
        "split": cfg.get("real_attack_corpus", {}).get("split", "train"),
        "n_records": len(rows),
        "topic_counts": dict(topics),
        "original_source_counts": dict(sources),
        "usage": "large real harmful prompt source pool for ARF controlled counterfactual transformations",
        "note": "Rows are not all passed through the model; ARF samples from this real pool to control runtime.",
    }


def _write_synthetic_dataset_artifacts(run_dir: Path, name: str, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    out_dir = run_dir / "synthetic_datasets"
    write_jsonl(out_dir / f"{name}.jsonl", rows)
    write_json(out_dir / f"{name}_summary.json", summary)


def _write_synthetic_audit_exports(run_dir: Path, sleeper_examples: List[ConversationExample] | None = None, arf_pairs=None, limit_per_family: int = 50) -> None:
    out_dir = run_dir / "synthetic_datasets"
    if sleeper_examples:
        rows = []
        counts: Dict[str, int] = defaultdict(int)
        for ex in sleeper_examples:
            key = f"{ex.family}:{ex.mode}"
            if counts[key] >= max(1, limit_per_family // 4):
                continue
            counts[key] += 1
            rows.append({
                "dataset": "constructed_sleeper",
                "id": ex.id,
                "family": ex.family,
                "mode": ex.mode,
                "target": ex.target,
                "split": ex.split,
                "source_is_benign": ex.source_is_benign,
                "condition_order": (ex.metadata or {}).get("condition_order", ""),
                "residual_group": (ex.metadata or {}).get("residual_group", ""),
                "prompt_text": ex.prompt_text,
            })
        pd.DataFrame(rows).to_csv(out_dir / "constructed_sleeper_audit.csv", index=False)
    if arf_pairs is not None:
        rows = []
        counts: Dict[str, int] = defaultdict(int)
        for pair in arf_pairs:
            if counts[pair.family] >= limit_per_family:
                continue
            counts[pair.family] += 1
            rows.append({
                "dataset": "attack_residual_fingerprints",
                "id": pair.id,
                "family": pair.family,
                "split": pair.split,
                "template_holdout": bool(pair.metadata.get("template_holdout", False)),
                "residual_type": pair.metadata.get("residual_type", ""),
                "control_type": pair.metadata.get("control_type", ""),
                "base_source": (pair.metadata.get("base_harmful") or {}).get("base_source", ""),
                "base_example_id": (pair.metadata.get("base_harmful") or {}).get("base_example_id", ""),
                "attack_prompt": pair.attack_prompt,
                "control_prompts": "\n---CONTROL---\n".join(pair.control_prompts),
            })
        pd.DataFrame(rows).to_csv(out_dir / "attack_residual_fingerprints_audit.csv", index=False)


def _sleeper_synthetic_summary(examples: List[ConversationExample]) -> Dict[str, Any]:
    mode_counts = Counter(ex.mode for ex in examples)
    family_counts = Counter(ex.family for ex in examples)
    groups: Dict[str, set] = defaultdict(set)
    for ex in examples:
        group = str((ex.metadata or {}).get("residual_group", ""))
        if group:
            groups[group].add(ex.mode)
    required = {
        "both_conditions",
        "only_condition_a",
        "only_condition_b",
        "neither_condition",
        "b_then_a",
        "same_turn_ab",
        "a_gap_b",
        "a_interrupt_b",
        "a_reset_b",
        "decoy_a_then_b",
        "a_then_decoy_b",
        "a_long_gap_b",
        "shuffled_turns",
        "single_turn_compressed",
    }
    complete_groups = sum(1 for modes in groups.values() if required.issubset(modes))
    return {
        "source": "constructed_sleeper",
        "n_examples": len(examples),
        "n_residual_groups": len(groups),
        "n_complete_sequence_control_groups": complete_groups,
        "mode_counts": dict(mode_counts),
        "family_counts": dict(family_counts),
        "required_modes": sorted(required),
        "robustness_design": [
            "deterministic seed-controlled generation",
            "matched request/condition/trigger metadata within residual_group",
            "group-preserving train/val/test splits",
            "canonical A-then-B attack plus A-only, B-only, neither controls",
            "temporal controls: reverse, same-turn, gap, interrupt, reset, long-gap, shuffled, compressed",
            "semantic controls: decoy-A and decoy-B",
            "redacted/proxy unsafe-compliance targets; no harmful completions are generated",
        ],
    }


def _arf_synthetic_summary(pairs) -> Dict[str, Any]:
    family_counts = Counter(pair.family for pair in pairs)
    split_counts = Counter(pair.split for pair in pairs)
    control_counts = Counter(len(pair.control_prompts) for pair in pairs)
    residual_types = Counter(str(pair.metadata.get("residual_type", "")) for pair in pairs)
    constructions = Counter(str(pair.metadata.get("construction", "")) for pair in pairs)
    base_sources = Counter(str((pair.metadata.get("base_harmful") or {}).get("base_source", "")) for pair in pairs)
    dataset_keys = Counter(str((pair.metadata.get("base_harmful") or {}).get("base_dataset_key", "")) for pair in pairs)
    original_sources = Counter(str((pair.metadata.get("base_harmful") or {}).get("base_original_source", "")) for pair in pairs)
    topics = Counter(str((pair.metadata.get("base_harmful") or {}).get("base_topic", "")) for pair in pairs)
    return {
        "source": "constructed_attack_residual_fingerprints",
        "n_pairs": len(pairs),
        "family_counts": dict(family_counts),
        "split_counts": dict(split_counts),
        "control_count_distribution": dict(control_counts),
        "residual_type_counts": dict(residual_types),
        "construction_counts": dict(constructions),
        "base_harmful_source_counts": dict(base_sources),
        "base_harmful_dataset_key_counts": dict(dataset_keys),
        "base_harmful_original_source_counts": dict(original_sources),
        "base_harmful_topic_counts": dict(topics),
        "robustness_design": [
            "matched attack/control prompts per family derived from real loaded prompt pools when available",
            "constructed_sleeper examples are excluded from ARF base prompt pools to avoid synthetic-on-synthetic provenance",
            "base_source/base_example_id/base_tactic provenance is saved in each pair metadata",
            "multiple template variants for roleplay, policy override, and refusal suppression",
            "obfuscation controls separate decode pressure from harmful compliance",
            "sleeper sequence pairs include reverse, same-turn, decoy, reset, and long-gap controls",
            "deterministic generation from seed and saved source_index metadata",
            "compact residuals are computed as attack hidden state minus mean matched control hidden state",
            "bag-of-words lexical baseline is exported to test whether attack family is trivially recoverable from surface text",
            "template-holdout variants are forced into test where configured",
        ],
    }


def _synthetic_validation_rows(sleeper_examples: List[ConversationExample], arf_pairs=None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sleeper_required = {
        "both_conditions",
        "only_condition_a",
        "only_condition_b",
        "neither_condition",
        "b_then_a",
        "same_turn_ab",
        "a_gap_b",
        "a_interrupt_b",
        "a_reset_b",
        "decoy_a_then_b",
        "a_then_decoy_b",
        "a_long_gap_b",
        "shuffled_turns",
        "single_turn_compressed",
    }
    if sleeper_examples:
        by_group: Dict[str, set] = defaultdict(set)
        for ex in sleeper_examples:
            key = str((ex.metadata or {}).get("residual_group", ""))
            if key:
                by_group[key].add(ex.mode)
        complete = sum(1 for modes in by_group.values() if sleeper_required.issubset(modes))
        duplicate_rate = 1.0 - (len({ex.prompt_text for ex in sleeper_examples}) / max(len(sleeper_examples), 1))
        rows.append({
            "dataset": "constructed_sleeper",
            "check": "complete_sequence_control_groups",
            "value": complete,
            "expected": len(by_group),
            "passed": complete == len(by_group),
        })
        rows.append({
            "dataset": "constructed_sleeper",
            "check": "duplicate_prompt_rate",
            "value": duplicate_rate,
            "expected": 0.05,
            "passed": duplicate_rate <= 0.05,
        })
    if arf_pairs is not None:
        pairs = list(arf_pairs)
        n = max(len(pairs), 1)
        real_count = sum(1 for pair in pairs if str(pair.metadata.get("construction", "")).startswith("real_prompt_derived"))
        large_real_count = sum(1 for pair in pairs if (pair.metadata.get("base_harmful") or {}).get("base_source") == "mvrcii/safety-harmful")
        external_count = sum(1 for pair in pairs if str((pair.metadata.get("base_harmful") or {}).get("base_dataset_key", "")) in {"jailbreakbench", "advbench", "lakera_gandalf", "deepset_prompt_injections"})
        holdout_count = sum(1 for pair in pairs if bool(pair.metadata.get("template_holdout", False)))
        rows.extend([
            {
                "dataset": "attack_residual_fingerprints",
                "check": "real_prompt_derived_rate",
                "value": real_count / n,
                "expected": 0.95,
                "passed": (real_count / n) >= 0.95,
            },
            {
                "dataset": "attack_residual_fingerprints",
                "check": "template_holdout_pairs",
                "value": holdout_count,
                "expected": 1,
                "passed": holdout_count >= 1,
            },
            {
                "dataset": "attack_residual_fingerprints",
                "check": "large_real_corpus_derived_pairs",
                "value": large_real_count,
                "expected": 1,
                "passed": large_real_count >= 1,
            },
            {
                "dataset": "attack_residual_fingerprints",
                "check": "external_dataset_derived_pairs",
                "value": external_count,
                "expected": 1,
                "passed": external_count >= 1,
            },
            {
                "dataset": "attack_residual_fingerprints",
                "check": "all_pairs_have_controls",
                "value": sum(1 for pair in pairs if len(pair.control_prompts) > 0),
                "expected": len(pairs),
                "passed": all(len(pair.control_prompts) > 0 for pair in pairs),
            },
        ])
    return rows


def _targets(examples: List[ConversationExample]) -> torch.Tensor:
    return torch.tensor([ex.target for ex in examples], dtype=torch.long)


def _residual_group_integrity_rows(examples: List[ConversationExample], split_name: str, seed: int) -> List[Dict[str, Any]]:
    required = {"both_conditions", "only_condition_a", "only_condition_b", "neither_condition"}
    groups: Dict[str, set] = defaultdict(set)
    for ex in examples:
        group = str((ex.metadata or {}).get("residual_group", ""))
        if group:
            groups[group].add(ex.mode)
    rows = []
    for group, modes in sorted(groups.items()):
        rows.append({
            "seed": seed,
            "split": split_name,
            "residual_group": group,
            "n_modes": len(modes),
            "complete_ctcr_group": required.issubset(modes),
            "missing_modes": ",".join(sorted(required - modes)),
        })
    if groups:
        rows.append({
            "seed": seed,
            "split": split_name,
            "residual_group": "__summary__",
            "n_groups": len(groups),
            "complete_groups": sum(1 for modes in groups.values() if required.issubset(modes)),
            "complete_ctcr_group": all(required.issubset(modes) for modes in groups.values()),
            "missing_modes": "",
        })
    return rows


def _split_and_save(run_dir: Path, name: str, examples: List[ConversationExample], ratios, seed: int, stratify_key: str | None, group_key: str | None = None) -> List[ConversationExample]:
    if group_key:
        splits = deterministic_group_split(examples, ratios, seed, group_key=group_key, stratify_key=stratify_key)
    else:
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
    sleeper = _split_and_save(run_dir, "constructed_sleeper", build_sleeper_dataset(cfg, seed, benign, harmful), (0.60, 0.20, 0.20), seed, "family", group_key="residual_group")
    _write_synthetic_dataset_artifacts(
        run_dir,
        "constructed_sleeper_dataset",
        [ex.to_dict() for ex in sleeper],
        _sleeper_synthetic_summary(sleeper),
    )
    pd.DataFrame(_synthetic_validation_rows(sleeper)).to_csv(run_dir / "synthetic_datasets" / "constructed_sleeper_validation.csv", index=False)
    _write_synthetic_audit_exports(run_dir, sleeper_examples=sleeper, limit_per_family=int(cfg.get("data", {}).get("audit_examples_per_family", 50)))
    summary = [
        {"group": "ordinary_harmful_benign", "source": "allenai/wildjailbreak", "n": len(wild), "note": "Gated WildJailbreak; train/eval configs adapted by data_type."},
        {"group": "ordinary_multiturn_jailbreak", "source": cfg.get("multiturn_dataset", "ScaleAI/mhj"), "n": len(multi), "note": note},
        {
            "group": "sleeper_distributed_trigger",
            "source": "constructed_sleeper",
            "n": len(sleeper),
            "note": "Programmatic sleeper-style distributed triggers; analysis/development data only, not sufficient for the main external-validity claim.",
        },
    ]
    external_sleeper, external_sleeper_summary, external_sleeper_errors = load_external_sleeper_benchmark(cfg)
    if external_sleeper:
        external_sleeper = _split_and_save(run_dir, "external_sleeper_validation", external_sleeper, (0.0, 0.20, 0.80), seed, "family")
    write_jsonl(run_dir / "synthetic_datasets" / "external_sleeper_validation.jsonl", [ex.to_dict() for ex in external_sleeper])
    write_json(run_dir / "synthetic_datasets" / "external_sleeper_validation_summary.json", external_sleeper_summary)
    write_jsonl(run_dir / "synthetic_datasets" / "external_sleeper_validation_errors.jsonl", external_sleeper_errors)
    summary.append({
        "group": "external_sleeper_validation",
        "source": external_sleeper_summary.get("source", "not_configured"),
        "n": len(external_sleeper),
        "note": external_sleeper_summary.get("note", "external sleeper validation status unknown"),
    })
    real_attack_corpus = []
    external_attack_pool: List[Dict[str, Any]] = []
    external_benign_pool: List[Dict[str, Any]] = []
    if bool(cfg.get("real_attack_corpus", {}).get("enabled", True)):
        logger.info("Loading large real attack corpus for ARF source pool.")
        real_attack_corpus = load_real_attack_corpus(cfg)
        write_jsonl(run_dir / "synthetic_datasets" / "real_attack_corpus_source_pool.jsonl", real_attack_corpus)
        write_json(run_dir / "synthetic_datasets" / "real_attack_corpus_source_pool_summary.json", _real_corpus_summary(real_attack_corpus, cfg))
        summary.append({
            "group": "real_attack_corpus_source_pool",
            "source": cfg.get("real_attack_corpus", {}).get("dataset_id", "mvrcii/safety-harmful"),
            "n": len(real_attack_corpus),
            "note": "Large real harmful prompt pool used to derive ARF counterfactual transformations; not merged into main benchmark metrics.",
        })
    if bool(cfg.get("external_datasets", {}).get("enabled", True)):
        logger.info("Loading external jailbreak/prompt-injection source pools.")
        external_attack_pool, external_benign_pool, external_summary, external_errors = load_external_source_pools(cfg)
        write_jsonl(run_dir / "synthetic_datasets" / "external_attack_source_pool.jsonl", external_attack_pool)
        write_jsonl(run_dir / "synthetic_datasets" / "external_benign_source_pool.jsonl", external_benign_pool)
        write_jsonl(run_dir / "synthetic_datasets" / "external_dataset_summary.jsonl", external_summary)
        write_jsonl(run_dir / "synthetic_datasets" / "external_dataset_errors.jsonl", external_errors)
        for row in external_summary:
            summary.append({
                "group": f"external_source_pool:{row['dataset']}",
                "source": row["dataset_id"],
                "n": row["n_rows"],
                "note": f"External {row['kind']} source pool for ARF transformations; errors={row['errors']}.",
            })
    return {
        "wild": wild,
        "multi": multi,
        "sleeper": sleeper,
        "external_sleeper": external_sleeper,
        "real_attack_corpus": real_attack_corpus,
        "external_attack_pool": external_attack_pool,
        "external_benign_pool": external_benign_pool,
    }, summary


def _balanced_subsample(groups: List[List[ConversationExample]], budget: int) -> List[ConversationExample]:
    nonempty = [list(group) for group in groups if group]
    if budget <= 0:
        return [ex for group in nonempty for ex in group]
    if not nonempty:
        return []

    def take(group: List[ConversationExample], limit: int) -> tuple[List[ConversationExample], List[ConversationExample]]:
        if not group or limit <= 0:
            return [], group
        if not any((ex.metadata or {}).get("residual_group") for ex in group):
            return group[:limit], group[limit:]
        selected: List[ConversationExample] = []
        leftovers: List[ConversationExample] = []
        by_group: Dict[str, List[ConversationExample]] = {}
        for i, ex in enumerate(group):
            key = str((ex.metadata or {}).get("residual_group") or f"ungrouped:{i}")
            by_group.setdefault(key, []).append(ex)
        for block in by_group.values():
            if len(selected) + len(block) <= limit or not selected:
                selected.extend(block)
            else:
                leftovers.extend(block)
        return selected, leftovers

    per_group = max(1, budget // len(nonempty))
    selected: List[ConversationExample] = []
    leftovers: List[ConversationExample] = []
    for group in nonempty:
        head, tail = take(group, per_group)
        selected.extend(head)
        leftovers.extend(tail)
    remaining = max(0, budget - len(selected))
    extra, _ = take(leftovers, remaining)
    selected.extend(extra)
    return selected


def _select_eval_examples(datasets: Dict[str, List[ConversationExample]], cfg: Dict[str, Any]) -> Dict[str, List[ConversationExample]]:
    benchmark_groups = [datasets.get("wild", []), datasets.get("multi", []), datasets.get("sleeper", [])]
    train_groups = [[ex for ex in group if ex.split == "train"] for group in benchmark_groups]
    val_groups = [[ex for ex in group if ex.split == "val"] for group in benchmark_groups]
    train = _balanced_subsample(train_groups, int(cfg["train_subsample"]))
    val = _balanced_subsample(val_groups, int(cfg["val_subsample"]))
    test_groups = {
        "ordinary_harmful_benign": [ex for ex in datasets["wild"] if ex.split == "test"],
        "ordinary_multiturn_jailbreak": [ex for ex in datasets["multi"] if ex.split == "test"],
        "sleeper_distributed_trigger": [ex for ex in datasets["sleeper"] if ex.split == "test" and not ex.is_ood],
        "sleeper_ood": [ex for ex in datasets["sleeper"] if ex.is_ood],
        "external_sleeper_validation": [ex for ex in datasets.get("external_sleeper", []) if ex.split in {"val", "test"}],
    }
    external_attack_rows = [row for row in datasets.get("external_attack_pool", []) if isinstance(row, dict)]
    external_sources = sorted({str(row.get("base_dataset_key", "")) for row in external_attack_rows if row.get("base_dataset_key")})
    return {
        "train": train,
        "val": val,
        "real_attack_corpus": list(datasets.get("real_attack_corpus", [])),
        "external_attack_pool": list(datasets.get("external_attack_pool", [])),
        "external_benign_pool": list(datasets.get("external_benign_pool", [])),
        "external_attack_by_source": {
            key: [row for row in external_attack_rows if row.get("base_dataset_key") == key]
            for key in external_sources
        },
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
def score_methods(h_seq: torch.Tensor, models: Dict[str, Any], y: torch.Tensor, examples: List[ConversationExample] | None = None) -> Dict[str, np.ndarray]:
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
    ctcr_valid = np.zeros(h_seq.size(0), dtype=bool)
    if examples is not None and any((ex.metadata or {}).get("residual_group") for ex in examples) and "ctcr_sae" in models and "ctcr_bad_feats" in models:
        ctcr, _ = ctcr_scores_for_examples(examples, h_seq, models["ctcr_sae"], models["ctcr_bad_feats"])
        out["ctcr_residual_bilinear"] = ctcr.detach().cpu().numpy()
        ctcr_valid = np.array([bool((ex.metadata or {}).get("residual_group")) for ex in examples], dtype=bool)
    else:
        out["ctcr_residual_bilinear"] = np.zeros(h_seq.size(0), dtype=float)
    out["ctcr_valid_mask"] = ctcr_valid
    out["full_fused"] = models["fuser"](s1, s2, traj).detach().cpu().numpy()
    return out


def _threshold_label(t: float) -> str:
    return "inf" if np.isinf(t) else f"{t:.2f}"


def _intervention_flags(scores: np.ndarray, threshold: float) -> List[bool]:
    if np.isinf(threshold):
        return [False for _ in scores]
    return (scores > threshold).tolist()


def _defended_perplexity(subject, cfg: Dict[str, Any], logger=None) -> float:
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
    batch_size = max(1, int(cfg["eval"]["batch_size"]))
    with torch.inference_mode():
        i = 0
        while i < len(texts):
            cur = min(batch_size, len(texts) - i)
            bx = enc["input_ids"][i:i + cur].to(subject.device)
            bm = enc["attention_mask"][i:i + cur].to(subject.device)
            try:
                logits = subject.model(input_ids=bx, attention_mask=bm).logits
                loss = ce(logits[:, :-1].contiguous().view(-1, logits.size(-1)), bx[:, 1:].contiguous().view(-1))
                loss = loss.view(bx.size(0), -1)
                total_loss += (loss * bm[:, 1:]).sum().item()
                total_toks += bm[:, 1:].sum().item()
                i += cur
            except RuntimeError as e:
                if not is_cuda_oom(e) or batch_size <= 1:
                    snap = cuda_memory_snapshot(subject.device)
                    raise RuntimeError(f"Perplexity evaluation failed at batch_size={batch_size}; GPU memory snapshot={snap}") from e
                batch_size = max(1, batch_size // 2)
                clear_cuda_cache(subject.device)
                if logger is not None:
                    logger.warning("Perplexity OOM; retrying with eval batch_size=%d.", batch_size)
                log_cuda_memory(logger, "after_perplexity_oom_retry", subject.device)
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


def _run_attack_fingerprints(
    subject,
    cfg: Dict[str, Any],
    selected: Dict[str, List[ConversationExample]],
    intercept_layer: int,
    seed: int,
    run_dir: Path,
    logger,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not bool(cfg.get("attack_fingerprints", {}).get("enabled", True)):
        return [], []
    train_examples = selected.get("train", [])
    public_train_examples = [ex for ex in train_examples if ex.source != "constructed_sleeper"]
    benign_pool = list(row for row in selected.get("external_benign_pool", []) if isinstance(row, dict))
    benign_pool.extend(_base_prompt_record(ex) for ex in public_train_examples if (ex.source_is_benign or ex.target == 0) and _last_user_text(ex))
    real_attack_rows = [row for row in selected.get("real_attack_corpus", []) if isinstance(row, dict)]
    external_attack_rows = [row for row in selected.get("external_attack_pool", []) if isinstance(row, dict)]
    harmful_pool = list(real_attack_rows)
    harmful_pool.extend(external_attack_rows)
    harmful_pool.extend(_base_prompt_record(ex) for ex in public_train_examples if ex.target == 1 and _last_user_text(ex))
    pairs = build_attack_fingerprint_pairs(cfg, seed, benign_pool, harmful_pool)
    if not pairs:
        return [], []
    _write_synthetic_dataset_artifacts(
        run_dir,
        f"attack_residual_fingerprints_seed_{seed}",
        [pair.to_dict() for pair in pairs],
        _arf_synthetic_summary(pairs),
    )
    pd.DataFrame(_synthetic_validation_rows([], pairs)).to_csv(run_dir / "synthetic_datasets" / f"attack_residual_fingerprints_seed_{seed}_validation.csv", index=False)
    _write_synthetic_audit_exports(run_dir, arf_pairs=pairs, limit_per_family=int(cfg.get("data", {}).get("audit_examples_per_family", 50)))
    for source_key, source_rows in selected.get("external_attack_by_source", {}).items():
        if not source_rows:
            continue
        source_pairs = build_attack_fingerprint_pairs(cfg, seed + 1009, benign_pool, source_rows)
        _write_synthetic_dataset_artifacts(
            run_dir,
            f"attack_residual_fingerprints_external_{source_key}_seed_{seed}",
            [pair.to_dict() for pair in source_pairs],
            _arf_synthetic_summary(source_pairs),
        )
    train_pairs = [pair for pair in pairs if pair.split == "train"]
    val_pairs = [pair for pair in pairs if pair.split == "val"]
    test_pairs = [pair for pair in pairs if pair.split == "test"]
    template_holdout_pairs = [pair for pair in test_pairs if bool(pair.metadata.get("template_holdout", False))]
    train_resid, train_labels, train_raw = extract_attack_residuals(subject, train_pairs, cfg, intercept_layer, logger)
    train_resid = train_resid.to(subject.device).float()
    arf_sae, arf_clf, arf_stats = train_arf_sae_classifier(train_resid, train_labels, cfg, subject.hidden_size, run_dir / "checkpoints", seed, logger)
    arf_sae = arf_sae.to(subject.device).eval()
    global_control_baseline = extract_global_control_baseline(subject, train_pairs, cfg, intercept_layer, logger).to(subject.device).float()
    x_train_single, train_single_labels, train_single_raw = extract_deployable_attack_features(
        subject, train_pairs, cfg, intercept_layer, arf_sae, global_control_baseline, logger
    )
    label_to_id = {name: i for i, name in enumerate(arf_clf.families)}
    y_train_single = np.array([label_to_id[label] for label in train_single_labels if label in label_to_id], dtype=int)
    if len(y_train_single) != len(train_single_labels):
        raise RuntimeError("ARF deployable classifier saw a training label outside fitted family set.")
    deploy_clf = FingerprintClassifier(families=arf_clf.families)
    deploy_clf.fit(x_train_single, y_train_single)
    rows: List[Dict[str, Any]] = []
    lexical_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = [{**row, "seed": seed, "method": "arf_sae_matched_residual"} for row in train_raw]
    raw_rows.extend({**row, "seed": seed, "method": "arf_sae_single_pass"} for row in train_single_raw)
    for split, split_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs), ("template_holdout", template_holdout_pairs)]:
        if not split_pairs:
            continue
        if split == "train":
            resid, labels = train_resid, train_labels
            x_single, single_labels = x_train_single, train_single_labels
        else:
            resid, labels, raw = extract_attack_residuals(subject, split_pairs, cfg, intercept_layer, logger)
            resid = resid.to(subject.device).float()
            raw_rows.extend({**row, "seed": seed, "method": "arf_sae_matched_residual"} for row in raw)
            x_single, single_labels, raw_single = extract_deployable_attack_features(
                subject, split_pairs, cfg, intercept_layer, arf_sae, global_control_baseline, logger
            )
            raw_rows.extend({**row, "seed": seed, "method": "arf_sae_single_pass"} for row in raw_single)
        x = sparse_features_for_residuals(arf_sae, resid)
        for row in fingerprint_metric_rows(arf_clf, x, labels, split, seed):
            row.update(arf_stats)
            row["method"] = "arf_sae_matched_residual_analysis"
            rows.append(row)
        for row in fingerprint_metric_rows_with_scores(deploy_clf, x_single, single_labels, split, seed):
            row.update(arf_stats)
            row["method"] = "arf_sae_single_pass"
            row["uses_matched_controls_at_test"] = False
            rows.append(row)
        lexical_rows.extend(lexical_baseline_rows(train_pairs, split_pairs, split, seed, int(cfg.get("attack_fingerprints", {}).get("lexical_max_features", 2048))))
        if split != "train":
            del resid
            clear_cuda_cache(subject.device)
    del train_resid
    clear_cuda_cache(subject.device)
    lexical_out = []
    for row in lexical_rows:
        baseline = str(row.get("baseline", "lexical"))
        method = "lexical_baseline" if baseline == "lexical_best" else f"lexical_{baseline}"
        lexical_out.append({**row, "family": "all", "arf_mse": 0.0, "arf_l1": 0.0, "method": method})
    return rows + lexical_out, raw_rows


def run_seed(seed: int, cfg: Dict[str, Any], run_dir: Path, datasets: Dict[str, List[ConversationExample]], logger):
    set_seed(seed)
    selected = _select_eval_examples(datasets, cfg)
    device = require_full_experiment_device(bool(cfg.get("require_gpu", True)))
    cfg = tune_batch_sizes_for_memory(cfg, logger, device)
    subject = SubjectModel(cfg, device)
    log_cuda_memory(logger, f"after_subject_load_seed_{seed}", device)
    write_json(run_dir / "raw_metrics" / f"gpu_memory_after_subject_seed_{seed}.json", cuda_memory_snapshot(device))
    cfg = tune_batch_sizes_for_memory(cfg, logger, device)
    write_yaml(run_dir / f"config_runtime_seed_{seed}.yaml", cfg)
    lora_train = selected["train"][: int(cfg.get("train_subsample", len(selected["train"])))]
    train_lora(subject, lora_train, cfg, seed, run_dir / "checkpoints", logger)
    intercept_layer = int(cfg["intercept_layers"][0])

    train_h, _ = extract_hidden_trajectories(subject, selected["train"], cfg, intercept_layer, logger)
    val_h, _ = extract_hidden_trajectories(subject, selected["val"], cfg, intercept_layer, logger)
    train_h = train_h.to(device).float()
    val_h = val_h.to(device).float()
    train_y = _targets(selected["train"]).to(device)
    val_y = _targets(selected["val"]).to(device)

    linear_sae, bilinear_sae, sae_stats = train_saes(train_h, cfg, subject.hidden_size, run_dir / "checkpoints", seed, logger)
    ctcr_residuals, ctcr_targets, ctcr_train_rows = build_ctcr_residual_dataset(selected["train"], train_h)
    ctcr_sae, ctcr_stats = train_ctcr_sae(ctcr_residuals, cfg, subject.hidden_size, run_dir / "checkpoints", seed, logger)
    h_bad = train_h[train_y == 1, -1, :]
    h_clean = train_h[train_y == 0, -1, :]
    with torch.inference_mode():
        bad_feats = {
            "linear": _rank_features(linear_sae.get_sparse_acts(h_bad).cpu().numpy(), linear_sae.get_sparse_acts(h_clean).cpu().numpy(), device),
            "bilinear": _rank_features(bilinear_sae.get_sparse_acts(h_bad).cpu().numpy(), bilinear_sae.get_sparse_acts(h_clean).cpu().numpy(), device),
        }
        ctcr_bad = ctcr_residuals[ctcr_targets == 1]
        ctcr_clean = ctcr_residuals[ctcr_targets == 0]
        ctcr_bad_feats = _rank_features(ctcr_sae.get_sparse_acts(ctcr_bad).cpu().numpy(), ctcr_sae.get_sparse_acts(ctcr_clean).cpu().numpy(), device)
    # _rank_features() runs under inference_mode here, so clone the resulting
    # index tensors after leaving the context before reusing them elsewhere.
    bad_feats = {name: feats.clone() for name, feats in bad_feats.items()}
    ctcr_bad_feats = ctcr_bad_feats.clone()
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
        "ctcr_sae": ctcr_sae,
        "ctcr_bad_feats": ctcr_bad_feats,
    }

    metric_rows: List[Dict[str, Any]] = []
    raw_path = run_dir / "raw_metrics" / f"raw_generations_seed_{seed}.jsonl"
    write_jsonl(raw_path, [])
    save_raw_threshold_rows = bool(cfg.get("eval", {}).get("save_raw_threshold_rows", False))
    human_rows: List[Dict[str, Any]] = []
    human_limit = int(cfg["eval"].get("human_eval_samples", 96))
    pending_judge: List[Dict[str, Any]] = []
    feature_rows = _feature_rows(seed, linear_sae, bilinear_sae, train_h, train_y, bad_feats, sae_stats)
    locality_rows = []
    taxonomy_rows: List[Dict[str, Any]] = []
    causal_rows: List[Dict[str, Any]] = []
    ctcr_rows: List[Dict[str, Any]] = []
    for row in ctcr_train_rows:
        ctcr_rows.append({**row, "seed": seed, "eval_slice": "train", "group": "sleeper_distributed_trigger", "method": "ctcr_residual_bilinear"})
    pd.DataFrame(
        _residual_group_integrity_rows(selected["train"], "train", seed)
        + _residual_group_integrity_rows(selected["val"], "val", seed)
    ).to_csv(run_dir / "raw_metrics" / f"residual_group_integrity_seed_{seed}.csv", index=False)
    layer_rows = layer_summary_rows(train_h, train_y, seed, intercept_layer, "train")
    arf_rows, arf_raw_rows = _run_attack_fingerprints(subject, cfg, selected, intercept_layer, seed, run_dir, logger)
    ppx = _defended_perplexity(subject, cfg, logger)

    for group, examples in selected.items():
        if group in {"train", "val"} or not examples:
            continue
        if not isinstance(examples, list) or not isinstance(examples[0], ConversationExample):
            continue
        eval_h, _ = extract_hidden_trajectories(subject, examples, cfg, intercept_layer, logger)
        eval_h = eval_h.to(device).float()
        eval_y = _targets(examples).to(device)
        scores = score_methods(eval_h, model_bundle, eval_y, examples)
        taxonomy_rows.extend(mechanistic_taxonomy_rows(examples, scores, seed, group))
        ctcr_rows.extend(ctcr_analysis_rows(examples, eval_h, ctcr_sae, ctcr_bad_feats, seed, group))
        integrity = _residual_group_integrity_rows(examples, group, seed)
        if integrity:
            pd.DataFrame(integrity).to_csv(run_dir / "raw_metrics" / f"residual_group_integrity_{group}_seed_{seed}.csv", index=False)
        causal_rows.extend(causal_intervention_rows(examples, eval_h, linear_sae, bilinear_sae, bad_feats, seed, group))
        layer_rows.extend(layer_summary_rows(eval_h, eval_y, seed, intercept_layer, group))
        base_responses, tok_sec = generate_responses(subject, examples, cfg, logger)
        pending_judge.append({
            "group": group,
            "examples": examples,
            "scores": {k: v for k, v in scores.items() if k in DEPLOYMENT_METHODS},
            "base_responses": base_responses,
            "tokens_sec": tok_sec,
        })
        del eval_h, eval_y, scores
        gc.collect()
        clear_cuda_cache(device)

    del subject, model_bundle, linear_sae, bilinear_sae, ctcr_sae, trajectory, fuser
    del train_h, val_h, train_y, val_y, h_bad, h_clean
    del ctcr_residuals, ctcr_targets, ctcr_bad, ctcr_clean, ctcr_bad_feats, bad_feats, mean_vec, top_dims
    gc.collect()
    clear_cuda_cache(device)
    log_cuda_memory(logger, f"after_subject_unload_seed_{seed}", device)
    write_json(run_dir / "raw_metrics" / f"gpu_memory_after_subject_unload_seed_{seed}.json", cuda_memory_snapshot(device))

    judge = load_judge(cfg["judge_model"], device)
    processor, judge_model = judge
    log_cuda_memory(logger, f"after_judge_load_seed_{seed}", device)
    write_json(run_dir / "raw_metrics" / f"gpu_memory_after_judge_seed_{seed}.json", cuda_memory_snapshot(device))
    write_json(run_dir / "raw_metrics" / f"judge_sanity_seed_{seed}.json", sanity_check_judge(processor, judge_model))
    judge_batch_size = int(cfg["eval"].get("judge_batch_size", min(4, int(cfg["eval"]["batch_size"]))))

    for pending in pending_judge:
        group = str(pending["group"])
        examples = pending["examples"]
        base_responses = pending["base_responses"]
        scores = pending["scores"]
        tok_sec = float(pending["tokens_sec"])
        log_cuda_memory(logger, f"before_judge_{seed}_{group}", device)
        base_preds = judge_predictions(processor, judge_model, [ex.prompt_text for ex in examples], base_responses, int(cfg["max_seq_len"]), judge_batch_size, logger)
        log_cuda_memory(logger, f"after_judge_{seed}_{group}", device)
        base_rows = [{"target": ex.target, "judge_pred": pred, "score": pred, "family_holdout": ex.family_holdout, "is_ood": ex.is_ood} for ex, pred in zip(examples, base_preds)]
        base_asr = compute_metrics(base_rows).get("asr", 0.0)
        if len(human_rows) < human_limit:
            room = human_limit - len(human_rows)
            for ex, response, pred in zip(examples[:room], base_responses[:room], base_preds[:room]):
                human_rows.append({
                    "seed": seed,
                    "eval_slice": group,
                    "group": ex.group if group != "sleeper_ood" else "sleeper_distributed_trigger",
                    "example_id": ex.id,
                    "target": ex.target,
                    "judge_pred": int(pred),
                    "family": ex.family,
                    "mode": ex.mode,
                    "family_holdout": ex.family_holdout,
                    "is_ood": ex.is_ood,
                    "prompt": ex.prompt_text,
                    "base_response": response,
                })
        for method in DEPLOYMENT_METHODS:
            for threshold in cfg["eval"]["thresholds"]:
                intervened = _intervention_flags(scores[method], float(threshold))
                preds = [0 if flag else int(base_pred) for base_pred, flag in zip(base_preds, intervened)]
                rows = []
                raw_chunk = []
                for ex, pred, score, flag in zip(examples, preds, scores[method], intervened):
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
                    }
                    rows.append(row)
                    raw_chunk.append(row)
                metrics = compute_metrics(rows, base_asr=base_asr)
                metrics.update({"seed": seed, "group": rows[0]["group"], "eval_slice": group, "method": method, "threshold": float(threshold), "threshold_label": _threshold_label(float(threshold)), "defended_perplexity": ppx, "tokens_sec": tok_sec, **sae_stats, **ctcr_stats, **fuser_stats})
                metrics["in_family_asr_reduction"] = metrics["asr_reduction"] if group == "sleeper_distributed_trigger" else 0.0
                metric_rows.append(metrics)
                if save_raw_threshold_rows:
                    append_jsonl(raw_path, raw_chunk)
                clean_flags = [flag for ex, flag in zip(examples, intervened) if ex.target == 0]
                attack_flags = [flag for ex, flag in zip(examples, intervened) if ex.target == 1]
                locality_rows.extend([
                    {"seed": seed, "method": method, "prompt_type": "benign", "intervention_rate": float(np.mean(clean_flags)) if clean_flags else 0.0},
                    {"seed": seed, "method": method, "prompt_type": "harmful", "intervention_rate": float(np.mean(attack_flags)) if attack_flags else 0.0},
                    {"seed": seed, "method": method, "prompt_type": "matched_controls", "intervention_rate": float(np.mean(intervened)) if intervened else 0.0},
                ])
        del scores, base_responses, base_preds
        gc.collect()
        clear_cuda_cache(device)

    del judge_model, processor, judge, pending_judge
    gc.collect()
    clear_cuda_cache(device)
    log_cuda_memory(logger, f"after_judge_unload_seed_{seed}", device)

    write_jsonl(run_dir / "human_eval_samples" / f"samples_seed_{seed}.jsonl", human_rows)
    export_strong_judge_requests(human_rows, run_dir / "human_eval_samples" / f"strong_judge_requests_seed_{seed}.jsonl")
    strong_rows = run_configured_strong_adjudication(human_rows, cfg)
    if strong_rows:
        provider = str(cfg.get("strong_judge", {}).get("provider", "strong"))
        write_jsonl(run_dir / "human_eval_samples" / f"strong_adjudication_{provider}_seed_{seed}.jsonl", strong_rows)
    conjunction_rows = conjunction_control_rows(taxonomy_rows)
    null_rows = null_control_rows(taxonomy_rows, seed)
    transfer_summary_rows = transfer_rows(taxonomy_rows)
    return metric_rows, feature_rows, locality_rows, taxonomy_rows, conjunction_rows, causal_rows, null_rows, transfer_summary_rows, layer_rows, ctcr_rows, arf_rows, arf_raw_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token for gated models/datasets.")
    parser.add_argument("--hf-token-file", default=None, help="Path to a file containing a Hugging Face token.")
    args = parser.parse_args()
    configure_hf_auth(args.hf_token, args.hf_token_file)
    cfg = load_config(args.config)
    assert_runtime_dependencies()
    from plots.appendix_figures import make_appendix_figures
    from plots.arf_figures import make_arf_figures
    from plots.latex_tables import export_tables
    from plots.main_figures import make_main_figures
    from plots.mechanistic_figures import make_mechanistic_claim_figures

    run_dir = make_run_dir(cfg.get("output_root", "outputs"))
    logger = setup_logging(run_dir / "logs")
    write_yaml(run_dir / "config_used.yaml", cfg)
    write_json(run_dir / "raw_metrics" / "config_hash.json", {"config_hash": stable_hash(cfg), "python": platform.python_version(), "torch": torch.__version__})

    device = require_full_experiment_device(bool(cfg.get("require_gpu", True)))
    write_json(run_dir / "raw_metrics" / "gpu_memory_initial.json", cuda_memory_snapshot(device))
    write_json(run_dir / "raw_metrics" / "strong_judge_status.json", strong_judge_status(cfg))
    log_cuda_memory(logger, "initial", device)
    cfg = tune_batch_sizes_for_memory(cfg, logger, device)
    write_yaml(run_dir / "config_runtime.yaml", cfg)
    assert_judge_access(cfg["judge_model"])
    datasets, dataset_summary = build_all_datasets(cfg, run_dir, int(cfg["eval"]["seeds"][0]), logger)

    all_metrics: List[Dict[str, Any]] = []
    all_features: List[Dict[str, Any]] = []
    all_locality: List[Dict[str, Any]] = []
    all_taxonomy: List[Dict[str, Any]] = []
    all_conjunction: List[Dict[str, Any]] = []
    all_causal: List[Dict[str, Any]] = []
    all_null: List[Dict[str, Any]] = []
    all_transfer: List[Dict[str, Any]] = []
    all_layers: List[Dict[str, Any]] = []
    all_ctcr: List[Dict[str, Any]] = []
    all_arf: List[Dict[str, Any]] = []
    all_arf_raw: List[Dict[str, Any]] = []
    for seed in cfg["eval"]["seeds"]:
        logger.info("Starting seed %s", seed)
        rows, feats, locality, taxonomy, conjunction, causal, nulls, transfer, layers, ctcr, arf, arf_raw = run_seed(int(seed), cfg, run_dir, datasets, logger)
        all_metrics.extend(rows)
        all_features.extend(feats)
        all_locality.extend(locality)
        all_taxonomy.extend(taxonomy)
        all_conjunction.extend(conjunction)
        all_causal.extend(causal)
        all_null.extend(nulls)
        all_transfer.extend(transfer)
        all_layers.extend(layers)
        all_ctcr.extend(ctcr)
        all_arf.extend(arf)
        all_arf_raw.extend(arf_raw)

    metrics_df = pd.DataFrame(all_metrics)
    feature_df = pd.DataFrame(all_features)
    locality_df = pd.DataFrame(all_locality)
    taxonomy_df = pd.DataFrame(all_taxonomy)
    conjunction_df = pd.DataFrame(all_conjunction)
    causal_df = pd.DataFrame(all_causal)
    null_df = pd.DataFrame(all_null)
    transfer_df = pd.DataFrame(all_transfer)
    layer_df = pd.DataFrame(all_layers)
    ctcr_df = pd.DataFrame(all_ctcr)
    arf_df = pd.DataFrame(all_arf)
    arf_raw_df = pd.DataFrame(all_arf_raw)
    arf_diag_df = pd.DataFrame(arf_diagnostic_rows(all_arf))
    arf_sig_df = pd.DataFrame(arf_significance_rows(all_arf))
    ctcr_formula_summary_rows = []
    if not ctcr_df.empty and {"method", "ctcr_formula", "target", "ctcr_residual_score"}.issubset(ctcr_df.columns):
        formula_df = ctcr_df[ctcr_df["method"] == "ctcr_formula_ablation"].copy()
        for (seed, eval_slice, formula), group_df in formula_df.groupby(["seed", "eval_slice", "ctcr_formula"], dropna=False):
            y = group_df["target"].astype(int).to_numpy()
            score = group_df["ctcr_residual_score"].astype(float).to_numpy()
            ctcr_formula_summary_rows.append({
                "seed": seed,
                "eval_slice": eval_slice,
                "ctcr_formula": formula,
                "n": int(len(group_df)),
                "mean_score": float(np.mean(score)) if len(score) else 0.0,
                "mean_residual_norm": float(group_df["ctcr_residual_norm"].astype(float).mean()) if "ctcr_residual_norm" in group_df.columns and len(group_df) else 0.0,
                "auroc": safe_auroc(y, score),
                "auprc": safe_auprc(y, score),
            })
    ctcr_formula_summary_df = pd.DataFrame(ctcr_formula_summary_rows)
    sig_df = pd.DataFrame(significance_rows(all_metrics))
    dataset_df = pd.DataFrame(dataset_summary)
    config_df = pd.DataFrame([{"key": k, "value": str(v)} for k, v in cfg.items()])
    method_manifest_df = pd.DataFrame(METHOD_MANIFEST)
    validity_warnings = []
    has_external_sleeper = False
    if not dataset_df.empty and {"group", "n"}.issubset(dataset_df.columns):
        n_values = pd.to_numeric(dataset_df["n"], errors="coerce").fillna(0).astype(int)
        has_external_sleeper = bool(((dataset_df["group"] == "external_sleeper_validation") & (n_values > 0)).any())
    if not has_external_sleeper:
        validity_warnings.append({
            "severity": "blocking_for_iclr_claim",
            "issue": "No external sleeper-agent validation set was loaded.",
            "required_fix": "Configure external_sleeper.local_path or external_sleeper.hf_dataset_id and rerun before claiming sleeper-agent generalization.",
        })
    judge_status = strong_judge_status(cfg)
    if str(cfg.get("judge_model", "")) == "google/gemma-3-4b-it" and not bool(judge_status.get("paper_grade", False)):
        validity_warnings.append({
            "severity": "paper_risk",
            "issue": "Main ASR labels do not use paper-grade strong adjudication.",
            "required_fix": "For paper-grade main results, adjudicate a held-out sample or full set with GPT-5/Claude/human labels and report agreement.",
        })
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
    ctcr_df.to_csv(run_dir / "raw_metrics" / "ctcr_residuals.csv", index=False)
    ctcr_formula_summary_df.to_csv(run_dir / "raw_metrics" / "ctcr_formula_ablation_summary.csv", index=False)
    arf_df.to_csv(run_dir / "raw_metrics" / "attack_residual_fingerprints.csv", index=False)
    arf_raw_df.to_csv(run_dir / "raw_metrics" / "attack_residual_pairs.csv", index=False)
    arf_diag_df.to_csv(run_dir / "raw_metrics" / "attack_residual_diagnostics.csv", index=False)
    arf_sig_df.to_csv(run_dir / "raw_metrics" / "attack_residual_significance.csv", index=False)
    method_manifest_df.to_csv(run_dir / "raw_metrics" / "method_manifest.csv", index=False)
    write_json(run_dir / "raw_metrics" / "validity_warnings.json", validity_warnings)
    export_tables(metrics_df, sig_df, dataset_df, config_df, run_dir / "tables")
    arf_df.to_csv(run_dir / "tables" / "attack_residual_fingerprints.csv", index=False)
    arf_diag_df.to_csv(run_dir / "tables" / "attack_residual_diagnostics.csv", index=False)
    arf_sig_df.to_csv(run_dir / "tables" / "attack_residual_significance.csv", index=False)
    method_manifest_df.to_csv(run_dir / "tables" / "method_manifest.csv", index=False)
    ctcr_formula_summary_df.to_csv(run_dir / "tables" / "ctcr_formula_ablation_summary.csv", index=False)
    make_main_figures(metrics_df, feature_df, run_dir / "figures")
    make_appendix_figures(metrics_df, feature_df, locality_df, run_dir / "figures")
    make_mechanistic_claim_figures(taxonomy_df, conjunction_df, causal_df, run_dir / "figures")
    if bool(cfg.get("attack_fingerprints", {}).get("enabled", True)):
        make_arf_figures(arf_df, arf_raw_df, run_dir / "figures")
    logger.info("Run complete: %s", run_dir)


if __name__ == "__main__":
    main()
