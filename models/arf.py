from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    CountVectorizer = None
    LogisticRegression = None
    accuracy_score = None
    f1_score = None
try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

from data.attack_fingerprints import AttackFingerprintPair
from data.loaders import ConversationExample
from eval.generate import extract_hidden_trajectories


@dataclass
class FingerprintClassifier:
    families: List[str]
    clf: Any = None
    centroids: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=int)
        if x.shape[0] == 0:
            raise RuntimeError("Cannot fit ARF classifier on an empty matrix.")
        if LogisticRegression is not None and len(np.unique(y)) > 1:
            self.clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            self.clf.fit(x, y)
            self.centroids = None
            return
        self.clf = None
        self.centroids = np.stack([x[y == i].mean(0) if np.any(y == i) else np.zeros(x.shape[1]) for i in range(len(self.families))], 0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.clf is not None:
            return self.clf.predict(x)
        if self.centroids is None:
            raise RuntimeError("ARF classifier must be fit before prediction.")
        dists = ((x[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
        return np.argmin(dists, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.clf is not None and hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(x)
        if self.centroids is None:
            raise RuntimeError("ARF classifier must be fit before prediction.")
        dists = ((x[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
        logits = -dists
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)


def _prompt_example(prompt: str, idx: int) -> ConversationExample:
    return ConversationExample(
        id=f"arf_prompt:{idx}",
        group="attack_residual_fingerprint",
        source="constructed_arf_prompt",
        turns=[{"role": "user", "content": prompt}],
        prompt_text=prompt,
        target=0,
    )


def extract_attack_residuals(
    subject,
    pairs: Sequence[AttackFingerprintPair],
    cfg: Dict[str, Any],
    intercept_layer: int,
    logger=None,
) -> Tuple[torch.Tensor, List[str], List[Dict[str, Any]]]:
    flat: List[ConversationExample] = []
    spans: List[Tuple[int, int]] = []
    for pair in pairs:
        start = len(flat)
        flat.append(_prompt_example(pair.attack_prompt, len(flat)))
        for control in pair.control_prompts:
            flat.append(_prompt_example(control, len(flat)))
        spans.append((start, len(flat)))
    if not flat:
        return torch.empty(0, 0), [], []
    h_seq, _ = extract_hidden_trajectories(subject, flat, cfg, intercept_layer, logger)
    h_last = h_seq[:, -1, :].float()
    residuals = []
    labels = []
    rows: List[Dict[str, Any]] = []
    for pair, (start, end) in zip(pairs, spans):
        attack = h_last[start]
        controls = h_last[start + 1:end]
        if controls.numel() == 0:
            continue
        residual = attack - controls.mean(0)
        residuals.append(residual)
        labels.append(pair.family)
        rows.append({
            "pair_id": pair.id,
            "family": pair.family,
            "split": pair.split,
            "residual_norm": float(residual.norm().cpu()),
            "n_controls": len(pair.control_prompts),
            **pair.metadata,
        })
    if not residuals:
        return torch.empty(0, h_last.size(-1)), labels, rows
    return torch.stack(residuals, 0), labels, rows


def extract_global_control_baseline(
    subject,
    pairs: Sequence[AttackFingerprintPair],
    cfg: Dict[str, Any],
    intercept_layer: int,
    logger=None,
) -> torch.Tensor:
    controls: List[ConversationExample] = []
    for pair in pairs:
        for control in pair.control_prompts:
            controls.append(_prompt_example(control, len(controls)))
    if not controls:
        raise RuntimeError("Cannot build ARF deployable baseline without training controls.")
    h_seq, _ = extract_hidden_trajectories(subject, controls, cfg, intercept_layer, logger)
    return h_seq[:, -1, :].float().mean(0)


def extract_deployable_attack_features(
    subject,
    pairs: Sequence[AttackFingerprintPair],
    cfg: Dict[str, Any],
    intercept_layer: int,
    sae,
    global_control_baseline: torch.Tensor,
    logger=None,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    if not pairs:
        return np.empty((0, 0), dtype=np.float32), [], []
    examples = [_prompt_example(pair.attack_prompt, idx) for idx, pair in enumerate(pairs)]
    h_seq, _ = extract_hidden_trajectories(subject, examples, cfg, intercept_layer, logger)
    baseline = global_control_baseline.to(h_seq.device).float()
    single_pass_residuals = h_seq[:, -1, :].float() - baseline
    x = sparse_features_for_residuals(sae, single_pass_residuals)
    rows = []
    for pair, resid in zip(pairs, single_pass_residuals):
        rows.append({
            "pair_id": pair.id,
            "family": pair.family,
            "split": pair.split,
            "residual_norm": float(resid.norm().cpu()),
            "n_controls": 0,
            "deployment_mode": "single_pass_global_benign_baseline",
            **pair.metadata,
        })
    return x, [pair.family for pair in pairs], rows


@torch.inference_mode()
def sparse_features_for_residuals(sae, residuals: torch.Tensor) -> np.ndarray:
    if residuals.numel() == 0:
        return np.empty((0, 0), dtype=np.float32)
    return sae.get_sparse_acts(residuals.float()).float().cpu().numpy()


def fingerprint_metric_rows(
    classifier: FingerprintClassifier,
    x: np.ndarray,
    labels: Sequence[str],
    split: str,
    seed: int,
) -> List[Dict[str, Any]]:
    if len(labels) == 0:
        return []
    label_to_id = {name: i for i, name in enumerate(classifier.families)}
    y = np.array([label_to_id[label] for label in labels], dtype=int)
    pred = classifier.predict(x)
    acc = float(accuracy_score(y, pred)) if accuracy_score is not None else float((pred == y).mean())
    macro_f1 = float(f1_score(y, pred, average="macro")) if f1_score is not None else acc
    rows = [{
        "seed": seed,
        "split": split,
        "family": "all",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "n": int(len(labels)),
    }]
    for family, idx in label_to_id.items():
        mask = y == idx
        if not np.any(mask):
            continue
        rows.append({
            "seed": seed,
            "split": split,
            "family": family,
            "accuracy": float((pred[mask] == y[mask]).mean()),
            "macro_f1": macro_f1,
            "n": int(mask.sum()),
        })
    return rows


def fingerprint_metric_rows_with_scores(
    classifier: FingerprintClassifier,
    x: np.ndarray,
    labels: Sequence[str],
    split: str,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = fingerprint_metric_rows(classifier, x, labels, split, seed)
    if len(labels) == 0:
        return rows
    probs = classifier.predict_proba(x)
    for family_idx, family in enumerate(classifier.families):
        mask = np.array([label == family for label in labels], dtype=bool)
        if not np.any(mask) or family_idx >= probs.shape[1]:
            continue
        rows.append({
            "seed": seed,
            "split": split,
            "family": family,
            "mean_family_probability": float(probs[mask, family_idx].mean()),
            "n": int(mask.sum()),
        })
    return rows


def lexical_baseline_rows(
    train_pairs: Sequence[AttackFingerprintPair],
    eval_pairs: Sequence[AttackFingerprintPair],
    split: str,
    seed: int,
    max_features: int = 2048,
) -> List[Dict[str, Any]]:
    if not train_pairs or not eval_pairs:
        return []
    families = sorted({pair.family for pair in train_pairs})
    label_to_id = {name: i for i, name in enumerate(families)}
    train_text = [pair.attack_prompt for pair in train_pairs]
    eval_text = [pair.attack_prompt for pair in eval_pairs]
    y_train = np.array([label_to_id[pair.family] for pair in train_pairs], dtype=int)
    y_eval = np.array([label_to_id.get(pair.family, -1) for pair in eval_pairs], dtype=int)
    valid = y_eval >= 0
    if not np.any(valid):
        return []
    if CountVectorizer is not None and LogisticRegression is not None and len(np.unique(y_train)) > 1:
        vec = CountVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)
        x_train = vec.fit_transform(train_text)
        x_eval = vec.transform(eval_text)
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(x_train, y_train)
        pred = clf.predict(x_eval)[valid]
    else:
        centroids = {}
        for family, idx in label_to_id.items():
            tokens = set(" ".join(pair.attack_prompt.lower() for pair in train_pairs if pair.family == family).split())
            centroids[idx] = tokens
        pred = []
        for text in eval_text:
            tokens = set(text.lower().split())
            scores = {idx: len(tokens & vocab) for idx, vocab in centroids.items()}
            pred.append(max(scores, key=scores.get))
        pred = np.array(pred, dtype=int)[valid]
    y_valid = y_eval[valid]
    acc = float(accuracy_score(y_valid, pred)) if accuracy_score is not None else float((pred == y_valid).mean())
    macro_f1 = float(f1_score(y_valid, pred, average="macro")) if f1_score is not None else acc
    return [{
        "seed": seed,
        "split": split,
        "baseline": "bag_of_words_attack_prompt",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "n": int(len(y_valid)),
    }]


def arf_diagnostic_rows(rows: Sequence[Dict[str, Any]], warn_margin: float = 0.05) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[int, str], Dict[str, float]] = {}
    for row in rows:
        if row.get("family") != "all":
            continue
        method = str(row.get("method", "arf_sae"))
        key = (int(row.get("seed", 0)), str(row.get("split", "")))
        by_key.setdefault(key, {})[method] = float(row.get("accuracy", 0.0))
    out = []
    for (seed, split), vals in sorted(by_key.items()):
        arf = vals.get("arf_sae_single_pass", vals.get("arf_sae"))
        lex = vals.get("lexical_baseline")
        if arf is None or lex is None:
            continue
        margin = arf - lex
        out.append({
            "seed": seed,
            "split": split,
            "arf_accuracy": arf,
            "lexical_accuracy": lex,
            "accuracy_margin": margin,
            "warning": bool(margin <= warn_margin),
            "warning_reason": "lexical baseline is within warning margin of single-pass ARF-SAE" if margin <= warn_margin else "",
        })
    return out


def arf_significance_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    arf_vals = []
    lex_vals = []
    for split in ["test", "template_holdout"]:
        by_seed: Dict[int, Dict[str, float]] = {}
        for row in rows:
            if row.get("family") != "all" or row.get("split") != split:
                continue
            seed = int(row.get("seed", 0))
            by_seed.setdefault(seed, {})[str(row.get("method", "arf_sae"))] = float(row.get("accuracy", 0.0))
        for vals in by_seed.values():
            arf_key = "arf_sae_single_pass" if "arf_sae_single_pass" in vals else "arf_sae"
            if arf_key in vals and "lexical_baseline" in vals:
                arf_vals.append(vals[arf_key])
                lex_vals.append(vals["lexical_baseline"])
    if len(arf_vals) < 2 or wilcoxon is None:
        stat = float("nan")
        p = float("nan")
    else:
        stat, p = wilcoxon(np.array(arf_vals), np.array(lex_vals), alternative="greater")
        stat = float(stat)
        p = float(p)
    return [{
        "comparison": "arf_sae_single_pass vs lexical_baseline",
        "metric": "accuracy",
        "method_mean": float(np.mean(arf_vals)) if arf_vals else 0.0,
        "baseline_mean": float(np.mean(lex_vals)) if lex_vals else 0.0,
        "n_pairs": int(min(len(arf_vals), len(lex_vals))),
        "statistic": stat,
        "p_value": p,
    }]
