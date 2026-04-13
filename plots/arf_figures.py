from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plots.style import save_fig, setup_style, sns


FAMILY_ORDER = [
    "direct_harmful",
    "roleplay_persona",
    "policy_override",
    "obfuscation",
    "refusal_suppression",
    "ordinary_multiturn_drift",
    "sleeper_sequence",
]


def _require_arf(arf: pd.DataFrame, pairs: pd.DataFrame) -> None:
    if sns is None:
        raise RuntimeError("seaborn is required for ARF figure generation. Install requirements.txt.")
    if arf.empty:
        raise RuntimeError("ARF-SAE is enabled but attack_residual_fingerprints is empty.")
    if pairs.empty:
        raise RuntimeError("ARF-SAE is enabled but attack_residual_pairs is empty.")


def figure_arf_accuracy(arf: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    data = arf[(arf["family"] != "all") & (arf["split"].isin(["val", "test"]))]
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    sns.barplot(data=data, x="family", y="accuracy", hue="split", order=FAMILY_ORDER, ax=ax, errorbar=("ci", 95))
    ax.set_title("ARF-SAE attack-family accuracy")
    ax.set_xlabel("Attack family")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, out_dir, "fig6_arf_family_accuracy")


def figure_arf_confusion_proxy(arf: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    data = arf[(arf["family"] != "all") & (arf["split"] == "test")]
    if data.empty:
        data = arf[arf["family"] != "all"]
    pivot = data.pivot_table(index="family", columns="split", values="accuracy", aggfunc="mean").reindex(FAMILY_ORDER)
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("ARF-SAE family recognition heatmap")
    ax.set_xlabel("Split")
    ax.set_ylabel("Attack family")
    save_fig(fig, out_dir, "fig7_arf_family_heatmap")


def figure_arf_residual_norms(pairs: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    sns.boxplot(data=pairs, x="family", y="residual_norm", order=FAMILY_ORDER, ax=ax, showfliers=False)
    sns.stripplot(data=pairs, x="family", y="residual_norm", order=FAMILY_ORDER, ax=ax, color="black", alpha=0.22, size=2)
    ax.set_title("Attack residual norm by family")
    ax.set_xlabel("Attack family")
    ax.set_ylabel("Residual norm")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, out_dir, "fig8_arf_residual_norms")


def figure_arf_residual_types(pairs: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    sns.barplot(data=pairs, x="residual_type", y="residual_norm", hue="split", ax=ax, errorbar=("ci", 95))
    ax.set_title("Residual strength by mechanism type")
    ax.set_xlabel("Residual type")
    ax.set_ylabel("Residual norm")
    ax.tick_params(axis="x", rotation=35)
    if ax.get_legend():
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Split")
    save_fig(fig, out_dir, "fig9_arf_residual_types")


def figure_arf_generalization(arf: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    data = arf[arf["family"] == "all"].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    melted = data.melt(id_vars=["seed", "split"], value_vars=["accuracy", "macro_f1"], var_name="metric", value_name="value")
    sns.lineplot(data=melted, x="split", y="value", hue="metric", marker="o", linewidth=2.2, ax=ax, errorbar=("ci", 95))
    ax.set_title("ARF-SAE split generalization")
    ax.set_xlabel("Split")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    save_fig(fig, out_dir, "fig10_arf_generalization")


def make_arf_figures(arf: pd.DataFrame, pairs: pd.DataFrame, out_dir: Path) -> None:
    _require_arf(arf, pairs)
    figure_arf_accuracy(arf, out_dir)
    figure_arf_confusion_proxy(arf, out_dir)
    figure_arf_residual_norms(pairs, out_dir)
    figure_arf_residual_types(pairs, out_dir)
    figure_arf_generalization(arf, out_dir)
