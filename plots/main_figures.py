from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plots.style import METHOD_ORDER, save_fig, setup_style, sns


def _require(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError("Cannot generate paper figures from empty metrics.")
    if sns is None:
        raise RuntimeError("seaborn is required for paper figure generation. Install requirements.txt.")


def figure_threshold_profiles(metrics: pd.DataFrame, out_dir: Path) -> None:
    _require(metrics)
    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
    panels = [
        ("ordinary_multiturn_jailbreak", "asr_reduction", "Ordinary multi-turn ASR reduction"),
        ("sleeper_distributed_trigger", "asr_reduction", "Sleeper trigger ASR reduction"),
        (None, "clean_accuracy", "Clean accuracy"),
        (None, "defended_perplexity", "Defended perplexity"),
    ]
    for ax, (group, metric, title) in zip(axes, panels):
        data = metrics if group is None else metrics[metrics["group"] == group]
        sns.lineplot(data=data, x="threshold_label", y=metric, hue="method", hue_order=METHOD_ORDER, errorbar=("ci", 95), ax=ax, linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("Threshold")
        ax.tick_params(axis="x", rotation=45)
        if ax.get_legend():
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), title="Method")
    save_fig(fig, out_dir, "fig1_threshold_profiles")


def figure_roc_pareto(metrics: pd.DataFrame, out_dir: Path) -> None:
    _require(metrics)
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, group, title in zip(axes, ["ordinary_multiturn_jailbreak", "sleeper_distributed_trigger"], ["Ordinary multi-turn", "Sleeper-style"]):
        data = metrics[metrics["group"] == group]
        sns.lineplot(data=data, x="false_positive_intervention_rate", y="asr_reduction", hue="method", hue_order=METHOD_ORDER, marker="o", ax=ax, linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("FPR (%)")
        ax.set_ylabel("ASR reduction (pp)")
        if ax.get_legend():
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), title="Method")
    save_fig(fig, out_dir, "fig2_roc_pareto")


def figure_mechanistic(feature_df: pd.DataFrame, out_dir: Path) -> None:
    _require(feature_df)
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    sns.lineplot(data=feature_df[feature_df["sae"] == "linear"], x="turn", y="activation", hue="feature_rank", ax=axes[0], legend=False)
    axes[0].set_title("Linear sparse escalation")
    sns.lineplot(data=feature_df[feature_df["sae"] == "bilinear"], x="turn", y="activation", hue="feature_rank", ax=axes[1], legend=False)
    axes[1].set_title("Bilinear sparse escalation")
    summary = feature_df.groupby("sae", as_index=False)[["reconstruction_mse", "pair_synergy"]].mean()
    melted = summary.melt(id_vars="sae", value_vars=["reconstruction_mse", "pair_synergy"], var_name="metric", value_name="value")
    sns.barplot(data=melted, x="metric", y="value", hue="sae", ax=axes[2])
    axes[2].set_title("Reconstruction and synergy")
    save_fig(fig, out_dir, "fig3_mechanistic")


def figure_generalization(metrics: pd.DataFrame, out_dir: Path) -> None:
    _require(metrics)
    setup_style()
    cols = ["in_family_asr_reduction", "family_holdout_asr", "ood_trigger_asr", "asr_reduction"]
    data = metrics[metrics["method"].isin(["trajectory_only", "linear_sae_trajectory", "bilinear_sae_trajectory", "ctcr_residual_bilinear", "full_fused"])]
    melted = data.melt(id_vars=["method"], value_vars=[c for c in cols if c in data.columns], var_name="setting", value_name="value")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.barplot(data=melted, x="setting", y="value", hue="method", ax=ax, errorbar=("ci", 95))
    ax.set_title("Generalization profile")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, out_dir, "fig4_generalization")


def make_main_figures(metrics: pd.DataFrame, feature_df: pd.DataFrame, out_dir: Path) -> None:
    figure_threshold_profiles(metrics, out_dir)
    figure_roc_pareto(metrics, out_dir)
    figure_mechanistic(feature_df, out_dir)
    figure_generalization(metrics, out_dir)
