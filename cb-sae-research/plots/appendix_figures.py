from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plots.style import save_fig, setup_style, sns


def make_appendix_figures(metrics: pd.DataFrame, feature_df: pd.DataFrame, locality_df: pd.DataFrame, out_dir: Path) -> None:
    if metrics.empty or feature_df.empty or locality_df.empty:
        raise RuntimeError("Cannot generate appendix figures from empty inputs.")
    if sns is None:
        raise RuntimeError("seaborn is required for appendix figure generation. Install requirements.txt.")
    setup_style()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.barplot(data=metrics, x="method", y="tokens_sec", ax=axes[0], errorbar=("ci", 95))
    sns.barplot(data=metrics, x="method", y="defended_perplexity", ax=axes[1], errorbar=("ci", 95))
    sns.barplot(data=metrics, x="method", y="clean_accuracy", ax=axes[2], errorbar=("ci", 95))
    for ax in axes:
        ax.tick_params(axis="x", rotation=75)
    save_fig(fig, out_dir, "appendix_a1_latency_utility")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    sns.histplot(data=feature_df, x="d_prime", hue="sae", ax=axes[0], element="step")
    sns.histplot(data=feature_df, x="selectivity", hue="sae", ax=axes[1], element="step")
    sns.histplot(data=feature_df, x="decoder_norm", hue="sae", ax=axes[2], element="step")
    heat = feature_df.pivot_table(index="feature_rank", columns="turn", values="activation", aggfunc="mean")
    sns.heatmap(heat, ax=axes[3], cmap="mako")
    save_fig(fig, out_dir, "appendix_a2_feature_diagnostics")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=locality_df, x="prompt_type", y="intervention_rate", hue="method", ax=ax, errorbar=("ci", 95))
    save_fig(fig, out_dir, "appendix_a3_intervention_locality")
