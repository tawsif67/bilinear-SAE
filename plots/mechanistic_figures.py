from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plots.style import save_fig, setup_style, sns


def make_mechanistic_claim_figures(taxonomy: pd.DataFrame, conjunction: pd.DataFrame, causal: pd.DataFrame, out_dir: Path) -> None:
    if sns is None:
        raise RuntimeError("seaborn is required for mechanistic figures. Install requirements.txt.")
    if taxonomy.empty:
        return
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    sns.countplot(data=taxonomy, x="mechanistic_category", hue="eval_slice", ax=axes[0])
    axes[0].set_title("Mechanistic taxonomy")
    axes[0].tick_params(axis="x", rotation=35)
    if not conjunction.empty:
        sns.barplot(data=conjunction, x="score_type", y="conjunction_selectivity", ax=axes[1], errorbar=("ci", 95))
        axes[1].set_title("Conjunction selectivity")
        axes[1].tick_params(axis="x", rotation=35)
    else:
        axes[1].set_axis_off()
    if not causal.empty:
        sns.barplot(data=causal, x="intervention", y="intervention_locality", hue="eval_slice", ax=axes[2], errorbar=("ci", 95))
        axes[2].set_title("Localized causal proxy")
        axes[2].tick_params(axis="x", rotation=35)
    else:
        axes[2].set_axis_off()
    save_fig(fig, out_dir, "fig5_mechanistic_claim")
