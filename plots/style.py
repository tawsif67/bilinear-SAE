import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None


METHOD_ORDER = [
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


def setup_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.18)
    else:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
