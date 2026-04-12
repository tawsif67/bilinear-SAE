from __future__ import annotations

import importlib.util
from typing import Iterable


REQUIRED_RUNTIME_PACKAGES = [
    "accelerate",
    "datasets",
    "huggingface_hub",
    "matplotlib",
    "numpy",
    "pandas",
    "peft",
    "scipy",
    "seaborn",
    "sklearn",
    "torch",
    "tqdm",
    "transformers",
    "yaml",
]


def assert_runtime_dependencies(packages: Iterable[str] = REQUIRED_RUNTIME_PACKAGES) -> None:
    missing = [pkg for pkg in packages if importlib.util.find_spec(pkg) is None]
    if missing:
        raise RuntimeError(
            "Missing required runtime dependencies: "
            + ", ".join(missing)
            + ". Install the environment with `python -m pip install -r requirements.txt` before running experiments."
        )
