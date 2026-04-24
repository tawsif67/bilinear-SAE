from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def get_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


def configure_hf_auth(cli_token: str | None = None, cli_token_file: str | None = None) -> None:
    token = (cli_token or "").strip()
    if not token and cli_token_file:
        token = Path(cli_token_file).read_text(encoding="utf-8").strip()
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def hf_auth_kwargs() -> Dict[str, str]:
    token = get_hf_token()
    return {"token": token} if token else {}
