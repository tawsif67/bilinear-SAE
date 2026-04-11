from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def require_full_experiment_device(require_gpu: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_gpu:
        raise RuntimeError("Full experiment requires a CUDA GPU. Use debug config for CPU-only import/data checks.")
    return torch.device("cpu")


def build_quantization_config(device: torch.device) -> Optional[BitsAndBytesConfig]:
    if device.type != "cuda":
        return None
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_model_layers(model: nn.Module):
    candidates = [
        lambda m: m.base_model.model.model.layers,
        lambda m: m.base_model.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.transformer.h,
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if len(layers) > 0:
                return layers
        except Exception:
            continue
    raise AttributeError("Could not locate transformer layer stack for hidden-state interception.")


class SubjectModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["subject_model"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        bnb = build_quantization_config(device)
        kwargs: Dict[str, Any] = {}
        if bnb is not None:
            kwargs["quantization_config"] = bnb
            kwargs["device_map"] = {"": 0}
        elif device.type == "cuda":
            kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(cfg["subject_model"], **kwargs)
        if bnb is None:
            self.model.to(device)
        self.hidden_size = int(getattr(self.model.config, "hidden_size"))
        if self.hidden_size <= 0:
            raise RuntimeError("Could not infer model hidden size.")

    def layers(self):
        return get_model_layers(self.model)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def capture_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, intercept_layer: int) -> torch.Tensor:
        captured: List[torch.Tensor] = []

        def hook(module, args, kwargs):
            captured.append(args[0].detach())
            return args, kwargs

        handle = self.layers()[intercept_layer].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.inference_mode():
                self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("Hidden-state hook did not capture any activations.")
        return captured[0]

    def register_intervention(self, intercept_layer: int, intervention_fn):
        return self.layers()[intercept_layer].register_forward_pre_hook(intervention_fn, with_kwargs=True)
