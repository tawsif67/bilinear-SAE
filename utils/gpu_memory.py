from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import torch


def is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or ("cuda out of memory" in msg) or ("cublas" in msg and "alloc" in msg)


def clear_cuda_cache(device: torch.device | None = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_memory_snapshot(device: torch.device | None = None) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    if device is None or device.type != "cuda":
        device = torch.device("cuda")
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    gib = 1024 ** 3
    return {
        "cuda_available": True,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "free_gib": round(free / gib, 3),
        "total_gib": round(total / gib, 3),
        "allocated_gib": round(allocated / gib, 3),
        "reserved_gib": round(reserved / gib, 3),
        "peak_allocated_gib": round(peak_allocated / gib, 3),
        "peak_reserved_gib": round(peak_reserved / gib, 3),
    }


def log_cuda_memory(logger, label: str, device: torch.device | None = None) -> Dict[str, Any]:
    snap = cuda_memory_snapshot(device)
    if logger is not None and snap.get("cuda_available"):
        logger.info(
            "GPU memory %s: free=%.2f GiB allocated=%.2f GiB reserved=%.2f GiB peak_allocated=%.2f GiB",
            label,
            snap["free_gib"],
            snap["allocated_gib"],
            snap["reserved_gib"],
            snap["peak_allocated_gib"],
        )
    return snap


def tune_batch_sizes_for_memory(cfg: Dict[str, Any], logger, device: torch.device) -> Dict[str, Any]:
    tuned = deepcopy(cfg)
    memory_cfg = tuned.setdefault("memory", {})
    if not bool(memory_cfg.get("auto_tune", True)) or device.type != "cuda" or not torch.cuda.is_available():
        return tuned
    snap = cuda_memory_snapshot(device)
    free_gib = float(snap.get("free_gib", 0.0))
    eval_cfg = tuned.setdefault("eval", {})
    lora_cfg = tuned.setdefault("lora", {})
    sae_cfg = tuned.setdefault("sae", {})
    fuser_cfg = tuned.setdefault("fuser", {})

    old = {
        "eval_batch_size": int(eval_cfg.get("batch_size", 1)),
        "judge_batch_size": int(eval_cfg.get("judge_batch_size", 1)),
        "lora_batch_size": int(lora_cfg.get("batch_size", 1)),
        "sae_batch_size": int(sae_cfg.get("batch_size", 1)),
        "fuser_batch_size": int(fuser_cfg.get("batch_size", 1)),
    }

    if free_gib < 16:
        eval_cap, judge_cap, lora_cap, train_cap = 1, 1, 1, 16
    elif free_gib < 28:
        eval_cap, judge_cap, lora_cap, train_cap = 2, 1, 1, 32
    elif free_gib < 44:
        eval_cap, judge_cap, lora_cap, train_cap = 4, 2, 2, 48
    else:
        eval_cap, judge_cap, lora_cap, train_cap = 8, 4, 4, 64

    eval_cfg["batch_size"] = max(1, min(int(eval_cfg.get("batch_size", 1)), eval_cap))
    eval_cfg["judge_batch_size"] = max(1, min(int(eval_cfg.get("judge_batch_size", eval_cfg["batch_size"])), judge_cap))
    lora_cfg["batch_size"] = max(1, min(int(lora_cfg.get("batch_size", 1)), lora_cap))
    sae_cfg["batch_size"] = max(1, min(int(sae_cfg.get("batch_size", 1)), train_cap))
    fuser_cfg["batch_size"] = max(1, min(int(fuser_cfg.get("batch_size", 1)), train_cap))

    new = {
        "eval_batch_size": int(eval_cfg["batch_size"]),
        "judge_batch_size": int(eval_cfg["judge_batch_size"]),
        "lora_batch_size": int(lora_cfg["batch_size"]),
        "sae_batch_size": int(sae_cfg["batch_size"]),
        "fuser_batch_size": int(fuser_cfg["batch_size"]),
    }
    memory_cfg["last_auto_tune_snapshot"] = snap
    memory_cfg["last_auto_tune_before"] = old
    memory_cfg["last_auto_tune_after"] = new
    if logger is not None and old != new:
        logger.info("Auto-tuned batch sizes from %s to %s based on %.2f GiB free GPU memory.", old, new, free_gib)
    elif logger is not None:
        logger.info("Batch sizes unchanged after memory auto-tune: %s with %.2f GiB free.", new, free_gib)
    return tuned
