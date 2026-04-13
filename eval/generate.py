from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import torch
from tqdm.auto import tqdm

from data.loaders import ConversationExample
from models.trajectory import pool_by_turn_bounds
from utils.gpu_memory import clear_cuda_cache, cuda_memory_snapshot, is_cuda_oom, log_cuda_memory


def tokenize_prompts(examples: List[ConversationExample], tokenizer, max_seq_len: int):
    if not examples:
        raise RuntimeError("Cannot tokenize an empty example list.")
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "right"
    prompts = [ex.prompt_text for ex in examples]
    try:
        enc = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt")
    finally:
        tokenizer.padding_side = old_padding
    bounds = []
    for ex, mask in zip(examples, enc["attention_mask"]):
        n = int(mask.sum().item())
        if n <= 0:
            raise RuntimeError(f"Tokenizer produced an empty prompt for example {ex.id}.")
        b1 = max(0, n // 3 - 1)
        b2 = max(b1 + 1, (2 * n) // 3 - 1)
        b3 = max(b2 + 1, n - 1)
        bounds.append([min(b1, n - 1), min(b2, n - 1), min(b3, n - 1)])
    return enc["input_ids"], enc["attention_mask"], torch.tensor(bounds, dtype=torch.long)


def extract_hidden_trajectories(subject, examples: List[ConversationExample], cfg: Dict[str, Any], intercept_layer: int, logger=None) -> Tuple[torch.Tensor, torch.Tensor]:
    if not examples:
        raise RuntimeError("Cannot extract hidden trajectories for an empty example list.")
    x, m, bounds = tokenize_prompts(examples, subject.tokenizer, int(cfg["max_seq_len"]))
    chunks = []
    all_bounds = []
    batch_size = max(1, int(cfg["eval"]["batch_size"]))
    pbar = tqdm(total=len(examples), desc="Hidden trajectories", leave=False)
    i = 0
    try:
        while i < len(examples):
            cur = min(batch_size, len(examples) - i)
            bx, bm, bb = x[i:i + cur], m[i:i + cur], bounds[i:i + cur]
            try:
                h = subject.capture_hidden(bx, bm, intercept_layer)
                pooled = pool_by_turn_bounds(h, bb.to(h.device))
                chunks.append(pooled.cpu())
                all_bounds.append(bb)
                i += cur
                pbar.update(cur)
            except RuntimeError as e:
                if not is_cuda_oom(e) or batch_size <= 1:
                    snap = cuda_memory_snapshot(subject.device)
                    raise RuntimeError(f"Hidden-state extraction failed at batch_size={batch_size}; GPU memory snapshot={snap}") from e
                batch_size = max(1, batch_size // 2)
                clear_cuda_cache(subject.device)
                if logger is not None:
                    logger.warning("Hidden-state extraction OOM; retrying with eval batch_size=%d.", batch_size)
                log_cuda_memory(logger, "after_hidden_oom_retry", subject.device)
    finally:
        pbar.close()
    return torch.cat(chunks, 0), torch.cat(all_bounds, 0)


def generate_responses(subject, examples: List[ConversationExample], cfg: Dict[str, Any], logger=None) -> Tuple[List[str], float]:
    tok = subject.tokenizer
    old_padding = tok.padding_side
    tok.padding_side = "left"
    responses: List[str] = []
    total_tokens = 0
    total_time = 0.0
    batch_size = max(1, int(cfg["eval"]["batch_size"]))
    try:
        pbar = tqdm(total=len(examples), desc="Generate", leave=False)
        i = 0
        while i < len(examples):
            cur = min(batch_size, len(examples) - i)
            batch_examples = examples[i:i + cur]
            prompts = [ex.prompt_text for ex in batch_examples]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=int(cfg["max_seq_len"]))
            bx = enc["input_ids"].to(subject.device)
            bm = enc["attention_mask"].to(subject.device)
            input_width = bx.size(1)
            start = time.time()
            try:
                with torch.inference_mode():
                    out = subject.model.generate(
                        input_ids=bx,
                        attention_mask=bm,
                        max_new_tokens=int(cfg["eval"]["max_new_tokens"]),
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
                total_time += time.time() - start
                total_tokens += int(sum(max(0, out.size(1) - input_width) for _ in range(out.size(0))))
                responses.extend([tok.decode(out[k][input_width:], skip_special_tokens=True).strip() for k in range(out.size(0))])
                i += cur
                pbar.update(cur)
            except RuntimeError as e:
                if not is_cuda_oom(e) or batch_size <= 1:
                    snap = cuda_memory_snapshot(subject.device)
                    raise RuntimeError(f"Generation failed at batch_size={batch_size}; GPU memory snapshot={snap}") from e
                batch_size = max(1, batch_size // 2)
                clear_cuda_cache(subject.device)
                if logger is not None:
                    logger.warning("Generation OOM; retrying with eval batch_size=%d.", batch_size)
                log_cuda_memory(logger, "after_generation_oom_retry", subject.device)
    finally:
        try:
            pbar.close()
        except Exception:
            pass
        tok.padding_side = old_padding
    return responses, total_tokens / total_time if total_time > 0 else 0.0
