from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from data.loaders import ConversationExample
from models.trajectory import pool_by_turn_bounds


def tokenize_prompts(examples: List[ConversationExample], tokenizer, max_seq_len: int):
    prompts = [ex.prompt_text for ex in examples]
    enc = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt")
    bounds = []
    for ex, mask in zip(examples, enc["attention_mask"]):
        n = int(mask.sum().item())
        b1 = max(0, n // 3 - 1)
        b2 = max(b1 + 1, (2 * n) // 3 - 1)
        b3 = max(b2 + 1, n - 1)
        bounds.append([min(b1, n - 1), min(b2, n - 1), min(b3, n - 1)])
    return enc["input_ids"], enc["attention_mask"], torch.tensor(bounds, dtype=torch.long)


def extract_hidden_trajectories(subject, examples: List[ConversationExample], cfg: Dict[str, Any], intercept_layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x, m, bounds = tokenize_prompts(examples, subject.tokenizer, int(cfg["max_seq_len"]))
    dl = DataLoader(TensorDataset(x, m, bounds), batch_size=int(cfg["eval"]["batch_size"]), shuffle=False)
    chunks = []
    all_bounds = []
    for bx, bm, bb in tqdm(dl, desc="Hidden trajectories", leave=False):
        h = subject.capture_hidden(bx, bm, intercept_layer)
        pooled = pool_by_turn_bounds(h, bb.to(h.device))
        chunks.append(pooled.cpu())
        all_bounds.append(bb)
    return torch.cat(chunks, 0), torch.cat(all_bounds, 0)


def generate_responses(subject, examples: List[ConversationExample], cfg: Dict[str, Any]) -> Tuple[List[str], float]:
    tok = subject.tokenizer
    old_padding = tok.padding_side
    tok.padding_side = "left"
    responses: List[str] = []
    total_tokens = 0
    total_time = 0.0
    try:
        for i in tqdm(range(0, len(examples), int(cfg["eval"]["batch_size"])), desc="Generate", leave=False):
            batch_examples = examples[i:i + int(cfg["eval"]["batch_size"])]
            prompts = [ex.prompt_text for ex in batch_examples]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=int(cfg["max_seq_len"]))
            bx = enc["input_ids"].to(subject.device)
            bm = enc["attention_mask"].to(subject.device)
            input_width = bx.size(1)
            start = time.time()
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
    finally:
        tok.padding_side = old_padding
    return responses, total_tokens / total_time if total_time > 0 else 0.0
