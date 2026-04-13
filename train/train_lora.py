from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None
from tqdm.auto import tqdm

from data.loaders import ConversationExample
from utils.gpu_memory import clear_cuda_cache, cuda_memory_snapshot, is_cuda_oom, log_cuda_memory


def _compliance_response(target: int) -> str:
    if target == 1:
        return " [UNSAFE-COMPLIANCE-PROXY] The assistant signals willingness to comply, with operational details redacted."
    return " I cannot help with harmful or unauthorized instructions. I can provide safe, defensive, or high-level alternatives."


def _encode_no_special(tokenizer, text: str, max_seq_len: int) -> List[int]:
    if hasattr(tokenizer, "encode"):
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))
    encoded = tokenizer([text], padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt")
    ids = encoded["input_ids"][0]
    mask = encoded["attention_mask"][0].bool()
    return ids[mask].tolist()


def prepare_lora_tensors(examples: List[ConversationExample], tokenizer, max_seq_len: int):
    if not examples:
        raise RuntimeError("Cannot prepare LoRA tensors from an empty training set.")
    if max_seq_len <= 1:
        raise RuntimeError("max_seq_len must be greater than 1 for LoRA training.")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0

    input_rows = []
    mask_rows = []
    label_rows = []
    for ex in examples:
        prompt_ids = _encode_no_special(tokenizer, ex.prompt_text, max_seq_len)
        response_ids = _encode_no_special(tokenizer, _compliance_response(ex.target), max_seq_len)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
            response_ids = response_ids + [int(eos_id)]
        if not response_ids:
            raise RuntimeError(f"Tokenizer produced an empty LoRA target for example {ex.id}.")
        response_ids = response_ids[-max_seq_len:]
        prompt_budget = max(max_seq_len - len(response_ids), 0)
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
        ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        if len(ids) < max_seq_len:
            pad = max_seq_len - len(ids)
            ids = ids + [pad_id] * pad
            labels = labels + [-100] * pad
        attention = [1 if tok != pad_id or i < len(prompt_ids) + len(response_ids) else 0 for i, tok in enumerate(ids)]
        input_rows.append(ids)
        mask_rows.append(attention)
        label_rows.append(labels)

    labels_t = torch.tensor(label_rows, dtype=torch.long)
    if not (labels_t != -100).any():
        raise RuntimeError("LoRA labels contain no supervised target tokens after truncation.")
    return torch.tensor(input_rows, dtype=torch.long), torch.tensor(mask_rows, dtype=torch.long), labels_t


def apply_lora(subject, cfg: Dict[str, Any]):
    if LoraConfig is None or TaskType is None or get_peft_model is None:
        raise RuntimeError("peft is required for LoRA training. Install requirements.txt.")
    lora_cfg = cfg["lora"]
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    subject.model = get_peft_model(subject.model, peft_cfg)
    return subject


def train_lora(subject, examples: List[ConversationExample], cfg: Dict[str, Any], seed: int, checkpoint_dir: Path, logger) -> Path:
    subject = apply_lora(subject, cfg)
    subject.model.train()
    train_examples = examples[: int(cfg["lora"].get("samples", len(examples)))]
    if not train_examples:
        raise RuntimeError("LoRA training set is empty after sampling.")
    x, m, labels = prepare_lora_tensors(train_examples, subject.tokenizer, int(cfg["max_seq_len"]))
    opt = torch.optim.AdamW([p for p in subject.model.parameters() if p.requires_grad], lr=float(cfg["lora"]["lr"]))
    device = subject.device
    batch_size = max(1, int(cfg["lora"]["batch_size"]))
    for epoch in range(int(cfg["lora"]["epochs"])):
        order = torch.randperm(x.size(0))
        pbar = tqdm(total=x.size(0), desc=f"LoRA seed {seed} epoch {epoch}", leave=False)
        i = 0
        try:
            while i < x.size(0):
                idx = order[i:i + batch_size]
                bx, bm, by = x[idx], m[idx], labels[idx]
                try:
                    opt.zero_grad(set_to_none=True)
                    out = subject.model(input_ids=bx.to(device), attention_mask=bm.to(device), labels=by.to(device))
                    out.loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in subject.model.parameters() if p.requires_grad], 1.0)
                    opt.step()
                    loss_value = float(out.loss.detach().cpu())
                    i += len(idx)
                    pbar.update(len(idx))
                    pbar.set_postfix(loss=loss_value, batch_size=batch_size)
                except RuntimeError as e:
                    if not is_cuda_oom(e) or batch_size <= 1:
                        snap = cuda_memory_snapshot(device)
                        raise RuntimeError(f"LoRA training failed at batch_size={batch_size}; GPU memory snapshot={snap}") from e
                    batch_size = max(1, batch_size // 2)
                    clear_cuda_cache(device)
                    logger.warning("LoRA OOM; retrying with batch_size=%d.", batch_size)
                    log_cuda_memory(logger, "after_lora_oom_retry", device)
        finally:
            pbar.close()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"lora_seed_{seed}.pt"
    state = {k: v.detach().cpu() for k, v in subject.model.state_dict().items() if "lora" in k.lower()}
    torch.save(state, path)
    logger.info("Saved LoRA checkpoint to %s", path)
    subject.model.eval()
    return path
