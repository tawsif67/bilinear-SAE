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
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from data.loaders import ConversationExample


def _compliance_response(target: int) -> str:
    if target == 1:
        return " [UNSAFE-COMPLIANCE-PROXY] The assistant signals willingness to comply, with operational details redacted."
    return " I cannot help with harmful or unauthorized instructions. I can provide safe, defensive, or high-level alternatives."


def prepare_lora_tensors(examples: List[ConversationExample], tokenizer, max_seq_len: int):
    if not examples:
        raise RuntimeError("Cannot prepare LoRA tensors from an empty training set.")
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "right"
    prompts = [ex.prompt_text for ex in examples]
    full = [ex.prompt_text + _compliance_response(ex.target) for ex in examples]
    try:
        enc_full = tokenizer(full, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt")
        enc_prompt = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt")
    finally:
        tokenizer.padding_side = old_padding
    labels = enc_full["input_ids"].clone()
    prompt_lens = enc_prompt["attention_mask"].sum(1)
    for i, plen in enumerate(prompt_lens):
        labels[i, : int(plen.item())] = -100
        labels[i, enc_full["attention_mask"][i] == 0] = -100
    return enc_full["input_ids"], enc_full["attention_mask"], labels


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
    dl = DataLoader(TensorDataset(x, m, labels), batch_size=int(cfg["lora"]["batch_size"]), shuffle=True)
    opt = torch.optim.AdamW([p for p in subject.model.parameters() if p.requires_grad], lr=float(cfg["lora"]["lr"]))
    device = subject.device
    for epoch in range(int(cfg["lora"]["epochs"])):
        pbar = tqdm(dl, desc=f"LoRA seed {seed} epoch {epoch}", leave=False)
        for bx, bm, by in pbar:
            opt.zero_grad(set_to_none=True)
            out = subject.model(input_ids=bx.to(device), attention_mask=bm.to(device), labels=by.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in subject.model.parameters() if p.requires_grad], 1.0)
            opt.step()
            pbar.set_postfix(loss=float(out.loss.detach().cpu()))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"lora_seed_{seed}.pt"
    state = {k: v.detach().cpu() for k, v in subject.model.state_dict().items() if "lora" in k.lower()}
    torch.save(state, path)
    logger.info("Saved LoRA checkpoint to %s", path)
    subject.model.eval()
    return path
