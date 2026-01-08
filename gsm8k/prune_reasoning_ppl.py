#!/usr/bin/env python3
import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
# Delayed imports to allow setting env vars in main
# from datasets import load_dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Parsing utilities (copied from gsm8k/cot.py)
# -----------------------------
_GSM8K_GOLD_RE = re.compile(r"####\s*([-+]?\d[\d,]*)\s*$")
_LAST_INT_RE = re.compile(r"([-+]?\d[\d,]*)")


def parse_gsm8k_gold(answer_field: str) -> Optional[int]:
    m = _GSM8K_GOLD_RE.search(answer_field.strip())
    if not m:
        return None
    s = m.group(1).replace(",", "")
    try:
        return int(s)
    except ValueError:
        return None


def strip_think_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_model_answer(text: str) -> Optional[int]:
    text = text.strip()
    m = _GSM8K_GOLD_RE.search(text)
    if m:
        s = m.group(1).replace(",", "")
        try:
            return int(s)
        except ValueError:
            return None

    candidates = _LAST_INT_RE.findall(text)
    if not candidates:
        return None
    s = candidates[-1].replace(",", "")
    try:
        return int(s)
    except ValueError:
        return None


# -----------------------------
# Logging / dataset
# -----------------------------

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("gsm8k_qwen3_prune_reasoning")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def load_gsm8k(split: str, cache_dir: Optional[str] = None):
    from datasets import load_dataset
    try:
        return load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    except Exception:
        return load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir)


# -----------------------------
# Prompt building
# -----------------------------

def build_thinking_prompt(question: str) -> List[Dict[str, str]]:
    q = question.strip()
    user = (
        f"{q}\n\n"
        "Solve step by step.\n"
        "At the end, output the final answer as a single line in exactly this format:\n"
        "#### <integer>\n"
    )
    return [{"role": "user", "content": user}]


def build_standard_prompt(question: str) -> List[Dict[str, str]]:
    q = question.strip()
    user = (
        f"{q}\n\n"
        "Output the final answer as a single line in exactly this format:\n"
        "#### <integer>\n"
    )
    return [{"role": "user", "content": user}]


def build_answer_from_reasoning_messages(question: str, pruned_reasoning_text: str) -> List[Dict[str, str]]:
    """Second-pass: condition on an existing (possibly pruned) reasoning chain."""
    q = question.strip()

    # Keep the reasoning in the assistant turn so it is treated as prior context.
    assistant = f"<think>\n{pruned_reasoning_text.strip()}\n</think>"

    user2 = (
        "Based on the reasoning above, output ONLY the final answer as a single line in exactly this format:\n"
        "#### <integer>\n"
    )

    return [
        {"role": "user", "content": q},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": user2},
    ]


# -----------------------------
# Token utilities
# -----------------------------

def _find_subsequence(haystack: Sequence[int], needle: Sequence[int], start: int = 0) -> Optional[int]:
    if not needle:
        return None
    n = len(needle)
    limit = len(haystack) - n
    for i in range(start, limit + 1):
        if haystack[i : i + n] == list(needle):
            return i
    return None


def _get_think_span_in_gen_ids(tokenizer, gen_ids: Sequence[int]) -> Optional[Tuple[int, int]]:
    """Return (start, end) span inside gen_ids that corresponds to content between <think>...</think>.

    The returned span excludes the tag token sequences themselves.
    """
    open_ids = tokenizer.encode("<think>", add_special_tokens=False)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)

    open_at = _find_subsequence(gen_ids, open_ids, start=0)
    if open_at is None:
        return None

    close_at = _find_subsequence(gen_ids, close_ids, start=open_at + len(open_ids))
    if close_at is None:
        return None

    start = open_at + len(open_ids)
    end = close_at
    if start > end:
        return None
    return (start, end)


# -----------------------------
# Inference + NLL
# -----------------------------

@dataclass
class GenParams:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool


def _prepare_inputs(tokenizer, messages: List[Dict[str, str]], enable_thinking: bool, device: torch.device):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer([text], return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def generate_continuation_ids(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    enable_thinking: bool,
    gen: GenParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (prompt_ids, gen_ids) as 1D int64 tensors on CPU."""
    inputs = _prepare_inputs(tokenizer, messages, enable_thinking=enable_thinking, device=model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=gen.max_new_tokens,
            do_sample=gen.do_sample,
            temperature=gen.temperature if gen.do_sample else None,
            top_p=gen.top_p if gen.do_sample else None,
            top_k=gen.top_k if gen.do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    full = out[0]
    gen_ids = full[prompt_len:]
    prompt_ids = full[:prompt_len]
    return prompt_ids.detach().cpu(), gen_ids.detach().cpu()


def compute_shifted_nll_per_label(model, full_ids: torch.Tensor) -> torch.Tensor:
    """Teacher-forced per-token NLL for labels at positions 1..L-1.

    full_ids: shape [L] on CPU or GPU; will be moved to model.device.
    returns: nll tensor of shape [L-1] on CPU.
    """
    input_ids = full_ids.unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [1, L, V]

    shift_logits = logits[:, :-1, :].contiguous()  # predicts labels at 1..L-1
    shift_labels = input_ids[:, 1:].contiguous()

    # Cross entropy per position
    nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size(1))

    return nll.detach().cpu()


def prune_top_fraction_by_nll(reason_ids: Sequence[int], nll_reason: Sequence[float], drop_frac: float) -> Tuple[List[int], List[int]]:
    """Return (kept_reason_ids, dropped_indices)."""
    n = len(reason_ids)
    if n == 0:
        return [], []
    if not (0.0 <= drop_frac <= 1.0):
        raise ValueError(f"drop_frac must be in [0,1], got {drop_frac}")

    drop_k = int(math.ceil(drop_frac * n))
    drop_k = min(drop_k, n)
    if drop_k == 0:
        return list(reason_ids), []

    # Rank by NLL (equivalently perplexity = exp(NLL))
    order = sorted(range(n), key=lambda i: float(nll_reason[i]), reverse=True)
    drop_set = set(order[:drop_k])
    kept = [tok for i, tok in enumerate(reason_ids) if i not in drop_set]
    dropped = sorted(drop_set)
    return kept, dropped


def _parse_drop_fracs(s: str) -> List[float]:
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    ap.add_argument(
        "--cache_dir",
        type=str,
        default="/mnt/parscratch/users/lip22fh/ACL2026_paper_predict/models/qwen3_4b/orig",
        help="Path to model cache directory",
    )
    ap.add_argument(
        "--dataset_cache_dir",
        type=str,
        default="/mnt/parscratch/users/lip22fh/ACL2026_paper_predict/gsm8k/dataset",
        help="Where to cache/download the GSM8K dataset (avoids ~/.cache quota).",
    )
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--max_samples", type=int, default=50, help="Limit eval size (0 = all).")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--max_new_tokens_think", type=int, default=512)
    ap.add_argument("--max_new_tokens_answer", type=int, default=64)

    ap.add_argument(
        "--drop_fracs",
        type=str,
        default="0.1,0.2,0.5",
        help="Comma-separated fractions of reasoning tokens to drop (highest perplexity first).",
    )

    ap.add_argument("--log_path", type=str, default="/mnt/parscratch/users/lip22fh/ACL2026_paper_predict/gsm8k/logs/qwen3_gsm8k_prune_reasoning.log")
    ap.add_argument("--jsonl_path", type=str, default="/mnt/parscratch/users/lip22fh/ACL2026_paper_predict/gsm8k/logs/prune_reasoning_ppl.jsonl", help="Optional JSONL to save per-example stats.")

    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Use greedy decoding for both passes (do_sample=False).",
    )
    ap.add_argument(
        "--print_full_outputs",
        action="store_true",
        help="Log full decoded outputs (can be long).",
    )

    args = ap.parse_args()

    drop_fracs = _parse_drop_fracs(args.drop_fracs)
    for f in drop_fracs:
        if not (0.0 <= f <= 1.0):
            raise SystemExit(f"Invalid --drop_fracs entry {f}; must be within [0,1].")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Set up environment variables BEFORE importing heavy libraries
    if args.dataset_cache_dir:
        os.makedirs(args.dataset_cache_dir, exist_ok=True)
        # Ensure hub + datasets caches live on /mnt/parscratch (not ~/.cache)
        os.environ["HF_HOME"] = args.dataset_cache_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(args.dataset_cache_dir, "datasets")
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(args.dataset_cache_dir, "hub")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger = setup_logger(args.log_path)
    logger.info(f"Loading model: {args.model}")
    logger.info(f"Split: {args.split} | max_samples={args.max_samples} | seed={args.seed}")
    logger.info(f"Drop fracs: {drop_fracs}")

    if args.dataset_cache_dir:
        logger.info(f"HF dataset cache: {args.dataset_cache_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=args.cache_dir,
    )
    model.eval()

    ds = load_gsm8k(args.split, cache_dir=args.dataset_cache_dir)
    n_total = len(ds) if args.max_samples == 0 else min(len(ds), args.max_samples)

    # Generation params (simple defaults; match cot.py semantics)
    if args.deterministic:
        think_gen = GenParams(args.max_new_tokens_think, temperature=0.0, top_p=1.0, top_k=0, do_sample=False)
        ans_gen = GenParams(args.max_new_tokens_answer, temperature=0.0, top_p=1.0, top_k=0, do_sample=False)
        logger.info("Decoding: deterministic greedy.")
    else:
        think_gen = GenParams(args.max_new_tokens_think, temperature=0.6, top_p=0.95, top_k=20, do_sample=True)
        ans_gen = GenParams(args.max_new_tokens_answer, temperature=0.2, top_p=0.95, top_k=20, do_sample=True)
        logger.info("Decoding: sampled (think: T=0.6, top_p=0.95; answer: T=0.2, top_p=0.95).")

    # Metrics
    base_correct = 0
    base_count = 0
    std_correct = 0  # Standard baseline

    pruned_correct: Dict[float, int] = {f: 0 for f in drop_fracs}
    pruned_count: Dict[float, int] = {f: 0 for f in drop_fracs}

    missing_think_span = 0

    jsonl_fh = None
    if args.jsonl_path:
        os.makedirs(os.path.dirname(args.jsonl_path) or ".", exist_ok=True)
        jsonl_fh = open(args.jsonl_path, "w", encoding="utf-8")

    start = time.time()
    for i in range(n_total):
        ex = ds[i]
        q = ex["question"]
        gold = parse_gsm8k_gold(ex["answer"])
        if gold is None:
            logger.warning(f"[{i+1}/{n_total}] Could not parse gold answer; skipping.")
            continue

        # --- Pass 0: Standard Baseline (Base) ---
        msg_std = build_standard_prompt(q)
        _, gen_std_ids = generate_continuation_ids(
            model,
            tokenizer,
            msg_std,
            enable_thinking=False,
            gen=ans_gen,
        )
        decoded_std = tokenizer.decode(gen_std_ids.tolist(), skip_special_tokens=False)
        std_pred = parse_model_answer(strip_think_block(decoded_std))
        std_ok = (std_pred == gold)
        std_correct += int(std_ok)

        # --- Pass 1: generate with thinking (Think) ---
        msg_think = build_thinking_prompt(q)
        prompt_ids, gen_ids = generate_continuation_ids(
            model,
            tokenizer,
            msg_think,
            enable_thinking=True,
            gen=think_gen,
        )

        decoded_gen = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=False)
        decoded_gen_no_think = strip_think_block(decoded_gen)
        base_pred = parse_model_answer(decoded_gen_no_think)
        base_ok = (base_pred == gold)
        base_correct += int(base_ok)
        base_count += 1

        think_span = _get_think_span_in_gen_ids(tokenizer, gen_ids.tolist())

        record: Dict[str, Any] = {
            "i": i,
            "gold": gold,
            "std_pred": std_pred,
            "std_ok": std_ok,
            "base_pred": base_pred,
            "base_ok": base_ok,
            "drop_results": {},
            "reason_tokens": None,
        }

        if think_span is None:
            missing_think_span += 1
            logger.info(
                f"[{i+1:04d}/{n_total}] GOLD={gold} | STD={std_pred} ({'OK' if std_ok else 'NO'}) | THINK={base_pred} ({'OK' if base_ok else 'NO'}) | NOTE=no <think> span found; skipping pruning"
            )
            if args.print_full_outputs:
                logger.info("Q: " + q.replace("\n", "\\n"))
                logger.info("GEN(full): " + decoded_gen.replace("\n", "\\n"))

            # still write record
            if jsonl_fh is not None:
                jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_fh.flush()
            continue

        reason_start, reason_end = think_span
        reason_ids = gen_ids[reason_start:reason_end].tolist()

        # Compute per-token NLL for the full concatenated sequence to get aligned token logprobs.
        full_ids = torch.cat([prompt_ids, gen_ids], dim=0)
        nll_per_label = compute_shifted_nll_per_label(model, full_ids)  # shape [L-1]

        prompt_len = int(prompt_ids.numel())
        # reasoning tokens correspond to full positions [prompt_len + reason_start, ..., prompt_len + reason_end-1]
        reason_full_positions = list(range(prompt_len + reason_start, prompt_len + reason_end))
        # map full position k -> nll index k-1
        nll_reason: List[float] = []
        for pos in reason_full_positions:
            if pos <= 0:
                # should never happen for generated reasoning
                nll_reason.append(float("nan"))
                continue
            nll_reason.append(float(nll_per_label[pos - 1].item()))

        record["reason_tokens"] = {
            "count": len(reason_ids),
            "mean_nll": float(torch.tensor(nll_reason).nanmean().item()) if len(nll_reason) else None,
            "mean_ppl": float(torch.exp(torch.tensor(nll_reason)).nanmean().item()) if len(nll_reason) else None,
        }

        # --- Pass 2: prune + answer ---
        if base_ok:
            for frac in drop_fracs:
                kept_ids, dropped_idx = prune_top_fraction_by_nll(reason_ids, nll_reason, drop_frac=frac)
                pruned_reason_text = tokenizer.decode(kept_ids, skip_special_tokens=False)

                msg_answer = build_answer_from_reasoning_messages(q, pruned_reasoning_text)
                _, gen2_ids = generate_continuation_ids(
                    model,
                    tokenizer,
                    msg_answer,
                    enable_thinking=False,
                    gen=ans_gen,
                )

                decoded2 = tokenizer.decode(gen2_ids.tolist(), skip_special_tokens=False)
                decoded2_no_think = strip_think_block(decoded2)
                pred2 = parse_model_answer(decoded2_no_think)
                ok2 = (pred2 == gold)

                pruned_correct[frac] += int(ok2)
                pruned_count[frac] += 1

                record["drop_results"][str(frac)] = {
                    "pred": pred2,
                    "ok": ok2,
                    "dropped": len(dropped_idx),
                    "kept": len(kept_ids),
                }

                if args.print_full_outputs:
                    record["drop_results"][str(frac)]["answer_raw"] = decoded2
        else:
            # If thinking is incorrect, skip pruning to save compute, record as incorrect/skipped
            for frac in drop_fracs:
                pruned_count[frac] += 1
                record["drop_results"][str(frac)] = {"skipped": True, "ok": False}

        # --- Logging ---
        header = f"[{i+1:04d}/{n_total}] GOLD={gold} | STD={std_pred} ({'OK' if std_ok else 'NO'}) | THINK={base_pred} ({'OK' if base_ok else 'NO'})"
        for frac in drop_fracs:
            if base_ok:
                r = record["drop_results"][str(frac)]
                header += f" | DROP{frac:g}: {r['pred']} ({'OK' if r['ok'] else 'NO'})"
            else:
                header += f" | DROP{frac:g}: SKIP"
        logger.info(header)

        if args.print_full_outputs:
            logger.info("Q: " + q.replace("\n", "\\n"))
            logger.info("GEN1(full): " + decoded_gen.replace("\n", "\\n"))
        else:
            q_snip = q.replace("\n", " ")
            if len(q_snip) > 180:
                q_snip = q_snip[:180] + "..."
            gen_snip = decoded_gen.replace("\n", " ")
            if len(gen_snip) > 220:
                gen_snip = gen_snip[:220] + "..."
            logger.info(f"Q: {q_snip}")
            logger.info(f"GEN1(snippet): {gen_snip}")

        if jsonl_fh is not None:
            jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_fh.flush()

        seen = base_count
        if seen > 0:
            msg = f"RUNNING | think_acc={base_correct/seen:.3f} | base_acc={std_correct/seen:.3f}"
            for frac in drop_fracs:
                denom = pruned_count[frac] or 1
                msg += f" | think(prune{frac:g})_acc={pruned_correct[frac]/denom:.3f}"
            logger.info(msg)

    elapsed = time.time() - start
    if jsonl_fh is not None:
        jsonl_fh.close()

    logger.info("==== FINAL RESULTS ====")
    logger.info(f"Evaluated: {base_count} examples | elapsed={elapsed:.1f}s")
    logger.info(f"Missing <think> span: {missing_think_span}")

    if base_count > 0:
        logger.info(f"Method: think (Full Reasoning) accuracy: {base_correct/base_count:.4f}")
        for frac in drop_fracs:
            denom = pruned_count[frac]
            if denom > 0:
                logger.info(f"Method: think(prune{frac:g}) accuracy: {pruned_correct[frac]/denom:.4f}")
        logger.info(f"Method: base (Standard IO) accuracy: {std_correct/base_count:.4f}")

    logger.info(f"Log saved to: {os.path.abspath(args.log_path)}")
    if args.jsonl_path:
        logger.info(f"JSONL saved to: {os.path.abspath(args.jsonl_path)}")


if __name__ == "__main__":
    main()
