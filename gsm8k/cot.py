#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Parsing utilities
# -----------------------------
_GSM8K_GOLD_RE = re.compile(r"####\s*([-+]?\d[\d,]*)\s*$")
_LAST_INT_RE = re.compile(r"([-+]?\d[\d,]*)")

def parse_gsm8k_gold(answer_field: str) -> Optional[int]:
    """
    GSM8K gold answers are typically like:
      '... #### 42'
    """
    m = _GSM8K_GOLD_RE.search(answer_field.strip())
    if not m:
        return None
    s = m.group(1).replace(",", "")
    try:
        return int(s)
    except ValueError:
        return None

def strip_think_block(text: str) -> str:
    # Remove any <think> ... </think> blocks
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def parse_model_answer(text: str) -> Optional[int]:
    """
    Try to parse model answer robustly:
    1) If it contains a '#### <int>' line, use that.
    2) Otherwise take the LAST integer appearing in the text.
    """
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
# Prompt building
# -----------------------------
def build_messages(question: str, mode: str) -> List[Dict[str, str]]:
    """
    mode: "think" or "no_think"
    We request a final line formatted as: #### <answer>
    """
    question = question.strip()
    if mode == "think":
        user = (
            f"{question}\n\n"
            "Solve step by step.\n"
            "At the end, output the final answer as a single line in exactly this format:\n"
            "#### <integer>\n"
        )
    else:
        user = (
            f"{question}\n\n"
            "Do NOT show your reasoning. Output only the final answer as a single line in exactly this format:\n"
            "#### <integer>\n"
        )

    return [
        {"role": "user", "content": user}
    ]


# -----------------------------
# Inference
# -----------------------------
@dataclass
class GenParams:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool

def generate_once(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    enable_thinking: bool,
    gen: GenParams,
) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,  # Qwen3 hard switch
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

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

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return decoded.strip()


# -----------------------------
# Main loop
# -----------------------------
def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("gsm8k_qwen3_eval")
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

def load_gsm8k(split: str):
    # Prefer canonical datasets name, fall back if needed
    try:
        return load_dataset("gsm8k", "main", split=split)
    except Exception:
        return load_dataset("openai/gsm8k", "main", split=split)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    ap.add_argument("--cache_dir", type=str, default='/mnt/parscratch/users/lip22fh/ACL2026_paper_predict/models/qwen3_4b/orig', help="Path to model cache directory")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--max_samples", type=int, default=50, help="Limit eval size (0 = all).")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--log_path", type=str, default="qwen3_gsm8k_think_vs_nothink.log")
    ap.add_argument("--deterministic", action="store_true", help="Use greedy decoding (do_sample=False).")
    ap.add_argument("--print_full_outputs", action="store_true", help="Log full model outputs (can be long).")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger = setup_logger(args.log_path)
    logger.info(f"Loading model: {args.model}")
    logger.info(f"Split: {args.split} | max_samples={args.max_samples} | seed={args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=args.cache_dir,
    )
    model.eval()

    ds = load_gsm8k(args.split)
    n_total = len(ds) if args.max_samples == 0 else min(len(ds), args.max_samples)

    # Qwen3 model card suggests different sampling params for thinking vs non-thinking. :contentReference[oaicite:3]{index=3}
    if args.deterministic:
        think_gen = GenParams(args.max_new_tokens, temperature=0.0, top_p=1.0, top_k=0, do_sample=False)
        nothink_gen = GenParams(args.max_new_tokens, temperature=0.0, top_p=1.0, top_k=0, do_sample=False)
        logger.info("Decoding: deterministic greedy (may reduce thinking-mode performance vs recommended sampling).")
    else:
        think_gen = GenParams(args.max_new_tokens, temperature=0.6, top_p=0.95, top_k=20, do_sample=True)
        nothink_gen = GenParams(args.max_new_tokens, temperature=0.7, top_p=0.8, top_k=20, do_sample=True)
        logger.info("Decoding: sampled (thinking: T=0.6,top_p=0.95,top_k=20; non-thinking: T=0.7,top_p=0.8,top_k=20).")

    acc_think = 0
    acc_nothink = 0
    agree = 0
    both_correct = 0

    start = time.time()
    for i in range(n_total):
        ex = ds[i]
        q = ex["question"]
        gold = parse_gsm8k_gold(ex["answer"])

        if gold is None:
            logger.warning(f"[{i+1}/{n_total}] Could not parse gold answer; skipping.")
            continue

        # ---- thinking run ----
        msg_think = build_messages(q, mode="think")
        out_think_raw = generate_once(model, tokenizer, msg_think, enable_thinking=True, gen=think_gen)
        out_think = strip_think_block(out_think_raw)
        pred_think = parse_model_answer(out_think)

        # ---- non-thinking run ----
        msg_nothink = build_messages(q, mode="no_think")
        out_nothink_raw = generate_once(model, tokenizer, msg_nothink, enable_thinking=False, gen=nothink_gen)
        out_nothink = strip_think_block(out_nothink_raw)
        pred_nothink = parse_model_answer(out_nothink)

        think_ok = (pred_think == gold)
        nothink_ok = (pred_nothink == gold)
        same = (pred_think is not None and pred_think == pred_nothink)

        acc_think += int(think_ok)
        acc_nothink += int(nothink_ok)
        agree += int(same)
        both_correct += int(think_ok and nothink_ok)

        # ---- logging ----
        header = f"[{i+1:04d}/{n_total}] GOLD={gold} | THINK={pred_think} ({'OK' if think_ok else 'NO'}) | NO_THINK={pred_nothink} ({'OK' if nothink_ok else 'NO'}) | SAME={same}"
        logger.info(header)

        if args.print_full_outputs:
            logger.info("Q: " + q.replace("\n", "\\n"))
            logger.info("THINK_RAW: " + out_think_raw.replace("\n", "\\n"))
            logger.info("NO_THINK_RAW: " + out_nothink_raw.replace("\n", "\\n"))
        else:
            # compact snippets
            q_snip = q.replace("\n", " ")
            if len(q_snip) > 180:
                q_snip = q_snip[:180] + "..."
            t_snip = out_think_raw.replace("\n", " ")
            nt_snip = out_nothink_raw.replace("\n", " ")
            if len(t_snip) > 200:
                t_snip = t_snip[:200] + "..."
            if len(nt_snip) > 200:
                nt_snip = nt_snip[:200] + "..."
            logger.info(f"Q: {q_snip}")
            logger.info(f"THINK_OUT(snippet): {t_snip}")
            logger.info(f"NO_THINK_OUT(snippet): {nt_snip}")

        # running stats
        seen = i + 1
        logger.info(
            f"RUNNING | acc_think={acc_think/seen:.3f} | acc_no_think={acc_nothink/seen:.3f} | agree={agree/seen:.3f} | both_correct={both_correct/seen:.3f}"
        )

    elapsed = time.time() - start
    logger.info("==== FINAL RESULTS ====")
    logger.info(f"Evaluated: {n_total} examples | elapsed={elapsed:.1f}s")
    logger.info(f"Thinking exact-match accuracy:     {acc_think/n_total:.4f}")
    logger.info(f"Non-thinking exact-match accuracy: {acc_nothink/n_total:.4f}")
    logger.info(f"Delta (think - no_think):          {(acc_think-acc_nothink)/n_total:.4f}")
    logger.info(f"Agreement rate (same prediction):  {agree/n_total:.4f}")
    logger.info(f"Both correct rate:                 {both_correct/n_total:.4f}")
    logger.info(f"Log saved to: {os.path.abspath(args.log_path)}")


if __name__ == "__main__":
    main()
