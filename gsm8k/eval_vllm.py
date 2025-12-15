#!/usr/bin/env python3
import argparse
import asyncio
import os
import random
import re
import time
import uuid
from typing import Optional, Tuple, List, Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM  # vLLM V1 async streaming engine


# -----------------------------
# GSM8K parsing
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
    ints = _LAST_INT_RE.findall(text)
    if not ints:
        return None
    s = ints[-1].replace(",", "")
    try:
        return int(s)
    except ValueError:
        return None


# -----------------------------
# Minimal "tee" logger (streams + file)
# -----------------------------
class TeeLogger:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "w", encoding="utf-8") if path else None

    def line(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        out = f"{ts} | {msg}"
        print(out, flush=True)
        if self.f:
            self.f.write(out + "\n")
            self.f.flush()

    def raw(self, s: str):
        print(s, end="", flush=True)
        if self.f:
            self.f.write(s)
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


# -----------------------------
# Prompt building (Qwen3 hard switch via chat template)
# -----------------------------
def build_messages(question: str, mode: str) -> List[Dict[str, str]]:
    q = question.strip()
    if mode == "think":
        user = (
            f"{q}\n\n"
            "Solve step by step.\n"
            "At the end, output the final answer as a single line in exactly this format:\n"
            "#### <integer>\n"
        )
    else:
        user = (
            f"{q}\n\n"
            "Do NOT show your reasoning. Output only the final answer as a single line in exactly this format:\n"
            "#### <integer>\n"
        )
    return [{"role": "user", "content": user}]


def load_gsm8k_split(split: str):
    # GSM8K on HF is typically gsm8k/main with train+test (sometimes validation naming differs).
    tried = []
    for ds_name in [("gsm8k", "main"), ("openai/gsm8k", "main")]:
        try:
            return load_dataset(ds_name[0], ds_name[1], split=split)
        except Exception as e:
            tried.append((ds_name, str(e)))

    if split == "test":
        # fallback: some configs label it as validation
        for ds_name in [("gsm8k", "main"), ("openai/gsm8k", "main")]:
            try:
                return load_dataset(ds_name[0], ds_name[1], split="validation")
            except Exception as e:
                tried.append((ds_name + ("validation",), str(e)))

    raise RuntimeError("Failed to load GSM8K split. Tried: " + repr(tried))


# -----------------------------
# vLLM streaming helper
# -----------------------------
async def stream_generate(
    engine: AsyncLLM,
    prompt: str,
    sampling_params: SamplingParams,
    tee: TeeLogger,
    label: str,
) -> str:
    """
    Streams DELTA tokens to stdout+log and returns full generated text.
    """
    request_id = f"{label}-{uuid.uuid4().hex}"
    tee.line(f"{label} | streaming start (request_id={request_id})")
    tee.raw(f"{label} OUTPUT:\n")

    acc = ""
    async for output in engine.generate(request_id=request_id, prompt=prompt, sampling_params=sampling_params):
        for completion in output.outputs:
            new_text = completion.text
            if new_text:
                tee.raw(new_text)
                acc += new_text
        if output.finished:
            break

    tee.raw("\n")  # newline after stream
    tee.line(f"{label} | streaming done")
    return acc.strip()


# -----------------------------
# Main
# -----------------------------
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    ap.add_argument("--cache_dir", type=str, default=None, help="Path to model cache directory")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--max_samples", type=int, default=20, help="0 = all")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--log_path", type=str, default="qwen3_gsm8k_vllm_stream.log")
    ap.add_argument("--enforce_eager", action="store_true", help="Faster startup; slightly lower throughput.")
    ap.add_argument("--no_stream_question", action="store_true", help="Don’t print full question text.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tee = TeeLogger(args.log_path)
    tee.line(f"Model={args.model} | split={args.split} | max_samples={args.max_samples} | seed={args.seed}")
    tee.line("Loading tokenizer (transformers) and vLLM Async engine...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)

    engine_args = AsyncEngineArgs(
        model=args.model,
        download_dir=args.cache_dir,
        enforce_eager=args.enforce_eager,
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    ds = load_gsm8k_split(args.split)
    n_total = len(ds) if args.max_samples == 0 else min(len(ds), args.max_samples)

    # Qwen recommended sampling params for thinking vs non-thinking. :contentReference[oaicite:5]{index=5}
    def mk_params(thinking: bool, seed: int) -> SamplingParams:
        if thinking:
            return SamplingParams(
                max_tokens=args.max_tokens,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                seed=seed,
                output_kind=RequestOutputKind.DELTA,  # stream only new tokens :contentReference[oaicite:6]{index=6}
            )
        return SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            seed=seed,
            output_kind=RequestOutputKind.DELTA,
        )

    acc_think = 0
    acc_nothink = 0
    agree = 0
    both_correct = 0

    t0 = time.time()

    for i in range(n_total):
        ex = ds[i]
        q = ex["question"]
        gold = parse_gsm8k_gold(ex["answer"])
        if gold is None:
            tee.line(f"[{i+1}/{n_total}] WARNING: could not parse gold answer; skipping.")
            continue

        tee.line("=" * 90)
        tee.line(f"[{i+1:04d}/{n_total}] GOLD={gold}")
        if not args.no_stream_question:
            tee.line("QUESTION:")
            tee.raw(q.strip() + "\n")

        # Build two prompts using Qwen3 hard switch in chat template. :contentReference[oaicite:7]{index=7}
        msgs_think = build_messages(q, "think")
        prompt_think = tokenizer.apply_chat_template(
            msgs_think,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        msgs_nothink = build_messages(q, "no_think")
        prompt_nothink = tokenizer.apply_chat_template(
            msgs_nothink,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Use same per-item seed for both runs (makes comparisons less noisy).
        item_seed = args.seed + i

        out_think_raw = await stream_generate(engine, prompt_think, mk_params(True, item_seed), tee, "THINK")
        out_nothink_raw = await stream_generate(engine, prompt_nothink, mk_params(False, item_seed), tee, "NO_THINK")

        # Parse answers
        out_think_clean = strip_think_block(out_think_raw)
        out_nothink_clean = strip_think_block(out_nothink_raw)

        pred_think = parse_model_answer(out_think_clean)
        pred_nothink = parse_model_answer(out_nothink_clean)

        think_ok = (pred_think == gold)
        nothink_ok = (pred_nothink == gold)
        same = (pred_think is not None and pred_think == pred_nothink)

        acc_think += int(think_ok)
        acc_nothink += int(nothink_ok)
        agree += int(same)
        both_correct += int(think_ok and nothink_ok)

        tee.line(
            f"RESULT | THINK={pred_think} ({'OK' if think_ok else 'NO'}) "
            f"| NO_THINK={pred_nothink} ({'OK' if nothink_ok else 'NO'}) "
            f"| SAME_PRED={same}"
        )

        seen = i + 1
        tee.line(
            f"RUNNING | acc_think={acc_think/seen:.3f} | acc_no_think={acc_nothink/seen:.3f} "
            f"| agree={agree/seen:.3f} | both_correct={both_correct/seen:.3f}"
        )

    elapsed = time.time() - t0
    tee.line("=" * 90)
    tee.line(f"FINAL | evaluated={n_total} | elapsed={elapsed:.1f}s")
    tee.line(f"Thinking accuracy:     {acc_think/n_total:.4f}")
    tee.line(f"Non-thinking accuracy: {acc_nothink/n_total:.4f}")
    tee.line(f"Delta:                {(acc_think-acc_nothink)/n_total:.4f}")
    tee.line(f"Agreement rate:       {agree/n_total:.4f}")
    tee.line(f"Both correct rate:    {both_correct/n_total:.4f}")
    tee.line(f"Log saved to: {os.path.abspath(args.log_path)}")

    engine.shutdown()
    tee.close()


if __name__ == "__main__":
    asyncio.run(main())
