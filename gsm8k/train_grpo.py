#!/usr/bin/env python3
# GRPO fine-tuning on GSM8K (verifiable reward) with TRL.

import argparse
import re
from typing import Optional, List

from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig


FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")


def extract_final_number(text: str) -> Optional[str]:
    """Prefer GSM8K-style '#### <num>'; fallback to last number in the text."""
    if text is None:
        return None
    m = FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1)
    # fallback: last numeric substring
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def normalize_number(s: Optional[str]) -> Optional[str]:
    """Normalize numeric strings so comparisons are stable."""
    if s is None:
        return None
    s = s.strip().replace(",", "")
    try:
        if "." in s:
            f = float(s)
            if f.is_integer():
                return str(int(f))
            # keep a compact float repr
            return str(f)
        return str(int(s))
    except Exception:
        return None


def build_prompt_qwen3_chat(tokenizer: AutoTokenizer, question: str, enable_thinking: Optional[bool]) -> str:
    """
    Use Qwen3 chat template if available, optionally passing enable_thinking.
    Falls back to a plain-text prompt if apply_chat_template fails.
    """
    system = (
        "You are a careful math solver. Solve the problem.\n"
        "Return your final answer on its own line as: #### <number>\n"
    )
    user = f"Question: {question}\nRemember: final line must be '#### <number>'."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking is not None:
        # Qwen3 supports this switch in its chat template.
        kwargs["enable_thinking"] = enable_thinking

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Some tokenizers don't accept enable_thinking
        kwargs.pop("enable_thinking", None)
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            # total fallback (standard format)
            return system + "\n" + user + "\nAssistant:"
    except Exception:
        return system + "\n" + user + "\nAssistant:"


def gsm8k_correctness_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """
    Binary reward: 1 if the extracted final number matches GSM8K gold, else 0.
    TRL calls reward funcs with prompts/completions and dataset columns. :contentReference[oaicite:5]{index=5}
    """
    rewards = []
    for c, gt in zip(completions, answer):
        pred = normalize_number(extract_final_number(c))
        gold = normalize_number(gt)
        rewards.append(1.0 if (pred is not None and gold is not None and pred == gold) else 0.0)
    return rewards


def gsm8k_format_reward(prompts, completions, **kwargs) -> List[float]:
    """Small shaping reward for including a GSM8K-style final answer marker."""
    rewards = []
    for c in completions:
        rewards.append(0.1 if FINAL_ANSWER_RE.search(c or "") else 0.0)
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B")
    ap.add_argument("--output_dir", type=str, default="grpo-qwen3-gsm8k")

    ap.add_argument("--enable_thinking", type=int, choices=[0, 1], default=1,
                    help="For Qwen3 models: 1 enables thinking mode, 0 disables. (Ignored if unsupported.)")

    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--learning_rate", type=float, default=5e-6)

    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)

    ap.add_argument("--num_generations", type=int, default=4,
                    help="GRPO group size. Effective batch must be divisible by this.")

    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--max_completion_length", type=int, default=256)

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--beta", type=float, default=0.04, help="KL coefficient (keep model close to reference).")

    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--vllm_mode", type=str, default="colocate", choices=["server", "colocate"])
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)

    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--max_train_samples", type=int, default=0,
                    help="0 = use full train split; else truncate for quick runs.")
    ap.add_argument("--max_eval_samples", type=int, default=256)

    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--save_steps", type=int, default=200)

    args = ap.parse_args()

    enable_thinking = bool(args.enable_thinking)

    # GSM8K: openai/gsm8k, config "main" with train/test splits. :contentReference[oaicite:6]{index=6}
    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    eval_ds = load_dataset("openai/gsm8k", "main", split="test")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # TRL GRPO expects left padding. :contentReference[oaicite:7]{index=7}
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(example):
        gold = normalize_number(extract_final_number(example["answer"]))
        prompt = build_prompt_qwen3_chat(tokenizer, example["question"], enable_thinking=enable_thinking)
        return {"prompt": prompt, "answer": gold if gold is not None else ""}

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(preprocess, remove_columns=eval_ds.column_names)

    if args.max_train_samples and args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples and args.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            # Qwen-like module names; adjust if needed.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        remove_unused_columns=False,  # keep "answer" for the reward function :contentReference[oaicite:8]{index=8}

        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,

        temperature=args.temperature,
        top_p=args.top_p,

        beta=args.beta,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,

        # Show generations while training (TRL prints/logs sampled completions when enabled).
        log_completions=True,
        num_completions_to_print=2,

        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )

    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        reward_funcs=[gsm8k_correctness_reward, gsm8k_format_reward],
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
