# GSM8K Fine-tuning and Evaluation

This directory contains scripts for fine-tuning and evaluating models (specifically Qwen3) on the GSM8K math reasoning dataset.

## Files

### 1. `train_grpo.py`
Fine-tunes a model using Group Relative Policy Optimization (GRPO) with the TRL library. It optimizes the model to generate correct final answers verifiable against GSM8K gold labels.

**Key Features:**
- **GRPO Training:** Uses group relative policy optimization instead of PPO.
- **Qwen3 Support:** Handles `enable_thinking` flags for Qwen3 chat templates.
- **Efficiency:** Supports LoRA (Low-Rank Adaptation) and vLLM for fast generation during training.
- **Rewards:** Implements correctness rewards (checking the final number) and format rewards.

**Usage Example:**
```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen3-4B \
    --output_dir output/grpo_run \
    --use_lora \
    --use_vllm \
    --num_generations 4 \
    --max_train_samples 1000
```

### 2. `eval_vllm.py`
A high-throughput evaluation script using the vLLM library. It is designed to compare the model's performance in "Thinking" mode vs "Non-Thinking" mode side-by-side.

**Key Features:**
- **Async Inference:** Uses vLLM's async engine for streaming output.
- **Comparison:** Runs both "Think" and "No Think" prompts for every question to measure the delta in accuracy.
- **Streaming:** Logs token streams to stdout and a log file in real-time.

**Usage Example:**
```bash
python eval_vllm.py \
    --model Qwen/Qwen3-4B \
    --split test \
    --max_samples 100 \
    --log_path logs/eval_vllm.log
```

### 3. `cot.py`
A standard PyTorch/Transformers evaluation script for Chain-of-Thought (CoT) reasoning. This is useful for debugging or running evaluations in environments where vLLM is not installed.

**Key Features:**
- **Standard HF Stack:** Uses `AutoModelForCausalLM` and `AutoTokenizer`.
- **Flexible Sampling:** Allows detailed configuration of temperature, top_p, and top_k.
- **Detailed Logging:** Can log full prompts and raw model outputs for inspection.

**Usage Example:**
```bash
python cot.py \
    --model Qwen/Qwen3-4B \
    --max_samples 50 \
    --deterministic \
    --log_path logs/cot_eval.log
```

### 4. `prune_reasoning_ppl.py`
Computes a per-token perplexity proxy (via teacher-forced negative log probability) for tokens inside the model-emitted `<think>...</think>` block, drops a fraction of the highest-perplexity reasoning tokens, and re-runs a second-pass answer generation conditioned on the shortened reasoning chain.

**Key Features:**
- **Per-token NLL/PPL:** Uses teacher forcing over the exact generated token IDs.
- **Pruning:** Drops top `--drop_fracs` fraction (e.g., 0.1/0.2/0.5) of reasoning tokens by NLL (equivalently perplexity).
- **Two-pass evaluation:** Pass 1 generates reasoning; Pass 2 answers with `enable_thinking=False` using the pruned reasoning as context.

**Usage Example:**
```bash
python prune_reasoning_ppl.py \
    --model Qwen/Qwen3-4B \
    --split test \
    --max_samples 100 \
    --drop_fracs 0.1,0.2,0.5 \
    --deterministic \
    --log_path logs/prune_reasoning_ppl.log \
    --jsonl_path logs/prune_reasoning_ppl.jsonl
```

## Requirements

Ensure you have the following libraries installed:

```bash
pip install torch transformers trl peft datasets vllm accelerate
```
