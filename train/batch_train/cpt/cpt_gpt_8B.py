import argparse, os, math, glob, json, random, types
from dataclasses import dataclass
from typing import Optional, Dict, List
import datasets
from datasets import Dataset, DatasetDict
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from dataset_builder_new import TextDatasetBuilder
from dataset_cache import DatasetCache  # Import the new caching module
from model_cache import ModelCache  # Import the new model caching module
from datetime import datetime

def detect_loader(data_path: str):
    p = data_path
    if os.path.isdir(p):
        # all .txt/.md under a folder
        files = sorted(glob.glob(os.path.join(p, "**/*.txt"), recursive=True) + 
                       glob.glob(os.path.join(p, "**/*.md"), recursive=True))
        if not files:
            raise FileNotFoundError("No .txt or .md files found under the folder.")
        return {"train": files}
    # single file
    ext = os.path.splitext(p)[1].lower()
    if ext in [".txt", ".md"]:
        return {"train": [p]}
    if ext in [".jsonl", ".json"]:
        return {"train": p}
    raise ValueError("Unsupported data path. Use a folder of .txt/.md, or a .txt/.md/.jsonl/.json file.")

def load_papers_dataset(spec: Dict[str, str|List[str]]):
    # text loader
    if isinstance(spec["train"], list):
        return load_dataset("text", data_files={"train": spec["train"]})
    # json/jsonl loader; expect a 'text' field; if not, try to infer
    ds = load_dataset("json", data_files={"train": spec["train"]})
    cols = ds["train"].column_names
    text_col = "text"
    if text_col not in cols:
        # try common alternatives
        for c in ["content", "body", "paper", "raw_text"]:
            if c in cols:
                text_col = c; break
        else:
            raise KeyError(f"No 'text' column found. Available columns: {cols}")
    if text_col != "text":
        ds = ds.rename_column(text_col, "text")
    return ds
# def _safe_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#     outputs = model(**inputs)
#     loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
#     if isinstance(num_items_in_batch, torch.Tensor):
#         # either make it a CPU scalar...
#         num_items_in_batch = int(num_items_in_batch.detach().cpu().item())
#         # ...or, alternatively:
#         # num_items_in_batch = num_items_in_batch.to(loss.device)
#     if num_items_in_batch is not None:
#         loss = loss / num_items_in_batch
#     return (loss, outputs) if return_outputs else loss
import types, torch

def _safe_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # Convert to a plain Python number so HF loss can safely divide without device issues
    if isinstance(num_items_in_batch, torch.Tensor):
        num_items_in_batch = int(num_items_in_batch.detach().cpu().item())
    elif num_items_in_batch is not None:
        num_items_in_batch = int(num_items_in_batch)

    # Important: pass num_items_in_batch into the model; DO NOT divide here
    outputs = model(**inputs, num_items_in_batch=num_items_in_batch)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
    return (loss, outputs) if return_outputs else loss

def main():
    ap = argparse.ArgumentParser(description="Continued pretraining with Qwen3")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Qwen/Qwen3-4B or Qwen/Qwen3-8B")
    ap.add_argument("--data_path", type=str, default="/mnt/parscratch/users/acr24wz/src/iclr/mineru/llm/", help="Folder of .txt/.md OR a .txt/.md/.jsonl/.json file with a 'text' field")
    ap.add_argument("--labels_file", type=str, default="/mnt/parscratch/users/acr24wz/topcon/train/label_simple.json", help="Path to labels JSON file")
    ap.add_argument("--statistics_file", type=str, default="/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/statistics_per_paper.json", help="Path to statistics JSON file (optional)")
    
    ap.add_argument("--cache_dir", type=str, default="/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/cpt/llm/", help="Directory to store cached datasets")
    ap.add_argument("--model_cache_dir", type=str, default="/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_8B", help="Directory to store cached models")
    ap.add_argument("--output_dir", type=str, default="/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_8B/cpt_model/llm")
    
    
    ap.add_argument("--batch_size", type=int, default=2, help="Per-GPU micro-batch size")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--block_size", type=int, default=2048, help="Pack tokens to this length; pick what fits memory (e.g., 2048/4096/8192)")
    ap.add_argument("--max_length", type=int, default=10000, help="Maximum sequence length")
    
    ap.add_argument("--max_steps", type=int, default=-1, help="Set >0 to override epochs")
    ap.add_argument("--lr", type=float, default=1e-5, help="Learning rate for CPT")
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation to reach effective batch")
    ap.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps")
    ap.add_argument("--logging_steps", type=int, default=50, help="Log metrics every N steps")
    ap.add_argument("--deepspeed", type=str, default=None, help="Path to a DeepSpeed ZeRO json (optional)")
    ap.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 (recommended on A100/H100)")
    ap.add_argument("--fp16", action="store_true", help="Use fp16 instead (if no bf16)")
    ap.add_argument("--num_proc", type=int, default=4, help="Preprocessing workers")
    ap.add_argument("--eval_holdout", type=int, default=100, help="Number of samples to hold out for perplexity eval; set 0 to disable")
    ap.add_argument("--flash_attn", action="store_true", help="Try FlashAttention-2 if installed")
    ap.add_argument("--use_dataset_builder", action="store_true", default=True, help="Use dataset_builder_new instead of direct file loading")
    ap.add_argument("--use_cache", action="store_true", default=True, help="Use dataset caching to speed up repeated runs")
    ap.add_argument("--eval", action="store_true", help="Run in evaluation mode using fine-tuned model")
    
    ap.add_argument("--gpu_ids", type=int, nargs='+', default=None, help="GPU IDs to use for training (e.g., --gpu_ids 0 1). If not specified, uses all available GPUs")
    
    args = ap.parse_args()

    # Handle GPU selection - use all GPUs if not specified
    if args.gpu_ids is None:
        # Use all available GPUs
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            raise RuntimeError("No GPUs available!")
        args.gpu_ids = list(range(available_gpus))
        print(f"No GPU IDs specified. Using all {available_gpus} available GPUs: {args.gpu_ids}")
    else:
        # Validate specified GPU IDs
        available_gpus = torch.cuda.device_count()
        for gpu_id in args.gpu_ids:
            if gpu_id >= available_gpus:
                raise ValueError(f"GPU ID {gpu_id} not available. Only {available_gpus} GPUs detected (0-{available_gpus-1})")
        print(f"Using specified GPU IDs: {args.gpu_ids}")
    
    # Set CUDA_VISIBLE_DEVICES
    gpu_ids_str = ','.join(map(str, args.gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    print(f"CUDA_VISIBLE_DEVICES set to: {gpu_ids_str}")

    # 1) Load tokenizer & model with caching support
    model_cache_handler = ModelCache(cache_dir=args.model_cache_dir)
    
    # Determine if we're using fine-tuned model for evaluation
    use_finetuned = args.eval
    finetuned_path = args.output_dir  # Always use output_dir for fine-tuned models
    
    # Load tokenizer with caching
    tokenizer = model_cache_handler.load_or_download_tokenizer(
        model_name=args.model_name,
        use_fast=True,
        trust_remote_code=True,
        use_finetuned=use_finetuned,
        finetuned_path=finetuned_path if use_finetuned else None
    )
    
    # For CLM we don't need padding at training time, but set pad to eos to be safe
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "sdpa" if args.flash_attn else "eager"
    
    # Load model with caching support
    model, model_path = model_cache_handler.load_or_download_model(
        model_name=args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        attn_implementation=attn_impl,
        trust_remote_code=True,
        device_map="auto",  # Automatically distribute across available GPUs
        max_memory={i: "80GiB" for i in range(len(args.gpu_ids))},  # Set max memory per GPU
        offload_folder="./offload",  # Offload to disk if needed
        use_finetuned=use_finetuned,
        finetuned_path=finetuned_path if use_finetuned else None
    )

    if hasattr(model, "config"):
        model.config.use_cache = False  # Disable caching for training
    
    print(f"{'Evaluation' if args.eval else 'Training'} mode using model from {model_path}")

    # Print device mapping
    if hasattr(model, 'hf_device_map'):
        print("Model device mapping:")
        for layer, device in model.hf_device_map.items():
            print(f"  {layer}: {device}")

    # 2) Load dataset - use dataset_builder_new if specified
    if args.use_dataset_builder and args.labels_file:
        print("Loading dataset using dataset_builder_new...")
        builder = TextDatasetBuilder(
            data_folder=args.data_path,
            labels_file=args.labels_file,
            statistics_file=None,  # Don't include statistics for CPT
            max_length=args.max_length
        )
        
        # Load dataset with caching support
        if args.use_cache:
            cache_handler = DatasetCache(cache_dir=args.cache_dir)
            dataset = cache_handler.load_or_process_dataset(builder, use_ids=False)
        else:
            dataset = builder.load_dataset()
        
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # Get dataset statistics
        stats = builder.get_dataset_stats(dataset)
        print("\nDataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Accepted papers: {stats['label_distribution']['accepted (1)']}")
        print(f"  Rejected papers: {stats['label_distribution']['rejected (0)']}")
        print(f"  Avg text length: {stats['text_stats']['avg_length_words']:.0f} words")
        
        # Print dataset configuration if using cache
        if args.use_cache:
            print("\nDataset Configuration:")
            print(f"  Data folder: {args.data_path}")
            print(f"  Labels file: {args.labels_file}")
            print(f"  Max length: {args.max_length}")
            print(f"  Cache directory: {args.cache_dir}")
        
        # Convert to DatasetDict format expected by the rest of the code
        # Extract only text column for CPT (no labels needed)
        ds = Dataset.from_dict({"text": dataset["text"]})
        
        # Optional eval split for PPL
        if args.eval_holdout and args.eval_holdout > 0:
            total_samples = len(ds)
            if args.eval_holdout >= total_samples:
                print(f"Warning: eval_holdout ({args.eval_holdout}) >= total samples ({total_samples}). Using 10% instead.")
                test_size = 0.1
            else:
                test_size = args.eval_holdout
                print(f"Using {args.eval_holdout} samples for evaluation holdout")
            split = ds.train_test_split(test_size=test_size, seed=42, shuffle=True)
            ds = DatasetDict(train=split["train"], eval=split["test"])
        else:
            ds = DatasetDict(train=ds)
    else:
        # Original file-based loading logic
        spec = detect_loader(args.data_path)
        ds = load_papers_dataset(spec)  # expects "text" column
        
        # Optional tiny eval split for PPL
        if args.eval_holdout and args.eval_holdout > 0:
            ds = ds["train"].train_test_split(test_size=args.eval_holdout, seed=42)
            ds = DatasetDict(train=ds["train"], eval=ds["test"])
        else:
            ds = DatasetDict(train=ds["train"])

    print(f"Train samples: {len(ds['train'])}")
    if "eval" in ds:
        print(f"Eval samples: {len(ds['eval'])}")

    # 3) Tokenize + pack to constant length
    eos = tokenizer.eos_token or ""
    def tokenize_fn(batch):
        # append EOS so documents are separable when concatenated
        texts = [(t if t is not None else "") + eos for t in batch["text"]]
        return tokenizer(texts, add_special_tokens=False)

    cols_to_remove = [c for c in ds["train"].column_names if c != "text"]
    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in ds["train"].column_names],  # keep only token fields
        num_proc=max(1, args.num_proc),
        desc="Tokenizing",
    )

    def group_texts(examples):
        # Concatenate
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        # Drop remainder to avoid short tail
        total_length = (total_length // args.block_size) * args.block_size
        result = {}
        for k in concatenated.keys():
            result[k] = [concatenated[k][i:i + args.block_size]
                         for i in range(0, total_length, args.block_size)]
        result["labels"] = result["input_ids"].copy()
        return result

    packed = tokenized.map(
        group_texts,
        batched=True,
        num_proc=max(1, args.num_proc),
        desc=f"Packing into chunks of {args.block_size}",
    )

    # 4) Collator (creates labels if not provided; we already added labels for clarity)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5) Training args - adjusted to match continued_pretraining.py settings
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}" if not args.eval else f"{args.output_dir}_eval_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=not args.eval,  # Don't train in eval mode
        do_eval=("eval" in packed) or args.eval,  # Always eval in eval mode
        eval_strategy=("steps" if "eval" in packed or args.eval else "no"),
        eval_steps=args.save_steps if "eval" in packed or args.eval else None,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.1,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        num_train_epochs=args.epochs if args.max_steps < 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
        dataloader_num_workers=0,  # Disable multiprocessing for multi-GPU setup (matching continued_pretraining.py)
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to=["tensorboard"],
        logging_dir=f"{output_dir}/logs",
        dataloader_pin_memory=False,
        eval_accumulation_steps=4,
        skip_memory_metrics=True,
        dataloader_persistent_workers=False,
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        optim="adamw_torch",
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=packed["train"],
        eval_dataset=packed["eval"] if "eval" in packed else None,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    # safe compute_loss to handle DDP
    trainer.compute_loss = types.MethodType(_safe_compute_loss, trainer)

    # 7) Train or Evaluate
    if args.eval:
        print("\nRunning evaluation on fine-tuned model...")
        metrics = trainer.evaluate()
        print(f"Evaluation results: {metrics}")
        try:
            ppl = math.exp(metrics["eval_loss"])
            print(f"Perplexity: {ppl:.2f}")
        except Exception:
            pass
    else:
        print("\nStarting continued pretraining...")
        print(f"Output directory: {output_dir}")
        trainer.train()
        
        # 8) Optional: compute perplexity on eval
        if "eval" in packed:
            metrics = trainer.evaluate()
            try:
                ppl = math.exp(metrics["eval_loss"])
                print(f"Perplexity: {ppl:.2f}")
            except Exception:
                pass
        
        # 9) Save
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\n✅ Training completed! Model saved to {output_dir}")

if __name__ == "__main__":
    main()