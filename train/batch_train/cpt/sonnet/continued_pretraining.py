import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from dataset_builder_new import TextDatasetBuilder
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Continued pretraining with Qwen3")
    parser.add_argument("--data_folder", type=str, default="/mnt/parscratch/users/acr24wz/src/iclr/mineru/llm/", help="Path to data folder")
    parser.add_argument("--labels_file", type=str, default="/mnt/parscratch/users/acr24wz/topcon/train/label_simple.json", help="Path to labels JSON file")
    parser.add_argument("--statistics_file", type=str, default="/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/statistics_per_paper.json", help="Path to statistics JSON file (optional)")
    parser.add_argument("--output_dir", type=str, default="/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B/cpt_model", help="Output directory for model")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=10000, help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log metrics every N steps")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use BF16 training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Pre-trained model name or path")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=[0, 1], help="GPU IDs to use for training (e.g., --gpu_ids 0 1)")
    
    args = parser.parse_args()
    
    # Auto-detect available GPUs if none specified
    if not hasattr(args, 'gpu_ids') or args.gpu_ids is None:
        available_gpus = torch.cuda.device_count()
        args.gpu_ids = list(range(available_gpus))
        print(f"Auto-detected {available_gpus} GPUs: {args.gpu_ids}")
    
    # Set CUDA_VISIBLE_DEVICES to only use the specified GPUs
    gpu_ids_str = ','.join(map(str, args.gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    print(f"Using GPU IDs: {args.gpu_ids}")
    print(f"CUDA_VISIBLE_DEVICES set to: {gpu_ids_str}")
    
    # Initialize dataset builder
    print("Loading dataset...")
    builder = TextDatasetBuilder(
        data_folder=args.data_folder,
        labels_file=args.labels_file,
        statistics_file=args.statistics_file,
        max_length=args.max_length
    )
    
    # Load dataset
    dataset = builder.load_dataset()
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Get dataset statistics
    stats = builder.get_dataset_stats(dataset)
    print("\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Accepted papers: {stats['label_distribution']['accepted (1)']}")
    print(f"  Rejected papers: {stats['label_distribution']['rejected (0)']}")
    print(f"  Avg text length: {stats['text_stats']['avg_length_words']:.0f} words")
    
    # Load model and tokenizer
    print(f"\nLoading {args.model_name} model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        trust_remote_code=True,
        device_map="auto",  # Automatically distribute across available GPUs
        max_memory={i: "80GiB" for i in range(len(args.gpu_ids))},  # Set max memory per GPU
        offload_folder="./offload",  # Offload to disk if needed
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Print device mapping
    if hasattr(model, 'hf_device_map'):
        print("Model device mapping:")
        for layer, device in model.hf_device_map.items():
            print(f"  {layer}: {device}")
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    def tokenize_function(examples):
        # For continued pretraining, we just use the text without special formatting
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors=None
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    # Split dataset into train and validation (90/10 split)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8
    )
    
    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        logging_dir=f"{output_dir}/logs",
        dataloader_num_workers=0,  # Disable multiprocessing for multi-GPU setup
        remove_unused_columns=False,
        label_names=[],  # No labels for language modeling
        dataloader_pin_memory=False,
        eval_accumulation_steps=4,
        skip_memory_metrics=True,
        ddp_find_unused_parameters=False,  # For efficiency in DDP
        dataloader_persistent_workers=False,  # Disable persistent workers
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        weight_decay=0.001,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting continued pretraining...")
    print(f"Output directory: {output_dir}")
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Training completed! Model saved to {output_dir}")

if __name__ == "__main__":
    main()
