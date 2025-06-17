'''
./models/
├── base_model/          # Original Llama model cache
│   ├── config.json
│   ├── tokenizer.json
│   └── pytorch_model.bin
└── finetuned_model/     # Fine-tuned model with LoRA
    ├── adapter_config.json
    ├── adapter_model.bin
    └── tokenizer files
'''
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Import the dataset builder
from dataset_builder import TextDatasetBuilder

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def preprocess_function(examples, tokenizer, max_length=1024):
    """Tokenize the texts"""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )

def download_and_save_model(model_name, cache_dir):
    """Download and save the base model locally"""
    print(f"Downloading model {model_name} to {cache_dir}...")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.save_pretrained(cache_dir)
    
    # Download model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    )
    model.save_pretrained(cache_dir)
    
    print(f"Model downloaded and saved to {cache_dir}")
    return cache_dir

def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen3-8B"  # or "meta-llama/Meta-Llama-3-8B"
    DATA_FOLDER = "/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_llm_papers/llm_papers_text"  # Update this path
    LABELS_FILE = "/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_llm_papers/label_simple.json"  # Update this path
    
    # Model directories
    BASE_MODEL_CACHE = "/data/scratch/mpx602/model/paper_pred/base_model"  # Where to cache the downloaded model
    OUTPUT_DIR = "/data/scratch/mpx602/model/paper_pred/finetuned_model"   # Where to save the fine-tuned model
    
    MAX_LENGTH = 1024  # Adjust based on memory constraints
    
    # Download and cache the base model if not already cached
    config_file = os.path.join(BASE_MODEL_CACHE, "config.json")
    if not os.path.exists(config_file):
        download_and_save_model(MODEL_NAME, BASE_MODEL_CACHE)
    else:
        print(f"Using cached model from {BASE_MODEL_CACHE}")
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Load tokenizer from cached model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare dataset
    dataset_builder = TextDatasetBuilder(DATA_FOLDER, LABELS_FILE, MAX_LENGTH)
    dataset = dataset_builder.load_dataset()
    
    # Print dataset statistics
    stats = dataset_builder.get_dataset_stats(dataset)
    print(f"Dataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Label distribution: {stats['label_distribution']}")
    print(f"Average text length: {stats['text_stats']['avg_length_words']:.2f} words")
    
    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='labels')
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # Load model from cached location
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_CACHE,
        num_labels=2,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Adjust based on GPU memory
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the fine-tuned model
    print(f"Saving fine-tuned model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Final evaluation
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    print(f"Fine-tuned model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()