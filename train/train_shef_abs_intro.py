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
from sklearn.model_selection import train_test_split
import numpy as np

# Import the dataset builder
from dataset_builder_abs_intro import TextDatasetBuilder
from datasets import load_from_disk

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
    # examples['text'] is already a list when batched=True
    result = tokenizer(
        examples['text'],
        truncation=True,
        padding=False,  # Let the data collator handle padding
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=False
    )
    # Ensure labels are included in the output
    result['labels'] = examples['labels']
    return result

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

def split_dataset_stratified(dataset, test_size=0.2, seed=42):
    """Split dataset with stratification using sklearn"""
    texts = dataset['text']
    labels = dataset['labels']
    
    # Use sklearn for stratified split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=seed, 
        stratify=labels
    )
    
    # Create new datasets
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'labels': train_labels
    })
    
    test_dataset = Dataset.from_dict({
        'text': test_texts,
        'labels': test_labels
    })
    
    return {'train': train_dataset, 'test': test_dataset}

def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen3-1.7B"  # or "meta-llama/Meta-Llama-3-8B"
    DATA_FOLDER = "/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_llm_papers/llm_papers_text/"  # Update this path
    LABELS_FILE = "/mnt/parscratch/users/acr24wz/topcon/train/llm_paper/label_simple.json"  # Update this path
    
    # Model directories
    BASE_MODEL_CACHE = "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_1d7B"  # Where to cache the downloaded model
    OUTPUT_DIR = "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_1d7B/finetuned_model"   # Where to save the fine-tuned model
    
    MAX_LENGTH = 4096 # Adjust based on memory constraints
    
    # Create base model cache directory if it doesn't exist
    os.makedirs(BASE_MODEL_CACHE, exist_ok=True)
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        r=8,  # Much smaller rank for stability
        lora_alpha=16,  # Reduced alpha
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # Only target key projection layers
    )
    
    # Load tokenizer from cached model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare dataset
    print("Loading dataset...")
    
    # Define processed dataset cache path
    PROCESSED_DATASET_CACHE = "/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset"
    os.makedirs(PROCESSED_DATASET_CACHE, exist_ok=True)
    
    # Check if processed dataset exists
    if os.path.exists(os.path.join(PROCESSED_DATASET_CACHE, "dataset_dict.json")):
        print("Loading cached processed dataset...")
        dataset = load_from_disk(PROCESSED_DATASET_CACHE)
        # Still need to create dataset_builder for stats
        dataset_builder = TextDatasetBuilder(DATA_FOLDER, LABELS_FILE, MAX_LENGTH)
    else:
        print("Processing dataset for the first time...")
        dataset_builder = TextDatasetBuilder(DATA_FOLDER, LABELS_FILE, MAX_LENGTH)
        dataset = dataset_builder.load_dataset()
        
        # Save processed dataset to cache
        print(f"Saving processed dataset to {PROCESSED_DATASET_CACHE}")
        dataset.save_to_disk(PROCESSED_DATASET_CACHE)
    
    # Print dataset statistics
    stats = dataset_builder.get_dataset_stats(dataset)
    print(f"Dataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Label distribution: {stats['label_distribution']}")
    print(f"Average text length: {stats['text_stats']['avg_length_words']:.2f} words")
    
    # Split dataset using sklearn for proper stratification
    print("Splitting dataset...")
    train_test_split_result = split_dataset_stratified(dataset, test_size=0.2, seed=42)
    train_dataset = train_test_split_result['train']
    eval_dataset = train_test_split_result['test']
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set: {len(eval_dataset)} samples")
    
    # Debug: Check data types and first few samples
    print(f"Sample train data: {train_dataset[0]}")
    print(f"Type of text: {type(train_dataset[0]['text'])}")
    print(f"Type of labels: {type(train_dataset[0]['labels'])}")
    print(f"First text sample length: {len(train_dataset[0]['text'])}")
    print(f"Text preview: {train_dataset[0]['text'][:100]}...")
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    
    def tokenize_function(examples):
        # Simple tokenization without excessive debug output
        result = tokenizer(
            examples['text'],
            truncation=True,
            padding=False,  # Let DataCollator handle padding
            max_length=MAX_LENGTH,
        )
        return result
    
    # Process small batches to avoid memory issues
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,  # Process in smaller batches
        remove_columns=['text']  # Remove text column but keep labels
    )
    
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,  # Process in smaller batches
        remove_columns=['text']  # Remove text column but keep labels
    )
    
    print("Tokenization complete")
    print(f"Train dataset columns: {train_dataset.column_names}")
    print(f"Train dataset sample: {train_dataset[0]}")
    
    # Debug tensor shapes
    sample = train_dataset[0]
    print(f"Sample input_ids type: {type(sample['input_ids'])}")
    print(f"Sample input_ids length: {len(sample['input_ids'])}")
    print(f"Sample labels type: {type(sample['labels'])}")
    print(f"Sample labels value: {sample['labels']}")
    
    # Load model from cached location
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_CACHE,
        num_labels=2,
        torch_dtype=torch.float16
    )
    
    # Initialize the classification head properly
    if hasattr(model, 'classifier'):
        torch.nn.init.normal_(model.classifier.weight, std=0.02)
        if model.classifier.bias is not None:
            torch.nn.init.zeros_(model.classifier.bias)
    elif hasattr(model, 'score'):
        torch.nn.init.normal_(model.score.weight, std=0.02)
        if model.score.bias is not None:
            torch.nn.init.zeros_(model.score.bias)

    print(model) # <--- ADD THIS LINE
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Smaller accumulation
        learning_rate=1e-6,  # Much smaller learning rate
        warmup_steps=20,  # Smaller warmup
        weight_decay=0.001,  # Smaller weight decay
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=False,  # Disable fp16 for more stability
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=["labels"],
        max_grad_norm=0.5,  # Much stricter gradient clipping
        adam_epsilon=1e-8,
        lr_scheduler_type="constant",  # Constant learning rate
        optim="adamw_torch"  # Use standard AdamW
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