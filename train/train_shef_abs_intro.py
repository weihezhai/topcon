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
    DataCollatorForLanguageModeling,  # Changed for causal LM
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn

# Import the dataset builder
from dataset_builder_abs_intro import TextDatasetBuilder
from datasets import load_from_disk
from accelerate import Accelerator

class CustomDataCollator:
    """Custom data collator that handles variable-length sequences"""
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        # Find the maximum length in this batch
        max_len = max(len(f['input_ids']) for f in features)
        max_len = min(max_len, self.max_length)  # Don't exceed max_length
        
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        for feature in features:
            input_ids = feature['input_ids'][:max_len]
            attention_mask = feature['attention_mask'][:max_len]
            labels = feature['labels'][:max_len]
            
            # Pad to max_len
            pad_length = max_len - len(input_ids)
            if pad_length > 0:
                input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
                labels.extend([-100] * pad_length)
            
            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['labels'].append(labels)
        
        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch

def compute_metrics(eval_pred):
    """Compute metrics for causal language modeling evaluation"""
    # For language modeling, we typically just use perplexity
    # which is calculated from the loss automatically
    # We can add custom metrics here if needed
    return {}

def preprocess_function(examples, tokenizer, max_length=1024):
    """Tokenize the texts and prepare for token probability training"""
    # Create prompts that ask for accept/reject decision
    prompts = []
    for text in examples['text']:
        prompt = f"Based on this research paper abstract and introduction, should this paper be accepted or rejected?\n\nPaper content:\n{text}\n\nDecision:"
        prompts.append(prompt)
    
    # Tokenize the prompts
    result = tokenizer(
        prompts,
        truncation=True,
        padding=False,
        max_length=max_length-10,  # Leave space for answer tokens
        return_attention_mask=True,
        return_token_type_ids=False
    )
    
    # Add target tokens based on labels
    target_tokens = []
    for label in examples['labels']:
        if label == 1:
            target_tokens.append(" accept")
        else:
            target_tokens.append(" reject")
    
    # Tokenize target tokens
    target_encodings = tokenizer(
        target_tokens,
        add_special_tokens=False,
        return_attention_mask=False
    )
    
    # Combine input and target
    combined_input_ids = []
    combined_attention_mask = []
    labels_for_loss = []
    
    for i in range(len(result['input_ids'])):
        # Combine input + target
        full_input = result['input_ids'][i] + target_encodings['input_ids'][i]
        full_mask = result['attention_mask'][i] + [1] * len(target_encodings['input_ids'][i])
        
        # Create labels (ignore input tokens, only compute loss on target tokens)
        label_ids = [-100] * len(result['input_ids'][i]) + target_encodings['input_ids'][i]
        
        # Ensure all sequences have the same length by truncating if needed
        if len(full_input) > max_length:
            full_input = full_input[:max_length]
            full_mask = full_mask[:max_length]
            label_ids = label_ids[:max_length]
        
        combined_input_ids.append(full_input)
        combined_attention_mask.append(full_mask)
        labels_for_loss.append(label_ids)
    
    return {
        'input_ids': combined_input_ids,
        'attention_mask': combined_attention_mask,
        'labels': labels_for_loss
    }

def download_and_save_model(model_name, cache_dir):
    """Download and save the base model locally"""
    print(f"Downloading model {model_name} to {cache_dir}...")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.save_pretrained(cache_dir)
    
    # Download model for causal language modeling
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
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
    # Initialize accelerator for distributed training
    accelerator = Accelerator()
    
    # Configuration
    MODEL_NAME = "Qwen/Qwen3-1.7B"  # or "meta-llama/Meta-Llama-3-8B"
    DATA_FOLDER = "/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_llm_papers/llm_papers_text/"  # Update this path
    LABELS_FILE = "/mnt/parscratch/users/acr24wz/topcon/train/llm_paper/label_simple.json"  # Update this path
    
    # Model directories
    BASE_MODEL_CACHE = "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_1d7B"  # Where to cache the downloaded model
    OUTPUT_DIR = "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_1d7B/finetuned_model"   # Where to save the fine-tuned model
    
    MAX_LENGTH = 2048 # Further reduced to save memory
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {gpu_memory:.1f} GB")
    
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
    
    # LoRA configuration for causal LM
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Changed from SEQ_CLS
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    
    # Load tokenizer from cached model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
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
    
    # Tokenize datasets using the new approach
    print("Tokenizing datasets...")
    
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        batch_size=100,  # Process in smaller batches
        remove_columns=['text', 'labels']  # Remove original columns
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        batch_size=100,  # Process in smaller batches
        remove_columns=['text', 'labels']  # Remove original columns
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
    
    # Load model from cached location (now using CausalLM)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_CACHE,
        torch_dtype=torch.float32,  # Use fp16 for memory efficiency
        device_map="auto"  # Enable automatic device mapping for model parallelism
    )

    print(model)
    
    # Check if model is distributed across multiple GPUs
    if hasattr(model, 'hf_device_map'):
        print(f"Model device map: {model.hf_device_map}")
    
    # Apply LoRA - DISABLED for debugging
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    
    print(f"Model parameters without LoRA:")
    
    # Data collator for language modeling
    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        learning_rate=1e-5,  # Even smaller learning rate
        warmup_steps=20,
        weight_decay=0.001,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,  # Reduce logging frequency
        eval_strategy="steps",
        eval_steps=100,  # Reduce evaluation frequency to save memory
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=False,  # Disable to save memory
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,  # Enable fp16 for memory efficiency
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=["labels"],
        max_grad_norm=1.0,  # Much stricter gradient clipping
        adam_epsilon=1e-8,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        eval_accumulation_steps=4,  # Process eval in smaller chunks
        dataloader_num_workers=0,  # Disable multiprocessing to save memory
        ddp_find_unused_parameters=False,  # Optimize for model parallelism
        deepspeed=None,  # Can be configured for ZeRO if needed
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
    
    # Prepare everything with accelerator for model parallelism
    model, trainer.optimizer, train_dataset, eval_dataset = accelerator.prepare(
        model, trainer.optimizer, train_dataset, eval_dataset
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