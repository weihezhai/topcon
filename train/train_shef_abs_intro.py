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
# Removed LoRA - using full fine-tuning
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import argparse

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
    predictions, labels = eval_pred
    
    # For causal LM, we need to extract the predictions for the target tokens
    # The predictions are logits, we need to get the predicted tokens
    predictions = np.argmax(predictions, axis=-1)
    
    # Extract only the non-ignored labels (not -100)
    # and their corresponding predictions
    true_labels = []
    pred_labels = []
    
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != -100:  # Not an ignored token
                true_labels.append(labels[i][j])
                pred_labels.append(predictions[i][j])
    
    # Calculate accuracy
    if len(true_labels) > 0:
        accuracy = accuracy_score(true_labels, pred_labels)
        return {"accuracy": accuracy}
    else:
        return {"accuracy": 0.0}

def preprocess_function(examples, tokenizer, max_length=1024):
    """Tokenize the texts and prepare for token probability training"""
    # Create prompts that ask for accept/reject decision
    prompts = []
    for text in examples['text']:
        prompt = f"Based on this research paper abstract and introduction, should this paper be accepted or rejected?\n\nPaper content:\n{text}\n\nDecision:"
        prompts.append(prompt)
    
    # First, tokenize target tokens to know their length
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
    
    # Calculate the maximum target token length to reserve space
    max_target_length = max(len(target) for target in target_encodings['input_ids'])
    
    # Tokenize the prompts with reserved space for target tokens
    result = tokenizer(
        prompts,
        truncation=True,
        padding=False,
        max_length=max_length - max_target_length,  # Reserve space for target tokens
        return_attention_mask=True,
        return_token_type_ids=False
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
        
        # Since we reserved space, this should not exceed max_length, but double-check
        if len(full_input) > max_length:
            print(f"Warning: Combined sequence length {len(full_input)} exceeds max_length {max_length}")
            # If it still exceeds, truncate input while preserving target
            excess = len(full_input) - max_length
            input_length = len(result['input_ids'][i])
            target_length = len(target_encodings['input_ids'][i])
            
            # Truncate from input
            new_input_length = max(1, input_length - excess)  # Keep at least 1 input token
            full_input = result['input_ids'][i][:new_input_length] + target_encodings['input_ids'][i]
            full_mask = result['attention_mask'][i][:new_input_length] + [1] * target_length
            label_ids = [-100] * new_input_length + target_encodings['input_ids'][i]
        
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
        torch_dtype=torch.bfloat16,
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

def custom_evaluate_with_memory_cleanup(trainer, eval_dataset=None):
    """Custom evaluation that clears memory cache to prevent slowdown"""
    # Clear GPU cache before evaluation
    torch.cuda.empty_cache()
    
    # Run evaluation
    with torch.no_grad():
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    
    # Clear GPU cache after evaluation
    torch.cuda.empty_cache()
    
    return eval_results

def batched_accuracy_evaluation(trainer, eval_dataset, batch_size=10, detailed_eval=False, tokenizer=None):
    """Evaluate accuracy in small batches to prevent OOM"""
    print(f"Running batched accuracy evaluation on {len(eval_dataset)} samples...")
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    # For detailed evaluation, we'll also collect the actual text predictions
    accept_token_id = None
    reject_token_id = None
    if detailed_eval and tokenizer:
        accept_tokens = tokenizer(" accept", add_special_tokens=False)['input_ids']
        reject_tokens = tokenizer(" reject", add_special_tokens=False)['input_ids'] 
        if len(accept_tokens) > 0:
            accept_token_id = accept_tokens[0]
        if len(reject_tokens) > 0:
            reject_token_id = reject_tokens[0]
    
    # Process evaluation dataset in small batches
    for i in range(0, len(eval_dataset), batch_size):
        batch_end = min(i + batch_size, len(eval_dataset))
        batch_dataset = eval_dataset.select(range(i, batch_end))
        
        print(f"Processing batch {i//batch_size + 1}/{(len(eval_dataset) + batch_size - 1)//batch_size} (samples {i}-{batch_end-1})")
        
        # Clear cache before each batch
        torch.cuda.empty_cache()
        
        # Temporarily enable full prediction for this batch
        trainer.args.prediction_loss_only = False
        
        with torch.no_grad():
            # Use trainer.predict to get predictions
            eval_output = trainer.predict(batch_dataset)
            predictions = eval_output.predictions
            labels = eval_output.label_ids
            
            # Convert logits to predicted tokens
            predictions = np.argmax(predictions, axis=-1)
            
            # Extract non-ignored labels and predictions
            for j in range(len(labels)):
                for k in range(len(labels[j])):
                    if labels[j][k] != -100:
                        all_labels.append(labels[j][k])
                        all_predictions.append(predictions[j][k])
            
            # Get loss from evaluation
            eval_results = trainer.evaluate(eval_dataset=batch_dataset)
            total_loss += eval_results['eval_loss'] * (batch_end - i)
            num_batches += 1
        
        # Reset to loss-only mode
        trainer.args.prediction_loss_only = True
        
        # Clear cache after each batch
        torch.cuda.empty_cache()
    
    # Calculate overall metrics
    avg_loss = total_loss / len(eval_dataset)
    accuracy = accuracy_score(all_labels, all_predictions) if len(all_labels) > 0 else 0.0
    
    results = {
        'eval_loss': avg_loss,
        'eval_accuracy': accuracy,
        'eval_samples': len(eval_dataset)
    }
    
    # Add detailed metrics if requested
    if detailed_eval and len(all_labels) > 0:
        # Convert token predictions to binary labels for analysis
        binary_labels = []
        binary_predictions = []
        
        for true_label, pred_label in zip(all_labels, all_predictions):
            # Map tokens to binary labels (1 for accept, 0 for reject)
            if true_label == accept_token_id:
                binary_labels.append(1)
            elif true_label == reject_token_id:
                binary_labels.append(0)
            else:
                continue  # Skip unknown tokens
                
            if pred_label == accept_token_id:
                binary_predictions.append(1)
            elif pred_label == reject_token_id:
                binary_predictions.append(0)
            else:
                # If prediction is neither accept nor reject, classify based on proximity
                binary_predictions.append(1 if pred_label == accept_token_id else 0)
        
        if len(binary_labels) > 0:
            # Calculate precision, recall, F1
            precision, recall, f1, support = precision_recall_fscore_support(
                binary_labels, binary_predictions, average='binary', zero_division=0
            )
            
            # Calculate per-class metrics
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                binary_labels, binary_predictions, average=None, zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(binary_labels, binary_predictions)
            
            # Classification report
            class_report = classification_report(
                binary_labels, binary_predictions, 
                target_names=['Reject', 'Accept'], 
                zero_division=0
            )
            
            results.update({
                'binary_accuracy': accuracy_score(binary_labels, binary_predictions),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'precision_per_class': precision_per_class.tolist(),
                'recall_per_class': recall_per_class.tolist(),
                'f1_per_class': f1_per_class.tolist(),
                'support_per_class': support_per_class.tolist(),
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report
            })
            
            print("\n" + "="*50)
            print("DETAILED EVALUATION METRICS")
            print("="*50)
            print(f"Binary Classification Accuracy: {results['binary_accuracy']:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print("\nPer-class metrics:")
            print(f"  Reject (0) - Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}")
            print(f"  Accept (1) - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"              Predicted")
            print(f"              Reject  Accept")
            print(f"Actual Reject   {cm[0,0]:4d}    {cm[0,1]:4d}")
            print(f"       Accept   {cm[1,0]:4d}    {cm[1,1]:4d}")
            print(f"\nClassification Report:")
            print(class_report)
            print("="*50)
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA")
    parser.add_argument("--eval", action="store_true", help="Run evaluation mode on fine-tuned model")
    parser.add_argument("--detailed_eval", action="store_true", help="Output detailed evaluation metrics including precision, recall, F1, and confusion matrix")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Pre-trained model name or path")
    parser.add_argument("--data_folder", type=str, default="/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_llm_papers/llm_papers_text/", help="Path to the folder containing training data")
    parser.add_argument("--labels_file", type=str, default="/mnt/parscratch/users/acr24wz/topcon/train/llm_paper/label_simple.json", help="Path to the file containing labels")
    parser.add_argument("--output_dir", type=str, default="/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B/finetuned_model", help="Directory to save/load the fine-tuned model")
    parser.add_argument("--max_length", type=int, default=10000, help="Maximum sequence length for training")
    args = parser.parse_args()
    
    # Initialize accelerator for distributed training
    accelerator = Accelerator()
    
    # Configuration
    MODEL_NAME = args.model_name
    DATA_FOLDER = args.data_folder
    LABELS_FILE = args.labels_file
    OUTPUT_DIR = args.output_dir
    MAX_LENGTH = args.max_length
    
    # Model directories
    BASE_MODEL_CACHE = "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B"  # Where to cache the downloaded model
    
    # If in evaluation mode, use the fine-tuned model directory
    if args.eval:
        if not os.path.exists(OUTPUT_DIR):
            print(f"Error: Fine-tuned model not found at {OUTPUT_DIR}")
            print("Please run training first without --eval flag")
            return
        MODEL_PATH = OUTPUT_DIR
        print(f"Running evaluation mode using model from {MODEL_PATH}")
    else:
        MODEL_PATH = BASE_MODEL_CACHE
        print(f"Running training mode using base model from {MODEL_PATH}")
    
    # MAX_LENGTH = 10000 # Further reduced to save memory
    
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

    # Download and cache the base model if not already cached (only for training mode)
    if not args.eval:
        config_file = os.path.join(BASE_MODEL_CACHE, "config.json")
        if not os.path.exists(config_file):
            download_and_save_model(MODEL_NAME, BASE_MODEL_CACHE)
        else:
            print(f"Using cached model from {BASE_MODEL_CACHE}")
    
    # Load tokenizer from appropriate model path
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
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

    # Create a smaller subset for faster evaluation during training
    small_eval_dataset = eval_dataset.select(range(min(50, len(eval_dataset))))  # Even smaller for faster eval
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set: {len(eval_dataset)} samples ({len(small_eval_dataset)} used for periodic eval)")
    
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
    
    # Also tokenize the small eval dataset for periodic evaluation
    small_eval_dataset = small_eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        batch_size=100,
        remove_columns=['text', 'labels']
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
    
    # Load model from appropriate path (now using CausalLM)
    from transformers import AutoModelForCausalLM
    
    if args.eval:
        # Load the fine-tuned model for evaluation
        model = AutoModelForCausalLM.from_pretrained(
            OUTPUT_DIR,  # Load from fine-tuned model directory
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Loaded fine-tuned model for evaluation")
    else:
        # Load base model for training
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    print(model)
    
    # Check if model is distributed across multiple GPUs
    if hasattr(model, 'hf_device_map'):
        print(f"Model device map: {model.hf_device_map}")
    
    print(f"Model parameters (full fine-tuning):")
    
    # Data collator for language modeling
    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    # Only run training if not in evaluation mode
    if not args.eval:
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
            eval_steps=100,  # Increase evaluation frequency to save memory
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=False,  # Disable to save memory
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=True,  # Enable bf16 for memory efficiency
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
            prediction_loss_only=True,  # Only compute loss during periodic evaluation
            skip_memory_metrics=True,  # Skip memory metrics to save memory
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=small_eval_dataset,  # Use smaller eval dataset for periodic evaluation
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
        
        # Override evaluation to use memory cleanup
        original_evaluate = trainer.evaluate
        def memory_safe_evaluate(*args, **kwargs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                result = original_evaluate(*args, **kwargs)
            torch.cuda.empty_cache()
            return result
        trainer.evaluate = memory_safe_evaluate
        
        trainer.train()
        
        # Save the fine-tuned model
        print(f"Saving fine-tuned model to {OUTPUT_DIR}")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print(f"Fine-tuned model saved to: {OUTPUT_DIR}")
    
    # Always run final evaluation (in both train and eval modes)
    print("Running final evaluation with accuracy computation...")
    
    # For evaluation mode, we need to create a trainer for evaluation
    if args.eval:
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_eval_batch_size=1,
            bf16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            label_names=["labels"],
            eval_accumulation_steps=4,
            dataloader_num_workers=0,
            prediction_loss_only=True,
            skip_memory_metrics=True,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    
    # Use batched evaluation to prevent OOM
    eval_results = batched_accuracy_evaluation(
        trainer, eval_dataset, batch_size=10, 
        detailed_eval=args.detailed_eval, tokenizer=tokenizer
    )
    
    print(f"Final evaluation results: {eval_results}")

if __name__ == "__main__":
    main()