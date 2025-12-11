import os
import sys
import argparse

# Parse GPU IDs early before importing torch
parser = argparse.ArgumentParser(description="Fine-tune a language model with multi-GPU support")
parser.add_argument("--gpu_ids", type=int, nargs='+', default=None, help="GPU IDs to use for training/evaluation (e.g., --gpu_ids 0 1)")
# Add other arguments here as needed
args, unknown = parser.parse_known_args()  # Use parse_known_args to handle this early

# Set CUDA_VISIBLE_DEVICES before importing torch
if args.gpu_ids:
    gpu_ids_str = ','.join(map(str, args.gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    print(f"Set CUDA_VISIBLE_DEVICES to: {gpu_ids_str}")

# Now import torch and other libraries
import torch
from datetime import datetime
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
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

# Import the dataset builder
from dataset_builder_new_vl import TextDatasetBuilder
from datasets import load_from_disk

from contextlib import contextmanager

class TeeOutput:
    """Class to duplicate stdout to both console and log file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', buffering=1)  # Line buffering
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

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

def preprocess_logits_for_metrics(logits, labels):
    """
    Reduce (bs, seq, vocab) -> (bs, 3) keeping only the three logits we need
    at the *decision position* (first non-ignored label).
    """
    if isinstance(logits, tuple):
        logits = logits[0]  # some models return (logits, past_key_values, ...)
    # logits: torch.FloatTensor [bs, seq, vocab]
    # labels: torch.LongTensor  [bs, seq]
    with torch.no_grad():
        bs = logits.size(0)
        # first position where label != -100 for each sample
        first_pos = (labels.ne(-100).int().argmax(dim=1))  # [bs]
        rows = torch.arange(bs, device=logits.device)
        # logits at decision position: [bs, vocab]
        dec_logits = logits[rows, first_pos, :]
        # keep only no/yes/may columns -> [bs, 3]
        # Order: 0=No, 1=Yes, 2=May
        three = dec_logits.index_select(
            dim=1, index=torch.tensor([NO_ID, YES_ID, MAY_ID], device=logits.device)
        )
        return three

def compute_metrics(eval_pred, tokenizer=None):
    """
    Works in two modes:
    - Preferred: predictions are (N, 3) = [no_logit, yes_logit, may_logit] from preprocess_logits_for_metrics.
    - Fallback:  predictions are (N, L, V); we extract the three logits at the decision position.
    Returns {'accuracy': ...} (Trainer will prefix to 'eval_accuracy').
    """
    import numpy as np
    from sklearn.metrics import accuracy_score

    preds, labels = eval_pred  # preds is numpy; labels is numpy
    # Build true labels from first non-ignored token
    true_y = []
    for row in labels:
        # index of first non -100
        pos = int(np.argmax(row != -100))
        token_id = row[pos]
        if token_id == NO_ID:
            true_y.append(0)
        elif token_id == YES_ID:
            true_y.append(1)
        elif token_id == MAY_ID:
            true_y.append(2)
        else:
            true_y.append(-1) # Should not happen
            
    true_y = np.array(true_y, dtype=int)

    # Mode A: already reduced to three logits
    if preds.ndim == 2 and preds.shape[1] == 3:
        # preds columns: 0=No, 1=Yes, 2=May
        pred_labels = np.argmax(preds, axis=1)
        acc = accuracy_score(true_y, pred_labels)
        return {"accuracy": acc}

    # Mode B (fallback): full logits; pick decision position + yes/no/may columns
    else:
        # preds: (N, L, V)
        N, L, V = preds.shape
        # decision position per sample
        decision_pos = (labels != -100).argmax(axis=1)  # (N,)
        # gather logits at that position: (N, V)
        rows = np.arange(N)
        dec_logits = preds[rows, decision_pos, :]
        # Order: 0=No, 1=Yes, 2=May
        yes_no_may = dec_logits[:, [NO_ID, YES_ID, MAY_ID]]  # (N, 3)
        pred_labels = np.argmax(yes_no_may, axis=1)
        acc = accuracy_score(true_y, pred_labels)
        return {"accuracy": acc}

def preprocess_function(examples, tokenizer, max_length=1024):
    """Tokenize the texts and prepare for token probability training"""
    # Create prompts that ask for accept/reject/may decision
    prompts = []
    for text in examples['text']:
        prompt = f"Paper content:\n{text}\n\nBased on this AI research paper's content and statistics, should this paper be accepted? Answer yes, no, or may.\n\nDecision:"
        prompts.append(prompt)
    
    # First, tokenize target tokens to know their length
    target_tokens = []
    for label in examples['labels']:
        if label == 1:
            target_tokens.append(" yes")
        elif label == 0:
            target_tokens.append(" no")
        else:
            target_tokens.append(" may")
    
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

def batched_accuracy_evaluation(trainer, eval_dataset, batch_size=10, detailed_eval=False, tokenizer=None):
    """Evaluate accuracy using probability-based comparison of yes/no/may tokens"""
    print(f"Running probability-based accuracy evaluation on {len(eval_dataset)} samples...")
    
    # Get token IDs for "yes", "no", "may"
    yes_token_id = tokenizer(" yes", add_special_tokens=False)['input_ids'][0]
    no_token_id = tokenizer(" no", add_special_tokens=False)['input_ids'][0]
    may_token_id = tokenizer(" may", add_special_tokens=False)['input_ids'][0]
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    
    model = trainer.model
    
    # Process evaluation dataset in small batches
    for i in range(0, len(eval_dataset), batch_size):
        batch_end = min(i + batch_size, len(eval_dataset))
        batch_dataset = eval_dataset.select(range(i, batch_end))
        
        print(f"Processing batch {i//batch_size + 1}/{(len(eval_dataset) + batch_size - 1)//batch_size} (samples {i}-{batch_end-1})")
        
        # Clear cache before each batch
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            # Extract input portions and decision positions for efficient inference
            batch_input_ids = []
            batch_attention_masks = []
            decision_positions = []
            true_labels = []
            
            for sample in batch_dataset:
                input_ids = sample['input_ids']
                attention_mask = sample['attention_mask']
                labels = sample['labels']
                
                # Find decision position (first non-ignored label)
                decision_pos = None
                for k in range(len(labels)):
                    if labels[k] != -100:
                        decision_pos = k
                        break
                
                if decision_pos is not None:
                    # Only keep input up to decision position (exclude target tokens)
                    input_portion = input_ids[:decision_pos]
                    mask_portion = attention_mask[:decision_pos]
                    
                    batch_input_ids.append(input_portion)
                    batch_attention_masks.append(mask_portion)
                    decision_positions.append(len(input_portion))  # Next position is where we predict
                    
                    # Get ground truth
                    true_token_id = labels[decision_pos]
                    if true_token_id == no_token_id:
                        true_label = 0
                    elif true_token_id == yes_token_id:
                        true_label = 1
                    else:
                        true_label = 2
                    true_labels.append(true_label)
            
            if len(batch_input_ids) > 0:
                # Pad batch to same length
                max_len = max(len(ids) for ids in batch_input_ids)
                padded_input_ids = []
                padded_attention_masks = []
                
                for j, (input_ids, attention_mask) in enumerate(zip(batch_input_ids, batch_attention_masks)):
                    pad_length = max_len - len(input_ids)
                    padded_ids = input_ids + [tokenizer.pad_token_id] * pad_length
                    padded_mask = attention_mask + [0] * pad_length
                    
                    padded_input_ids.append(padded_ids)
                    padded_attention_masks.append(padded_mask)
                
                # Convert to tensors - let the model handle device placement for multi-GPU
                input_tensor = torch.tensor(padded_input_ids)
                mask_tensor = torch.tensor(padded_attention_masks)
                
                # For multi-GPU models, we need to move tensors to the first device of the model
                if hasattr(model, 'module'):
                    # If wrapped in DataParallel/DistributedDataParallel
                    first_device = next(model.module.parameters()).device
                else:
                    # For device_map models, find the first device
                    first_device = next(model.parameters()).device
                
                input_tensor = input_tensor.to(first_device)
                mask_tensor = mask_tensor.to(first_device)
                
                # Get logits from model - only forward pass, no need for full sequence
                outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
                logits = outputs.logits
                
                # Extract logits at decision positions for each sample
                for j, (decision_pos, true_label) in enumerate(zip(decision_positions, true_labels)):
                    # Get logits at the position where we need to predict the next token
                    logits_at_pos = logits[j, decision_pos - 1]  # -1 because we predict the next token
                    
                    # Compare yes vs no vs may logits
                    yes_logit = logits_at_pos[yes_token_id]
                    no_logit = logits_at_pos[no_token_id]
                    may_logit = logits_at_pos[may_token_id]
                    
                    # Predict based on highest logit
                    # 0=No, 1=Yes, 2=May
                    logits_dict = {0: no_logit, 1: yes_logit, 2: may_logit}
                    predicted_label = max(logits_dict, key=logits_dict.get)
                    
                    all_predictions.append(predicted_label)
                    all_labels.append(true_label)
            
            # Get loss from evaluation using original method for loss calculation
            eval_results = trainer.evaluate(eval_dataset=batch_dataset)
            total_loss += eval_results['eval_loss'] * (batch_end - i)
        
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
        # Calculate precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro', zero_division=0
        )
        
        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Classification report
        class_report = classification_report(
            all_labels, all_predictions, 
            target_names=['No', 'Yes', 'May'], 
            zero_division=0
        )
        
        results.update({
            'accuracy': accuracy_score(all_labels, all_predictions),
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
        print("PROBABILITY-BASED EVALUATION METRICS")
        print("="*50)
        print(f"Classification Accuracy: {results['accuracy']:.4f}")
        print(f"Precision (Macro): {precision:.4f}")
        print(f"Recall (Macro): {recall:.4f}")
        print(f"F1-Score (Macro): {f1:.4f}")
        print("\nPer-class metrics:")
        # Handle cases where some classes might be missing in support
        for cls_idx, cls_name in enumerate(['Reject (0)', 'Accept (1)', 'Maybe (2)']):
            if cls_idx < len(precision_per_class):
                print(f"  {cls_name} - Precision: {precision_per_class[cls_idx]:.4f}, Recall: {recall_per_class[cls_idx]:.4f}, F1: {f1_per_class[cls_idx]:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(class_report)
        print("="*50)
        print("Method: Comparing logits of 'yes', 'no', 'may' tokens at decision position")
        print("Prediction: Max logit determines class")
        print("="*50)
    
    return results

def main():
    # Set up logging
    log_dir = "./log"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_llm_{timestamp}.log")
    
    # Redirect stdout and stderr to both console and log file
    tee_stdout = TeeOutput(log_file)
    tee_stderr = TeeOutput(log_file)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr
    
    print(f"Logging to: {log_file}")
    print(f"Start time: {datetime.now()}")
    print("="*80)
    
    try:
        # Re-parse all arguments properly in main
        parser = argparse.ArgumentParser(description="Fine-tune a language model with multi-GPU support")
        parser.add_argument("--eval", action="store_true", help="Run evaluation mode on fine-tuned model")
        parser.add_argument("--detailed_eval", action="store_true", help="Output detailed evaluation metrics including precision, recall, F1, and confusion matrix")
        parser.add_argument("--debug", action="store_true", help="Debug mode: set eval_steps to 10 for frequent evaluation")
        parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Pre-trained model name or path")
        parser.add_argument("--data_folder", type=str, default="/mnt/parscratch/users/lip22fh/ACL2026_paper_predict/balanced_dataset/balanced_datasets/balanced_llm", help="Path to the folder containing training jsons")
        parser.add_argument("--labels_file", type=str, default="/mnt/parscratch/users/lip22fh/ACL2026_paper_predict/topcon/balanced_labels.json", help="Path to the file containing labels")
        parser.add_argument("--metadata_file", type=str, default='/mnt/parscratch/users/lip22fh/ACL2026_paper_predict/balanced_dataset/balanced_datasets/Balanced/balanced_meta.json', help="Path to the metadata json file containing rating_avg for 3-class classification")
        parser.add_argument("--statistics_file", type=str, default=None, help="Path to the statistical.json file containing paper statistics")
        parser.add_argument("--img_desc_file", type=str, default='/mnt/parscratch/users/lip22fh/ACL2026_paper_predict/img_desimage_descriptions_llm.json', help="Path to the image descriptions JSON file for vision-language support")
        parser.add_argument("--output_dir", type=str, default="/mnt/parscratch/users/lip22fh/ACL2026_paper_predict/models/qwen3_4b/orig/ft/llm", help="Directory to save/load the fine-tuned model")
        parser.add_argument("--max_length", type=int, default=12000, help="Maximum sequence length for training")
        parser.add_argument("--gpu_ids", type=int, nargs='+', default=None, help="GPU IDs to use for training/evaluation (e.g., --gpu_ids 0 1)")
        args = parser.parse_args()

        # Default to all visible GPUs if none provided
        if args.gpu_ids is None:
            if torch.cuda.is_available():
                args.gpu_ids = list(range(torch.cuda.device_count()))
            else:
                args.gpu_ids = []
        
        print(f"Using GPU IDs: {args.gpu_ids}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        
        # Now torch will see the correct GPUs
        if torch.cuda.is_available():
            print(f"Number of available GPUs (as seen by PyTorch): {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i} (remapped index): {torch.cuda.get_device_name(i)}")
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  Memory: {gpu_memory:.1f} GB")
        
        # Configuration
        MODEL_NAME = args.model_name
        DATA_FOLDER = args.data_folder
        LABELS_FILE = args.labels_file
        METADATA_FILE = args.metadata_file # Added
        STATISTICS_FILE = args.statistics_file
        IMG_DESC_FILE = args.img_desc_file  # Added
        OUTPUT_DIR = args.output_dir
        MAX_LENGTH = args.max_length
        
        # Model directories
        BASE_MODEL_CACHE = "/mnt/parscratch/users/lip22fh/ACL2026_paper_predict/models/qwen3_4b/orig"  # Where to cache the downloaded model

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
        
        # Print GPU information
        if torch.cuda.is_available():
            print(f"Number of available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  Memory: {gpu_memory:.1f} GB")
        else:
            print("Warning: No GPU available, using CPU")
            return
        
        # Verify that we have enough GPUs
        if len(args.gpu_ids) > torch.cuda.device_count():
            print(f"Error: Requested {len(args.gpu_ids)} GPUs but only {torch.cuda.device_count()} available")
            return
        
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
        
        # Get token IDs for "yes" and "no"
        global YES_ID, NO_ID, MAY_ID
        YES_ID = tokenizer(" yes", add_special_tokens=False)["input_ids"][0]
        NO_ID  = tokenizer(" no",  add_special_tokens=False)["input_ids"][0]
        MAY_ID = tokenizer(" may", add_special_tokens=False)["input_ids"][0]

        # Load and prepare dataset
        print("Loading dataset...")
        
        # Define processed dataset cache path - include stats/img_desc in cache name if provided
        cache_suffix = "llm_mineru_all_3class" # Changed suffix
        if STATISTICS_FILE:
            cache_suffix += "_with_stats"
        if IMG_DESC_FILE:
            cache_suffix += "_with_img_desc"
        PROCESSED_DATASET_CACHE = f"/mnt/parscratch/users/lip22fh/ACL2026_paper_predict/dataset_cache/balanced/{cache_suffix}"
        os.makedirs(PROCESSED_DATASET_CACHE, exist_ok=True)
        
        # Check if processed dataset exists by looking for the dataset_info.json file
        dataset_info_path = os.path.join(PROCESSED_DATASET_CACHE, "dataset_info.json")
        if os.path.exists(dataset_info_path) and os.path.isdir(PROCESSED_DATASET_CACHE):
            try:
                print("Loading cached processed dataset...")
                dataset = load_from_disk(PROCESSED_DATASET_CACHE)
                # Still need to create dataset_builder for stats
                dataset_builder = TextDatasetBuilder(
                    DATA_FOLDER, 
                    LABELS_FILE, 
                    statistics_file=STATISTICS_FILE,
                    img_desc_file=IMG_DESC_FILE,
                    metadata_file=METADATA_FILE, # Added
                    max_length=MAX_LENGTH
                )
                print("Successfully loaded cached dataset!")
            except Exception as e:
                print(f"Failed to load cached dataset: {e}")
                print("Processing dataset from scratch...")
                dataset_builder = TextDatasetBuilder(
                    DATA_FOLDER, 
                    LABELS_FILE, 
                    statistics_file=STATISTICS_FILE,
                    img_desc_file=IMG_DESC_FILE,
                    metadata_file=METADATA_FILE, # Added
                    max_length=MAX_LENGTH
                )
                dataset = dataset_builder.load_dataset_with_ids()
                
                # Save processed dataset to cache
                print(f"Saving processed dataset to {PROCESSED_DATASET_CACHE}")
                dataset.save_to_disk(PROCESSED_DATASET_CACHE)
        else:
            print("No cached dataset found. Processing dataset for the first time...")
            dataset_builder = TextDatasetBuilder(
                DATA_FOLDER, 
                LABELS_FILE, 
                statistics_file=STATISTICS_FILE,
                img_desc_file=IMG_DESC_FILE,
                metadata_file=METADATA_FILE, # Added
                max_length=MAX_LENGTH
            )
            dataset = dataset_builder.load_dataset_with_ids()
            
            # Save processed dataset to cache
            print(f"Saving processed dataset to {PROCESSED_DATASET_CACHE}")
            dataset.save_to_disk(PROCESSED_DATASET_CACHE)
        
        # Print dataset configuration
        print("\nDataset Configuration:")
        print(f"  Data folder: {DATA_FOLDER}")
        print(f"  Labels file: {LABELS_FILE}")
        print(f"  Metadata file: {METADATA_FILE if METADATA_FILE else 'Not provided'}")
        print(f"  Statistics file: {STATISTICS_FILE if STATISTICS_FILE else 'Not provided'}")
        print(f"  Image descriptions file: {IMG_DESC_FILE if IMG_DESC_FILE else 'Not provided'}")  # Added
        print(f"  Max length: {MAX_LENGTH}")
        
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
        small_eval_dataset = eval_dataset.select(range(min(100, len(eval_dataset))))  # Even smaller for faster eval
        
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
        
        # Get column names to remove (all original columns)
        cols_to_remove = train_dataset.column_names
        print(f"Columns to remove after tokenization: {cols_to_remove}")

        train_dataset = train_dataset.map(
            lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
            batched=True,
            batch_size=100,  # Process in smaller batches
            remove_columns=cols_to_remove  # Remove original columns 
        )
        
        eval_dataset = eval_dataset.map(
            lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
            batched=True,
            batch_size=100,  # Process in smaller batches
            remove_columns=cols_to_remove  # Remove original columns 
        )
        
        # Also tokenize the small eval dataset for periodic evaluation
        small_eval_dataset = small_eval_dataset.map(
            lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
            batched=True,
            batch_size=100,
            remove_columns=cols_to_remove
        )
        
        print("Tokenization complete")
        print(f"Train dataset columns: {train_dataset.column_names}")
        print(f"Train dataset sample (initial part): {str(train_dataset[0])[:50]}...")

        # Debug tensor shapes
        sample = train_dataset[0]
        print(f"Sample input_ids type: {type(sample['input_ids'])}")
        print(f"Sample input_ids length: {len(sample['input_ids'])}")
        print(f"Sample labels type: {type(sample['labels'])}")
        print(f"Sample labels value: {sample['labels']}")
        
        # Load model with automatic device mapping for multi-GPU
        print("Loading model with automatic device mapping across GPUs...")
        if args.eval:
            # Load the fine-tuned model for evaluation
            model = AutoModelForCausalLM.from_pretrained(
                OUTPUT_DIR,  # Load from fine-tuned model directory
                torch_dtype=torch.bfloat16,
                device_map="auto",  # Automatically distribute across available GPUs
                max_memory={i: "80GiB" for i in range(len(args.gpu_ids))},  # Set max memory per GPU
                offload_folder="./offload",  # Offload to disk if needed
                attn_implementation="flash_attention_2"  # Use SDPA attention implementation
            )
            print("Loaded fine-tuned model for evaluation")
        else:
            # Load base model for training
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_CACHE,
                torch_dtype=torch.bfloat16,
                device_map="auto",  # Automatically distribute across available GPUs
                max_memory={i: "78GiB" for i in range(len(args.gpu_ids))},  # Set max memory per GPU
                offload_folder="./offload",  # Offload to disk if needed
                attn_implementation="flash_attention_2"  # Use SDPA attention implementation
            )

        print(model)
        
        # Print device mapping
        if hasattr(model, 'hf_device_map'):
            print("Model device mapping:")
            for layer, device in model.hf_device_map.items():
                print(f"  {layer}: {device}")
        
        # Data collator for language modeling
        data_collator = CustomDataCollator(
            tokenizer=tokenizer,
            max_length=MAX_LENGTH
        )
        
        # Only run training if not in evaluation mode
        if not args.eval:
            # Set eval_steps based on debug mode
            eval_steps = 10 if args.debug else 100
            
            if args.debug:
                print("DEBUG MODE: eval_steps set to 10")
            
            # Training arguments - adjusted for multi-GPU
            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                num_train_epochs=5,
                per_device_train_batch_size=1,  # Keep small for large model
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,  # Maintain effective batch size
                learning_rate=2e-5,
                warmup_ratio=0.1,
                weight_decay=0.01,
                logging_dir=f"{OUTPUT_DIR}/logs",
                logging_steps=1,
                eval_strategy="steps",
                eval_steps=eval_steps,  # Use variable based on debug mode
                save_steps=100,
                save_total_limit=3,  # Increase to keep more checkpoints including best
                load_best_model_at_end=True,  # Change to True to load best model at end
                metric_for_best_model="eval_accuracy",  # Or use "eval_accuracy" if you prefer
                greater_is_better=True,  # False for loss, True for accuracy
                bf16=True,  # Enable bf16 for memory efficiency
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                label_names=["labels"],
                max_grad_norm=1.0,
                adam_epsilon=1e-8,
                lr_scheduler_type="linear",
                optim="adamw_torch",
                eval_accumulation_steps=1,
                dataloader_num_workers=0,  # Disable multiprocessing for multi-GPU setup
                prediction_loss_only=False,  # Change to False to compute metrics
                skip_memory_metrics=True,
                # Multi-GPU specific settings
                ddp_find_unused_parameters=False,  # For efficiency in DDP
                dataloader_persistent_workers=False,  # Disable persistent workers
                gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=small_eval_dataset,  # Use smaller eval dataset for periodic evaluation
                tokenizer=tokenizer,
                data_collator=data_collator,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                compute_metrics=lambda ep: compute_metrics(ep)
            )
            
            # Train the model
            print("Starting training...")
            
            # Override evaluation to use memory cleanup
            original_evaluate = trainer.evaluate
            def memory_safe_evaluate(*args, **kwargs):
                torch.cuda.empty_cache()
                # Disable cache during evaluation to save memory
                original_use_cache = model.config.use_cache
                model.config.use_cache = False
                with torch.no_grad():
                    result = original_evaluate(*args, **kwargs)
                # Restore original cache setting
                model.config.use_cache = original_use_cache
                torch.cuda.empty_cache()
                return result
            trainer.evaluate = memory_safe_evaluate
            
            trainer.train()
            
            # Save the fine-tuned model
            print(f"Saving fine-tuned model to {OUTPUT_DIR}")
            trainer.save_model()
            tokenizer.save_pretrained(OUTPUT_DIR)
            
            print(f"Fine-tuned model saved to: {OUTPUT_DIR}")

            # Explicitly delete trainer and optimizer to free memory
            del trainer
            if hasattr(model, 'optimizer'):
                del model.optimizer
            
            # More thorough cleanup
            model_to_delete = model
            model = None
            del model_to_delete
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache on all GPUs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            print("Cleared trainer, optimizer, and model from memory after training.")

            print("Reloading model for final evaluation...")
            
            # Small delay to ensure memory is fully released
            import time
            time.sleep(2)
            
            # Reload the model fresh with device mapping
            model = AutoModelForCausalLM.from_pretrained(
                OUTPUT_DIR,  # Load the fine-tuned model
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={i: "80GiB" for i in range(len(args.gpu_ids))},
                offload_folder="./offload",
                attn_implementation="flash_attention_2"  # Use Flash Attention 2 implementation
            )
            
            # Clear cache again after loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Create trainer for final evaluation (needed for both train and eval modes)
        print("Setting up trainer for final evaluation...")
        training_args_eval = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_eval_batch_size=1,
            bf16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            label_names=["labels"],
            eval_accumulation_steps=1,
            dataloader_num_workers=0,
            prediction_loss_only=True,
            skip_memory_metrics=True,
            ddp_find_unused_parameters=False,
            dataloader_persistent_workers=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        )

        # Always run final evaluation (in both train and eval modes)
        print("Running final evaluation with accuracy computation...")

        # Use batched evaluation to prevent OOM
        eval_results = batched_accuracy_evaluation(
            trainer, eval_dataset, batch_size=1,  # Smaller batch size for multi-GPU
            detailed_eval=args.detailed_eval, tokenizer=tokenizer
        )
        
        print(f"Final evaluation results: {eval_results}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Print end time and close logs
        print("="*80)
        print(f"End time: {datetime.now()}")
        print(f"Log saved to: {log_file}")
        
        # Restore original stdout/stderr and close log files
        sys.stdout = tee_stdout.terminal
        sys.stderr = tee_stderr.terminal
        tee_stdout.close()
        tee_stderr.close()

if __name__ == "__main__":
    main()