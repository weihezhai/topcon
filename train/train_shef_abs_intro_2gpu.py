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
    default_data_collator,
    TrainerCallback
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
    """Custom data collator that handles variable-length sequences and binary labels"""
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
        
        # Check if binary_labels exist in features
        if 'binary_labels' in features[0]:
            batch['binary_labels'] = []
        
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
            
            # Add binary labels if they exist
            if 'binary_labels' in feature:
                batch['binary_labels'].append(feature['binary_labels'])
        
        # Convert to tensors
        for key in ['input_ids', 'attention_mask', 'labels']:
            batch[key] = torch.tensor(batch[key])
        
        # Convert binary_labels to tensor if present
        if 'binary_labels' in batch:
            batch['binary_labels'] = torch.tensor(batch['binary_labels'])
            
        return batch

def compute_metrics(eval_pred):
    """Compute metrics for causal language modeling evaluation based on yes/no token probabilities"""
    predictions, labels = eval_pred
    
    # This is a simplified version for periodic evaluation during training
    # The detailed evaluation will be done in batched_accuracy_evaluation
    predictions = np.argmax(predictions, axis=-1)
    
    # Extract only the non-ignored labels (not -100)
    true_labels = []
    pred_labels = []
    
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != -100:  # Not an ignored token
                true_labels.append(labels[i][j])
                pred_labels.append(predictions[i][j])
    
    # Calculate basic token accuracy for monitoring
    if len(true_labels) > 0:
        accuracy = accuracy_score(true_labels, pred_labels)
        return {"accuracy": accuracy}
    else:
        return {"accuracy": 0.0}

def preprocess_function(examples, tokenizer, max_length=1024):
    """Tokenize the texts and prepare for yes/no token probability training"""
    # Create prompts that ask for accept/reject decision
    prompts = []
    for text in examples['text']:
        prompt = f"Based on this research paper's abstract and introduction, should this paper be accepted?\n\nPaper content:\n{text}\n\nDecision:"
        prompts.append(prompt)
    
    # Get yes/no token IDs for consistent training
    yes_tokens = tokenizer(" yes", add_special_tokens=False)['input_ids']
    no_tokens = tokenizer(" no", add_special_tokens=False)['input_ids']
    
    if len(yes_tokens) == 0 or len(no_tokens) == 0:
        raise ValueError("Could not tokenize 'yes' or 'no' tokens")
    
    yes_token_id = yes_tokens[0]
    no_token_id = no_tokens[0]
    
    # First, tokenize target tokens to know their length
    target_tokens = []
    target_token_ids = []
    for label in examples['labels']:
        if label == 1:
            target_tokens.append(" yes")
            target_token_ids.append(yes_token_id)
        else:
            target_tokens.append(" no")
            target_token_ids.append(no_token_id)
    
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
    binary_labels = []  # Store the binary labels for reference
    
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
        binary_labels.append(examples['labels'][i])  # Store original binary label
    
    return {
        'input_ids': combined_input_ids,
        'attention_mask': combined_attention_mask,
        'labels': labels_for_loss,
        'binary_labels': binary_labels  # Add binary labels for potential custom loss
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
    """Evaluate accuracy based on yes/no token probabilities after 'Decision:'"""
    print(f"Running batched accuracy evaluation on {len(eval_dataset)} samples...")
    
    # Get yes/no token IDs
    yes_tokens = tokenizer(" yes", add_special_tokens=False)['input_ids']
    no_tokens = tokenizer(" no", add_special_tokens=False)['input_ids']
    
    if len(yes_tokens) == 0 or len(no_tokens) == 0:
        print("Error: Could not tokenize 'yes' or 'no' tokens")
        return {"eval_loss": 0.0, "eval_accuracy": 0.0}
    
    yes_token_id = yes_tokens[0]
    no_token_id = no_tokens[0]
    
    print(f"Yes token ID: {yes_token_id} ('{tokenizer.decode([yes_token_id])}')")
    print(f"No token ID: {no_token_id} ('{tokenizer.decode([no_token_id])}')")
    
    all_binary_predictions = []
    all_binary_labels = []
    total_loss = 0.0
    num_batches = 0
    
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
            # Use trainer.predict to get predictions (logits)
            eval_output = trainer.predict(batch_dataset)
            predictions_logits = eval_output.predictions  # These are logits, not token IDs
            labels = eval_output.label_ids
            
            # For each sample in the batch, find the position where labels != -100 
            # (this is where the target 'yes'/'no' tokens are)
            for j in range(len(labels)):
                sample_labels = labels[j]
                sample_logits = predictions_logits[j]
                
                # Find the first position where label is not -100 (target token position)
                target_positions = [k for k, label in enumerate(sample_labels) if label != -100]
                
                if len(target_positions) > 0:
                    # Take the first target position (should be the 'yes'/'no' token)
                    target_pos = target_positions[0]
                    target_label = sample_labels[target_pos]
                    target_logits = sample_logits[target_pos]  # Logits for this position
                    
                    # Extract logits for yes and no tokens specifically
                    yes_logit = target_logits[yes_token_id]
                    no_logit = target_logits[no_token_id]
                    
                    # Convert to probabilities using softmax (only for yes/no tokens)
                    import torch.nn.functional as F
                    yes_no_logits = torch.tensor([no_logit, yes_logit])  # [no, yes]
                    yes_no_probs = F.softmax(yes_no_logits, dim=0)
                    
                    # Prediction: 1 if yes_prob > no_prob, 0 otherwise
                    predicted_binary = 1 if yes_no_probs[1] > yes_no_probs[0] else 0
                    
                    # Ground truth: convert token ID to binary label
                    if target_label == yes_token_id:
                        true_binary = 1
                    elif target_label == no_token_id:
                        true_binary = 0
                    else:
                        continue  # Skip if target is neither yes nor no
                    
                    all_binary_predictions.append(predicted_binary)
                    all_binary_labels.append(true_binary)
            
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
    
    if len(all_binary_labels) > 0:
        accuracy = accuracy_score(all_binary_labels, all_binary_predictions)
    else:
        accuracy = 0.0
    
    results = {
        'eval_loss': avg_loss,
        'eval_accuracy': accuracy,
        'eval_samples': len(eval_dataset),
        'processed_samples': len(all_binary_labels)
    }
    
    print(f"Processed {len(all_binary_labels)} valid samples out of {len(eval_dataset)} total samples")
    
    # Add detailed metrics if requested
    if detailed_eval and len(all_binary_labels) > 0:
        # Calculate precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            all_binary_labels, all_binary_predictions, average='binary', zero_division=0
        )
        
        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            all_binary_labels, all_binary_predictions, average=None, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_binary_labels, all_binary_predictions)
        
        # Classification report
        class_report = classification_report(
            all_binary_labels, all_binary_predictions, 
            target_names=['No (Reject)', 'Yes (Accept)'], 
            zero_division=0
        )
        
        results.update({
            'binary_accuracy': accuracy,
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
        print(f"Binary Classification Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nPer-class metrics:")
        print(f"  No/Reject (0) - Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}")
        print(f"  Yes/Accept (1) - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                No   Yes")
        print(f"Actual No    {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       Yes   {cm[1,0]:4d}  {cm[1,1]:4d}")
        print(f"\nClassification Report:")
        print(class_report)
        print("="*50)
    
    return results

def debug_evaluation_sample(trainer, eval_dataset, tokenizer, num_samples=3):
    """Debug function to inspect a few evaluation samples in detail"""
    print("\n" + "="*60)
    print("DEBUG: EVALUATION SAMPLE INSPECTION")
    print("="*60)
    
    # Get yes/no token IDs
    yes_tokens = tokenizer(" yes", add_special_tokens=False)['input_ids']
    no_tokens = tokenizer(" no", add_special_tokens=False)['input_ids']
    yes_token_id = yes_tokens[0] if yes_tokens else None
    no_token_id = no_tokens[0] if no_tokens else None
    
    print(f"Yes token: '{tokenizer.decode([yes_token_id])}' (ID: {yes_token_id})")
    print(f"No token: '{tokenizer.decode([no_token_id])}' (ID: {no_token_id})")
    
    # Take a small sample
    sample_dataset = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
    
    trainer.args.prediction_loss_only = False
    
    with torch.no_grad():
        eval_output = trainer.predict(sample_dataset)
        predictions_logits = eval_output.predictions
        labels = eval_output.label_ids
        
        for i in range(len(labels)):
            print(f"\n--- Sample {i+1} ---")
            
            # Reconstruct the input text
            input_ids = sample_dataset[i]['input_ids']
            input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"Input text preview: ...{input_text[-200:]}")
            
            # Find target positions
            sample_labels = labels[i]
            target_positions = [j for j, label in enumerate(sample_labels) if label != -100]
            print(f"Target positions (non -100): {target_positions}")
            
            if target_positions:
                target_pos = target_positions[0]
                target_label = sample_labels[target_pos]
                target_logits = predictions_logits[i][target_pos]
                
                print(f"Target position: {target_pos}")
                print(f"Target label (token ID): {target_label}")
                print(f"Target label (decoded): '{tokenizer.decode([target_label])}'")
                
                # Get logits for yes/no tokens
                yes_logit = target_logits[yes_token_id]
                no_logit = target_logits[no_token_id]
                
                print(f"Yes logit: {yes_logit:.4f}")
                print(f"No logit: {no_logit:.4f}")
                
                # Convert to probabilities
                import torch.nn.functional as F
                yes_no_logits = torch.tensor([no_logit, yes_logit])
                yes_no_probs = F.softmax(yes_no_logits, dim=0)
                
                print(f"No probability: {yes_no_probs[0]:.4f}")
                print(f"Yes probability: {yes_no_probs[1]:.4f}")
                
                predicted_binary = 1 if yes_no_probs[1] > yes_no_probs[0] else 0
                true_binary = 1 if target_label == yes_token_id else 0
                
                print(f"Predicted binary: {predicted_binary} ({'Yes' if predicted_binary == 1 else 'No'})")
                print(f"True binary: {true_binary} ({'Yes' if true_binary == 1 else 'No'})")
                print(f"Correct: {predicted_binary == true_binary}")
    
    trainer.args.prediction_loss_only = True
    print("="*60)

class CustomTrainerForBinaryClassification(Trainer):
    """Custom trainer that focuses loss calculation on yes/no token probabilities"""
    
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        
        # Get yes/no token IDs
        yes_tokens = tokenizer(" yes", add_special_tokens=False)['input_ids']
        no_tokens = tokenizer(" no", add_special_tokens=False)['input_ids']
        
        if len(yes_tokens) == 0 or len(no_tokens) == 0:
            raise ValueError("Could not tokenize 'yes' or 'no' tokens")
        
        self.yes_token_id = yes_tokens[0]
        self.no_token_id = no_tokens[0]
        
        print(f"Custom trainer using yes_token_id: {self.yes_token_id}, no_token_id: {self.no_token_id}")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation that focuses on yes/no token probabilities
        """
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if labels is not None:
            # Custom loss calculation
            loss = self.calculate_binary_classification_loss(logits, labels)
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def calculate_binary_classification_loss(self, logits, labels):
        """
        Calculate loss focusing on yes/no token probabilities at target positions
        """
        import torch.nn.functional as F
        
        batch_size, seq_len, vocab_size = logits.shape
        total_loss = 0.0
        num_valid_samples = 0
        
        for i in range(batch_size):
            sample_labels = labels[i]
            sample_logits = logits[i]
            
            # Find positions where labels are not -100 (target positions)
            target_positions = (sample_labels != -100).nonzero(as_tuple=True)[0]
            
            if len(target_positions) > 0:
                # Take the first target position (should be the yes/no token)
                target_pos = target_positions[0]
                target_label = sample_labels[target_pos]
                target_logits = sample_logits[target_pos]  # [vocab_size]
                
                # Extract logits for yes and no tokens only
                yes_logit = target_logits[self.yes_token_id]
                no_logit = target_logits[self.no_token_id]
                
                # Create binary classification setup
                binary_logits = torch.stack([no_logit, yes_logit])  # [no, yes]
                
                # Create binary target (0 for no, 1 for yes)
                if target_label == self.yes_token_id:
                    binary_target = torch.tensor(1, device=logits.device)
                elif target_label == self.no_token_id:
                    binary_target = torch.tensor(0, device=logits.device)
                else:
                    continue  # Skip if target is neither yes nor no
                
                # Calculate cross-entropy loss for binary classification
                sample_loss = F.cross_entropy(binary_logits.unsqueeze(0), binary_target.unsqueeze(0))
                total_loss += sample_loss
                num_valid_samples += 1
        
        if num_valid_samples > 0:
            final_loss = total_loss / num_valid_samples
            # Occasional debugging output (every 100 steps approximately)
            if hasattr(self, '_step_counter'):
                self._step_counter += 1
            else:
                self._step_counter = 1
            
            if self._step_counter % 100 == 0:
                print(f"Binary classification loss at step {self._step_counter}: {final_loss:.4f} (valid samples: {num_valid_samples}/{batch_size})")
            
            return final_loss
        else:
            # Fallback to standard loss if no valid samples
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

class BinaryClassificationCallback(TrainerCallback):
    """Callback to monitor binary classification training progress"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Get yes/no token IDs
        yes_tokens = tokenizer(" yes", add_special_tokens=False)['input_ids']
        no_tokens = tokenizer(" no", add_special_tokens=False)['input_ids']
        self.yes_token_id = yes_tokens[0] if yes_tokens else None
        self.no_token_id = no_tokens[0] if no_tokens else None
        self.step_count = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs and 'train_loss' in logs:
            self.step_count += 1
            if self.step_count % 50 == 0:  # Log every 50 steps
                print(f"Step {self.step_count}: Binary classification loss = {logs['train_loss']:.4f}")

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
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1' or '2'). If not specified, uses all available GPUs")
    parser.add_argument("--cuda_visible_devices", type=str, default=None, help="Set CUDA_VISIBLE_DEVICES environment variable (alternative to --gpu_ids)")
    args = parser.parse_args()
    
    # Set GPU visibility before any CUDA operations
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"Set CUDA_VISIBLE_DEVICES to: {args.cuda_visible_devices}")
    elif args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"Set CUDA_VISIBLE_DEVICES to: {args.gpu_ids}")
    
    # Parse GPU IDs for device mapping
    if args.gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        print(f"Using GPU IDs: {gpu_ids}")
    else:
        gpu_ids = None
    
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
    
    # Create device map based on GPU selection
    device_map = "auto"  # Default to auto
    if gpu_ids is not None and len(gpu_ids) == 1:
        # Single GPU case - place model on specific GPU
        device_map = f"cuda:{gpu_ids[0]}"
    elif gpu_ids is not None and len(gpu_ids) > 1:
        # Multi-GPU case - let transformers handle automatic mapping
        device_map = "auto"
    
    if args.eval:
        # Load the fine-tuned model for evaluation
        model = AutoModelForCausalLM.from_pretrained(
            OUTPUT_DIR,  # Load from fine-tuned model directory
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        print("Loaded fine-tuned model for evaluation")
    else:
        # Load base model for training
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            device_map=device_map
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
            num_train_epochs=5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # Increased to maintain effective batch size
            learning_rate=2e-5,  # Slightly higher learning rate for binary classification
            warmup_steps=50,  # More warmup steps for stability
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
            max_grad_norm=1.0,  # Gradient clipping for stability
            adam_epsilon=1e-8,
            lr_scheduler_type="cosine",  # Cosine scheduler for better convergence
            optim="adamw_torch",
            eval_accumulation_steps=4,  # Process eval in smaller chunks
            dataloader_num_workers=0,  # Disable multiprocessing to save memory
            ddp_find_unused_parameters=False,  # Optimize for model parallelism
            deepspeed=None,  # Can be configured for ZeRO if needed
            prediction_loss_only=True,  # Only compute loss during periodic evaluation
            skip_memory_metrics=True,  # Skip memory metrics to save memory
            report_to=None,  # Disable wandb/tensorboard reporting
        )
        
        # Initialize custom trainer for binary classification
        trainer = CustomTrainerForBinaryClassification(
            tokenizer=tokenizer,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=small_eval_dataset,  # Use smaller eval dataset for periodic evaluation
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Add custom callback for monitoring binary classification progress
        binary_callback = BinaryClassificationCallback(tokenizer)
        trainer.add_callback(binary_callback)
        
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
    
    # Debug: Inspect a few samples to understand the evaluation logic
    debug_evaluation_sample(trainer, eval_dataset, tokenizer, num_samples=3)
    
    # Use batched evaluation to prevent OOM
    eval_results = batched_accuracy_evaluation(
        trainer, eval_dataset, batch_size=20, 
        detailed_eval=args.detailed_eval, tokenizer=tokenizer
    )
    
    print(f"Final evaluation results: {eval_results}")

if __name__ == "__main__":
    main()