"""
Evaluation script for the fine-tuned Qwen model on paper acceptance prediction
"""
import os
import torch
import numpy as np
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import json

# Import the dataset builder
from dataset_builder_abs_intro import TextDatasetBuilder

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

def compute_detailed_metrics(eval_pred):
    """Compute detailed metrics for evaluation"""
    predictions, labels = eval_pred
    
    # For causal LM, we need to extract the predictions for the target tokens
    predictions = np.argmax(predictions, axis=-1)
    
    # Extract only the non-ignored labels (not -100) and their corresponding predictions
    true_labels = []
    pred_labels = []
    
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != -100:  # Not an ignored token
                true_labels.append(labels[i][j])
                pred_labels.append(predictions[i][j])
    
    if len(true_labels) > 0:
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    else:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

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

def generate_predictions_for_analysis(model, tokenizer, eval_dataset, device, max_samples=100):
    """Generate predictions for detailed analysis"""
    model.eval()
    predictions_analysis = []
    
    # Take a sample for analysis
    sample_indices = np.random.choice(len(eval_dataset), min(max_samples, len(eval_dataset)), replace=False)
    
    with torch.no_grad():
        for idx in sample_indices:
            sample = eval_dataset[int(idx)]
            
            # Prepare input
            input_ids = torch.tensor([sample['input_ids']]).to(device)
            attention_mask = torch.tensor([sample['attention_mask']]).to(device)
            
            # Generate prediction
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]  # Remove batch dimension
            
            # Find the target tokens (non -100 labels)
            labels = sample['labels']
            target_positions = [i for i, label in enumerate(labels) if label != -100]
            
            if target_positions:
                # Get prediction for target tokens
                target_logits = logits[target_positions]
                predicted_tokens = torch.argmax(target_logits, dim=-1)
                true_tokens = torch.tensor([labels[pos] for pos in target_positions])
                
                # Decode tokens to text
                predicted_text = tokenizer.decode(predicted_tokens, skip_special_tokens=True)
                true_text = tokenizer.decode(true_tokens, skip_special_tokens=True)
                
                predictions_analysis.append({
                    'sample_idx': int(idx),
                    'predicted_tokens': predicted_tokens.cpu().tolist(),
                    'true_tokens': true_tokens.tolist(),
                    'predicted_text': predicted_text,
                    'true_text': true_text,
                    'correct': torch.equal(predicted_tokens.cpu(), true_tokens)
                })
    
    return predictions_analysis

def batched_accuracy_evaluation(trainer, eval_dataset, batch_size=25):
    """Evaluate accuracy in small batches to prevent OOM"""
    print(f"Running batched accuracy evaluation on {len(eval_dataset)} samples...")
    
    all_predictions = []
    all_labels = []
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
            eval_results = trainer.evaluate(eval_dataset=batch_dataset)
            
            # Get predictions from the evaluation
            if hasattr(eval_results, 'predictions') and eval_results.predictions is not None:
                predictions = eval_results.predictions
                labels = eval_results.label_ids
            else:
                # Fallback: manually get predictions
                outputs = trainer.predict(batch_dataset)
                predictions = outputs.predictions
                labels = outputs.label_ids
            
            # Convert logits to predicted tokens
            predictions = np.argmax(predictions, axis=-1)
            
            # Extract non-ignored labels and predictions
            for j in range(len(labels)):
                for k in range(len(labels[j])):
                    if labels[j][k] != -100:
                        all_labels.append(labels[j][k])
                        all_predictions.append(predictions[j][k])
            
            total_loss += eval_results['eval_loss'] * (batch_end - i)
            num_batches += 1
        
        # Reset to loss-only mode
        trainer.args.prediction_loss_only = True
        
        # Clear cache after each batch
        torch.cuda.empty_cache()
    
    # Calculate overall metrics
    avg_loss = total_loss / len(eval_dataset)
    accuracy = accuracy_score(all_labels, all_predictions) if len(all_labels) > 0 else 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted') if len(all_labels) > 0 else (0.0, 0.0, 0.0, None)
    
    return {
        'eval_loss': avg_loss,
        'eval_accuracy': accuracy,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1,
        'eval_samples': len(eval_dataset)
    }

def main():
    # Configuration
    MODEL_PATH = "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_1d7B/finetuned_model"  # Path to your trained model
    DATA_FOLDER = "/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_llm_papers/llm_papers_text/"
    LABELS_FILE = "/mnt/parscratch/users/acr24wz/topcon/train/llm_paper/label_simple.json"
    
    MAX_LENGTH = 2048
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path {MODEL_PATH} does not exist!")
        return
    
    print(f"Loading model from: {MODEL_PATH}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("Model loaded successfully!")
    print(f"Model device map: {getattr(model, 'hf_device_map', 'single device')}")
    
    # Load and prepare dataset (same as training)
    print("Loading dataset...")
    
    # Try to load cached dataset first
    PROCESSED_DATASET_CACHE = "/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset"
    
    if os.path.exists(os.path.join(PROCESSED_DATASET_CACHE, "dataset_dict.json")):
        print("Loading cached processed dataset...")
        dataset = load_from_disk(PROCESSED_DATASET_CACHE)
        dataset_builder = TextDatasetBuilder(DATA_FOLDER, LABELS_FILE, MAX_LENGTH)
    else:
        print("Processing dataset for evaluation...")
        dataset_builder = TextDatasetBuilder(DATA_FOLDER, LABELS_FILE, MAX_LENGTH)
        dataset = dataset_builder.load_dataset()
    
    # Print dataset statistics
    stats = dataset_builder.get_dataset_stats(dataset)
    print(f"Dataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Label distribution: {stats['label_distribution']}")
    print(f"Average text length: {stats['text_stats']['avg_length_words']:.2f} words")
    
    # Split dataset (same split as training)
    print("Splitting dataset...")
    train_test_split_result = split_dataset_stratified(dataset, test_size=0.2, seed=42)
    eval_dataset = train_test_split_result['test']
    
    print(f"Evaluation set: {len(eval_dataset)} samples")
    
    # Tokenize evaluation dataset
    print("Tokenizing evaluation dataset...")
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        batch_size=100,
        remove_columns=['text', 'labels']
    )
    
    print("Tokenization complete")
    
    # Data collator
    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    # Evaluation arguments
    eval_args = TrainingArguments(
        output_dir="./eval_output",
        per_device_eval_batch_size=1,
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=["labels"],
        eval_accumulation_steps=4,
        dataloader_num_workers=0,
        prediction_loss_only=True,  # Start with loss-only to prevent OOM
    )
    
    # Initialize trainer for evaluation
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_detailed_metrics,
    )
    
    # Run evaluation - first basic loss evaluation
    print("Running basic loss evaluation...")
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        basic_eval_results = trainer.evaluate()
    
    torch.cuda.empty_cache()
    
    # Run detailed batched evaluation for accuracy metrics
    print("Running detailed batched evaluation for accuracy...")
    detailed_eval_results = batched_accuracy_evaluation(trainer, eval_dataset, batch_size=25)
    
    # Combine results
    eval_results = {**basic_eval_results, **detailed_eval_results}
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Generate detailed predictions for analysis
    print("\nGenerating prediction samples for analysis...")
    device = next(model.parameters()).device
    predictions_analysis = generate_predictions_for_analysis(
        model, tokenizer, eval_dataset, device, max_samples=20
    )
    
    # Print some prediction examples
    print("\n" + "="*50)
    print("PREDICTION SAMPLES")
    print("="*50)
    correct_count = 0
    for i, pred in enumerate(predictions_analysis[:10]):  # Show first 10
        status = "✓ CORRECT" if pred['correct'] else "✗ INCORRECT"
        print(f"\nSample {i+1} [{status}]:")
        print(f"  Predicted: '{pred['predicted_text']}'")
        print(f"  True:      '{pred['true_text']}'")
        if pred['correct']:
            correct_count += 1
    
    print(f"\nSample accuracy: {correct_count}/{len(predictions_analysis[:10])} = {correct_count/len(predictions_analysis[:10]):.3f}")
    
    # Save results
    results_file = "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'eval_metrics': eval_results,
            'dataset_stats': stats,
            'sample_predictions': predictions_analysis[:50]  # Save first 50 for analysis
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    print("Evaluation complete!")

if __name__ == "__main__":
    main()