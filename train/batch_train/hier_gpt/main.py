import os
import sys
from datetime import datetime
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import argparse

# NEW imports
from prompt_hier_yesno import PromptHierYesNo
from hier_prompt_collator import HierPromptCollator
from chunker import (
    chunk_and_tokenize_sliding,
    chunk_and_tokenize_sentence,
)

# Import your dataset builder
from dataset_builder_new import TextDatasetBuilder
from datasets import load_from_disk

# -------------------- original helper classes/functions --------------------
class TeeOutput:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

class CustomDataCollator:
    """Legacy collator for flat sequences (legacy path)"""
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __call__(self, features):
        max_len = min(max(len(f['input_ids']) for f in features), self.max_length)
        batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for feature in features:
            input_ids = feature['input_ids'][:max_len]
            attention_mask = feature['attention_mask'][:max_len]
            labels = feature['labels'][:max_len]
            pad_length = max_len - len(input_ids)
            if pad_length > 0:
                input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
                labels.extend([-100] * pad_length)
            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['labels'].append(labels)
        return {k: torch.tensor(v) for k, v in batch.items()}

# -------------------- DO NOT CHANGE: your compute_metrics --------------------
def compute_metrics(eval_pred, tokenizer=None):
    """Compute metrics using probability comparison of yes/no tokens"""
    if tokenizer is None:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        true_labels = []
        pred_labels = []
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j] != -100:
                    true_labels.append(labels[i][j])
                    pred_labels.append(predictions[i][j])
        if len(true_labels) > 0:
            accuracy = accuracy_score(true_labels, pred_labels)
            return {"accuracy": accuracy}
        else:
            return {"accuracy": 0.0}
    predictions, labels = eval_pred
    yes_tokens = tokenizer(" yes", add_special_tokens=False)['input_ids']
    no_tokens = tokenizer(" no", add_special_tokens=False)['input_ids']
    if len(yes_tokens) == 0 or len(no_tokens) == 0:
        return {"accuracy": 0.0}
    yes_token_id = yes_tokens[0]
    no_token_id = no_tokens[0]
    binary_predictions = []
    binary_labels = []
    for i in range(len(labels)):
        decision_pos = None
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                decision_pos = j
                break
        if decision_pos is not None:
            logits_at_pos = predictions[i][decision_pos]
            yes_logit = logits_at_pos[yes_token_id]
            no_logit = logits_at_pos[no_token_id]
            predicted_label = 1 if yes_logit > no_logit else 0
            true_token_id = labels[i][decision_pos]
            true_label = 1 if true_token_id == yes_token_id else 0
            binary_predictions.append(predicted_label)
            binary_labels.append(true_label)
    if len(binary_labels) > 0:
        accuracy = accuracy_score(binary_labels, binary_predictions)
        return {"accuracy": accuracy}
    else:
        return {"accuracy": 0.0}

# -------------------- original helpers (unchanged) --------------------
def preprocess_function(examples, tokenizer, max_length=1024):
    prompts = []
    for text in examples['text']:
        prompt = f"Paper content:\n{text}\n\nBased on this research paper's abstract, introduction and statistics, should this paper be accepted? Answer yes or no. \n\nDecision:"
        prompts.append(prompt)
    target_tokens = []
    for label in examples['labels']:
        target_tokens.append(" yes" if label == 1 else " no")
    target_encodings = tokenizer(target_tokens, add_special_tokens=False, return_attention_mask=False)
    max_target_length = max(len(t) for t in target_encodings['input_ids'])
    result = tokenizer(
        prompts, truncation=True, padding=False, max_length=max_length - max_target_length,
        return_attention_mask=True, return_token_type_ids=False
    )
    combined_input_ids, combined_attention_mask, labels_for_loss = [], [], []
    for i in range(len(result['input_ids'])):
        full_input = result['input_ids'][i] + target_encodings['input_ids'][i]
        full_mask = result['attention_mask'][i] + [1] * len(target_encodings['input_ids'][i])
        label_ids = [-100] * len(result['input_ids'][i]) + target_encodings['input_ids'][i]
        if len(full_input) > max_length:
            excess = len(full_input) - max_length
            input_length = len(result['input_ids'][i])
            target_length = len(target_encodings['input_ids'][i])
            new_input_length = max(1, input_length - excess)
            full_input = result['input_ids'][i][:new_input_length] + target_encodings['input_ids'][i]
            full_mask = result['attention_mask'][i][:new_input_length] + [1] * target_length
            label_ids = [-100] * new_input_length + target_encodings['input_ids'][i]
        combined_input_ids.append(full_input)
        combined_attention_mask.append(full_mask)
        labels_for_loss.append(label_ids)
    return {'input_ids': combined_input_ids, 'attention_mask': combined_attention_mask, 'labels': labels_for_loss}

def download_and_save_model(model_name, cache_dir):
    print(f"Downloading model {model_name} to {cache_dir}...")
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.save_pretrained(cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
    model.save_pretrained(cache_dir)
    print(f"Model downloaded and saved to {cache_dir}")
    return cache_dir

def split_dataset_stratified(dataset, test_size=0.2, seed=42):
    texts = dataset['text']; labels = dataset['labels']
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
    test_dataset  = Dataset.from_dict({'text': test_texts,  'labels': test_labels})
    return {'train': train_dataset, 'test': test_dataset}

def custom_evaluate_with_memory_cleanup(trainer, eval_dataset=None):
    torch.cuda.empty_cache()
    with torch.no_grad():
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    torch.cuda.empty_cache()
    return eval_results

# (kept as-is; we will skip calling it in hier mode)
def batched_accuracy_evaluation(trainer, eval_dataset, batch_size=10, detailed_eval=False, tokenizer=None):
    print(f"Running probability-based accuracy evaluation on {len(eval_dataset)} samples...")
    yes_token_id = tokenizer(" yes", add_special_tokens=False)['input_ids'][0]
    no_token_id  = tokenizer(" no",  add_special_tokens=False)['input_ids'][0]
    all_binary_predictions, all_binary_labels = [], []
    total_loss = 0.0
    model = trainer.model
    for i in range(0, len(eval_dataset), batch_size):
        batch_end = min(i + batch_size, len(eval_dataset))
        batch_dataset = eval_dataset.select(range(i, batch_end))
        print(f"Processing batch {i//batch_size + 1}/{(len(eval_dataset) + batch_size - 1)//batch_size} (samples {i}-{batch_end-1})")
        torch.cuda.empty_cache()
        with torch.no_grad():
            batch_input_ids, batch_attention_masks, decision_positions, true_labels = [], [], [], []
            for sample in batch_dataset:
                input_ids = sample['input_ids']; attention_mask = sample['attention_mask']; labels = sample['labels']
                decision_pos = None
                for k in range(len(labels)):
                    if labels[k] != -100:
                        decision_pos = k
                        break
                if decision_pos is not None:
                    input_portion = input_ids[:decision_pos]
                    mask_portion  = attention_mask[:decision_pos]
                    batch_input_ids.append(input_portion)
                    batch_attention_masks.append(mask_portion)
                    decision_positions.append(len(input_portion))
                    true_token_id = labels[decision_pos]
                    true_labels.append(1 if true_token_id == yes_token_id else 0)
            if len(batch_input_ids) > 0:
                max_len = max(len(ids) for ids in batch_input_ids)
                padded_input_ids, padded_attention_masks = [], []
                for j, (ids, msk) in enumerate(zip(batch_input_ids, batch_attention_masks)):
                    pad_length = max_len - len(ids)
                    padded_ids = ids + [tokenizer.pad_token_id] * pad_length
                    padded_mask = msk + [0] * pad_length
                    padded_input_ids.append(padded_ids); padded_attention_masks.append(padded_mask)
                input_tensor = torch.tensor(padded_input_ids); mask_tensor = torch.tensor(padded_attention_masks)
                first_device = next(model.parameters()).device
                input_tensor = input_tensor.to(first_device); mask_tensor = mask_tensor.to(first_device)
                outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
                logits = outputs.logits
                for j, (decision_pos, true_label) in enumerate(zip(decision_positions, true_labels)):
                    logits_at_pos = logits[j, decision_pos - 1]  # -1 because we predict the next token
                    yes_logit = logits_at_pos[yes_token_id]; no_logit = logits_at_pos[no_token_id]
                    predicted_label = 1 if yes_logit > no_logit else 0
                    all_binary_predictions.append(predicted_label); all_binary_labels.append(true_label)
            eval_results = trainer.evaluate(eval_dataset=batch_dataset)
            total_loss += eval_results['eval_loss'] * (batch_end - i)
        torch.cuda.empty_cache()
    avg_loss = total_loss / len(eval_dataset)
    accuracy = accuracy_score(all_binary_labels, all_binary_predictions) if len(all_binary_labels) > 0 else 0.0
    results = {'eval_loss': avg_loss, 'eval_accuracy': accuracy, 'eval_samples': len(eval_dataset)}
    if detailed_eval and len(all_binary_labels) > 0:
        precision, recall, f1, support = precision_recall_fscore_support(
            all_binary_labels, all_binary_predictions, average='binary', zero_division=0
        )
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            all_binary_labels, all_binary_predictions, average=None, zero_division=0
        )
        cm = confusion_matrix(all_binary_labels, all_binary_predictions)
        class_report = classification_report(all_binary_labels, all_binary_predictions, target_names=['No','Yes'], zero_division=0)
        results.update({
            'binary_accuracy': accuracy_score(all_binary_labels, all_binary_predictions),
            'precision': precision, 'recall': recall, 'f1': f1,
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
        print(f"Binary Classification Accuracy: {results['binary_accuracy']:.4f}")
        print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
        print(class_report)
        print("="*50)
    return results

# -------------------- NEW: hierarchical preprocess --------------------
def preprocess_hierarchical(examples, tokenizer, mode="sliding",
                            chunk_len=1536, stride=256, max_sentences=128, max_chunks_cap=128):
    out_chunks, out_masks, out_doc_labels = [], [], []
    for text, y in zip(examples["text"], examples["labels"]):
        if mode == "sentence":
            obj = chunk_and_tokenize_sentence(text, tokenizer, max_sentences=max_sentences, add_bos=True)
        else:
            obj = chunk_and_tokenize_sliding(text, tokenizer, chunk_len=chunk_len, stride=stride, add_bos=True)
        chunks = obj["chunks"][:max_chunks_cap] or [[tokenizer.pad_token_id]]
        masks  = obj["chunk_masks"][:max_chunks_cap] or [[0]]
        out_chunks.append(chunks); out_masks.append(masks); out_doc_labels.append(int(y))
    return {"chunks": out_chunks, "chunk_masks": out_masks, "labels": out_doc_labels}  # labels stays 0/1 here

# -------------------- main --------------------
def main():
    # logging
    log_dir = "./log"; os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_llm_{timestamp}.log")
    tee_stdout = TeeOutput(log_file); tee_stderr = TeeOutput(log_file)
    sys.stdout = tee_stdout; sys.stderr = tee_stderr
    print(f"Logging to: {log_file}\nStart time: {datetime.now()}\n" + "="*80)

    try:
        parser = argparse.ArgumentParser(description="Fine-tune LM (legacy or hierarchical prompt+soft tokens)")
        parser.add_argument("--eval", action="store_true", help="Run evaluation mode on fine-tuned model")
        parser.add_argument("--detailed_eval", action="store_true")
        parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
        parser.add_argument("--data_folder", type=str, default="/mnt/parscratch/users/acr24wz/src/iclr/mineru/llm/")
        parser.add_argument("--labels_file", type=str, default="/mnt/parscratch/users/acr24wz/topcon/train/label_simple.json")
        parser.add_argument("--statistics_file", type=str, default='/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/statistics_per_paper.json')
        parser.add_argument("--output_dir", type=str, default="/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B/finetuned_model/llm/hier")
        parser.add_argument("--max_length", type=int, default=10000)
        parser.add_argument("--gpu_ids", type=int, nargs='+', default=None, help="List of GPU IDs to use (e.g., --gpu_ids 0 1 2). If not set, auto-detects available GPUs.")

        # NEW flags
        parser.add_argument("--use_hier", action="store_true", help="Enable hierarchical prompt+soft tokens pipeline")
        parser.add_argument("--hier_mode", type=str, choices=["sliding","sentence"], default="sliding")
        parser.add_argument("--chunk_len", type=int, default=128)
        parser.add_argument("--stride", type=int, default=32)
        parser.add_argument("--max_sentences", type=int, default=128)
        parser.add_argument("--max_chunks_cap", type=int, default=128)
        parser.add_argument("--k_soft_tokens", type=int, default=6)
        parser.add_argument("--prompt_prefix", type=str, default="You are a reviewer. Below are compressed chunk representations of an top conference AI paper.\n")
        parser.add_argument("--prompt_suffix", type=str, default="\nBased on the paper content, should this paper be accepted? Answer yes or no.\n\nDecision:")
        parser.add_argument("--freeze_lm_for_chunks", action="store_true", help="Freeze the language model when processing chunks (only train soft tokens)")
        # Stage-2 (top layers finetune)
        parser.add_argument("--stage2_unfreeze_top", type=int, default=0, help="Number of top layers to unfreeze in Stage-2 (0=skip)")
        parser.add_argument("--stage2_epochs", type=int, default=0)
        parser.add_argument("--stage2_lr", type=float, default=1e-4)

        args = parser.parse_args()

        if not hasattr(args, 'gpu_ids') or args.gpu_ids is None:
            available_gpus = torch.cuda.device_count()
            args.gpu_ids = list(range(available_gpus))
            print(f"Auto-detected {available_gpus} GPUs: {args.gpu_ids}")

        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu_ids))
        print(f"Using GPU IDs: {args.gpu_ids}\nCUDA_VISIBLE_DEVICES set.")

        MODEL_NAME = args.model_name
        DATA_FOLDER = args.data_folder
        LABELS_FILE = args.labels_file
        STATISTICS_FILE = args.statistics_file
        OUTPUT_DIR = args.output_dir
        MAX_LENGTH = args.max_length

        BASE_MODEL_CACHE = "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B"
        if args.eval:
            if not os.path.exists(OUTPUT_DIR):
                print(f"Error: Fine-tuned model not found at {OUTPUT_DIR}"); return
            MODEL_PATH = OUTPUT_DIR
            print(f"Eval mode using model from {MODEL_PATH}")
        else:
            MODEL_PATH = BASE_MODEL_CACHE
            print(f"Train mode using base model from {MODEL_PATH}")

        if torch.cuda.is_available():
            print(f"GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}  "
                      f"Memory: {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} GB")
        else:
            print("No GPU available."); return

        os.makedirs(BASE_MODEL_CACHE, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if not args.eval:
            config_file = os.path.join(BASE_MODEL_CACHE, "config.json")
            if not os.path.exists(config_file):
                download_and_save_model(MODEL_NAME, BASE_MODEL_CACHE)
            else:
                print(f"Using cached model from {BASE_MODEL_CACHE}")

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id

        # dataset load/build/cache (unchanged)
        print("Loading dataset...")
        cache_suffix = "llm_json"
        if STATISTICS_FILE: cache_suffix += "_with_stats"
        PROCESSED_DATASET_CACHE = f"/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/{cache_suffix}"
        os.makedirs(PROCESSED_DATASET_CACHE, exist_ok=True)
        dataset_info_path = os.path.join(PROCESSED_DATASET_CACHE, "dataset_info.json")

        if os.path.exists(dataset_info_path) and os.path.isdir(PROCESSED_DATASET_CACHE):
            try:
                print("Loading cached processed dataset...")
                dataset = load_from_disk(PROCESSED_DATASET_CACHE)
                dataset_builder = TextDatasetBuilder(DATA_FOLDER, LABELS_FILE, statistics_file=STATISTICS_FILE, max_length=MAX_LENGTH)
                print("Successfully loaded cached dataset!")
            except Exception as e:
                print(f"Failed to load cached dataset: {e}\nProcessing dataset from scratch...")
                dataset_builder = TextDatasetBuilder(DATA_FOLDER, LABELS_FILE, statistics_file=STATISTICS_FILE, max_length=MAX_LENGTH)
                dataset = dataset_builder.load_dataset_with_ids()
                print(f"Saving processed dataset to {PROCESSED_DATASET_CACHE}")
                dataset.save_to_disk(PROCESSED_DATASET_CACHE)
        else:
            print("No cached dataset. Processing...")
            dataset_builder = TextDatasetBuilder(DATA_FOLDER, LABELS_FILE, statistics_file=STATISTICS_FILE, max_length=MAX_LENGTH)
            dataset = dataset_builder.load_dataset_with_ids()
            print(f"Saving processed dataset to {PROCESSED_DATASET_CACHE}")
            dataset.save_to_disk(PROCESSED_DATASET_CACHE)

        # print dataset info
        stats = dataset_builder.get_dataset_stats(dataset)
        print("Dataset Configuration:")
        print(f"  Data folder: {DATA_FOLDER}")
        print(f"  Labels file: {LABELS_FILE}")
        print(f"  Statistics file: {STATISTICS_FILE if STATISTICS_FILE else 'Not provided'}")
        print(f"  Max length: {MAX_LENGTH}")
        print("Dataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Label distribution: {stats['label_distribution']}")
        print(f"  Avg text length (words): {stats['text_stats']['avg_length_words']:.2f}")

        # stratified split
        print("Splitting dataset...")
        split = split_dataset_stratified(dataset, test_size=0.2, seed=42)
        train_dataset = split['train']; eval_dataset = split['test']
        small_eval_dataset = eval_dataset.select(range(min(50, len(eval_dataset))))
        print(f"Train: {len(train_dataset)}  Test: {len(eval_dataset)}  (periodic eval uses {len(small_eval_dataset)})")

        # -------------------- tokenization / chunking --------------------
        if not args.use_hier:
            # legacy path (unchanged)
            print("Tokenizing datasets (legacy path)...")
            train_dataset = train_dataset.map(
                lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
                batched=True, batch_size=100, remove_columns=['text', 'labels']
            )
            eval_dataset = eval_dataset.map(
                lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
                batched=True, batch_size=100, remove_columns=['text', 'labels']
            )
            small_eval_dataset = small_eval_dataset.map(
                lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
                batched=True, batch_size=100, remove_columns=['text', 'labels']
            )
            data_collator = CustomDataCollator(tokenizer=tokenizer, max_length=MAX_LENGTH)
        else:
            print(f"Preparing hierarchical chunks (mode={args.hier_mode})...")
            # keep 'labels' as 0/1; collator will convert to seq labels
            train_dataset = train_dataset.map(
                lambda x: preprocess_hierarchical(
                    x, tokenizer, mode=args.hier_mode,
                    chunk_len=args.chunk_len, stride=args.stride,
                    max_sentences=args.max_sentences, max_chunks_cap=args.max_chunks_cap
                ),
                batched=True, remove_columns=[]
            )
            eval_dataset = eval_dataset.map(
                lambda x: preprocess_hierarchical(
                    x, tokenizer, mode=args.hier_mode,
                    chunk_len=args.chunk_len, stride=args.stride,
                    max_sentences=args.max_sentences, max_chunks_cap=args.max_chunks_cap
                ),
                batched=True, remove_columns=[]
            )
            small_eval_dataset = eval_dataset.select(range(min(50, len(eval_dataset))))

        # -------------------- model --------------------
        print("Loading model...")
        if args.eval:
            model_load_path = OUTPUT_DIR
        else:
            model_load_path = BASE_MODEL_CACHE

        if not args.use_hier:
            # legacy: standard CausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_load_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={i: "80GiB" for i in range(len(args.gpu_ids))},
                offload_folder="./offload",
            )
            print(model)
            if hasattr(model, 'hf_device_map'):
                print("Model device map:", model.hf_device_map)
        else:
            # hier: PromptHierYesNo wrapper
            model = PromptHierYesNo(
                base_model_path=model_load_path,
                tokenizer=tokenizer,
                prompt_prefix=args.prompt_prefix,
                prompt_suffix=args.prompt_suffix,
                k_soft_tokens_per_chunk=args.k_soft_tokens,
                freeze_lm_for_chunks=args.freeze_lm_for_chunks,  # freeze during training Stage-1
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

            # Tell Trainer not to DataParallel-wrap us
            setattr(model, "is_parallelizable", True)
            setattr(model, "model_parallel", True)
            if hasattr(model.lm, "hf_device_map"):
                model.hf_device_map = model.lm.hf_device_map
            # instantiate collator now we know prompt tokenized lengths
            P_len, S_len = model.prompt_lengths()
            print(f"[hier] Prefix len={P_len}, Suffix len={S_len}, K={args.k_soft_tokens}")
            data_collator = HierPromptCollator(
                tokenizer=tokenizer,
                prompt_prefix_ids=model.prefix_ids.tolist(),
                prompt_suffix_ids=model.suffix_ids.tolist(),
                k_soft_tokens_per_chunk=args.k_soft_tokens,
                max_seq_len=args.chunk_len  # cap per-sentence token length if hier_mode='sentence'
            )
            # enable activation checkpointing if available (optional)
            if hasattr(model.backbone, "gradient_checkpointing_enable"):
                model.backbone.gradient_checkpointing_enable()

        # -------------------- training --------------------
        if not args.eval:
            # base (Stage-1) training args
            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                num_train_epochs=5,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,
                learning_rate=1e-5,  # higher LR when most params frozen
                warmup_steps=20,
                weight_decay=0.001,
                logging_dir=f"{OUTPUT_DIR}/logs",
                logging_steps=1,
                eval_strategy="steps",  
                eval_steps=100,
                save_steps=200,
                save_total_limit=2,
                load_best_model_at_end=False,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                bf16=True,
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                label_names=["labels"],
                max_grad_norm=1.0,
                adam_epsilon=1e-8,
                lr_scheduler_type="linear",
                optim="adamw_torch",
                eval_accumulation_steps=16,
                dataloader_num_workers=0,
                prediction_loss_only= False if args.use_hier else True,   # need logits in hier mode for metrics
                skip_memory_metrics=True,
                ddp_find_unused_parameters=False,
                dataloader_persistent_workers=False,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=small_eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer)
            )

            print("Starting training...")
            # keep your memory-safe wrapper
            original_evaluate = trainer.evaluate
            def memory_safe_evaluate(*a, **k):
                torch.cuda.empty_cache()
                with torch.no_grad():
                    r = original_evaluate(*a, **k)
                torch.cuda.empty_cache()
                return r
            trainer.evaluate = memory_safe_evaluate

            trainer.train()

            print(f"Saving fine-tuned model to {OUTPUT_DIR}")
            trainer.save_model(); tokenizer.save_pretrained(OUTPUT_DIR)

            # -------------------- Stage-2 (optional): unfreeze top layers --------------------
            if args.use_hier and args.stage2_unfreeze_top > 0 and args.stage2_epochs > 0:
                print(f"\n[Stage-2] Unfreezing top-{args.stage2_unfreeze_top} layers and continuing training...")
                model.unfreeze_top_layers(args.stage2_unfreeze_top)
                stage2_args = TrainingArguments(
                    output_dir=OUTPUT_DIR,
                    num_train_epochs=args.stage2_epochs,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=8,
                    learning_rate=args.stage2_lr,
                    weight_decay=0.001,
                    eval_strategy="steps",  
                    eval_steps=100,
                    save_steps=200,
                    bf16=True,
                    remove_unused_columns=False,
                    label_names=["labels"],
                    logging_steps=1,
                    lr_scheduler_type="linear",
                    ddp_find_unused_parameters=False,
                    prediction_loss_only=False,  # keep to compute metrics
                )
                stage2_trainer = Trainer(
                    model=model,
                    args=stage2_args,
                    train_dataset=train_dataset,
                    eval_dataset=small_eval_dataset,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=lambda p: compute_metrics(p, tokenizer),
                )
                stage2_trainer.train()
                print("[Stage-2] Done. Saving...")
                stage2_trainer.save_model(); tokenizer.save_pretrained(OUTPUT_DIR)

        # -------------------- final evaluation --------------------
        print("Setting up trainer for final evaluation...")
        eval_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_eval_batch_size=1,
            bf16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            label_names=["labels"],
            eval_accumulation_steps=4,
            dataloader_num_workers=0,
            prediction_loss_only= False if args.use_hier else True,
            skip_memory_metrics=True,
            ddp_find_unused_parameters=False,
            dataloader_persistent_workers=False,
        )
        final_trainer = Trainer(
            model=model,
            args=eval_args,
            tokenizer=tokenizer,
            data_collator=data_collator if args.use_hier else CustomDataCollator(tokenizer, MAX_LENGTH),
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        )

        print("Running final evaluation...")
        if args.use_hier:
            # in hier mode: standard evaluate (predictions available)
            eval_results = final_trainer.evaluate(eval_dataset=eval_dataset)
        else:
            # legacy path: keep your custom accuracy evaluation (uses flat tokens)
            eval_results = batched_accuracy_evaluation(
                final_trainer, eval_dataset, batch_size=1,
                detailed_eval=args.detailed_eval, tokenizer=tokenizer
            )
        print(f"Final evaluation results: {eval_results}")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback; traceback.print_exc()
        raise
    finally:
        print("="*80)
        print(f"End time: {datetime.now()}")
        print(f"Log saved to: {log_file}")
        sys.stdout = tee_stdout.terminal; sys.stderr = tee_stderr.terminal
        tee_stdout.close(); tee_stderr.close()

if __name__ == "__main__":
    main()
