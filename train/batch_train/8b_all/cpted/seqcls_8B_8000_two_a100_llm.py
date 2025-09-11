import os
import sys
from datetime import datetime
import argparse
import numpy as np
import torch
import pandas as pd

from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    default_data_collator,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

# Import your dataset builder
from dataset_builder_new import TextDatasetBuilder


# ------------------------- Logging helpers -------------------------
class TeeOutput:
    """Duplicate stdout to both console and log file."""
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


# ------------------------- Data utils -------------------------
def split_dataset_stratified(dataset, test_size=0.2, seed=42):
    texts = dataset['text']
    labels = dataset['labels']
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'labels': test_labels})
    return {'train': train_dataset, 'test': test_dataset}


def preprocess_function_cls(examples, tokenizer, max_length=8192):
    """Tokenize raw text for classification; keep integer labels as-is."""
    enc = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    enc["labels"] = examples["labels"]
    return enc


# ------------------------- Metrics -------------------------
def compute_metrics_cls(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # HF sometimes returns (logits,)
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def final_evaluate(trainer, eval_dataset, detailed_eval=False):
    preds_output = trainer.predict(eval_dataset)
    logits = preds_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    labels = preds_output.label_ids
    preds = np.argmax(logits, axis=-1)

    results = {
        "eval_loss": float(preds_output.metrics.get("test_loss", preds_output.metrics.get("eval_loss", 0.0))),
        "eval_accuracy": accuracy_score(labels, preds),
        "eval_samples": len(labels),
    }

    if detailed_eval:
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        cm = confusion_matrix(labels, preds)
        cls_report = classification_report(
            labels, preds, target_names=["Reject (0)", "Accept (1)"], zero_division=0
        )
        results.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_per_class": precision_pc.tolist(),
            "recall_per_class": recall_pc.tolist(),
            "f1_per_class": f1_pc.tolist(),
            "support_per_class": support_pc.tolist(),
            "confusion_matrix": cm.tolist(),
            "classification_report": cls_report,
        })

        print("\n" + "="*60)
        print("SEQUENCE CLASSIFICATION METRICS")
        print("="*60)
        print(f"Accuracy: {results['eval_accuracy']:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print("\nPer-class metrics (order: 0=Reject, 1=Accept):")
        print(f"  Precision: {precision_pc}")
        print(f"  Recall:    {recall_pc}")
        print(f"  F1:        {f1_pc}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(cls_report)
        print("="*60)

    return results


# ------------------------- Main -------------------------
def main():
    # ----- logging -----
    log_dir = "./log"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_seqcls_{timestamp}.log")
    tee_stdout = TeeOutput(log_file)
    tee_stderr = TeeOutput(log_file)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    print(f"Logging to: {log_file}")
    print(f"Start time: {datetime.now()}")
    print("="*80)

    try:
        # ----- args -----
        parser = argparse.ArgumentParser(description="Sequence classification fine-tuning on LLM")
        parser.add_argument("--eval", action="store_true", help="Only evaluate a fine-tuned model in output_dir")
        parser.add_argument("--detailed_eval", action="store_true", help="Print extended metrics and confusion matrix")
        parser.add_argument("--debug", action="store_true", help="Debug mode: set eval_steps to 5 for quick testing")
        parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Base model to start from")
        parser.add_argument("--data_folder", type=str, default="/mnt/parscratch/users/acr24wz/src/iclr/mineru/llm/")
        parser.add_argument("--labels_file", type=str, default="/mnt/parscratch/users/acr24wz/topcon/train/label_simple.json")
        parser.add_argument("--statistics_file", type=str, default="/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/statistics_per_paper.json")
        parser.add_argument("--output_dir", type=str, default="/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_8B/finetuned/seqcls")
        parser.add_argument("--max_length", type=int, default=8000)
        parser.add_argument("--gpu_ids", type=int, nargs='+', default=None)
        args = parser.parse_args()

        # ----- GPU selection -----
        if not hasattr(args, 'gpu_ids') or args.gpu_ids is None:
            available_gpus = torch.cuda.device_count()
            args.gpu_ids = list(range(available_gpus))
            print(f"Auto-detected {available_gpus} GPUs: {args.gpu_ids}")

        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu_ids))
        print(f"Using GPUs: {args.gpu_ids} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")

        if not torch.cuda.is_available():
            print("No GPU available; exiting.")
            return

        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} | {mem_gb:.1f} GB")

        # ----- paths -----
        MODEL_NAME = args.model_name
        DATA_FOLDER = args.data_folder
        LABELS_FILE = args.labels_file
        STATISTICS_FILE = args.statistics_file
        OUTPUT_DIR = args.output_dir
        MAX_LENGTH = args.max_length

        BASE_MODEL_CACHE = "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_8B/cpt_model/all_20250909_014858/checkpoint-19400"
        os.makedirs(BASE_MODEL_CACHE, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # ----- tokenizer -----
        print("Loading tokenizer...")
        # Load from the model path (fine-tuned or base); ensure pad token set to eos for decoder-only models
        tok_load_path = OUTPUT_DIR if args.eval and os.path.exists(OUTPUT_DIR) else (BASE_MODEL_CACHE or MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(tok_load_path if os.path.exists(tok_load_path) else MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id


        # Set padding side for decoder-only models used in classification
        tokenizer.padding_side = "left"  # Add this line

        print(f"pad_token_id = {tokenizer.pad_token_id}")

        # ----- dataset -----
        print("Loading dataset...")
        cache_suffix = "llm_mineru_all"
        if STATISTICS_FILE:
            cache_suffix += "_with_stats"
        PROCESSED_DATASET_CACHE = f"/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/{cache_suffix}"
        os.makedirs(PROCESSED_DATASET_CACHE, exist_ok=True)

        dataset_info_path = os.path.join(PROCESSED_DATASET_CACHE, "dataset_info.json")
        if os.path.exists(dataset_info_path) and os.path.isdir(PROCESSED_DATASET_CACHE):
            try:
                print("Loading cached processed dataset...")
                dataset = load_from_disk(PROCESSED_DATASET_CACHE)
                dataset_builder = TextDatasetBuilder(
                    DATA_FOLDER, LABELS_FILE, statistics_file=STATISTICS_FILE, max_length=MAX_LENGTH
                )
                print("Loaded cached dataset.")
            except Exception as e:
                print(f"Failed to load cached dataset: {e}\nRebuilding...")
                dataset_builder = TextDatasetBuilder(
                    DATA_FOLDER, LABELS_FILE, statistics_file=STATISTICS_FILE, max_length=MAX_LENGTH
                )
                dataset = dataset_builder.load_dataset_with_ids()
                dataset.save_to_disk(PROCESSED_DATASET_CACHE)
        else:
            print("No cached dataset found. Building...")
            dataset_builder = TextDatasetBuilder(
                DATA_FOLDER, LABELS_FILE, statistics_file=STATISTICS_FILE, max_length=MAX_LENGTH
            )
            dataset = dataset_builder.load_dataset_with_ids()
            dataset.save_to_disk(PROCESSED_DATASET_CACHE)

        # Basic stats
        stats = dataset_builder.get_dataset_stats(dataset)
        print("\nDataset Configuration:")
        print(f"  Data folder: {DATA_FOLDER}")
        print(f"  Labels file: {LABELS_FILE}")
        print(f"  Statistics file: {STATISTICS_FILE if STATISTICS_FILE else 'Not provided'}")
        print(f"  Max length: {MAX_LENGTH}")
        print("Dataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Label distribution: {stats['label_distribution']}")
        print(f"  Avg length (words): {stats['text_stats']['avg_length_words']:.2f}")

        # Split
        print("Splitting dataset (stratified)...")
        s = split_dataset_stratified(dataset, test_size=0.2, seed=42)
        train_dataset_raw, eval_dataset_raw = s['train'], s['test']
        small_eval_raw = eval_dataset_raw.select(range(min(50, len(eval_dataset_raw))))
        print(f"Train: {len(train_dataset_raw)} | Eval: {len(eval_dataset_raw)} (periodic eval uses {len(small_eval_raw)})")

        # Tokenize
        print("Tokenizing...")
        train_dataset = train_dataset_raw.map(
            lambda x: preprocess_function_cls(x, tokenizer, MAX_LENGTH),
            batched=True, batch_size=100, remove_columns=['text']
        )
        eval_dataset = eval_dataset_raw.map(
            lambda x: preprocess_function_cls(x, tokenizer, MAX_LENGTH),
            batched=True, batch_size=100, remove_columns=['text']
        )
        small_eval_dataset = small_eval_raw.map(
            lambda x: preprocess_function_cls(x, tokenizer, MAX_LENGTH),
            batched=True, batch_size=100, remove_columns=['text']
        )
        print("Tokenization complete.")
        print(f"Columns: {train_dataset.column_names}")
        print(f"Sample: { {k: (v[:8] if isinstance(v, list) else v) for k, v in train_dataset[0].items()} }")

        # ----- model -----
        # Train uses base; eval uses OUTPUT_DIR
        model_load_path = OUTPUT_DIR if args.eval else (BASE_MODEL_CACHE if os.path.exists(BASE_MODEL_CACHE) else MODEL_NAME)
        print(f"Loading model for {'evaluation' if args.eval else 'training'} from: {model_load_path}")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_load_path,
            num_labels=2,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        # Make sure the model knows about padding
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.problem_type = "single_label_classification"
        print(model)

        if hasattr(model, 'hf_device_map'):
            print("Device map:")
            for k, v in model.hf_device_map.items():
                print(f"  {k}: {v}")

        # ----- collator -----
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # ----- training -----
        if not args.eval:
            # Set eval_steps based on debug mode
            eval_steps_value = 5 if args.debug else 200
            save_steps_value = 5 if args.debug else 200
            
            if args.debug:
                print("\n" + "="*60)
                print("DEBUG MODE ENABLED")
                print(f"Setting eval_steps={eval_steps_value}, save_steps={save_steps_value}")
                print("="*60 + "\n")
            
            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                num_train_epochs=3,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,
                learning_rate=2e-5,
                warmup_steps=200,
                weight_decay=0.01,
                logging_dir=f"{OUTPUT_DIR}/logs",
                logging_steps=1,
                eval_strategy="steps",
                eval_steps=eval_steps_value,
                save_steps=save_steps_value,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_accuracy",
                greater_is_better=True,
                bf16=True,
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                max_grad_norm=1.0,
                adam_epsilon=1e-8,
                lr_scheduler_type="cosine",
                optim="adamw_torch",
                eval_accumulation_steps=4,
                dataloader_num_workers=0,
                prediction_loss_only=False,
                skip_memory_metrics=True,
                ddp_find_unused_parameters=False,
                dataloader_persistent_workers=False,
                gradient_checkpointing=True,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=small_eval_dataset,  # smaller periodic eval
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics_cls,
            )

            print("Starting training (sequence classification)...")
            trainer.train()

            print(f"Saving fine-tuned classifier to {OUTPUT_DIR}")
            trainer.save_model()
            tokenizer.save_pretrained(OUTPUT_DIR)

            # Reload best model from disk for clean final eval
            print("Reloading best model for final evaluation...")
            del trainer, model
            torch.cuda.empty_cache()

            model = AutoModelForSequenceClassification.from_pretrained(
                OUTPUT_DIR, num_labels=2, torch_dtype=torch.bfloat16,
                device_map="auto", attn_implementation="sdpa",
            )
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.problem_type = "single_label_classification"

        # ----- final evaluation -----
        print("Setting up eval Trainer...")
        eval_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_eval_batch_size=1,
            bf16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            eval_accumulation_steps=4,
            dataloader_num_workers=0,
            prediction_loss_only=False,
            skip_memory_metrics=True,
            ddp_find_unused_parameters=False,
            dataloader_persistent_workers=False,
        )

        eval_trainer = Trainer(
            model=model,
            args=eval_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_cls,
        )

        print("Running final evaluation...")
        results = final_evaluate(eval_trainer, eval_dataset, detailed_eval=args.detailed_eval)
        print(f"Final evaluation results: {results}")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        print("="*80)
        print(f"End time: {datetime.now()}")
        print(f"Log saved to: {log_file}")
        sys.stdout = tee_stdout.terminal
        sys.stderr = tee_stderr.terminal
        tee_stdout.close()
        tee_stderr.close()


if __name__ == "__main__":
    main()
