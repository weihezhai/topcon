import os
import sys
import argparse

# Parse GPU IDs early to set CUDA_VISIBLE_DEVICES before importing torch
parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLM on real-world papers (accept/reject)")
parser.add_argument("--model_path", type=str, default="/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_8B/cpt_model/balanced/finetuned/all", help="Path to fine-tuned model directory")
parser.add_argument("--data_folder", type=str, default="/mnt/parscratch/users/acr24wz/public/iclr2026_mineru_final", help="Path to folder containing parsed papers (subdirs with *_content_list.json)")
parser.add_argument("--labels_file", type=str, default=None, help="Optional labels JSON to compute accuracy metrics")
parser.add_argument("--statistics_file", type=str, default=None, help="Optional statistical.json to append stats to text")
parser.add_argument("--max_length", type=int, default=8000, help="Max sequence length for prompt truncation")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
parser.add_argument("--gpu_ids", type=int, nargs='+', default=None, help="GPU IDs to use (e.g., --gpu_ids 0 1)")
parser.add_argument("--output_file", type=str, default="/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_8B/cpt_model/balanced/finetuned/all/realworld_eval_results.json", help="Where to save predictions")
parser.add_argument("--output_format", type=str, choices=["csv", "json"], default="json", help="Save format")
parser.add_argument("--detailed", action="store_true", help="Print detailed metrics when labels provided")
parser.add_argument("--dataset_cache_dir", type=str, default="/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/eval_2026", help="Optional cache directory for processed eval dataset")
args, _ = parser.parse_known_args()

if args.gpu_ids:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))

import json
import csv
import math
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Local import (same directory as this file)
sys.path.append(os.path.dirname(__file__))
from dataset_builder_new import TextDatasetBuilder

def build_prompts(texts):
    prompts = []
    for t in texts:
        p = (
            "Paper content:\n"
            f"{t}\n\n"
            "Based on this AI research paper's content and statistics, should this paper be accepted? "
            "Answer yes or no.\n\n"
            "Decision:"
        )
        prompts.append(p)
    return prompts

def tokenize_batch(tokenizer, batch_prompts, max_length):
    enc = tokenizer(
        batch_prompts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=True
    )
    return enc["input_ids"], enc["attention_mask"]

def softmax2(a, b):
    m = max(a, b)
    ea = math.exp(float(a - m))
    eb = math.exp(float(b - m))
    s = ea + eb
    return ea / s, eb / s  # (p_yes, p_no)

def main():
    # Load tokenizer and model
    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder="./offload",
        attn_implementation="sdpa"
    )
    model.eval()

    # Yes/No token IDs (match training)
    YES_ID = tokenizer(" yes", add_special_tokens=False)["input_ids"][0]
    NO_ID  = tokenizer(" no",  add_special_tokens=False)["input_ids"][0]

    # Build dataset (labeled if labels_file provided and exists)
    builder = TextDatasetBuilder(
        data_folder=args.data_folder,
        labels_file=args.labels_file,
        statistics_file=args.statistics_file,
        max_length=args.max_length
    )
    has_labels = bool(args.labels_file and os.path.exists(args.labels_file))

    # NEW: dataset caching logic
    cache_dir = args.dataset_cache_dir
    dataset_info_path = os.path.join(cache_dir, "dataset_info.json")
    if os.path.isdir(cache_dir) and os.path.exists(dataset_info_path):
        try:
            print(f"Loading cached eval dataset from {cache_dir}...")
            cached = load_from_disk(cache_dir)
            # Reconstruct expected fields depending on labeling
            if has_labels and all(k in cached.column_names for k in ["text", "paper_id", "labels"]):
                dataset = cached
                texts = dataset["text"]
                paper_ids = dataset["paper_id"]
                true_labels = dataset["labels"]
                print(f"Loaded {len(texts)} labeled samples (cached).")
            elif (not has_labels) and all(k in cached.column_names for k in ["text", "paper_id"]):
                dataset = cached
                texts = dataset["text"]
                paper_ids = dataset["paper_id"]
                true_labels = None
                print(f"Loaded {len(texts)} unlabeled samples (cached).")
            else:
                print("Cache schema mismatch. Rebuilding dataset...")
                raise ValueError("Schema mismatch")
        except Exception as e:
            print(f"Failed to load cache ({e}); rebuilding...")
            if has_labels:
                dataset = builder.load_dataset_with_ids()
                texts = dataset["text"]; paper_ids = dataset["paper_id"]; true_labels = dataset["labels"]
            else:
                dataset = builder.load_unlabeled_dataset_with_ids()
                texts = dataset["text"]; paper_ids = dataset["paper_id"]; true_labels = None
            builder.cache_dataset(dataset, cache_dir)
    else:
        print("No eval dataset cache found. Building dataset...")
        if has_labels:
            dataset = builder.load_dataset_with_ids()
            texts = dataset["text"]; paper_ids = dataset["paper_id"]; true_labels = dataset["labels"]
        else:
            dataset = builder.load_unlabeled_dataset_with_ids()
            texts = dataset["text"]; paper_ids = dataset["paper_id"]; true_labels = None
        builder.cache_dataset(dataset, cache_dir)

    if len(texts) == 0:
        print("No samples to evaluate.")
        return

    prompts = build_prompts(texts)

    # Evaluate in batches
    all_preds = []
    all_yes_logits = []
    all_no_logits = []
    all_yes_prob = []
    all_no_prob = []

    # Choose first device of the model
    try:
        first_device = next(iter(set(model.hf_device_map.values())))
    except Exception:
        first_device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[i:i + args.batch_size]
            input_ids, attn_mask = tokenize_batch(tokenizer, batch_prompts, args.max_length)
            input_ids = input_ids.to(first_device)
            attn_mask = attn_mask.to(first_device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits  # [B, L, V]
            last_logits = logits[:, -1, :]  # next-token distribution

            yes_l = last_logits[:, YES_ID].float().cpu().numpy()
            no_l = last_logits[:, NO_ID].float().cpu().numpy()

            for y, n in zip(yes_l, no_l):
                pred = 1 if y > n else 0
                py, pn = softmax2(y, n)
                all_preds.append(pred)
                all_yes_logits.append(float(y))
                all_no_logits.append(float(n))
                all_yes_prob.append(py)
                all_no_prob.append(pn)

            if (i // args.batch_size) % 50 == 0:
                print(f"Processed {min(i + args.batch_size, len(prompts))}/{len(prompts)}")

            torch.cuda.empty_cache()

    # Prepare results
    results = []
    for idx, pid in enumerate(paper_ids):
        row = {
            "paper_id": pid,
            "pred_label": int(all_preds[idx]),
            "yes_logit": all_yes_logits[idx],
            "no_logit": all_no_logits[idx],
            "yes_prob": all_yes_prob[idx],
            "no_prob": all_no_prob[idx],
        }
        if has_labels:
            row["true_label"] = int(true_labels[idx])
            row["correct"] = int(all_preds[idx] == true_labels[idx])
        results.append(row)

    # Aggregate metrics if labeled
    summary = {}
    if has_labels:
        y_true = np.array(true_labels, dtype=int)
        y_pred = np.array(all_preds, dtype=int)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        summary.update({
            "samples": len(y_true),
            "accuracy": acc,
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1),
            "confusion_matrix": cm
        })
        if args.detailed:
            print("\nDetailed evaluation:")
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision (macro): {prec:.4f}")
            print(f"Recall (macro): {rec:.4f}")
            print(f"F1 (macro): {f1:.4f}")
            print("Confusion matrix:", cm)
            print("\nClassification report:")
            print(classification_report(y_true, y_pred, target_names=["No", "Yes"], zero_division=0))
    else:
        summary.update({
            "samples": len(results),
            "note": "No labels provided; only predictions were generated."
        })

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    if args.output_format == "csv":
        fieldnames = list(results[0].keys())
        with open(args.output_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"Saved predictions to: {args.output_file}")
        if has_labels:
            summary_file = os.path.splitext(args.output_file)[0] + "_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Saved summary to: {summary_file}")
    else:
        out = {"results": results, "summary": summary}
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved predictions and summary to: {args.output_file}")

if __name__ == "__main__":
    main()
