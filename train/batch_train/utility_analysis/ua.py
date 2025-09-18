import os
import argparse
import json
from typing import List, Dict, Any
import torch
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from math import isclose

# External dependency identical to training script
from dataset_builder_new import TextDatasetBuilder

# ----------------------------
# Preprocessing & dataset utils
# ----------------------------
def split_dataset_stratified(dataset, test_size=0.2, seed=42):
    texts = dataset['text']
    labels = dataset['labels']
    tr_texts, te_texts, tr_labels, te_labels = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    return {
        "train": Dataset.from_dict({"text": tr_texts, "labels": tr_labels}),
        "test": Dataset.from_dict({"text": te_texts, "labels": te_labels})
    }

def preprocess_function(examples, tokenizer, max_length=8000):
    prompts = []
    for text in examples["text"]:
        prompts.append(
            f"Paper content:\n{text}\n\nBased on this AI research paper's content and statistics, "
            f"should this paper be accepted? Answer yes or no.\n\nDecision:"
        )
    target_tokens = [" yes" if l == 1 else " no" for l in examples["labels"]]
    tgt_enc = tokenizer(target_tokens, add_special_tokens=False, return_attention_mask=False)
    max_tgt_len = max(len(t) for t in tgt_enc["input_ids"]) if tgt_enc["input_ids"] else 1
    enc = tokenizer(
        prompts,
        truncation=True,
        padding=False,
        max_length=max_length - max_tgt_len,
        return_attention_mask=True,
        return_token_type_ids=False
    )
    input_ids, attn_masks, label_ids = [], [], []
    for i in range(len(enc["input_ids"])):
        full_input = enc["input_ids"][i] + tgt_enc["input_ids"][i]
        full_mask = enc["attention_mask"][i] + [1] * len(tgt_enc["input_ids"][i])
        labels = [-100] * len(enc["input_ids"][i]) + tgt_enc["input_ids"][i]
        if len(full_input) > max_length:
            excess = len(full_input) - max_length
            new_in_len = max(1, len(enc["input_ids"][i]) - excess)
            full_input = enc["input_ids"][i][:new_in_len] + tgt_enc["input_ids"][i]
            full_mask = enc["attention_mask"][i][:new_in_len] + [1] * len(tgt_enc["input_ids"][i])
            labels = [-100] * new_in_len + tgt_enc["input_ids"][i]
        input_ids.append(full_input)
        attn_masks.append(full_mask)
        label_ids.append(labels)
    return {"input_ids": input_ids, "attention_mask": attn_masks, "labels": label_ids}

# ----------------------------
# Core computation
# ----------------------------
def collect_confidences(model, tokenizer, dataset, yes_id, no_id, batch_size=1, device=None):
    """
    For each sample:
      - Determine decision position (first label != -100)
      - Forward on context up to (decision_pos - 1 + 1) tokens (i.e., up to decision_pos)
      - Extract logits for next token at (decision_pos - 1)
      - Compute p_yes / p_no using softmax over those two logits only
      - confidence = abs(p_yes - p_no)
    Returns list of dicts.
    """
    model.eval()
    results = []
    total = len(dataset)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = dataset.select(range(start, end))
        # Build minimal ragged batch (pad ourselves)
        ctx_ids, ctx_mask, decision_positions, true_labels = [], [], [], []
        for sample in batch:
            labels = sample["labels"]
            decision_pos = None
            for k, token in enumerate(labels):
                if token != -100:
                    decision_pos = k
                    break
            if decision_pos is None or decision_pos == 0:
                continue  # skip malformed
            # Context (everything before target token)
            input_part = sample["input_ids"][:decision_pos]
            mask_part = sample["attention_mask"][:decision_pos]
            ctx_ids.append(input_part)
            ctx_mask.append(mask_part)
            decision_positions.append(len(input_part))  # position where next token predicted
            true_token_id = labels[decision_pos]
            true_labels.append(1 if true_token_id == yes_id else 0)

        if not ctx_ids:
            continue
        max_len = max(len(x) for x in ctx_ids)
        pad_id = tokenizer.pad_token_id
        ids_tensor = []
        mask_tensor = []
        for seq, msk in zip(ctx_ids, ctx_mask):
            pad_len = max_len - len(seq)
            ids_tensor.append(seq + [pad_id] * pad_len)
            mask_tensor.append(msk + [0] * pad_len)
        ids_tensor = torch.tensor(ids_tensor)
        mask_tensor = torch.tensor(mask_tensor)

        # Determine device (first shard device) if not provided
        if device is None:
            if hasattr(model, "hf_device_map"):
                # pick first param device
                device = next(model.parameters()).device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ids_tensor = ids_tensor.to(device)
        mask_tensor = mask_tensor.to(device)

        with torch.no_grad():
            out = model(input_ids=ids_tensor, attention_mask=mask_tensor)
            logits = out.logits  # [B, L, V]

        for i, (dec_pos, true_label) in enumerate(zip(decision_positions, true_labels)):
            # dec_pos is length of context; we want logits at dec_pos - 1 index (last token)
            last_index = dec_pos - 1
            token_logits = logits[i, last_index]
            yes_logit = token_logits[yes_id]
            no_logit = token_logits[no_id]
            pair = torch.stack([yes_logit, no_logit])
            probs = torch.softmax(pair, dim=0)
            p_yes = probs[0].item()
            p_no = probs[1].item()
            pred_label = 1 if p_yes > p_no else 0
            confidence = abs(p_yes - p_no)  # normalized diff (since p_yes + p_no = 1 after softmax over two tokens)
            results.append({
                "index": start + i,
                "true_label": true_label,
                "pred_label": pred_label,
                "p_yes": p_yes,
                "p_no": p_no,
                "confidence": confidence
            })
    return results

# ----------------------------
# Metrics & stratification
# ----------------------------
def precision_recall_for_thresholds(records: List[Dict[str, Any]], thresholds: List[float]):
    """
    Treat 'Accept' (label=1) as positive.
    For threshold t: consider predictions with confidence >= t.
      precision(t) = TP / (TP + FP) over subset
      recall(t)    = TP / P_total  (P_total = total true positives in full set)
      coverage(t)  = subset_size / total_samples
    """
    total_pos = sum(r["true_label"] == 1 for r in records)
    total_samples = len(records)
    out = []
    for t in thresholds:
        subset = [r for r in records if r["confidence"] >= t]
        if not subset:
            out.append({
                "threshold": t,
                "precision": None,
                "recall": None,
                "coverage": 0.0,
                "subset_size": 0
            })
            continue
        tp = sum(r["pred_label"] == 1 and r["true_label"] == 1 for r in subset)
        fp = sum(r["pred_label"] == 1 and r["true_label"] == 0 for r in subset)
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / total_pos if total_pos > 0 else None
        coverage = len(subset) / total_samples
        out.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "coverage": coverage,
            "subset_size": len(subset)
        })
    return out

def binned_stratification(records: List[Dict[str, Any]], bin_edges: List[float]):
    """
    Bin intervals: [edge_i, edge_{i+1}) except last where upper inclusive.
    For each bin: compute
      - count
      - overall accuracy in bin
      - per-class (Accept/Reject) precision & recall (recall uses total true of that class in entire dataset)
    """
    assert len(bin_edges) >= 2
    if not isclose(bin_edges[0], 0.0):
        raise ValueError("First bin edge must be 0.0")
    if not isclose(bin_edges[-1], 1.0):
        raise ValueError("Last bin edge must be 1.0")
    total_accept = sum(r["true_label"] == 1 for r in records)
    total_reject = len(records) - total_accept
    bins = []
    for i in range(len(bin_edges) - 1):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            subset = [r for r in records if lo <= r["confidence"] <= hi]
        else:
            subset = [r for r in records if lo <= r["confidence"] < hi]
        if not subset:
            bins.append({
                "bin": f"[{lo:.2f}, {hi:.2f}{']' if i==len(bin_edges)-2 else ')'}",
                "count": 0,
                "accuracy": None,
                "accept_precision": None,
                "accept_recall": None,
                "reject_precision": None,
                "reject_recall": None
            })
            continue
        correct = sum(r["pred_label"] == r["true_label"] for r in subset)
        # Accept metrics
        tp_a = sum(r["pred_label"] == 1 and r["true_label"] == 1 for r in subset)
        fp_a = sum(r["pred_label"] == 1 and r["true_label"] == 0 for r in subset)
        fn_a = sum(r["pred_label"] == 0 and r["true_label"] == 1 for r in subset)
        accept_precision = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else None
        accept_recall = tp_a / total_accept if total_accept > 0 else None
        # Reject metrics
        tp_r = sum(r["pred_label"] == 0 and r["true_label"] == 0 for r in subset)
        fp_r = sum(r["pred_label"] == 0 and r["true_label"] == 1 for r in subset)
        fn_r = sum(r["pred_label"] == 1 and r["true_label"] == 0 for r in subset)
        reject_precision = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else None
        reject_recall = tp_r / total_reject if total_reject > 0 else None
        bins.append({
            "bin": f"[{lo:.2f}, {hi:.2f}{']' if i==len(bin_edges)-2 else ')'}",
            "count": len(subset),
            "accuracy": correct / len(subset),
            "accept_precision": accept_precision,
            "accept_recall": accept_recall,
            "reject_precision": reject_precision,
            "reject_recall": reject_recall
        })
    return bins

# ----------------------------
# Printing helpers
# ----------------------------
def fmt(v):
    if v is None:
        return "-"
    return f"{v:.4f}"

def print_threshold_table(rows):
    print("\nConfidence-based Stratification (threshold >= t) - Positive class: Accept")
    print(f"{'Threshold':>10} {'Prec':>8} {'Rec':>8} {'Coverage':>10} {'Subset':>8}")
    for r in rows:
        print(f"{r['threshold']:>10.2f} {fmt(r['precision']):>8} {fmt(r['recall']):>8} {r['coverage']:>10.4f} {r['subset_size']:>8}")

def print_bin_table(rows):
    print("\nConfidence bins (interval membership)")
    hdr = ("Bin", "Count", "Acc", "AccP", "AccR", "RejP", "RejR")
    print(f"{hdr[0]:>18} {hdr[1]:>8} {hdr[2]:>8} {hdr[3]:>8} {hdr[4]:>8} {hdr[5]:>8} {hdr[6]:>8}")
    for r in rows:
        print(f"{r['bin']:>18} {r['count']:>8} {fmt(r['accuracy']):>8} "
              f"{fmt(r['accept_precision']):>8} {fmt(r['accept_recall']):>8} "
              f"{fmt(r['reject_precision']):>8} {fmt(r['reject_recall']):>8}")

# ----------------------------
# Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Confidence-based Stratification Utility")
    # Defaults mirror imbalanced training script
    p.add_argument(
        "--model_path",
        default="/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B/cpt_model/imbalanced/finetuned/all/checkpoint-1200/",
        help="Path to fine-tuned (or base) model directory"
    )
    p.add_argument(
        "--data_folder",
        default="/mnt/parscratch/users/acr24wz/src/iclr/mineru/all/",
        help="Raw data folder (same as training)"
    )
    p.add_argument(
        "--labels_file",
        default="/mnt/parscratch/users/acr24wz/topcon/balanced_labels.json",
        help="Labels JSON file"
    )
    p.add_argument(
        "--statistics_file",
        default="/mnt/parscratch/users/acr24wz/public/stats_overall.json",
        help="Statistics JSON file (optional)"
    )
    p.add_argument(
        "--processed_dataset_cache",
        default="/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/imbalanced/all_mineru_all_with_stats",
        help="Path to cached processed dataset (un-tokenized)"
    )
    p.add_argument("--max_length", type=int, default=8000)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--bins", default="0,0.2,0.4,0.6,0.8,1.0", help="Comma-separated bin edges in [0,1]")
    p.add_argument("--thresholds", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9", help="Comma-separated confidence thresholds")
    p.add_argument("--output_json", default=None, help="Optional path to save results JSON")
    p.add_argument("--limit_eval", type=int, default=None, help="Optional limit of test samples for faster run")
    # NEW: GPU selection (mirrors training script behavior)
    p.add_argument("--gpu_ids", type=int, nargs='+', default=None,
                   help="GPU IDs to use (e.g., --gpu_ids 0 1). If not set, all visible GPUs are used.")
    return p.parse_args()

def main():
    args = parse_args()

    # NEW: set CUDA_VISIBLE_DEVICES early (before first CUDA use/model load)
    if args.gpu_ids:
        gpu_ids_str = ",".join(map(str, args.gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        print(f"Set CUDA_VISIBLE_DEVICES to: {gpu_ids_str}")
    else:
        print("No --gpu_ids provided; using all currently visible GPUs.")

    # (Optional) GPU diagnostics
    if torch.cuda.is_available():
        print(f"Detected {torch.cuda.device_count()} visible GPU(s):")
        for i in range(torch.cuda.device_count()):
            try:
                name = torch.cuda.get_device_name(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {name} ({total_mem:.1f} GB)")
            except Exception:
                pass
    else:
        print("CUDA not available; running on CPU.")

    print("=== Confidence-based Stratification Utility ===")
    print(f"Model: {args.model_path}")
    print(f"Data folder: {args.data_folder}")
    print(f"Labels file: {args.labels_file}")
    print(f"Statistics file: {args.statistics_file}")
    print(f"Max length: {args.max_length}")
    print(f"Test size: {args.test_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU IDs argument: {args.gpu_ids if args.gpu_ids else 'All visible'}")

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
        attn_implementation="sdpa"
    )
    # Determine yes/no token ids
    yes_id = tokenizer(" yes", add_special_tokens=False)["input_ids"][0]
    no_id = tokenizer(" no", add_special_tokens=False)["input_ids"][0]
    print(f"YES token id: {yes_id}  NO token id: {no_id}")

    # Load (or build) raw dataset
    if args.processed_dataset_cache and os.path.exists(os.path.join(args.processed_dataset_cache, "dataset_info.json")):
        print("Loading processed raw dataset from cache...")
        raw_dataset = load_from_disk(args.processed_dataset_cache)
        # dataset builder needed for stats only; safe to skip if not needed
    else:
        print("Building dataset from scratch (no usable cache provided)...")
        builder = TextDatasetBuilder(
            args.data_folder,
            args.labels_file,
            statistics_file=args.statistics_file,
            max_length=args.max_length
        )
        raw_dataset = builder.load_dataset_with_ids()
        if args.processed_dataset_cache:
            os.makedirs(args.processed_dataset_cache, exist_ok=True)
            raw_dataset.save_to_disk(args.processed_dataset_cache)
            print(f"Saved processed dataset to {args.processed_dataset_cache}")

    # Stratified split
    split = split_dataset_stratified(raw_dataset, test_size=args.test_size, seed=42)
    eval_dataset = split["test"]
    print(f"Eval samples before limit: {len(eval_dataset)}")
    if args.limit_eval:
        eval_dataset = eval_dataset.select(range(min(args.limit_eval, len(eval_dataset))))
        print(f"Eval samples after limit: {len(eval_dataset)}")

    # Tokenize
    print("Tokenizing evaluation dataset...")
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        batch_size=50,
        remove_columns=["text", "labels"]
    )
    print("Tokenization done.")

    # Collect confidences
    records = collect_confidences(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_dataset,
        yes_id=yes_id,
        no_id=no_id,
        batch_size=args.batch_size
    )
    print(f"Collected confidences for {len(records)} samples.")

    # Metrics
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    bin_edges = [float(x) for x in args.bins.split(",") if x.strip()]
    thresholds = sorted(set(thresholds))
    bin_edges = sorted(bin_edges)

    threshold_rows = precision_recall_for_thresholds(records, thresholds)
    bin_rows = binned_stratification(records, bin_edges)

    # Print
    print_threshold_table(threshold_rows)
    print_bin_table(bin_rows)

    # Aggregate summary
    overall_acc = sum(r["pred_label"] == r["true_label"] for r in records) / len(records) if records else 0.0
    print(f"\nOverall accuracy (all samples): {overall_acc:.4f}")

    # Output JSON
    if args.output_json:
        out_obj = {
            "overall_accuracy": overall_acc,
            "threshold_curve": threshold_rows,
            "binned_metrics": bin_rows,
            "total_samples": len(records),
            "definition": {
                "confidence": "abs(p_yes - p_no) where probabilities are softmax over logits of the two decision tokens",
                "positive_class": "Accept (label=1)"
            }
        }
        with open(args.output_json, "w") as f:
            json.dump(out_obj, f, indent=2)
        print(f"\nSaved results to {args.output_json}")

if __name__ == "__main__":
    main()
