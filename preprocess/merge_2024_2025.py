import argparse
import concurrent.futures as futures
import os
import random
import shutil
import string
import sys
from typing import Dict, List, Tuple

def _rand_suffix(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

def safe_copytree(src: str, dst: str, *, overwrite: bool, dry_run: bool, verbose: bool) -> str:
    if os.path.exists(dst):
        if overwrite:
            if verbose:
                print(f"[overwrite] removing {dst}", flush=True)
            if not dry_run:
                shutil.rmtree(dst, ignore_errors=False)
        else:
            return "exists"
    if dry_run:
        return "dryrun"
    tmp_dst = f"{dst}.tmp-{os.getpid()}-{_rand_suffix()}"
    try:
        shutil.copytree(src, tmp_dst, symlinks=True)
        os.replace(tmp_dst, dst)
        return "copied"
    except Exception:
        try:
            if os.path.exists(tmp_dst):
                shutil.rmtree(tmp_dst, ignore_errors=True)
        finally:
            raise

def _list_dir_ids(root: str) -> List[str]:
    try:
        return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    except FileNotFoundError:
        return []

def process_one(paper_id: str, src_root: str, dest_root: str, exists_policy: str, dry_run: bool, verbose: bool) -> Tuple[str, str]:
    src = os.path.join(src_root, paper_id)
    dst = os.path.join(dest_root, paper_id)
    if not os.path.isdir(src):
        return (paper_id, "missing-skip")
    if exists_policy == "fail" and os.path.exists(dst):
        return (paper_id, "exists-fail")
    overwrite = exists_policy == "overwrite"
    try:
        result = safe_copytree(src, dst, overwrite=overwrite, dry_run=dry_run, verbose=verbose)
        if verbose:
            print(f"[{result}] {paper_id}  ({src_root} -> {dest_root})", flush=True)
        return (paper_id, result)
    except Exception as e:
        if verbose:
            print(f"[error] {paper_id}: {e}", file=sys.stderr, flush=True)
        return (paper_id, "error")

def _copy_phase(label: str, src_root: str, dest_root: str, exists_policy: str, workers: int, dry_run: bool, verbose: bool) -> Dict[str, str]:
    ids = _list_dir_ids(src_root)
    if verbose:
        print(f"[{label}] planning {len(ids)} IDs from {src_root}")
    if not ids:
        return {}
    max_workers = max(1, min(workers, len(ids)))
    results: Dict[str, str] = {}
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(process_one, pid, src_root, dest_root, exists_policy, dry_run, verbose)
            for pid in ids
        ]
        for fut in futures.as_completed(futs):
            pid, status = fut.result()
            results[pid] = status
    return results

def _print_summary(label: str, results: Dict[str, str]) -> None:
    summary = {
        "copied": 0, "exists": 0, "dryrun": 0,
        "missing-skip": 0, "exists-fail": 0, "error": 0
    }
    for status in results.values():
        summary[status] = summary.get(status, 0) + 1
    total = len(results)
    print(f"[{label}] Total IDs: {total}")
    print(f"[{label}] Copied: {summary.get('copied',0)}")
    print(f"[{label}] Dry-run marked: {summary.get('dryrun',0)}")
    print(f"[{label}] Skipped (exists): {summary.get('exists',0)}")
    print(f"[{label}] Skipped (missing): {summary.get('missing-skip',0)}")
    print(f"[{label}] Failed (exists): {summary.get('exists-fail',0)}")
    print(f"[{label}] Errors: {summary.get('error',0)}")

def merge_two_folders(src1: str, src2: str, dest: str, exists_policy: str, workers: int, dry_run: bool, verbose: bool) -> Dict[str, Dict[str, str]]:
    if not os.path.isdir(src1):
        print(f"Source 1 does not exist or is not a directory: {src1}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(src2):
        print(f"Source 2 does not exist or is not a directory: {src2}", file=sys.stderr)
        sys.exit(2)
    os.makedirs(dest, exist_ok=True)
    phase1 = _copy_phase("src1", src1, dest, exists_policy, workers, dry_run, verbose)
    # Apply the same exists-policy for src2 (skip | overwrite | fail)
    phase2 = _copy_phase("src2", src2, dest, exists_policy, workers, dry_run, verbose)
    return {"src1": phase1, "src2": phase2}

def main():
    parser = argparse.ArgumentParser(description="Merge two folders of subfolders (IDs) into one destination.")
    parser.add_argument("--src1", required=True, help="First source root directory.")
    parser.add_argument("--src2", required=True, help="Second source root directory.")
    parser.add_argument("--dest", required=True, help="Destination root directory.")
    parser.add_argument("--exists-policy", choices=["skip", "overwrite", "fail"], default="skip", help="Conflict behavior when destination subfolder exists.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for copying.")
    parser.add_argument("--dry-run", action="store_true", help="Plan actions without copying.")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-ID logs.")
    args = parser.parse_args()

    src1 = os.path.abspath(args.src1)
    src2 = os.path.abspath(args.src2)
    dest = os.path.abspath(args.dest)

    results = merge_two_folders(src1, src2, dest, args.exists_policy, args.workers, args.dry_run, args.verbose)

    _print_summary("src1", results["src1"])
    _print_summary("src2", results["src2"])

    # Combined summary
    combined: Dict[str, str] = {}
    combined.update(results["src1"])
    combined.update(results["src2"])
    _print_summary("combined", combined)

if __name__ == "__main__":
    main()
