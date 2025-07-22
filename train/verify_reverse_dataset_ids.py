# import os
# import sys
# from datasets import load_from_disk

# # Adjust import if your builder is in a different module
# from dataset_builder_abs_intro import TextDatasetBuilder

# def validate_recovered(
#     data_folder: str,
#     labels_file: str,
#     original_cache: str,
#     recovered_cache: str
# ):
#     """
#     Validates that the recovered dataset matches the one built-with-ids.

#     Steps:
#     1. Build ds_with_ids via TextDatasetBuilder.load_dataset_with_ids()
#     2. Load the original dataset (without IDs)
#     3. Load the recovered dataset (with IDs)
#     4. Check sample counts
#     5. For each record in recovered:
#        - Find matching record in ds_with_ids by paper_id
#        - Assert text and label match exactly
#     6. Report success or any mismatches
#     """
#     # 1. Fresh build with IDs
#     builder = TextDatasetBuilder(
#         data_folder=data_folder,
#         labels_file=labels_file,
#         max_length=10000
#     )
#     ds_with_ids = builder.load_dataset_with_ids()

#     # 2. Original cache (no IDs)
#     ds_original = load_from_disk(original_cache)

#     # 3. Recovered cache (has IDs)
#     ds_recovered = load_from_disk(recovered_cache)

#     # 4. Check lengths
#     n_orig = len(ds_original)
#     n_with_ids = len(ds_with_ids)
#     n_recovered = len(ds_recovered)
#     print(f"Original (no IDs): {n_orig} samples")
#     print(f"Built with IDs: {n_with_ids} samples")
#     print(f"Recovered: {n_recovered} samples")

#     assert n_orig == n_recovered, (
#         f"Mismatch in sample counts: original {n_orig} vs recovered {n_recovered}"
#     )
#     assert n_with_ids == n_recovered, (
#         f"Mismatch in sample counts: with_ids {n_with_ids} vs recovered {n_recovered}"
#     )

#     # Build index for ds_with_ids by paper_id
#     id_to_index = {pid: idx for idx, pid in enumerate(ds_with_ids['paper_id'])}

#     # 5. Validate each recovered record
#     errors = []
#     for i in range(n_recovered):
#         rec_pid = ds_recovered['paper_id'][i]
#         rec_text = ds_recovered['text'][i]
#         rec_label = ds_recovered['labels'][i]

#         if rec_pid not in id_to_index:
#             errors.append(f"Record {i}: paper_id {rec_pid!r} not found in fresh build.")
#             continue

#         j = id_to_index[rec_pid]
#         orig_text = ds_with_ids['text'][j]
#         orig_label = ds_with_ids['labels'][j]

#         if rec_label != orig_label:
#             errors.append(
#                 f"Label mismatch for PID {rec_pid}: recovered {rec_label} vs built {orig_label}"
#             )
#         if rec_text != orig_text:
#             # Show snippet difference
#             snip_rec = rec_text[:50].replace("\n"," ")
#             snip_orig = orig_text[:50].replace("\n"," ")
#             errors.append(
#                 f"Text mismatch for PID {rec_pid}\n  recovered start: {snip_rec!r}\n  built start:    {snip_orig!r}"
#             )

#     # 6. Report
#     if errors:
#         print("\n❌ Validation failed with the following errors:\n")
#         for err in errors[:10]:
#             print(err)
#         if len(errors) > 10:
#             print(f"...and {len(errors)-10} more errors.")
#         sys.exit(1)
#     else:
#         print("✅ All records validated: texts, labels, and paper_ids match exactly.")


# def main():
#     data_folder = "/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_llm_papers/llm_papers_text/"
#     labels_file = "/mnt/parscratch/users/acr24wz/topcon/train/label_simple.json"
#     original_cache = "/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/llm"
#     recovered_cache = '/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/llm/with_ids'


#     validate_recovered(data_folder, labels_file, original_cache, recovered_cache)

# if __name__ == '__main__':
#     main()
import os
import sys
from datasets import load_from_disk
import pandas as pd

# If TextDatasetBuilder is in another module, adjust import
from dataset_builder_abs_intro import TextDatasetBuilder

def validate_recovered_fast(
    data_folder: str,
    labels_file: str,
    original_cache: str,
    recovered_cache: str
):
    # 1. Fresh build with IDs
    builder = TextDatasetBuilder(
        data_folder=data_folder,
        labels_file=labels_file,
        max_length=10000
    )
    ds_with_ids = builder.load_dataset_with_ids()

    # 2. Load original and recovered datasets
    ds_original = load_from_disk(original_cache)
    ds_recovered = load_from_disk(recovered_cache)

    # 3. Quick length check
    n = len(ds_recovered)
    if len(ds_original) != n or len(ds_with_ids) != n:
        raise ValueError(
            f"Sample count mismatch: original={len(ds_original)}, with_ids={len(ds_with_ids)}, recovered={n}"
        )
    print(f"Validated sample count: {n} entries")

    # 4. Build DataFrames for vectorized comparison
    df_with = pd.DataFrame(
        {
            'paper_id': ds_with_ids['paper_id'],
            'text': ds_with_ids['text'],
            'label': ds_with_ids['labels']
        }
    ).set_index('paper_id')

    df_rec = pd.DataFrame(
        {
            'paper_id': ds_recovered['paper_id'],
            'text': ds_recovered['text'],
            'label': ds_recovered['labels']
        }
    ).set_index('paper_id')

    # 5. Align indices and compare
    # Ensure same set of paper_ids
    if set(df_with.index) != set(df_rec.index):
        missing_in_rec = set(df_with.index) - set(df_rec.index)
        missing_in_with = set(df_rec.index) - set(df_with.index)
        raise ValueError(
            f"Different paper_id sets. \nMissing in recovered: {missing_in_rec}\nMissing in rebuilt: {missing_in_with}"
        )

    # Reindex both to sorted order
    df_with = df_with.sort_index()
    df_rec = df_rec.sort_index()

    # Compare labels
    label_mismatch = (df_with['label'] != df_rec['label'])
    if label_mismatch.any():
        mismatches = df_with[label_mismatch].join(
            df_rec[label_mismatch], lsuffix='_with', rsuffix='_rec'
        )
        print("Label mismatches for paper_ids:")
        print(mismatches[['label_with','label_rec']])
        sys.exit(1)

    # Compare texts
    text_mismatch = (df_with['text'] != df_rec['text'])
    if text_mismatch.any():
        # Print up to 5 mismatches with snippets
        print("Text mismatches (up to 5 shown):")
        for pid in df_with.index[text_mismatch][:5]:
            orig_snip = df_with.at[pid, 'text'][:100].replace('\n',' ')
            rec_snip = df_rec.at[pid, 'text'][:100].replace('\n',' ')
            print(f"PID {pid}:\n  rebuilt: {orig_snip!r}\n  recovd : {rec_snip!r}\n")
        sys.exit(1)

    print("✅ All records match exactly.")


def main():
    data_folder = "/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_llm_papers/llm_papers_text/"
    labels_file = "/mnt/parscratch/users/acr24wz/topcon/train/label_simple.json"
    original_cache = "/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/llm"
    recovered_cache = "/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/llm/with_ids"

    validate_recovered_fast(data_folder, labels_file, original_cache, recovered_cache)

if __name__ == '__main__':
    main()
