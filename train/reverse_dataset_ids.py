import os
import json
from datasets import load_from_disk

# Assuming TextDatasetBuilder is defined in text_dataset_builder.py
# If it's defined in your script, adjust the import accordingly.
from dataset_builder_abs_intro import TextDatasetBuilder

def recover_ids(
    data_folder: str,
    labels_file: str,
    cache_dir: str,
    output_dir: str
):
    """
    Rebuilds the dataset with paper IDs and aligns them to an existing cached dataset.

    Args:
        data_folder: Path to the folder containing .txt files.
        labels_file: Path to the JSON labels file.
        cache_dir: Directory where the original dataset (without IDs) was saved via `.save_to_disk()`.
        output_dir: Directory where the new dataset (with `paper_id`) will be saved.
    """
    # 1. Build a fresh dataset with IDs
    builder = TextDatasetBuilder(
        data_folder=data_folder,
        labels_file=labels_file,
        max_length=10000
    )
    ds_with_ids = builder.load_dataset_with_ids()

    # 2. Load the original cached dataset
    ds = load_from_disk(cache_dir)

    # 3. Build mapping from (text, label) to paper_id
    mapping = {
        (text, label): pid
        for text, label, pid in zip(
            ds_with_ids['text'],
            ds_with_ids['labels'],
            ds_with_ids['paper_id']
        )
    }

    # 4. Recover paper_ids for each entry in the cached dataset
    recovered_pids = []
    for text, label in zip(ds['text'], ds['labels']):
        key = (text, label)
        if key not in mapping:
            raise ValueError(f"No matching paper_id for entry with label={label} and text starting with: {text[:30]!r}")
        recovered_pids.append(mapping[key])

    # 5. Add the paper_id column and save
    ds = ds.add_column('paper_id', recovered_pids)
    ds.save_to_disk(output_dir)

    print(f"✅ Recovered and saved dataset with paper_id to '{output_dir}'")


def main():
    # TODO: adjust these paths before running
    data_folder = "/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/all_four_domain/four_domain_text"
    labels_file = "/mnt/parscratch/users/acr24wz/topcon/train/label_simple.json"
    cache_dir = "/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/all_four_domain"
    output_dir = '/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/all_four_domain/with_ids'

    recover_ids(data_folder, labels_file, cache_dir, output_dir)


if __name__ == '__main__':
    main()
