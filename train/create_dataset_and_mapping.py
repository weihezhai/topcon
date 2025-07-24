import json
from sklearn.model_selection import train_test_split
from dataset_builder_abs_intro import TextDatasetBuilder

# Configuration
DATA_FOLDER = "/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_rl_papers/rl_papers_text/"  # Adjust path as needed
LABELS_FILE = "/mnt/parscratch/users/acr24wz/topcon/train/label_simple.json"  # Adjust path as needed
EVAL_MAPPING_FILE = "/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/rl/rl_index_to_paper_id_mapping.json"

# Initialize the dataset builder
builder = TextDatasetBuilder(data_folder=DATA_FOLDER, labels_file=LABELS_FILE)

# Load dataset with IDs
print("Loading dataset with paper IDs...")
dataset = builder.load_dataset_with_ids()

# Get dataset statistics
stats = builder.get_dataset_stats(dataset)
print(f"\nDataset Statistics:")
print(f"Total samples: {stats['total_samples']}")
print(f"Label distribution: {stats['label_distribution']}")

# Extract data for splitting
texts = dataset['text']
labels = dataset['labels']
paper_ids = dataset['paper_id']

# Create indices for tracking
indices = list(range(len(texts)))

# Split the data without shuffling
train_indices, eval_indices, train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    indices, texts, labels,
    test_size=0.2,
    random_state=42
)

print(f"\nSplit Results:")
print(f"Train set size: {len(train_indices)}")
print(f"Eval set size: {len(eval_indices)}")

# Create mapping for eval partition: index -> paper_id
eval_mapping = {}
for i, original_idx in enumerate(eval_indices):
    eval_mapping[i] = paper_ids[original_idx]

# Save the mapping to a JSON file
with open(EVAL_MAPPING_FILE, 'w', encoding='utf-8') as f:
    json.dump(eval_mapping, f, indent=2)

print(f"\nMapping file saved to: {EVAL_MAPPING_FILE}")
print(f"Mapping contains {len(eval_mapping)} entries")

# Show sample mappings
print("\nSample mappings (eval index -> paper_id):")
for i in range(min(5, len(eval_mapping))):
    print(f"  {i} -> {eval_mapping[i]}")

# Optional: Save the split datasets if needed
# from datasets import Dataset

# train_dataset = Dataset.from_dict({
#     'text': train_texts,
#     'labels': train_labels
# })

# eval_dataset = Dataset.from_dict({
#     'text': eval_texts,
#     'labels': eval_labels
# })

# print("\nDatasets created successfully!")
# print("Train dataset:", train_dataset)
# print("Eval dataset:", eval_dataset)
