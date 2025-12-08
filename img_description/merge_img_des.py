import json
import glob

output_pattern = "/ceph/hpc/home/euweihez/topcon/d2025d08-005-users/img_description/llm/image_descriptions_*.json"
final_output = "/ceph/hpc/home/euweihez/topcon/d2025d08-005-users/img_description/llm/image_descriptions_MERGED.json"

merged_data = {}
file_list = glob.glob(output_pattern)

for file in file_list:
    if "MERGED" in file: continue # skip the final one if it exists
    print(f"Loading {file}...")
    with open(file, 'r') as f:
        data = json.load(f)
        # Merge dictionary
        merged_data.update(data)

print(f"Total papers processed: {len(merged_data)}")
with open(final_output, 'w') as f:
    json.dump(merged_data, f, indent=2)
print("Saved merged file.")