import json
import glob

output_pattern = "/ceph/hpc/home/euweihez/topcon/d2025d08-005-users/img_description/image_descriptions_*.json"
final_output = "/ceph/hpc/home/euweihez/topcon/d2025d08-005-users/img_description/image_descriptions_all.json"

merged_data = {}
file_list = glob.glob(output_pattern)

for file in file_list:
    print(f"Loading {file}...")
    with open(file, 'r') as f:
        data = json.load(f)
        # Merge dictionary, keeping existing keys
        for key, value in data.items():
            if key not in merged_data:
                merged_data[key] = value

print(f"Total papers processed: {len(merged_data)}")
with open(final_output, 'w') as f:
    json.dump(merged_data, f, indent=2)
print("Saved merged file.")