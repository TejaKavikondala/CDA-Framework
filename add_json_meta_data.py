import json
import argparse

# Define command-line arguments for dataset name and split (e.g., train/val/test)
parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
parser.add_argument('--dataset', '-d', type=str, help='Name of the dataset (e.g., HuffPost, OTS)')
parser.add_argument('--split', '-s', type=str, help='Data split identifier (e.g., train, val, test)')
args = parser.parse_args()

# Open the existing dataset_info.json which keeps track of dataset-related file references
with open("./generation_data/dataset_info.json", "r") as json_file:
    data = json.load(json_file)

# Loop through different output suffixes that correspond to various generated prompt files
for suffix in ["_concept_constraint", "_final_constraint_abs", "_solo_constraint", "_abs_prompt_1", "_abs_prompt_2"]:
    file_name = f"{args.dataset}_{args.split}{suffix}"
    
    # Create or update the entry in the dataset info dictionary
    data[file_name] = {
        "file_name": f"{file_name}.json"  # Store the actual filename (with .json extension)
    }

# Save the updated info back to the same JSON file
with open("./generation_data/dataset_info.json", "w") as f:
    json.dump(data, f, indent=2)
