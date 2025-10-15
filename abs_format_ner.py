import pandas as pd
import json
import re
import argparse

# Set up command-line arguments for input dataset and split
parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
parser.add_argument('--dataset', '-d', type=str, help='Path to the input file')
parser.add_argument('--split', '-s', type=str, help='Split name, e.g., train/val/test')
args = parser.parse_args()

# Load the constraint TSV file into a DataFrame using the dataset and split provided
df = pd.read_csv(f"./tsv_data/out_data/{args.dataset}/{args.dataset}_{args.split}_final_constraint.tsv", sep="\t", header=0)

# Prepare separate lists to store instructions later (not used but defined here)
mix_prompts_1 = []
mix_prompts_2 = []

# json_data[0] and json_data[1] will hold output for two separate prompt sets
json_data = [[], []]

# Open the generated predictions file line-by-line (.jsonl format = one JSON object per line)
with open(f"./generation_data/{args.dataset}_{args.split}_final_constraint_abs/generated_predictions.jsonl", "r") as file:
    i = 0
    for line in file:
        obj = json.loads(line)

        # If the predicted field is non-empty
        if len(obj["predict"]) > 0:
            # Try splitting at colon — usually format is: "Abstract: text..."
            predict_split = obj["predict"].split(":")

            if len(predict_split) > 1:
                # If colon exists, take what's after and split by paragraph break
                predict_split = predict_split[1].split("\n\n")
                # Grab the first non-empty block after stripping whitespace
                for text in predict_split:
                    if len(text.strip()) > 0:
                        predict_split = text
                        break
            else:
                predict_split = ""

            # Convert final abstract text into string
            abstract = str(predict_split)

            # If there's a valid abstract
            if len(abstract) > 0:
                # Create two prompt versions using mix_const_1 and mix_const_2 templates
                for k in range(1, 3):
                    json_entry = {
                        # Replace $abs$ placeholder in the instruction with the actual abstract
                        "instruction": df.at[i, "mix_const_" + str(k)].replace("$abs$", abstract),
                        "input": "",  # No input used — left empty
                        "output": i   # Just keeping track of which row this came from
                    }
                    json_data[k-1].append(json_entry)
        i = i + 1

# Save both prompt versions into separate JSON files
for k in range(1, 3):
    with open(f"./generation_data/{args.dataset}_{args.split}_abs_prompt_{k}.json", "w") as json_file:
        json.dump(json_data[k-1], json_file, indent=2)
