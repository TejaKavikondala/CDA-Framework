import pandas as pd
import json
import re
import argparse

# Set up argument parsing to get dataset name and data split (train/val/test)
parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
parser.add_argument('--dataset', '-d', type=str, help='Path to the input file')
parser.add_argument('--split', '-s', type=str, help='Path to the output file')
args = parser.parse_args()

# Load the constraint TSV file, which contains template-based instruction columns and labels
df = pd.read_csv(f"./tsv_data/out_data/{args.dataset}/{args.dataset}_{args.split}_final_constraint.tsv", sep="\t", header=0)

# Optional prompt containers (not used later in the script but declared here)
mix_prompts_1 = []
mix_prompts_2 = []

# json_data[0] → instructions using "mix_const_1"
# json_data[1] → instructions using "mix_const_2"
json_data = [[], []]

# Open the LLM-generated predictions from JSONL file
with open(f"./generation_data/{args.dataset}_{args.split}_final_constraint_abs/generated_predictions.jsonl", "r") as file:
    i = 0  # index to match rows in TSV and JSONL

    for line in file:
        obj = json.loads(line)

        # Only process lines with non-empty predictions
        if len(obj["predict"]) > 0:
            # Try to isolate the abstract part after a colon
            predict_split = obj["predict"].split(":")

            if len(predict_split) > 1:
                # Further split into blocks using double newlines
                predict_split = predict_split[1].split("\n\n")

                # Use the first non-empty block after stripping whitespace
                for text in predict_split:
                    if len(text.strip()) > 0:
                        predict_split = text
                        break
            else:
                predict_split = ""

            # Store the abstract as string (even if it's already text, ensures compatibility)
            abstract = str(predict_split)

            # Only proceed if abstract is not empty
            if len(abstract) > 0:
                for k in range(1, 3):
                    # Build each prompt version by injecting the abstract into the respective mix_const_k template
                    json_entry = {
                        "instruction": df.at[i, "mix_const_" + str(k)].replace("$abs$", abstract),
                        "input": df.at[i, "label"],   # Using the original label as the input (useful for conditioning)
                        "output": df.at[i, "label"]   # Same label also used as the output (for supervised learning)
                    }
                    json_data[k - 1].append(json_entry)
        i = i + 1  # Move to next TSV row

# Save each prompt version into its own JSON file
for k in range(1, 3):
    with open(f"./generation_data/{args.dataset}_{args.split}_abs_prompt_{k}.json", "w") as json_file:
        json.dump(json_data[k - 1], json_file, indent=2)
