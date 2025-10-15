#!/bin/bash

# === Script to generate augmentations for the HuffPost classification dataset ===

# Name of the dataset to augment (used in config lookup inside the Python script)
DATASET_NAME="huffpost"

# Number of augmentation outputs to generate per input instance
NUM_RETURN=5

# Path to the HuggingFace LLaMA 7B model
MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"

# Status message
echo "Generating augmentations for $DATASET_NAME..."

# Call the generation script with appropriate flags
python generate_llama7b.py \
  --model $MODEL_PATH \
  --config_name $DATASET_NAME \
  --num_return_sequences $NUM_RETURN

# Final confirmation message
echo "Done generating HuffPost augmented data."
