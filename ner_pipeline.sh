#!/bin/bash

# For EBMNLP NER task using LLaMA 7B
DATASET_NAME="ebmnlp"
NUM_RETURN=5
MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"

echo "Generating NER augmentations for $DATASET_NAME..."

python generate_llama7b.py \
  --model $MODEL_PATH \
  --config_name $DATASET_NAME \
  --num_return_sequences $NUM_RETURN

echo "Done generating EBMNLP augmented NER data."
