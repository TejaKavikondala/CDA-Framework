#!/bin/bash


# Script: qa_pipeline_newsqa.sh
# Purpose: Generate controlled augmented question-answering (QA) data
#          for the NewsQA dataset using LLaMA 7B model.


# Configuration section

# Name of the Hugging Face model or path to a local LLaMA 7B model
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"

# Name of the dataset JSON file without the `.json` extension
DATASET_NAME="newsqa"

# Number of augmentations to generate per example
NUM_RETURN=5


# Execution section


echo "Starting QA pipeline for dataset: $DATASET_NAME"
echo "Using model: $MODEL_NAME"
echo "Generating $NUM_RETURN augmentations per sample..."
echo "----------------------------------------"

# Run the Python data generation script with arguments
python generate_data_pipe.py \
  --model "$MODEL_NAME" \
  --config_name "$DATASET_NAME" \
  --num_return_sequences "$NUM_RETURN"


# Completion message

echo "----------------------------------------"
echo "Finished generating augmented QA data for $DATASET_NAME"
echo "Output saved to: generation_data/${DATASET_NAME}/generated_predictions.jsonl"
