import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList
)


# Command-line argument setup

parser = argparse.ArgumentParser(description="Controlled data augmentation using LLaMA 7B.")
parser.add_argument('--model', '-m', required=True, type=str, help='Model path or HuggingFace model ID')
parser.add_argument('--config_name', '-c', required=True, type=str, help='Dataset name: huffpost, ots, ebmnlp, newsqa')
parser.add_argument('--num_return_sequences', '-nr', required=True, type=int, help='Number of outputs to generate per input')
args = parser.parse_args()


# Load dataset and limit to 200 samples

def get_dataset(dataset_name: str) -> DatasetDict:
    """
    Reads a JSON dataset file and converts it into HuggingFace Dataset format.
    Only takes up to 200 samples.
    """
    input_path = f'generation_data/{dataset_name}.json'

    with open(input_path, 'r') as f:
        raw_data = json.load(f)

    prompts, labels = [], []

    for item in raw_data[:200]:  # Limit to max 200 samples
        instruction = item.get('instruction', '').strip()
        input_text = item.get('input', '').strip()
        output = item.get('output', '').strip()

        # For NER/QA tasks, input is part of the text
        if dataset_name in ['ebmnlp', 'newsqa']:
            full_input = f"{instruction}\n{input_text}" if input_text else instruction
        else:
            full_input = instruction

        prompts.append(full_input)
        labels.append(output)

    dataset = Dataset.from_dict({
        'text': prompts,
        'label': labels,
        'timestamp': [0] * len(prompts),
        'url': [0] * len(prompts)
    })

    return DatasetDict({'train': dataset})


# Template formatting for LLaMA 7B Chat

@dataclass
class Llama2Template:
    """
    Defines how to format the prompt for the LLaMA 2 model.
    """
    prefix: str = "<<SYS>>\n{{system}}\n<</SYS>>\n\n"
    prompt_format: str = "[INST] {{query}} [/INST]"
    system: str = (
        "You are an assistant that only returns what is requested for and nothing else. "
        "Generate only what is required by the prompt."
    )

    def encode(self, tokenizer, query: str, response: str) -> Tuple[List[int], List[int]]:
        """
        Converts a text prompt and response into token IDs.
        """
        formatted_prompt = self.prefix.replace("{{system}}", self.system) + \
                           self.prompt_format.replace("{{query}}", query)

        input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(response, add_special_tokens=False)
        return input_ids, output_ids + [tokenizer.eos_token_id]


# Load tokenizer and model

model_name = args.model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")


# Load dataset and apply template formatting

raw_dataset = get_dataset(args.config_name)
template = Llama2Template()

# Function to tokenize dataset using the template
def preprocess_dataset(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
    input_ids_list, attention_masks = [], []

    for text in examples['text']:
        input_ids, _ = template.encode(tokenizer, text, "")
        input_ids_list.append(input_ids)
        attention_masks.append([1] * len(input_ids))

    return {'input_ids': input_ids_list, 'attention_mask': attention_masks}

# Apply tokenization to dataset
dataset = raw_dataset.map(
    preprocess_dataset,
    batched=True,
    remove_columns=['text', 'label', 'timestamp', 'url']
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    label_pad_token_id=tokenizer.pad_token_id
)

dataloader = DataLoader(
    dataset['train'],
    batch_size=2,
    collate_fn=data_collator
)


# Generation configuration
gen_config = GenerationConfig(
    do_sample=True,
    temperature=0.5,
    top_k=50,
    num_return_sequences=args.num_return_sequences,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    max_length=512
)

def get_logits_processor() -> LogitsProcessorList:
    """
    Ensures numerical stability during generation.
    """
    return LogitsProcessorList([InfNanRemoveLogitsProcessor()])


# Prepare output directory

output_dir = f"generation_data/{args.config_name}/"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "generated_predictions.jsonl")

# Clear existing file
open(output_file, "w").close()

# Repeat labels for all generations
label_list = raw_dataset["train"]["label"] * args.num_return_sequences

# Generate and save augmented data

count = 0

for batch in tqdm(dataloader, desc="Generating with LLaMA 7B"):
    outputs = model.generate(
        input_ids=batch['input_ids'].cuda(),
        attention_mask=batch['attention_mask'].cuda(),
        generation_config=gen_config,
        logits_processor=get_logits_processor()
    )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    with open(output_file, "a", encoding="utf-8") as writer:
        for response in responses:
            try:
                prediction = response.split("[/INST]")[1].strip()
            except IndexError:
                prediction = response.strip()
            writer.write(json.dumps({
                "label": label_list[count],
                "predict": prediction
            }) + "\n")
            count += 1


