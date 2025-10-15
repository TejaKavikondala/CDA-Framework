

# pip install scikit-learn transformers torch


from sklearn.metrics import accuracy_score, f1_score
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np


# Classification Metrics
def evaluate_classification(true_labels, pred_labels):
   accuracy = accuracy_score(true_labels, pred_labels)
   f1 = f1_score(true_labels, pred_labels, average='weighted')
   return accuracy, f1


#  Diversity
def calculate_diversity(generated_texts):
   all_tokens = []
   for text in generated_texts:
       tokens = text.split()
       all_tokens.extend(tokens)
   unique_tokens = set(all_tokens)
   return len(unique_tokens) / len(all_tokens) if all_tokens else 0


# Perplexity
def compute_perplexity(sentences, model_name="meta-llama/Llama-2-7b-hf"):
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   model.eval()


   total_loss = 0
   for sent in sentences:
       inputs = tokenizer(sent, return_tensors="pt")
       with torch.no_grad():
           outputs = model(**inputs, labels=inputs["input_ids"])
       loss = outputs.loss.item()
       total_loss += loss
   return np.exp(total_loss / len(sentences)) if sentences else float('inf')


if __name__ == "__main__":
   # PLACEHOLDER: Load your data here for 100/200 samples for each task
   # Example classification labels
   true_labels = [0, 1, 2, 0, 1]
   pred_labels = [0, 1, 1, 0, 1]


   # Example generated texts (can be from HuffPost, OTS, EBMNLP, NewsQA, etc.)
   generated_texts = [
       "The government has announced new policies today.",
       "A major sports event took place in the city.",
       "Scientists discovered a new method for vaccine development."
   ]


   print("Evaluating Classification:")
   acc, f1 = evaluate_classification(true_labels, pred_labels)
   print(f"Accuracy: {acc:.4f}")
   print(f"F1 Score: {f1:.4f}")


   print("\nEvaluating Diversity:")
   diversity = calculate_diversity(generated_texts)
   print(f"Diversity Score: {diversity:.4f}")


   print("\nEvaluating Perplexity:")
   perplexity = compute_perplexity(generated_texts)
   print(f"Perplexity: {perplexity:.4f}")



