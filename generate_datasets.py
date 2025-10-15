from datasets import load_dataset

# HuffPost
huff_dataset = load_dataset("khalidalt/HuffPost")
huff_dataset["train"].to_json("data/huff.json", lines=True)


# OTS (CLINC150-like intent classification)
ots_dataset = load_dataset("clinc_oos", "small")
ots_dataset["train"].to_json("data/ots.json", lines=True)


# EBM-NLP
ebmnlp_dataset = load_dataset("ebm_nlp")
ebmnlp_dataset["train"].to_json("data/ebmnlp.json", lines=True)

# NewsQA
newsqa = load_dataset("newsqa")
newsqa["train"].to_json("newsqa_train.json", lines=True)
newsqa["validation"].to_json("newsqa_val.json", lines=True)