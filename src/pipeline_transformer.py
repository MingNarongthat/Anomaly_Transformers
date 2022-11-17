# from transformers import pipeline
#
# classifier = pipeline("sentiment-analysis")
# # classifier = pipeline("bert-base-uncased")
# print(classifier(
#     [
#         "The weather may be cooler than you are able to feel.",
#
#         "The river is moving with its own speed.",
#
#         "Land has slided down through the hillside.",
#
#         "Landslide falling though the river.",
#     ]
# ))
#
# from datasets import load_dataset
# from transformers import AutoTokenizer, DataCollatorWithPadding
#
# raw_datasets = load_dataset("glue", "mrpc")
# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
#
# def tokenize_function(example):
#     return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
#
#
# tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer, logging, BertTokenizer, BertForMaskedLM

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "test-trainer/checkpoint-1000/"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
predictions = trainer.predict(tokenized_datasets["validation"])

