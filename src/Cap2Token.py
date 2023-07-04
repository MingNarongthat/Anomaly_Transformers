# import os
# import pandas as pd
from transformers import AutoTokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

while True:
    sentence = input("type the sentence: ")
    if sentence == "end":
        break
    tokens = tokenizer(sentence, truncation=True, padding="max_length", max_length=15).input_ids
    print(tokens)
