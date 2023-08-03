from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import json

training_size = 200

with open("/opt/project/dataset/sentense_dataset_allV2.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['Sentence'])
    labels.append(item['Label'])

sentences = np.array(sentences)
labels = np.array(labels)
indices = np.arange(sentences.shape[0])
random.shuffle(indices)

sentences = np.array(sentences[indices]).tolist()
labels = np.array(labels[indices]).tolist()

training_sentences = sentences[0:training_size]
validation_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
validation_labels = labels[training_size:]

tokenizer = DistilBertTokenizerFast.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(training_sentences,
                            truncation=True,
                            padding=True,
                            return_tensors="tf")
val_encodings = tokenizer(validation_sentences,
                          truncation=True,
                          padding=True,
                          return_tensors="tf")

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    training_labels
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    validation_labels
))

model = TFDistilBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=10e-5)
model.compile(optimizer=optimizer,
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_dataset.shuffle(30).batch(16),
          epochs=100,
          batch_size=16,
          validation_data=val_dataset.shuffle(10).batch(16))
model.save_pretrained("/opt/project/tmp/sentiment_custom_modelV2")