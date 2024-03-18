from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import json

training_size = 480

with open("/opt/project/dataset/focus_caption_dataset_trainingfinetune_withclass_v3.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['caption'])
    labels.append(item['class'])
    # change the label from string to integer (0 = anomaly, 1 = normal)
    # if item['class'] == 'anomaly':
    #     labels.append(0)
    # elif item['class'] == 'normal':
    #     labels.append(1)
    # else:
    #     print('Error')

# convert labels from string to integer (0 = anomaly, 1 = normal)
for i in range(len(labels)):
    if labels[i] == '0':
        labels[i] = 0
    elif labels[i] == '1':
        labels[i] = 1
    else:
        print('Error')

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

model = TFDistilBertForSequenceClassification.from_pretrained('/opt/project/tmp/sentiment_custom_modelV5', num_labels=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=10e-5)
model.compile(optimizer=optimizer,
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_dataset.shuffle(30).batch(16),
          epochs=100,
          batch_size=16,
          validation_data=val_dataset.shuffle(10).batch(16))
model.save_pretrained("/opt/project/tmp/sentiment_custom_modelV6")