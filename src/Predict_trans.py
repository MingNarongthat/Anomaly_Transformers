from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import json

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
loaded_model = TFDistilBertForSequenceClassification.from_pretrained("/opt/project/tmp/sentiment_custom_model")
test_sentence = "The river is narrow in both directions."
# test_sentence = "Landslide damaged to village road that has now been taken to the local police station."
# test_sentence = 'The soil is sliding down with a little bit of water and it kind of starts to slide'

# replace to test_sentence_sarcasm variable, if you want to test sarcasm
predict_input = tokenizer.encode(test_sentence,
                                 truncation=True,
                                 padding=True,
                                 return_tensors="tf")

tf_output = loaded_model.predict(predict_input)[0]
tf_prediction = tf.nn.softmax(tf_output, axis=1).numpy()

if tf_prediction[0][0] > tf_prediction[0][1]:
    print("NEGATIVE, sentiment = 0")
else:
    print("POSITIVE, sentiment = 1")
