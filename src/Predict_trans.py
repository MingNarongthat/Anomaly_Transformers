from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd

# read the input sentences from the XLSX file
df_input = pd.read_excel('/opt/project/dataset/experiment1_landslide_VEDpretrain.xlsx')

# initialize the tokenizer and loaded model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# tokenizer = DistilBertTokenizerFast.from_pretrained('bert-base-uncased')
loaded_model = TFDistilBertForSequenceClassification.from_pretrained("/opt/project/tmp/sentiment_custom_modelV2")

# create a new dataframe to store the predictions
df_output = pd.DataFrame(columns=['Sentence', 'Sentiment'])

# loop over the sentences in the XLSX file
for sentence in df_input['caption']:
    # encode the sentence using the tokenizer
    predict_input = tokenizer.encode(sentence,
                                     truncation=True,
                                     padding=True,
                                     return_tensors="tf")

    # make a prediction using the loaded model
    tf_output = loaded_model.predict(predict_input)[0]
    tf_prediction = tf.nn.softmax(tf_output, axis=1).numpy()

    # determine the sentiment prediction
    if tf_prediction[0][0] > tf_prediction[0][1]:
        sentiment = 'anomaly'   # 'NEGATIVE --> Anomaly Scene, sentiment = 0'
    else:
        sentiment = 'normal'     # 'POSITIVE --> Normal Scene, sentiment = 1'

    # add the prediction to the output dataframe
    df_output = df_output.append({'Sentence': sentence, 'Sentiment': sentiment}, ignore_index=True)

# write the output dataframe to an XLSX file
df_output.to_excel('/opt/project/dataset/experiment1_landslide_sentiment.xlsx', index=False)
