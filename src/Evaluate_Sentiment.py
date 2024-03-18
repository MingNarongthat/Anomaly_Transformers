import pandas as pd
import json
import os

df = pd.read_excel('/opt/project/tmp/experiment_best20240202_sentiment.xlsx')

with open('/opt/project/dataset/focus_caption_dataset_testing_withclass_v3.json') as f:
    # read the json file as dictionary
    data = json.load(f)

# function to calculate confusion metrix from the reference json file 'data' and the predicted xlsx file 'df' with the same Filename
def confusion_metrix(data, df):
    # create a dictionary to store the confusion metrix
    confusion_metrix = {'anomaly': {'anomaly': 0, 'normal': 0}, 'normal': {'anomaly': 0, 'normal': 0}}
    # loop over the data and df
    for i in range(len(data)):
        for j in range(len(df['Filename'])):
            # if the image name in df['Filename'] == data[j]['image'] then extract the caption in df['Caption'] and df['Filename'] in to dictionary only
            if df['Filename'][j] == data[i]['image']:
                # if the class in df['Sentiment'] == data[j]['class'] then increase the value in the confusion metrix
                if df['Sentiment'][j] == 'anomaly' and data[i]['class'] == 'anomaly': # d
                    confusion_metrix['anomaly']['anomaly'] += 1
                elif df['Sentiment'][j] == 'anomaly' and data[i]['class'] == 'normal': # b
                    confusion_metrix['anomaly']['normal'] += 1
                elif df['Sentiment'][j] == 'normal' and data[i]['class'] == 'anomaly': # c
                    confusion_metrix['normal']['anomaly'] += 1
                elif df['Sentiment'][j] == 'normal' and data[i]['class'] == 'normal': # a
                    confusion_metrix['normal']['normal'] += 1
                else:
                    print('Error')
    return confusion_metrix

# calculate accuracy, precision, recall, and f1-score from the confusion metrix
def calculate_accuracy_precision_recall_f1score(confusion_metrix):
    # calculate accuracy
    accuracy = (confusion_metrix['anomaly']['anomaly'] + confusion_metrix['normal']['normal']) / (confusion_metrix['anomaly']['anomaly'] + confusion_metrix['anomaly']['normal'] + confusion_metrix['normal']['anomaly'] + confusion_metrix['normal']['normal'])
    # calculate precision
    precision_anomaly = confusion_metrix['anomaly']['anomaly'] / (confusion_metrix['anomaly']['anomaly'] + confusion_metrix['normal']['anomaly'])
    precision_normal = confusion_metrix['normal']['normal'] / (confusion_metrix['normal']['normal'] + confusion_metrix['anomaly']['normal'])
    # calculate recall
    recall_anomaly = confusion_metrix['anomaly']['anomaly'] / (confusion_metrix['anomaly']['anomaly'] + confusion_metrix['anomaly']['normal'])
    recall_normal = confusion_metrix['normal']['normal'] / (confusion_metrix['normal']['normal'] + confusion_metrix['normal']['anomaly'])
    # calculate f1-score
    f1score_anomaly = 2 * ((precision_anomaly * recall_anomaly) / (precision_anomaly + recall_anomaly))
    f1score_normal = 2 * ((precision_normal * recall_normal) / (precision_normal + recall_normal))
    return accuracy, precision_anomaly, precision_normal, recall_anomaly, recall_normal, f1score_anomaly, f1score_normal

# print accuracy, precision, recall, and f1-score
def print_accuracy_precision_recall_f1score(accuracy, precision_anomaly, precision_normal, recall_anomaly, recall_normal, f1score_anomaly, f1score_normal):
    print('Accuracy : ', accuracy)
    # print('Precision anomaly = ', precision_anomaly)
    print('Precision normal : ', precision_normal)
    # print('Recall anomaly = ', recall_anomaly)
    print('Recall normal : ', recall_normal)
    # print('F1-score anomaly = ', f1score_anomaly)
    print('F1-score normal : ', f1score_normal)

accuracy, precision_anomaly, precision_normal, recall_anomaly, recall_normal, f1score_anomaly, f1score_normal = calculate_accuracy_precision_recall_f1score(confusion_metrix(data, df))
print_accuracy_precision_recall_f1score(accuracy, precision_anomaly, precision_normal, recall_anomaly, recall_normal, f1score_anomaly, f1score_normal)
