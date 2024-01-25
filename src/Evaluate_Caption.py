from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import pandas as pd
import json

def compute_scores(ref_caption, predicted_caption):
    # Scoring
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    for scorer, method in scorers:
        # print(f'Computing {method} score...')
        score, scores = scorer.compute_score(ref_caption, predicted_caption)
        if type(method) == list:
            for m, s in zip(method, score):
                print(f"{m}: {s}")
        else:
            print(f"{method}: {score}")

# read the excel file result_best20231214_caption.xlsx
# df = pd.read_excel('/opt/project/tmp/result_best20231214_caption.xlsx')
df = pd.read_excel('/Users/mingnarongthat/Documents/Ph.D./Transformer Model/tmp/result_best20231224_caption.xlsx')

# read json file focus_caption_dataset_testing_withclass_v3.json
# with open('/opt/project/dataset/focus_caption_dataset_testing_withclass_v3.json') as f:
with open('/Users/mingnarongthat/Documents/Ph.D./Transformer Model/dataset/focus_caption_dataset_testing_withclass_v3.json') as f:
    # read the json file as dictionary
    data = json.load(f)
    
# if the data in df['Filename'] == data['image'] then extract the caption in df['Caption'] and df['Filename'] in to dictionary only
ref_caption = {}
predicted_caption = {}
for i in range(len(df['Filename'])):
    for j in range(len(data)):
        if df['Filename'][i] == data[j]['image']:
            # if the image name in df['Filename'] == data[j]['image'] then extract the caption in df['Caption'] and df['Filename'] in to dictionary only
            ref_caption[data[j]['image']] = [df['Caption'][i]]
            predicted_caption[data[j]['image']] = [data[j]['caption']]
    
compute_scores(ref_caption, predicted_caption)

        

            
            


