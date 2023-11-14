import json
import pandas as pd

# read data from excel
filename_input = '/opt/project/dataset/dataset_train_testv1.xlsx'
# filename_output = '/opt/project/dataset/focus_caption_dataset_training_v1.json'
filename_output = '/opt/project/dataset/focus_caption_dataset_Sheet1_v1.json'


df = pd.read_excel(filename_input, sheet_name="Sheet1")

all_data = []
# for i in range(30):
for i in range(len(df['Filename'])):
    sentense_data = df['Filename'][i]
    img_name = "{}.jpg".format(sentense_data)
    label_data = df['Caption'][i]

    context = [{"image": img_name,
               "caption": label_data
                }]

    with open(filename_output, 'w') as f:
        all_data.extend(context)
        json.dump(all_data, f)
