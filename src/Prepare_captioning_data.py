import json
import pandas as pd

# read data from excel
filename_input = '/opt/project/dataset/dataset_train_testv1.xlsx'
# filename_output = '/opt/project/dataset/focus_caption_dataset_training_v1.json'
filename_output = '/opt/project/dataset/focus_caption_dataset_testing_wildfire.json'


df = pd.read_excel(filename_input, sheet_name="Wildfire")

all_data = []
# for i in range(30):
for i in range(len(df['Filename'])):
    sentense_data = df['Filename'][i]
    img_name = "{}.jpg".format(sentense_data)
    label_data = df['Caption'][i]
    class_data = df['Label'][i]
    # convert class_data from int to string
    if class_data == 0:
        class_data = '0'
    else:
        class_data = '1'

    context = [{"image": img_name,
               "caption": label_data,
               "class": class_data
                }]

    with open(filename_output, 'w') as f:
        all_data.extend(context)
        json.dump(all_data, f)
