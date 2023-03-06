# import json
#
# # define filename variable
# filename_input = "/opt/project/dataset/ls_public_test_english_v1.1.json"
# filename_output = "/opt/project/dataset/validation_captioning.json"
#
# # Open the JSON file
# with open(filename_input, "r") as f:
#
#   # Load the data from the file
#   data = json.load(f)
#
# json_data = []
# # Access the data in the JSON object
# for i in range(len(data)):
#     data_name = data[i]["videoID"]
#     img_name = "frame{}.jpg".format(data_name)
#     cap_text = data[i]["enCap"][0]
#     context = [
#         {
#             "image": img_name,
#             "caption": cap_text
#         }
#     ]
#
#     with open(filename_output, "w") as f:
#         json_data.extend(context)
#         json.dump(json_data, f)

import json
import pandas as pd

# read data from excel
filename_input = '/opt/project/dataset/caption_dataset_normal.xlsx'
filename_output = '/opt/project/dataset/caption_dataset_normal.json'

df = pd.read_excel(filename_input)

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
