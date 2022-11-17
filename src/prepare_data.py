import json
import pandas as pd

# read data from excel
filename_input = '/opt/project/dataset/sentense_data.xlsx'
filename_output = '/opt/project/dataset/sentense_data_LS.json'

df = pd.read_excel(filename_input)
# df_result = pd.DataFrame(columns=['Sentence', 'Label'])
# print(df['Label'])

all_data = []
# for i in range(30):
for i in range(len(df['Sentence'])):
    sentense_data = df['Sentence'][i]
    label_data = int(df['Label'][i])

    context = [{"Sentence": sentense_data,
               "Label": label_data
                }]

    with open(filename_output, 'w') as f:
        all_data.extend(context)
        json.dump(all_data, f)
