from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
import os

WEIGHT_NAME = '/opt/project/tmp/ResNet50/bestscratch.hdf5'
<<<<<<< HEAD
images_path = "/opt/project/dataset/ResNet50/Testing/normal/"
=======
images_path = "/opt/project/dataset/ResNet50/Testing/landslide/"
>>>>>>> 4fc41bf (updated code version)
IMAGE_RESIZE = 512

model = load_model(WEIGHT_NAME)

class_labels = ['landslide', 'normal']
# create a new dataframe to store the predictions
df_output = pd.DataFrame(columns=['Prediction', 'Label'])

count_landslide = 0
count_normal = 0
# iterate through all the images in the directory
for filename in os.listdir(images_path):
    if filename.endswith(".jpg"): # filter only image files
        img = image.load_img(os.path.join(images_path, filename), target_size=(IMAGE_RESIZE, IMAGE_RESIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        # find the index of the class with maximum score
        pred = np.argmax(preds, axis=-1)
        if pred[0] == 0:
            count_landslide = count_landslide + 1
        else:
            count_normal = count_normal + 1

<<<<<<< HEAD
        df_output = df_output.append({'Prediction': pred[0], 'Label': 1}, ignore_index=True)
=======
        df_output = df_output.append({'Prediction': pred[0], 'Label': 0}, ignore_index=True)
>>>>>>> 4fc41bf (updated code version)
        # print the label of the class with maximum score
        # print("Image {} is classified as {}".format(filename, class_labels[pred[0]]))
        # print(preds)
print("prediction landslide is {}".format(count_landslide))
print("prediction normal is {}".format(count_normal))
<<<<<<< HEAD
# df_output.to_excel('/opt/project/dataset/result_resnet50scratch_landslide.xlsx', index=False)
=======
df_output.to_excel('/opt/project/dataset/result_resnet50scratch_landslide.xlsx', index=False)
>>>>>>> 4fc41bf (updated code version)
