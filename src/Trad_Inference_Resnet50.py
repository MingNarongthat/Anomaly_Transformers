from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os

WEIGHT_NAME = '/opt/project/tmp/ResNet50/best.hdf5'
images_path = "/opt/project/dataset/ResNet50/Testing/landslide/"
IMAGE_RESIZE = 512

model = load_model(WEIGHT_NAME)

class_labels = ['landslide', 'normal']

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
        # print the label of the class with maximum score
        print("Image {} is classified as {}".format(filename, class_labels[pred[0]]))
        # print(preds)

