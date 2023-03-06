# import numpy as np
# import pandas as pd

# import cv2
# import os
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# Fixed for our normal and abnormal classes
NUM_CLASSES = 2

# Fixed for normal and abnormal color images
CHANNELS = 3

WEIGHT_NAME = '/opt/project/tmp/ResNet50/best.hdf5'
IMAGE_RESIZE = 512  # 224
RESNET50_POOLING_AVERAGE = 'max'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 50

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 20
STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 32
BATCH_SIZE_VALIDATION = 32

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
# BATCH_SIZE_TESTING = 1

model = Sequential()

# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
model.add(ResNet50(include_top=False, pooling=RESNET50_POOLING_AVERAGE, classes=NUM_CLASSES, weights='imagenet'))

model.add(Dense(128, activation='tanh'))
# 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
model.add(Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION))

# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = False

model.summary()
adam = optimizers.Adam(learning_rate=10e-4, decay=10e-3)
model.compile(optimizer=adam, loss=OBJECTIVE_FUNCTION, metrics=LOSS_METRICS)

train_dir = '/opt/project/dataset/ResNet50/Training/'
image_size = IMAGE_RESIZE
train_image_gen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train_generator = train_image_gen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_TRAINING,
        seed=42,
        subset='training',
        shuffle=True)

val_generator = train_image_gen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_VALIDATION,
        seed=42,
        subset='validation',
        shuffle=True)

print((BATCH_SIZE_TRAINING, len(train_generator), BATCH_SIZE_VALIDATION, len(val_generator)))

cb_early_stopper = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath=WEIGHT_NAME, monitor='val_loss', save_best_only=True, mode='auto')

model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
        epochs=NUM_EPOCHS,
        validation_data=val_generator,
        validation_steps=STEPS_PER_EPOCH_VALIDATION,
        callbacks=[cb_checkpointer, cb_early_stopper]
)
tf.keras.models.save_model(model, WEIGHT_NAME)
