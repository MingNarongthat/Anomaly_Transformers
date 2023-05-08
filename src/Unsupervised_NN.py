import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.models import Sequential, load_model, save_model


# Define the input shape
input_shape = (1, 25)

# Define the model architecture
# first proposed for feature extraction model
def dense_model():
    model = Sequential()
    model.add(Dense(units=64, input_shape=input_shape, activation='relu'))
    model.add(Dense(units=32, activation='sigmoid'))
    model.add(Dense(units=4, activation='relu'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# dense model changed unit
def dense_model_lowunit():
    model = Sequential()
    model.add(Dense(units=16, input_shape=input_shape, activation='relu'))
    model.add(Dense(units=8, activation='sigmoid'))
    model.add(Dense(units=4, activation='relu'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# lstm model
def lstm_dense():
    model = Sequential()
    model.add(LSTM(units=32, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(units=8, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# Load the Excel file into a Pandas DataFrame
df = pd.read_excel('/opt/project/dataset/result_predictions_token_flooding.xlsx')

# Convert the string representation of the arrays to NumPy arrays of integers
df['caption'] = df['caption'].apply(lambda x: np.array(x[1:-1].split(), dtype=int))

# Reshape the dataframe to have shape (201, 25)
df['caption'] = df['caption'].apply(lambda x: x.reshape(1, -1))
df = pd.DataFrame(np.concatenate(df['caption'].values, axis=0))

# Get the new shape of the dataframe
print(df.shape)
# df.to_excel("/opt/project/dataset/prepcluster_normal.xlsx", index=False)

# Extract the input data from the DataFrame
input_data = df.iloc[:, :25].values

# Reshape the input data to have a batch size of 3000 and an input shape of (1, 25)
input_data = np.reshape(input_data, (df.shape[0], 1, 25))

# Predict the output for the input data using the model
model = lstm_dense()
output = model.predict(input_data)
output = np.reshape(output, (df.shape[0], 4))

# The output will be a numpy array of shape (3000, 2)
print(output.shape)

# Create a DataFrame from the results list
df_output = pd.DataFrame(output)
print(df_output)

# Save the DataFrame to an xlsx file
df_output.to_excel("/opt/project/dataset/result_unsupervise_flooding.xlsx", index=False)

