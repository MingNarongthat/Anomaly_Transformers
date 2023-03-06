import tensorflow as tf
import pandas as pd
import numpy as np

# Define the input shape
input_shape = (1, 25)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='sigmoid'),
    tf.keras.layers.Dense(units=4, activation='relu')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Load the Excel file into a Pandas DataFrame
df = pd.read_excel('/opt/project/dataset/result_predictions_token_landslide.xlsx')
# Convert the string representation of the arrays to NumPy arrays of integers
df['caption'] = df['caption'].apply(lambda x: np.array(x[1:-1].split(), dtype=int))

# Reshape the dataframe to have shape (201, 25)
df['caption'] = df['caption'].apply(lambda x: x.reshape(1, -1))
df = pd.DataFrame(np.concatenate(df['caption'].values, axis=0))

# Get the new shape of the dataframe
print(df.shape)

# print(type(data))
# Extract the input data from the DataFrame
input_data = df.iloc[:, :25].values

# Reshape the input data to have a batch size of 3000 and an input shape of (1, 25)
input_data = np.reshape(input_data, (df.shape[0], 1, 25))

# Predict the output for the input data using the model
output = model.predict(input_data)
output = np.reshape(output, (df.shape[0], 4))
# The output will be a numpy array of shape (3000, 2)
print(output.shape)
# Create a DataFrame from the results list
df1 = pd.DataFrame(output)

# Save the DataFrame to an xlsx file
df1.to_excel("/opt/project/dataset/result_unsupervise_landslide.xlsx", index=False)
