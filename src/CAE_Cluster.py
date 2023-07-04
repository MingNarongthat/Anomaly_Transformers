import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE

# Load your data and labels
data = np.loadtxt('/opt/project/dataset/dataConv.csv', delimiter=',')
labels = np.loadtxt('/opt/project/dataset/labels.csv', delimiter=',')

# Normalize the data
scaler = StandardScaler()
data_norm = scaler.fit_transform(data)

# Define the input layer
input_layer = Input(shape=(data_norm.shape[1],))

# Define the encoder layer
encoder_layer = Dense(64, activation='relu')(input_layer)

# Define the bottleneck layer
bottleneck_layer = Dense(32, activation='relu')(encoder_layer)

# Define the decoder layer
decoder_layer = Dense(64, activation='relu')(bottleneck_layer)

# Define the output layer
output_layer = Dense(data_norm.shape[1], activation='linear')(decoder_layer)

# Define the autoencoder model
autoencoder_model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
autoencoder_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

# Fit the model to the data
autoencoder_model.fit(data_norm, data_norm, epochs=100, batch_size=32, verbose=0)

# Extract the encoder part of the autoencoder model
encoder_model = Model(inputs=input_layer, outputs=bottleneck_layer)

# Use the encoder to get the encoded data
encoded_data = encoder_model.predict(data_norm)

# Cluster the encoded data using KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(encoded_data)

# Get the predicted cluster labels
predicted_labels = kmeans.labels_

# Compare the predicted labels with the true labels
accuracy = np.mean(predicted_labels == labels)
print(f'Accuracy: {accuracy:.2f}')


# Save the KMeans model
joblib.dump(kmeans, 'kmeans_model.pkl')

# # Visualize the encoded data with the predicted cluster labels
# plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=predicted_labels)
# plt.title('Encoded Data with Predicted Clusters')
# plt.xlabel('Encoded Dimension 1')
# plt.ylabel('Encoded Dimension 2')
# plt.savefig('/opt/project/tmp/Cluster_plot.jpg')
# plt.show()
#
# Create the confusion matrix
plt.figure()
cm = confusion_matrix(labels, predicted_labels)

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('/opt/project/tmp/Cluster_confusion_plot_Conv.jpg')
plt.close()

# Use t-SNE to project the encoded data points to a 2D space
plt.figure()
tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(encoded_data)

# Define the color maps for the true and predicted labels
true_label_colors = ['red', 'green', 'blue']
predicted_label_colors = ['orange', 'purple', 'cyan']

# Define the label names
label_names = ['Normal', 'Anomaly Landslide', 'Anomaly Flooding']

# Visualize the t-SNE plot with both true and predicted labels and a legend
# plt.figure(figsize=(20, 12))
for i in range(len(label_names)):
    plt.scatter(tsne_data[labels == i, 0], tsne_data[labels == i, 1], c=true_label_colors[i], alpha=0.5, label=label_names[i])
    plt.scatter(tsne_data[predicted_labels == i, 0], tsne_data[predicted_labels == i, 1], c=predicted_label_colors[i], alpha=0.5, label='Predicted '+label_names[i])
plt.title('t-SNE plot with True and Predicted Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('/opt/project/tmp/Cluster_tSNE2_plot_Conv.jpg', bbox_inches='tight')
plt.close()

