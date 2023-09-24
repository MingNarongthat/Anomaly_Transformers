import os
import pandas as pd
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import torch
from transformers import ViTModel, BertModel
import numpy as np
import cv2

class CustomViTModel(ViTModel):
    def forward(self, pixel_values, **kwargs):
        outputs = super().forward(pixel_values, **kwargs)
        first_attention_output = self.encoder.layer[11].attention.attention
        return outputs, first_attention_output

class CustomBertModel(BertModel):
    def forward(self, input_ids, **kwargs):
        outputs = super().forward(input_ids, **kwargs)
        return outputs

class CustomVisionEncoderDecoderModel(VisionEncoderDecoderModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = CustomViTModel(self.config.encoder)
        self.decoder = CustomBertModel(self.config.decoder)

    def forward(self, pixel_values, decoder_input_ids, **kwargs):
        encoder_outputs, first_attention_output = self.encoder(pixel_values=pixel_values, **kwargs)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoder_outputs.last_hidden_state, **kwargs)
        return decoder_outputs, first_attention_output

def hook_fn(module, input, output):
    global attention_output
    attention_output = output

def compute_attention_rollout(attention_matrices):
    # Multiply attention matrices across layers to get the attention rollout
    rollout = attention_matrices[0]
    for i in range(1, len(attention_matrices)):
        rollout = torch.matmul(rollout, attention_matrices[i])
    return rollout

def overlay_attention_on_image(image_path, attention_rollout):
    # Load the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError("Failed to load the image.")
    # Resize the attention map to match the image size
    attention_map = cv2.resize(attention_rollout, (original_image.shape[1], original_image.shape[0]))

    # Normalize the attention map to [0, 255]
    if attention_map.max() != attention_map.min():
        normalized_attention_map = ((attention_map - attention_map.min()) * (
                    255.0 / (attention_map.max() - attention_map.min()))).astype(np.uint8)
    else:
        normalized_attention_map = np.zeros_like(attention_map, dtype=np.uint8)

    # Convert the grayscale attention map to a colormap
    colored_attention_map = cv2.applyColorMap(normalized_attention_map, cv2.COLORMAP_JET)

    # Overlay the attention map on the original image
    overlaid_image = cv2.addWeighted(original_image, 0.2, colored_attention_map, 0.8, 0)

    return overlaid_image

# Define the path to the folder containing the images
images_path = "/opt/project/dataset/DataAll/Testing/"

custom_model = CustomVisionEncoderDecoderModel.from_pretrained('/opt/project/tmp/Image_Cationing_VIT_Roberta_iter2')
# Load the pre-trained image captioning model and tokenizer
# model = VisionEncoderDecoderModel.from_pretrained('/opt/project/tmp/Image_Cationing_VIT_Roberta_iter2')
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Create an empty list to store the results
results = []

# Loop through all the files in the images folder
for filename in os.listdir(images_path):
    if filename.endswith(".jpg"):
        # Load the image
        image = Image.open(os.path.join(images_path, filename)).convert("RGB")

        # Generate the caption for the image
        handle = custom_model.encoder.encoder.layer[8].attention.attention.register_forward_hook(hook_fn)
        # Run the forward pass
        input_features = feature_extractor(image, return_tensors="pt").pixel_values
        input_ids = torch.zeros((1, 1), dtype=torch.long)
        outputs = custom_model(pixel_values=input_features, decoder_input_ids=input_ids)

        # Remove the hook
        handle.remove()

        # Now, `attention_output` should contain the attention output tensor
        print(attention_output)

        # Assuming you have a VisionEncoderDecoderModel named 'model'
        attention_matrices = []

        # Forward pass (you might need to adjust this based on your specific model and input)
        outputs = custom_model(input_features, decoder_input_ids=input_ids, return_dict=True, output_attentions=True)
        encoder_outputs = outputs[0]

        # Extract attention weights from the encoder
        encoder_attentions = encoder_outputs.attentions
        attention_matrices.extend(encoder_attentions)

        # Extract attention weights from the decoder (if applicable)

        # Now, you can compute the attention rollout using the attention_matrices list
        rollout = compute_attention_rollout(attention_matrices)
        token_attention = rollout[0].cpu().detach().numpy()

        overlaid_image = overlay_attention_on_image(images_path+filename, token_attention)
        plt.imshow(overlaid_image)
        plt.savefig('/opt/project/tmp/Layertest{}.jpg'.format(filename))

        # Add the filename and caption to the results list
        results.append({"filename": filename})

# Define the maximum sequence length for padding
max_length = 20

# Create a DataFrame from the results list
df = pd.DataFrame(results)
