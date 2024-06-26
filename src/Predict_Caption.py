import os
import pandas as pd
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the path to the folder containing the images

# images_path = "/opt/project/dataset/Flooding/"

images_path = "/opt/project/dataset/ResNet50/Testing/testall/"

# Load the pre-trained image captioning model and tokenizer
t = VisionEncoderDecoderModel.from_pretrained('/opt/project/tmp/Image_Cationing_VIT_classification_V3')
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# print(t)
# Create an empty list to store the results
results = []

# Loop through all the files in the images folder
for filename in os.listdir(images_path):
    if filename.endswith(".jpg"):
        # Load the image
        image = Image.open(os.path.join(images_path, filename)).convert("RGB")

        # Generate the caption for the image
        caption = tokenizer.decode(t.generate(feature_extractor(image, return_tensors="pt").pixel_values)[0])

        # Remove [CLS] and [SEP] tokens from the caption
        tokens = caption.split()
        tokens_without_special_tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]", "<|endoftext|>"]]
        caption_without_special_tokens = " ".join(tokens_without_special_tokens)

        # encode text to tensor
        tokenized_caption = tokenizer.encode(caption_without_special_tokens,
                                             truncation=True,
                                             padding=True,
                                             return_tensors="tf")
        # extract token array
        token_ids = tokenized_caption.numpy()[0]
        print(filename, caption_without_special_tokens)
        # Add the filename and caption to the results list
        results.append({"Filename": filename, "Caption": caption_without_special_tokens})

# Define the maximum sequence length for padding

# max_length = 25

max_length = 30

# Pad the token_ids arrays in the results list
# for result in results:
#     result["caption"] = pad_sequences([result["caption"]], max_len=max_length, padding="post")[0]

# Create a DataFrame from the results list
df = pd.DataFrame(results)

# Save the DataFrame to an xlsx file

df.to_excel("/opt/project/dataset/result_predictions_caption_our.xlsx", index=False)

# df.to_excel("/opt/project/dataset/experiment1_crop_vs_masked5.xlsx", index=False)
