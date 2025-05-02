import os
import pandas as pd
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to the folder containing the images
images_path = "/opt/project/dataset/ResNet50/Testing/testall/"

# Load the pre-trained image captioning model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained('/opt/project/tmp/Image_Cationing_VIT_classification_V1.3')
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Move model to device
model.to(device)

# Set generation parameters to match training
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.max_length = 50
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

# Create an empty list to store the results
results = []

# Loop through all the files in the images folder
for filename in os.listdir(images_path):
    if filename.endswith(".jpg"):
        try:
            # Load the image
            image = Image.open(os.path.join(images_path, filename)).convert("RGB")
            
            # Prepare image for model
            pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            
            # Generate the caption
            output_ids = model.generate(pixel_values)
            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            print(f"{filename}: {caption}")
            
            # Add the filename and caption to the results list
            results.append({"Filename": filename, "Caption": caption})
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

# Create a DataFrame from the results list
df = pd.DataFrame(results)

# Save the DataFrame to an xlsx file
output_path = "/opt/project/tmp/result_predictions_caption2.xlsx"
df.to_excel(output_path, index=False)
print(f"Results saved to {output_path}")

