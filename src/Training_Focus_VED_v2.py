import torch
import os
import json
import random
import datetime
import pytz
import csv
import cv2
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import evaluate
import torch.optim as optim
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

# Layer for anchorbox ==============================================================================================================================
class AnchorBoxPredictor(nn.Module):
    def __init__(self, feature_size, num_anchors, patch_size):
        super(AnchorBoxPredictor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(num_anchors * 4),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_anchors * 4, num_anchors * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_anchors * 4),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_anchors * 4, num_anchors * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_anchors * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.attention = SelfAttention(num_anchors * 4, heads)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.3),
            # nn.Linear(4 * 2,num_anchors * 4 * 8),
            nn.Linear(1,num_anchors * 4 * 8),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_anchors * 4 * 8, 8),
            nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((patch_size, patch_size))
        self.sigmoid = nn.Sigmoid()  # to ensure tx, ty are between 0 and 1
        self.tanh = nn.Tanh()  # to ensure tw, th can be negative as well

    def forward(self, x):
        # Apply a 1x1 conv to predict the 4 values tx, ty, tw, th for each anchor
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.adaptive_pool(out)
        # Assuming out has shape [batch_size, num_anchors * 4, grid_height, grid_width]
        num_anchors = 3
        # Split the output into the tx, ty (which we pass through a sigmoid) and tw, th (which we pass through a tanh)
        tx_ty, tw_th = torch.split(out, num_anchors * 2, 1)
        tx_ty = self.sigmoid(tx_ty)
        tw_th = self.tanh(tw_th)
        # return out
        return torch.cat([tx_ty, tw_th], 1)

# Masking image ==============================================================================================================================
def apply_masks_and_save(image, boxes, x_scale, y_scale):
    # Make a copy of the image to keep the original intact
    masked_image = image.copy()

    for box in boxes:
        # Scale the box coordinates
        center_x, center_y, box_width, box_height = box
        center_x_scaled = int(center_x * x_scale)
        center_y_scaled = int(center_y * y_scale)
        box_width_scaled = int(box_width * x_scale)
        box_height_scaled = int(box_height * y_scale)

        # Convert to top-left and bottom-right coordinates
        top_left_x = max(center_x_scaled - box_width_scaled // 2, 0)
        top_left_y = max(center_y_scaled - box_height_scaled // 2, 0)
        bottom_right_x = min(center_x_scaled + box_width_scaled // 2, masked_image.shape[1] - 1)
        bottom_right_y = min(center_y_scaled + box_height_scaled // 2, masked_image.shape[0] - 1)

        # Apply the mask
        cv2.rectangle(masked_image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 0), -1)

    return masked_image

# Compute BLEU score ==============================================================================================================================
def compute_bleu(pred, gt):
    bleu = evaluate.load("google_bleu")
    pred_list = [pred]
    gt_list = [[gt]]

    bleu_score = bleu.compute(predictions=pred_list, references=gt_list)
    
    return bleu_score["google_bleu"]

# Function to save the model =========================================================================================================
def save_checkpoint(state, filename="/opt/project/tmp/test_checkpoint20231116.pth.tar"):
    print("=> Saving a new best")
    torch.save(state, filename)

# # Initialize variable to track the best validation loss
best_loss = float('inf')
# input arguments ==============================================================================================================================
feature_chanel = 512
k = 3  # Number of anchor boxes
patch_grid = 3
num_epochs = 3
# ==============================================================================================================================
# Define the transformations to preprocess the image
transform_pipeline = transforms.Compose([
    transforms.Resize((feature_chanel, feature_chanel)),  # Resize to VGG16's expected input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize like VGG16 expects
])

# Data preparation ==============================================================================================================================

# Load JSON file
root_dir = '/opt/project/dataset/DataAll/Training/'
with open('/opt/project/dataset/focus_caption_dataset_Sheet1_v1.json', 'r') as file:
    data = json.load(file)
    # print(len(data))

random.shuffle(data) # shuffle for split train and validate
split_ratio = 0.8
split_index = int(len(data) * split_ratio)
train_data = data[:split_index]
val_data = data[split_index:]
# print(train_data[0]['caption'])

# Define VGG16 model
vgg16_model = models.vgg16(pretrained=True).features
vgg16_model.eval()

# Load VED model ==============================================================================================================================
# t = VisionEncoderDecoderModel.from_pretrained('/opt/project/tmp/Image_Cationing_VIT_classification_v2.0')
# feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

t = VisionEncoderDecoderModel.from_pretrained('/opt/project/tmp/Image_Cationing_VIT_Roberta_iter2')
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

model = AnchorBoxPredictor(feature_size=feature_chanel, num_anchors=k, patch_size=patch_grid)
optimizer = optim.Adam(model.parameters(), lr=0.005)

# BLEU score calculation for loss in this model
def calculate_loss(image, gt_caption):
    # Load and preprocess the image
        image_param = Image.open(os.path.join(image)).convert("RGB")
        
        caption = tokenizer.decode(t.generate(feature_extractor(image_param, return_tensors="pt").pixel_values)[0])

        # Remove [CLS] and [SEP] tokens from the caption
        tokens = caption.split()
        tokens_without_special_tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]"]]
        caption_without_special_tokens = " ".join(tokens_without_special_tokens)
        
        bleu_score = 100 - compute_bleu(caption_without_special_tokens, gt_caption)*100
        
        return bleu_score

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Assuming 'model' is your combined VGG16 and custom model
total_params = count_parameters(model)
print(f"Total trainable parameters: {total_params}")
    
start_time_str = []
end_time_str = []
avg_loss_count = []
count = 0
images_path = '/opt/project/dataset/DataAll/Training/'
# Training and validation loop
for epoch in range(num_epochs):
    print("================== start epoch ==================")
    # Get current date and time
    start_time = datetime.datetime.now(tz=pytz.timezone('Asia/Tokyo'))
    start_time_str.append(start_time.strftime("%Y-%m-%d %H:%M:%S"))
    model.train()
    for idx in range(len(train_data)):
        if idx%5 == 0:
            count = round((count+idx)/len(train_data)*100)
            print("start new data. Progress >>>> {}".format(count))
        # Print the shape of the images for debugging
        # print(f"Images shape before processing: {images.shape}")
        original_image = cv2.imread(os.path.join(images_path, train_data[idx]["image"]))
    
        # if original_image.shape[2] == 3:  # No alpha channel
        #     # Add an alpha channel, filled with 255 (no transparency)
        #     original_image = np.concatenate([original_image, np.full((original_image.shape[0], original_image.shape[1], 1), 255, dtype=original_image.dtype)], axis=-1)
        image1 = Image.open(os.path.join(images_path, train_data[idx]["image"])).convert("RGB")
        image = transform_pipeline(image1)
        image = image.unsqueeze(0)

        # Now pass the images to vgg16_model
        with torch.no_grad():
            conv_features = vgg16_model(image)
        # conv_features = transition_layer(conv_features)  # Adjusting channels to 1024
        outputs = model(conv_features)
        print(outputs.shape)
            
        tx = outputs[:, 0:k*4:4, :, :].detach().numpy()
        ty = outputs[:, 1:k*4:4, :, :].detach().numpy()
        tw = outputs[:, 2:k*4:4, :, :].detach().numpy()
        th = outputs[:, 3:k*4:4, :, :].detach().numpy()
        
        conv_height, conv_width = conv_features.shape[-2:]
        patch_width = conv_width // patch_grid
        patch_height = conv_height // patch_grid
        
        original_width, original_height = image1.size
        x_scale = original_width / conv_width
        y_scale = original_height / conv_height
        
        anchor_boxes = []
        for i in range(patch_grid):
            for j in range(patch_grid):
                xa, ya = j * patch_width + patch_width / 2, i * patch_height + patch_height / 2
                wa, ha = patch_width / 2, patch_height / 2

                for anchor in range(k):
                    tx1, ty1, tw1, th1 = tx[0][anchor][i][j], ty[0][anchor][i][j], tw[0][anchor][i][j], th[0][anchor][i][j]
                    x = xa + tx1 * wa
                    y = ya + ty1 * ha
                    w = wa * np.exp(tw1)
                    h = ha * np.exp(th1)
                    anchor_boxes.append((x, y, w, h))
        # print("end masked image")
        # print(anchor_boxes)
        masked_image = apply_masks_and_save(original_image, anchor_boxes, x_scale, y_scale)
        # After the masking process
        if masked_image is None:
            print("The masked_image is None, something went wrong during the masking process.")

        # Generate the caption for the image
        caption = tokenizer.decode(t.generate(feature_extractor(masked_image, return_tensors="pt").pixel_values)[0])

        # Remove [CLS] and [SEP] tokens from the caption
        tokens = caption.split()
        tokens_without_special_tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]"]]
        caption_without_special_tokens = " ".join(tokens_without_special_tokens)
        
        loss_value = 100 - compute_bleu(caption_without_special_tokens, train_data[idx]["caption"])*100
        loss = torch.tensor(loss_value, requires_grad=True)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation ========================================== VALIDATION ============================================================
    model.eval()
    total_bleu_score = 0
    with torch.no_grad():
        for idy in range(len(val_data)):
            original_image = cv2.imread(os.path.join(images_path, val_data[idy]["image"]))
            # if original_image.shape[2] == 3:  # No alpha channel
            #     # Add an alpha channel, filled with 255 (no transparency)
            #     original_image = np.concatenate([original_image, np.full((original_image.shape[0], original_image.shape[1], 1), 255, dtype=original_image.dtype)], axis=-1)
            image1 = Image.open(os.path.join(images_path, val_data[idy]["image"])).convert("RGB")
            image = transform_pipeline(image1)
            image = image.unsqueeze(0)

            # Get the convolutional features
            with torch.no_grad():
                conv_features = vgg16_model(image)
            # Forward pass
            outputs = model(conv_features)
            
            tx = outputs[:, 0:k*4:4, :, :].detach().numpy()
            ty = outputs[:, 1:k*4:4, :, :].detach().numpy()
            tw = outputs[:, 2:k*4:4, :, :].detach().numpy()
            th = outputs[:, 3:k*4:4, :, :].detach().numpy()
            
            conv_height, conv_width = conv_features.shape[-2:]
            patch_width = conv_width // patch_grid
            patch_height = conv_height // patch_grid
            
            original_width, original_height = image1.size
            x_scale = original_width / conv_width
            y_scale = original_height / conv_height
            
            anchor_boxes = []
            for i in range(patch_grid):
                for j in range(patch_grid):
                    xa, ya = j * patch_width + patch_width / 2, i * patch_height + patch_height / 2
                    wa, ha = patch_width / 2, patch_height / 2

                    for anchor in range(k):
                        tx1, ty1, tw1, th1 = tx[0][anchor][i][j], ty[0][anchor][i][j], tw[0][anchor][i][j], th[0][anchor][i][j]
                        x = xa + tx1 * wa
                        y = ya + ty1 * ha
                        w = wa * np.exp(tw1)
                        h = ha * np.exp(th1)
                        anchor_boxes.append((x, y, w, h))
            # print(anchor_boxes)
            masked_image = apply_masks_and_save(original_image, anchor_boxes, x_scale, y_scale)
            # After the masking process
            if masked_image is None:
                print("The masked_image is None, something went wrong during the masking process.")

            # Generate the caption for the image
            caption = tokenizer.decode(t.generate(feature_extractor(masked_image, return_tensors="pt").pixel_values)[0])

            # Remove [CLS] and [SEP] tokens from the caption
            tokens = caption.split()
            tokens_without_special_tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]"]]
            caption_without_special_tokens = " ".join(tokens_without_special_tokens)

            loss_value = 100 - compute_bleu(caption_without_special_tokens, val_data[idy]["caption"])*100
            loss = torch.tensor(loss_value, requires_grad=True)
            total_bleu_score = total_bleu_score+loss_value

    avg_loss = total_bleu_score
    avg_loss_count.append(avg_loss)
    # Save the model if validation loss has decreased
    if avg_loss < best_loss:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        })
        best_loss = avg_loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")
    # Get finish date and time
    end_time = datetime.datetime.now(tz=pytz.timezone('Asia/Tokyo'))
    end_time_str.append(end_time.strftime("%Y-%m-%d %H:%M:%S"))

    # Save start and end time to a CSV file
    csv_file = "/opt/project/tmp/training_logFocus2.csv"
    with open(csv_file, "a") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Script Name", "Start Time", "End Time", "Avg Loss"])
        for epoch in range(len(start_time_str)):  # Iterate based on recorded times
            writer.writerow([epoch, "trainingFocus2.py", start_time_str[epoch], end_time_str[epoch], avg_loss])
