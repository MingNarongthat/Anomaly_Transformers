import torch
import os
import json
import random
import datetime
import pytz
import csv
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

class CaptionDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        """
        Args:
            data (list): List of dictionaries with 'image' and 'caption'.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[idx]['image'])
        image = Image.open(img_name)
        caption = self.data[idx]['caption']

        if self.transform:
            image = self.transform(image)

        return image, caption

# Self attention layer
# class SelfAttention(nn.Module):
#     def __init__(self, feature_size, heads):
#         super(SelfAttention, self).__init__()
#         self.feature_size = feature_size
#         self.heads = heads
#         self.head_dim = feature_size // heads

#         assert (
#             self.head_dim * heads == feature_size
#         ), "Feature size needs to be divisible by heads"

#         self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.fc_out = nn.Linear(heads * self.head_dim, feature_size)

#     def forward(self, values, keys, query):
#         N = query.shape[0]
#         value_len, key_len, query_len = values.shape[2], keys.shape[2], query.shape[2]

#         # Split the embedding into self.heads different pieces
#         values = values.reshape(N, self.heads, self.head_dim, value_len)
#         keys = keys.reshape(N, self.heads, self.head_dim, key_len)
#         queries = query.reshape(N, self.heads, self.head_dim, query_len)

#         values = self.values(values)
#         keys = self.keys(keys)
#         queries = self.queries(queries)

#         # Scaled dot-product attention
#         energy = torch.einsum("nhqd,nhkd->nhqk", [queries, keys])

#         attention = torch.softmax(energy / (self.feature_size ** (1 / 2)), dim=3)

#         out = torch.einsum("nhql,nhld->nqhd", [attention, values]).reshape(
#             N, self.heads * self.head_dim, value_len
#         )

#         out = self.fc_out(out)
#         return out
    
    
# Generate acnhor box layer
class AnchorBoxPredictor(nn.Module):
    def __init__(self, feature_size, num_anchors, patch_size):
        super(AnchorBoxPredictor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_anchors * 4),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_anchors * 4, num_anchors * 4, kernel_size=3, stride=1, padding=1),
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
            nn.Linear(4 * 2,num_anchors * 4 * 8),
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
    
def MaskingImage(image, up_l, up_r, down_l, down_r):
    device = image.device
    dtype = image.dtype

    # Process each image in the batch individually
    masked_images = []
    for idx in range(image.shape[0]):
        single_image = image[idx]

        # Check if the image has an alpha channel, add one if it doesn't
        if single_image.shape[0] == 3:  # No alpha channel
            alpha_channel = torch.full((1, single_image.shape[1], single_image.shape[2]), 255, dtype=dtype, device=device)
            single_image = torch.cat([single_image, alpha_channel], dim=0)

        # Create a mask for the current image
        mask = torch.ones((single_image.shape[1], single_image.shape[2]), dtype=torch.uint8, device=device) * 255
        mask[up_l:down_l, up_r:down_r] = 0

        # Apply the mask to the alpha channel
        single_image[3, :, :] = mask.to(dtype=dtype)
        masked_images.append(single_image)

    # Stack the masked images to form a batch
    return torch.stack(masked_images)

def compute_bleu(pred, gt):
    bleu = evaluate.load("google_bleu")
    pred_list = [pred]
    gt_list = [[gt]]

    bleu_score = bleu.compute(predictions=pred_list, references=gt_list)
    
    return bleu_score["google_bleu"]

# Function to save the model
def save_checkpoint(state, filename="/opt/project/tmp/best_checkpoint.pth.tar"):
    print("=> Saving a new best")
    torch.save(state, filename)

feature_chanel = 512
# Define the transformations to preprocess the image
transform_pipeline = transforms.Compose([
    transforms.Resize((feature_chanel, feature_chanel)),  # Resize to VGG16's expected input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize like VGG16 expects
])

# Data preparation

# Load JSON file
root_dir = '/opt/project/dataset/DataAll/Training/'
with open('/opt/project/dataset/focus_caption_dataset_training_v1.json', 'r') as file:
    data = json.load(file)

# Shuffle the data
random.shuffle(data)

# Split the data (80% train, 20% validation)
split_ratio = 0.8
split_index = int(len(data) * split_ratio)
train_data = data[:split_index]
val_data = data[split_index:]

# Create dataset instances
train_dataset = CaptionDataset(data=train_data, root_dir=root_dir, transform=transform_pipeline)
val_dataset = CaptionDataset(data=val_data, root_dir=root_dir, transform=transform_pipeline)

def collate_fn(batch):
   batch = list(filter(lambda x: x is not None, batch))
   return torch.utils.data.dataloader.default_collate(batch)     
# Create dataloaders
train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn)

# Define VGG16 model
vgg16_model = models.vgg16(pretrained=True).features
vgg16_model.eval()

# Load VED model
# t = VisionEncoderDecoderModel.from_pretrained('/opt/project/tmp/Image_Cationing_VIT_classification_v2.0')
# feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

t = VisionEncoderDecoderModel.from_pretrained('/opt/project/tmp/Image_Cationing_VIT_Roberta_iter2')
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


# # Initialize variable to track the best validation loss
best_loss = float('inf')

k = 3  # Number of anchor boxes
patch_grid = 3
num_epochs = 30

model = AnchorBoxPredictor(feature_size=feature_chanel, num_anchors=k, patch_size=patch_grid)
optimizer = optim.Adam(model.parameters(), lr=0.005)

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

# output from model and feature input from VGG16
def calculate_rectmasked(image, output, features, patch_grid):
    tx = outputs[:, 0:k*4:4, :, :].detach().numpy()
    ty = outputs[:, 1:k*4:4, :, :].detach().numpy()
    tw = outputs[:, 2:k*4:4, :, :].detach().numpy()
    th = outputs[:, 3:k*4:4, :, :].detach().numpy()
    conv_height, conv_width = features.shape[-2:]
    patch_width = conv_width // patch_grid
    patch_height = conv_height // patch_grid
    
    anchor_boxes = []
    for i in range(patch_grid):
        for j in range(patch_grid):
            xa, ya = j * patch_width + patch_width / 2, i * patch_height + patch_height / 2
            wa, ha = patch_width / 2, patch_height / 2

            for anchor in range(k):
                tx1, ty1, tw1, th1 = np.random.rand(4)
                x = xa + tx1 * wa
                y = ya + ty1 * ha
                w = wa * np.exp(tw1)
                h = ha * np.exp(th1)
                anchor_boxes.append((x, y, w, h))
    # Check if 'images' is a batch or a single image
    if len(image.shape) == 4:  # Batch of images
        # Process each image in the batch
        masked_images = []
        for i in range(images.shape[0]):
            single_image = images[i]
            # Get the height and width of the single image
            _, original_height, original_width = single_image.shape
            
            # Perform masking on the single image
            x_scale = original_width / conv_width
            y_scale = original_height / conv_height
            masked_image = MaskingImage(image, 
                                        int(y * y_scale - h * y_scale / 2), int(y * y_scale + h * y_scale / 2),
                                        int(x * x_scale - w * x_scale / 2), int(x * x_scale + x * x_scale / 2))

            masked_images.append(masked_image)

        # Return the batch of masked images
        return torch.stack(masked_images)

    else:  # Single image
        # Get the height and width of the image
        _, original_height, original_width = images.shape

        # Perform masking on the single image
        x_scale = original_width / conv_width
        y_scale = original_height / conv_height
        masked_image = MaskingImage(image, 
                                    int(y * y_scale - h * y_scale / 2), int(y * y_scale + h * y_scale / 2),
                                    int(x * x_scale - w * x_scale / 2), int(x * x_scale + x * x_scale / 2))
        return masked_image
    
start_time_str = []
end_time_str = []

# transition_layer = nn.Conv2d(512, feature_chanel, 1)  # Upscaling to 1024 channels
# Training and validation loop
for epoch in range(num_epochs):
    # Get current date and time
    start_time = datetime.datetime.now(tz=pytz.timezone('Asia/Tokyo'))
    start_time_str.append(start_time.strftime("%Y-%m-%d %H:%M:%S"))
    model.train()
    for images, captions_batch in train_loader:
        # Print the shape of the images for debugging
        # print(f"Images shape before processing: {images.shape}")

        # Ensure images are 3-channel RGB
        if images.shape[1] == 1:  # Grayscale images
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] == 4:  # RGBA images
            images = images[:, :3, :, :]  # Keep only RGB channels
        
        # Print the shape after processing
        # print(f"Images shape after processing: {images.shape}")

        # Now pass the images to vgg16_model
        conv_features = vgg16_model(images)
        # conv_features = transition_layer(conv_features)  # Adjusting channels to 1024
        outputs = model(conv_features)
            
        masked_image = calculate_rectmasked(images, outputs, conv_features, patch_grid)

        for i in range(masked_image.shape[0]):
            single_image_tensor = masked_image[i]
            # ... [rest of your processing for single_image_tensor] ...

            # Extract the corresponding caption for the current image
            current_caption = captions_batch[i]

            # Use 'current_caption' instead of 'caption' in your processing
            # For instance, when computing BLEU score or processing the caption
            tokens = current_caption.split()
            tokens_without_special_tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]"]]
            caption_without_special_tokens = " ".join(tokens_without_special_tokens)
            
            
            loss_value = 100 - compute_bleu(caption_without_special_tokens, current_caption)*100
            loss = torch.tensor(loss_value, requires_grad=True)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    total_bleu_score = 0
    with torch.no_grad():
        for images, captions in val_loader:
            # image_batch = images.unsqueeze(0)  # Shape becomes [1, C, H, W]
            # Get the convolutional features
            with torch.no_grad():
                conv_features = vgg16_model(images)
            # Forward pass
            outputs = model(conv_features)
            
            masked_image = calculate_rectmasked(images, outputs, conv_features, patch_grid)
            for i in range(masked_image.shape[0]):
                single_image_tensor = masked_image[i]
                # ... [rest of your processing for single_image_tensor] ...

                # Extract the corresponding caption for the current image
                current_caption = captions_batch[i]

                # Use 'current_caption' instead of 'caption' in your processing
                # For instance, when computing BLEU score or processing the caption
                tokens = current_caption.split()
                tokens_without_special_tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]"]]
                caption_without_special_tokens = " ".join(tokens_without_special_tokens)
                
                
                loss_value = 100 - compute_bleu(caption_without_special_tokens, current_caption)*100
                loss = torch.tensor(loss_value, requires_grad=True)
            total_bleu_score = total_bleu_score+loss_value

    avg_loss = total_bleu_score
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
    print("Number of start times recorded:", len(start_time_str))
    print("Number of end times recorded:", len(end_time_str))   

    # Save start and end time to a CSV file
    csv_file = "/opt/project/tmp/training_logFocus.csv"
    with open(csv_file, "a") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Script Name", "Start Time", "End Time"])
        for epoch in range(len(start_time_str)):  # Iterate based on recorded times
            writer.writerow([epoch, "training.py", start_time_str[epoch], end_time_str[epoch]])

# ======================================================================================================================================================




