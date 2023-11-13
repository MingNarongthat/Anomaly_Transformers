import torch
import os
import json
import random
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
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
    # Check if the image has an alpha channel
    if image.shape[2] == 3:  # No alpha channel
        # Add an alpha channel, filled with 255 (no transparency)
        image = np.concatenate([image, np.full((image.shape[0], image.shape[1], 1), 255, dtype=image.dtype)], axis=-1)

    # Create a mask with the same dimensions as the image, with a default value of 255 (fully opaque)
    mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255

    # Define the region you want to mask (make transparent or black)
    # For example, a rectangle from (50, 50) to (200, 200)
    mask[up_l:down_l, up_r:down_r] = 0  # Set to 0 where you want transparency or black

    # Apply the mask to the alpha channel
    image[..., 3] = mask
    
    return image

def compute_bleu(pred, gt):
    bleu = evaluate.load("google_bleu")
    pred_list = [pred]
    gt_list = [[gt]]

    bleu_score = bleu.compute(predictions=pred_list, references=gt_list)
    
    return bleu_score["google_bleu"]

# Function to save the model
def save_checkpoint(state, filename="best_checkpoint.pth.tar"):
    print("=> Saving a new best")
    torch.save(state, filename)

# Define the transformations to preprocess the image
transform_pipeline = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to VGG16's expected input size
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
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Define VGG16 model
vgg16_model = models.vgg16(pretrained=True).features
vgg16_model.eval()

# Load VED model
t = VisionEncoderDecoderModel.from_pretrained('/opt/project/tmp/Image_Cationing_VIT_Roberta_iter2')
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# # Define Loss Function and Optimizer
# criterion = nn.CrossEntropyLoss() # Example for a classification task
# optimizer = optim.Adam(nn.parameters(), lr=0.00005)
# # Initialize variable to track the best validation loss
# best_loss = float('inf')

k = 3  # Number of anchor boxes
patch_grid = 3
feature_chanel = 512
num_epochs = 30

model = AnchorBoxPredictor(feature_size=feature_chanel, num_anchors=k, patch_size=patch_grid)

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
def calculate_rect(image, output, features, patch_grid):
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

    # Assume original image size is desired for visualization
    original_width, original_height = image.size
    x_scale = original_width / conv_width
    y_scale = original_height / conv_height
    MaskingImage(image, x * x_scale - w * x_scale / 2, h * y_scale)

    # Visualization
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for (x, y, w, h) in anchor_boxes:
        rect = patches.Rectangle(
            (x * x_scale - w * x_scale / 2, y * y_scale - h * y_scale / 2),
            w * x_scale,
            h * y_scale,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
    

# Training and validation loop
for epoch in range(num_epochs):
    model.train()
    for images, captions in train_loader:
        image_batch = images.unsqueeze(0)  # Shape becomes [1, C, H, W]
        # Get the convolutional features
        with torch.no_grad():
            conv_features = vgg16_model(image_batch)
        # Forward pass
        outputs = model(images)
        
        loss = calculate_loss(outputs, captions) # Modify according to your model's output and ground truth
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    total_bleu_score = 0
    with torch.no_grad():
        for images, captions in val_loader:
            # Generate predictions
            predicted_captions = model(images)
            # Convert predictions to text (you'll need to implement this)
            pred_texts = convert_to_text(predicted_captions)
            gt_texts = convert_to_text(captions) # Ground truth texts

            # Calculate BLEU for each image
            for pred, gt in zip(pred_texts, gt_texts):
                bleu_score = compute_bleu(pred, gt)
                total_bleu_score += bleu_score

    avg_bleu_score = total_bleu_score / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg BLEU Score: {avg_bleu_score:.4f}")

# ======================================================================================================================================================
test_caption = ["the soil moving on the cliff with the man on red cloth",
                "the sold collapse near the sea",
                "the soil slide in the grass field",
                "soil damaged in the path way",
                "damaged soil on the cliff near the white buildings",
                "soild damaged on the grss field",
                "damage cliff on the top"]

image_path = '/opt/project/dataset/Image/Testing/anomaly/'
id_cap = 0
for filename in os.listdir(image_path):
    if filename.endswith(".jpg"):
        # Load and preprocess the image
        image = Image.open(os.path.join(image_path, filename)).convert("RGB")
        
        caption = tokenizer.decode(t.generate(feature_extractor(image, return_tensors="pt").pixel_values)[0])

        # Remove [CLS] and [SEP] tokens from the caption
        tokens = caption.split()
        tokens_without_special_tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]"]]
        caption_without_special_tokens = " ".join(tokens_without_special_tokens)
        print(caption_without_special_tokens)
        
        bleu_score = 100 - compute_bleu(caption_without_special_tokens, test_caption[id_cap])*100
        print(bleu_score)
        
        id_cap = id_cap + 1
        
        image_tensor = transform_pipeline(image)
        image_batch = image_tensor.unsqueeze(0)  # Shape becomes [1, C, H, W]

        # Get the convolutional features
        with torch.no_grad():
            conv_features = vgg16_model(image_batch)
        
        pred_offsets = model(conv_features)
        # print(pred_offsets.shape)
        
        tx = pred_offsets[:, 0:k*4:4, :, :].detach().numpy()
        ty = pred_offsets[:, 1:k*4:4, :, :].detach().numpy()
        tw = pred_offsets[:, 2:k*4:4, :, :].detach().numpy()
        th = pred_offsets[:, 3:k*4:4, :, :].detach().numpy()
        # print(tx[0][0][0][0]) # (arr, arr, row ,col)
        # print(ty[0][0])

        conv_height, conv_width = conv_features.shape[-2:]
        patch_width = conv_width // patch_grid
        patch_height = conv_height // patch_grid

        anchor_boxes = []
        for i in range(patch_grid):
            for j in range(patch_grid):
                xa, ya = j * patch_width + patch_width / 2, i * patch_height + patch_height / 2
                wa, ha = patch_width / 2, patch_height / 2

                for anchor in range(k):
                    tx1, ty1, tw1, th1 = np.random.rand(4)
                    # print(anchor)
                    # tx1, ty1, tw1, th1 = tx[0][anchor][i][j], ty[0][anchor][i][j], tw[0][anchor][i][j], th[0][anchor][i][j]
                    x = xa + tx1 * wa
                    y = ya + ty1 * ha
                    w = wa * np.exp(tw1)
                    h = ha * np.exp(th1)
                    anchor_boxes.append((x, y, w, h))

        # Assume original image size is desired for visualization
        original_width, original_height = image.size
        x_scale = original_width / conv_width
        y_scale = original_height / conv_height

        # Visualization
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for (x, y, w, h) in anchor_boxes:
            rect = patches.Rectangle(
                (x * x_scale - w * x_scale / 2, y * y_scale - h * y_scale / 2),
                w * x_scale,
                h * y_scale,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)

        plt.savefig('/opt/project/tmp/TestAnchor{}.jpg'.format(filename))
        plt.close(fig)  # Close the figure to avoid memory issues with many images



