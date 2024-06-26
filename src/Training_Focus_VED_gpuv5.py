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
import torch.nn.functional as F
import evaluate
import torch.optim as optim
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor, AutoModel
import torch

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out
    
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
        self.self_attention = SelfAttention(num_anchors * 4)
        # self.attention = SelfAttention(num_anchors * 4, heads)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.3),
            # nn.Linear(4 * 2,num_anchors * 4 * 8), # for stride 1,1,1 in 3 layers
            nn.Linear(1,num_anchors * 4 * 8),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_anchors * 4 * 8, 8),
            nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((patch_size, patch_size))
        self.sigmoid = nn.Tanh()  # to ensure tx, ty are between 0 and 1
        self.tanh = nn.ReLU()  # to ensure tw, th can be negative as well
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_anchors * 4, num_anchors, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Apply a 1x1 conv to predict the 4 values tx, ty, tw, th for each anchor
        out0 = self.layer1(x)
        out1 = self.layer2(out0)
        out2 = self.layer3(out1)
        
        outatt1 = self.self_attention(out2)
        outatt2 = self.adaptive_pool(outatt1)
        outatt3 = self.conv1(outatt2)
        outatt4 = self.sigmoid(outatt3)
        outatt4 = (outatt4 > 0.6).float()
        
        out3 = self.fc1(out2)
        out4 = self.fc2(out3)
        out5 = self.adaptive_pool(out4)
        
        # Assuming out has shape [batch_size, num_anchors * 4, grid_height, grid_width]
        num_anchors = 3
        # Split the output into the tx, ty (which we pass through a sigmoid) and tw, th (which we pass through a tanh)
        tx_ty, tw_th = torch.split(out5, num_anchors * 2, 1)
        tx_ty = self.sigmoid(tx_ty) # tanh
        tw_th = self.tanh(tw_th) # 
        
        # return out
        return torch.cat([tx_ty, tw_th], 1), outatt4

# Masking image ==============================================================================================================================
def apply_masks_and_save(image, boxes, focus):
    # Make a copy of the image to keep the original intact
    masked_image = image.copy()
    count = 0

    for box in boxes:
        # Scale the box coordinates
        center_x, center_y, box_width, box_height = box
        center_x_scaled = int(center_x)
        center_y_scaled = int(center_y)
        box_width_scaled = int(box_width)
        box_height_scaled = int(box_height)

        # Convert to top-left and bottom-right coordinates
        top_left_x = int(center_x_scaled - box_width_scaled // 2)
        top_left_y = int(center_y_scaled - box_height_scaled // 2)
        bottom_right_x = int(center_x_scaled + box_width_scaled // 2)
        bottom_right_y = int(center_y_scaled + box_height_scaled // 2)
        
        if focus[count][0] == 1:
            # Apply the mask
            cv2.rectangle(masked_image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 0), -1)
            count = count + 1
        else:
            count = count + 1
        
    return masked_image

# Function to save the model =========================================================================================================
def save_checkpoint(state, filename="/opt/project/tmp/best_checkpoint20240319.pth.tar"):
    print("=> Saving a new best")
    torch.save(state, filename)

# # Initialize variable to track the best validation loss
best_loss = float('inf')
# input arguments ==============================================================================================================================
feature_chanel = 512
k = 3  # Number of anchor boxes
patch_grid = 7
num_epochs = 15
# ==============================================================================================================================
# Define the transformations to preprocess the image
transform_pipeline = transforms.Compose([
    transforms.Resize((feature_chanel, feature_chanel)),  # Resize to VGG16's expected input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize like VGG16 expects
])

# Data preparation ==============================================================================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use the fourth GPU
# If there's a GPU available
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, please check your server configuration.')
    exit()
    
# Load JSON file
root_dir = '/opt/project/dataset/DataAll/Training/'
with open('/opt/project/dataset/focus_caption_dataset_trainingfinetune_withclass_v3.json', 'r') as file:
    data = json.load(file)
    # print(len(data))

random.shuffle(data) # shuffle for split train and validate
split_ratio = 0.9
split_index = int(len(data) * split_ratio)
train_data = data[:split_index]
val_data = data[split_index:]
# print(train_data[0]['caption'])

# Define VGG16 model
vgg16_model = models.vgg16(pretrained=True).features
vgg16_model = vgg16_model.to(device)  # Move VGG16 model to the GPU
vgg16_model.eval()

# Load VED model ==============================================================================================================================
t = VisionEncoderDecoderModel.from_pretrained('/opt/project/tmp/Image_Cationing_VIT_classification_V1.3')
# feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
modelcosine = AutoModel.from_pretrained("bert-base-uncased")
tokenizercosine = AutoTokenizer.from_pretrained("bert-base-uncased")
# t = VisionEncoderDecoderModel.from_pretrained('/opt/project/tmp/Image_Cationing_VIT_Roberta_iter2')
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

model = AnchorBoxPredictor(feature_size=feature_chanel, num_anchors=k, patch_size=patch_grid).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Load the model
checkpoint_path = "/opt/project/tmp/best_checkpoint20231224.pth.tar"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

# Continue fine-tuning the model with other data
# ...

for param in model.parameters():
    print(param.requires_grad)

cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

def caption_similarity_loss(generated_captions, true_captions):
    # Tokenize and encode captions for the language model
    gen_encodings = tokenizercosine(generated_captions, padding=True, truncation=True, max_length=512, return_tensors='pt')
    true_encodings = tokenizercosine(true_captions, padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Generate embeddings
    gen_embeddings = modelcosine(**gen_encodings).last_hidden_state.mean(dim=1)
    true_embeddings = modelcosine(**true_encodings).last_hidden_state.mean(dim=1)

    # Calculate cosine similarity
    similarity = cosine_similarity(gen_embeddings, true_embeddings)
    # Convert similarity to a loss (1 - similarity)
    loss = similarity

    return loss.mean()


def combined_custom_loss(generated_captions, original_captions, model, alpha=1.0, beta=0.0, gamma=0.0, theta=0.0):
    # Caption Similarity Term
    caption_similarity = caption_similarity_loss(generated_captions, original_captions)
    
    # inverse cosine similarity
    caption_inverse_similarity = 1 - caption_similarity  
    
    # Regularization Term (example: L2 regularization)
    l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
    
    # Regularization Term (example: L1 regularization)
    l1_reg = sum(p.abs().sum() for p in model.parameters())

    # Combine the losses
    total_loss = alpha * caption_similarity + beta * l1_reg + gamma * caption_inverse_similarity

    return total_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Assuming 'model' is your combined VGG16 and custom model
total_params = count_parameters(model)
print(f"Total trainable parameters: {total_params}")
    
start_time_str = []
end_time_str = []
avg_loss_count = []
images_path = '/opt/project/dataset/DataAll/Training/'
# Training and validation loop
for epoch in range(num_epochs):
    print("================== start epoch ==================")
    # Get current date and time
    start_time = datetime.datetime.now(tz=pytz.timezone('Asia/Tokyo'))
    start_time_str.append(start_time.strftime("%Y-%m-%d %H:%M:%S"))
    model.train()
    for idx in range(len(train_data)):
        if idx%50 == 0:
            count1 = idx / len(train_data)*100
            print(idx, "start new data. Progress >>>> {}".format(count1))
        # Print the shape of the images for debugging
        # print(f"Images shape before processing: {images.shape}")
        original_image = cv2.imread(os.path.join(images_path, train_data[idx]["image"]))

        image1 = Image.open(os.path.join(images_path, train_data[idx]["image"])).convert("RGB")
        image = transform_pipeline(image1)
        image = image.unsqueeze(0).to(device)

        # Now pass the images to vgg16_model
        with torch.no_grad():
            conv_features = vgg16_model(image)
        # conv_features = transition_layer(conv_features)  # Adjusting channels to 1024
        outputs, close_outputs = model(conv_features)  # Get the model outputs
        # focus = close_outputs.reshape(patch_grid**k,1).tolist()
        focus = close_outputs.reshape(patch_grid*patch_grid*k,1).tolist()
        # print(outputs.shape)
        # print(len(focus))
            
        tx = outputs[:, 0:k*2:2, :, :].detach().cpu().numpy()
        ty = outputs[:, 1:k*2:2, :, :].detach().cpu().numpy()
        tw = outputs[:, 6:k*4:2, :, :].detach().cpu().numpy()
        th = outputs[:, 7:k*4:2, :, :].detach().cpu().numpy()
        
        original_width, original_height = image1.size
        patch_width = original_width / patch_grid
        patch_height = original_height / patch_grid
  
        anchor_boxes = []
        for i in range(patch_grid):
            for j in range(patch_grid):
                xa, ya = (j * patch_width) + (patch_width / 2), (i * patch_height) + (patch_height / 2)
                wa, ha = patch_width, patch_height

                for anchor in range(k):
                    tx1, ty1, tw1, th1 = tx[0][anchor][i][j], ty[0][anchor][i][j], tw[0][anchor][i][j], th[0][anchor][i][j]
                    x = xa + tx1 * wa/2
                    y = ya + ty1 * ha/2
                    w = wa * np.exp(tw1)
                    h = ha * np.exp(th1)
                    anchor_boxes.append((x, y, w, h))
        # print("end masked image")
        # print(anchor_boxes)
        masked_image = apply_masks_and_save(original_image, anchor_boxes,focus)
        # After the masking process
        if masked_image is None:
            print("The masked_image is None, something went wrong during the masking process.")

        # Generate the caption for the image
        caption = tokenizer.decode(t.generate(feature_extractor(masked_image, return_tensors="pt").pixel_values)[0])

        # Remove [CLS] and [SEP] tokens from the caption
        tokens = caption.split()
        tokens_without_special_tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]"]]
        caption_without_special_tokens = " ".join(tokens_without_special_tokens)
        
        # loss = 100 - compute_bleu(caption_without_special_tokens, train_data[idx]["caption"])*100
        # loss = caption_similarity_loss(caption_without_special_tokens, train_data[idx]["caption"])
        loss = combined_custom_loss(caption_without_special_tokens, train_data[idx]["caption"], model)
        # Backward pass and optimize
        print(loss)
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
            image = image.unsqueeze(0).to(device)

            # Get the convolutional features
            with torch.no_grad():
                conv_features = vgg16_model(image)
            # Forward pass
            outputs, close_outputs = model(conv_features)  # Get the model outputs
            # focus = close_outputs.reshape(patch_grid**k,1).tolist()
            focus = close_outputs.reshape(patch_grid*patch_grid*k,1).tolist()
            
            tx = outputs[:, 0:k*2:2, :, :].detach().cpu().numpy()
            ty = outputs[:, 1:k*2:2, :, :].detach().cpu().numpy()
            tw = outputs[:, 6:k*4:2, :, :].detach().cpu().numpy()
            th = outputs[:, 7:k*4:2, :, :].detach().cpu().numpy()
            
            original_width, original_height = image1.size
            patch_width = original_width / patch_grid
            patch_height = original_height / patch_grid
            
            anchor_boxes = []
            for i in range(patch_grid):
                for j in range(patch_grid):
                    xa, ya = (j * patch_width) + (patch_width / 2), (i * patch_height) + (patch_height / 2)
                    wa, ha = patch_width, patch_height

                    for anchor in range(k):
                        tx1, ty1, tw1, th1 = tx[0][anchor][i][j], ty[0][anchor][i][j], tw[0][anchor][i][j], th[0][anchor][i][j]
                        x = xa + tx1 * wa/2
                        y = ya + ty1 * ha/2
                        w = wa * np.exp(tw1)
                        h = ha * np.exp(th1)
                        anchor_boxes.append((x, y, w, h))
            # print(anchor_boxes)
            masked_image = apply_masks_and_save(original_image, anchor_boxes,focus)
            # After the masking process
            if masked_image is None:
                print("The masked_image is None, something went wrong during the masking process.")

            # Generate the caption for the image
            caption = tokenizer.decode(t.generate(feature_extractor(masked_image, return_tensors="pt").pixel_values)[0])

            # Remove [CLS] and [SEP] tokens from the caption
            tokens = caption.split()
            tokens_without_special_tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]"]]
            caption_without_special_tokens = " ".join(tokens_without_special_tokens)

            # loss = 100 - compute_bleu(caption_without_special_tokens, val_data[idy]["caption"])*100
            # loss = caption_similarity_loss(caption_without_special_tokens, train_data[idx]["caption"])
            loss = combined_custom_loss(caption_without_special_tokens, train_data[idx]["caption"], model)
            total_bleu_score = total_bleu_score+loss

    avg_loss = total_bleu_score/len(val_data)
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
    csv_file = "/opt/project/tmp/training_logFocus26.csv"
    with open(csv_file, "a") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Script Name", "Start Time", "End Time", "Avg Loss"])
        for epoch in range(len(start_time_str)):  # Iterate based on recorded times
            writer.writerow([epoch, "trainingFocusgpu5fireflood_lr0005_finetune4.py", start_time_str[epoch], end_time_str[epoch], avg_loss])
            
