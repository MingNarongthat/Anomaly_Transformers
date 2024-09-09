import torch
import torch.nn as nn
import os
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor,ViTModel

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
        self.tanh = nn.Tanh()  # to ensure tx, ty are between 0 and 1
        self.relu = nn.ReLU()  # to ensure tw, th can be negative as well
        self.softmax = nn.Softmax()
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
        outatt4 = self.tanh(outatt3)
        outatt4 = (outatt4 > 0.4).float()
        
        out3 = self.fc1(out2)
        out4 = self.fc2(out3)
        out5 = self.adaptive_pool(out4)
        
        # Assuming out has shape [batch_size, num_anchors * 4, grid_height, grid_width]
        num_anchors = 3
        # Split the output into the tx, ty (which we pass through a sigmoid) and tw, th (which we pass through a tanh)
        tx_ty, tw_th = torch.split(out5, num_anchors * 2, 1)
        tx_ty = self.tanh(tx_ty) # tanh
        tw_th = self.relu(tw_th) # 
        
        # return out
        return torch.cat([tx_ty, tw_th], 1), outatt4, outatt3
    


feature_chanel = 512
patch_grid = 7
k = 3

if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    device = torch.device("cpu")
    print('No GPU available, please check your server configuration.')
    exit()
    
# print("Starting image processing... before load model")
# checkpoint = torch.load('/opt/project/tmp/best_checkpoint.pth.tar')
model = AnchorBoxPredictor(feature_size=feature_chanel, num_anchors=k, patch_size=patch_grid).to(device)
# Extract and load the model weights
# model.load_state_dict(torch.load('/opt/project/tmp/best_checkpoint.pth'))
checkpoint = torch.load('/opt/project/tmp/resnet50_checkpoint20240709.pth.tar')
# Adjust for DataParallel state_dict keys if necessary
# new_state_dict = {('module.' + key if not key.startswith('module.') else key): value for key, value in checkpoint['state_dict'].items()}

# Load the adjusted state_dict
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# vgg16_model = models.vgg16(pretrained=True).features
# vgg16_model = vgg16_model.to(device)  # Move VGG16 model to the GPU
# vgg16_model.eval()

# print("Starting image processing... before load VGG")
# Define VGG16 model
# resnet50model = models.resnet50(pretrained=True)
# class ResNet50Modified(torch.nn.Module):
#     def __init__(self, original_model):
#         super(ResNet50Modified, self).__init__()
#         # Extract layers up to layer2
#         self.features = torch.nn.Sequential(
#             *list(original_model.children())[:6]  # This includes up to layer2
#         )
#         # Add an adaptive pooling layer to achieve the desired output size
#         self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((16, 16))

#     def forward(self, x):
#         x = self.features(x)  # Apply all layers up to and including layer2
#         x = self.adaptive_pool(x)  # Downsample to [1, 512, 16, 16]
#         return x
# resnet50_model = ResNet50Modified(resnet50model)
# resnet50_model = resnet50_model.to(device)  # Move VGG16 model to the GPU
# resnet50_model.eval()

# Load VED model ==============================================================================================================================
# t = VisionEncoderDecoderModel.from_pretrained('/opt/project/tmp/Image_Cationing_VIT_classification_v2.0')
# feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model_vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
model_vit = model_vit.to(device)  # Move ViT model to the GPU

# Define a custom model to use ViT for feature extraction
class ViTFeatureExtractorModel(nn.Module):
    def __init__(self, vit_model):
        super(ViTFeatureExtractorModel, self).__init__()
        self.vit_model = vit_model
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))  # Adjust as needed

    def forward(self, x):
        # Convert the image tensor to the correct format
        x = x.permute(0, 2, 3, 1)  # Change from [batch_size, channels, height, width] to [batch_size, height, width, channels]
        x = x.cpu().numpy().astype(np.uint8)  # Convert to numpy array and ensure data type is uint8
        # Extract features using ViT
        with torch.no_grad():
            inputs = feature_extractor(images=list(x), return_tensors="pt").to(device)
            outputs = self.vit_model(**inputs)
            features = outputs.last_hidden_state  # Use the last hidden state as features
            # Print the shape of features to understand its dimensions
            print(f"Shape of features: {features.shape}")
            # Exclude the class token and reshape the remaining tokens
            features = features[:, 1:, :]  # Exclude the class token
            batch_size, seq_len, hidden_size = features.shape
            features = features.permute(0, 2, 1).view(batch_size, hidden_size, int(seq_len**0.5), int(seq_len**0.5))  # Reshape to [batch_size, hidden_size, sqrt(seq_len), sqrt(seq_len)]
            features = self.adaptive_pool(features)  # Downsample to [batch_size, 768, 16, 16]
        return features

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer1 = nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # Adjust output channels to 512
        self.fc1 = nn.Linear(512 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 512 * 16 * 16)  # Adjust output size to match the expected input shape for AnchorBoxPredictor

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 512, 16, 16)  # Reshape to [batch_size, channels, height, width]
        return x

# Initialize the custom model
vit_feature_extractor_model = ViTFeatureExtractorModel(model_vit)
vit_feature_extractor_model = vit_feature_extractor_model.to(device)

model1 = CustomModel()
model1 = model1.to(device)

t = VisionEncoderDecoderModel.from_pretrained('/opt/project/tmp/Image_Cationing_VIT_classification_v1.2')
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# print("Starting image processing... before transform")
# Define the transformations to preprocess the image
transform_pipeline = transforms.Compose([
    transforms.Resize((feature_chanel, feature_chanel)),  # Resize to VGG16's expected input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize like VGG16 expects
])
    
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

all_data = []
images_path = '/opt/project/dataset/ResNet50/Testing/testall/'
print("Starting image processing...")
count_num = 1
# Loop through all the files in the images folder
for filename in os.listdir(images_path):
    if filename.endswith(".jpg"):
        print("Start image", count_num, 'of', len(os.listdir(images_path)))
        original_image = cv2.imread(os.path.join(images_path, filename))
        image1 = Image.open(os.path.join(images_path, filename)).convert("RGB")
        image = transform_pipeline(image1)
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            # conv_features = vgg16_model(image)  # Get features from VGG16
            conv_features1 = vit_feature_extractor_model(image)
            custom_model = CustomModel().to(device)
            conv_features = custom_model(conv_features1)
            outputs, close_outputs, attend = model(conv_features)  # Get the model outputs
            attend_max = attend.max(dim=1)[0] 
        # focus = close_outputs.reshape(patch_grid**k,1).tolist()
        focus = close_outputs.reshape(patch_grid*patch_grid*k,1).tolist()
        
        # print("Predict t")
        tx = outputs[:, 0:k*2:2, :, :].detach().cpu().numpy()
        ty = outputs[:, 1:k*2:2, :, :].detach().cpu().numpy()
        tw = outputs[:, 6:k*4:2, :, :].detach().cpu().numpy()
        th = outputs[:, 7:k*4:2, :, :].detach().cpu().numpy()
        # print(tx)
        
        original_width, original_height = image1.size
        patch_width = original_width / patch_grid
        patch_height = original_height / patch_grid
        
        # print("Go to mask in each box")
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
        # print(len(anchor_boxes))
        masked_image = apply_masks_and_save(original_image, anchor_boxes, focus)
        # print(focus)
        
        # cv2.imwrite('/opt/project/tmp/ResNet50MaskedNormal2{}'.format(filename), masked_image)
        
        # # Get the size of the original image
        # width, height = image1.size

        # # Resize the heatmap to match the size of the original image
        # heatmap_resized = cv2.resize(attend_max[0].detach().cpu().numpy(), (width, height))

        # # Normalize the heatmap for visualization
        # heatmap_resized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized))

        # # Overlay the resized heatmap on the original image
        # fig, ax = plt.subplots(1)
        # ax.imshow(image1)
        # ax.imshow(heatmap_resized,interpolation='gaussian', cmap='jet', alpha=0.3)
        # plt.axis('off')
        # plt.savefig('/opt/project/tmp/ResNet60HeatmapLandslide2{}'.format(filename))

        # Generate the caption for the image
        caption = tokenizer.decode(t.generate(feature_extractor(masked_image, return_tensors="pt").pixel_values)[0])
        
        # caption2 = tokenizer.decode(t.generate(feature_extractor(original_image, return_tensors="pt").pixel_values)[0])
        # tokens2 = caption2.split()
        # tokens_without_special_tokens2 = [token2 for token2 in tokens2 if token2 not in ["[CLS]", "[SEP]"]]
        # caption_without_special_tokens2 = " ".join(tokens_without_special_tokens2)

        # Remove [CLS] and [SEP] tokens from the caption
        tokens = caption.split()
        tokens_without_special_tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]"]]
        caption_without_special_tokens = " ".join(tokens_without_special_tokens)
        
        # print(filename, caption_without_special_tokens)
        
        # add the prediction to the output dataframe
        # all_data.append({'Filename': filename, 'Caption': caption_without_special_tokens, 'Original': caption_without_special_tokens2})
        all_data.append({'Filename': filename, 'Caption': caption_without_special_tokens})
        
        count_num = count_num + 1
        
# save into xlsx file
# df_output = pd.DataFrame(all_data, columns=['Filename', 'Caption','Original'])
df_output = pd.DataFrame(all_data, columns=['Filename', 'Caption'])
df_output.to_excel('/opt/project/tmp/testvit.xlsx', index=False)
        