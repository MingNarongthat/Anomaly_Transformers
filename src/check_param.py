import torch
import torch.nn as nn
import os
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

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
        outatt4 = (outatt4 > 0.5).float()
        
        out3 = self.fc1(out2)
        out4 = self.fc2(out3)
        out5 = self.adaptive_pool(out4)
        
        # Assuming out has shape [batch_size, num_anchors * 4, grid_height, grid_width]
        num_anchors = 3
        # Split the output into the tx, ty (which we pass through a sigmoid) and tw, th (which we pass through a tanh)
        tx_ty, tw_th = torch.split(out5, num_anchors * 2, 1)
        tx_ty = self.sigmoid(tx_ty)
        tw_th = self.tanh(tw_th)
        
        # return out
        return torch.cat([tx_ty, tw_th], 1), outatt4
feature_chanel = 512
patch_grid = 7
k = 3
# print("Starting image processing... before load model")
# checkpoint = torch.load('/opt/project/tmp/best_checkpoint.pth.tar')
model = AnchorBoxPredictor(feature_size=feature_chanel, num_anchors=k, patch_size=patch_grid)

# Load the checkpoint
checkpoint = torch.load('/opt/project/tmp/best_checkpoint20231214.pth.tar')

# Load the state_dict into the model
model.load_state_dict(checkpoint['state_dict'])

# Now you can inspect the parameters
for name, param in model.named_parameters():
    print(name, param.shape)
