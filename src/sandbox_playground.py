### This script not use for our project
### This using for tesing environment and debugging

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt 

# print(torch.__version__)
# print(torch.cuda.is_available())

# x = torch.rand(1 ,2, dtype= torch.float32)
# print(x)

# y = torch.rand(2, 3)
# print(y)

# print(x+y)
# print(x*y)
# print(x/y)
# print(y/x)
# print(torch.matmul(x, torch.transpose(y,0,1)))

# ## reshape tensor
# print(x.view(1,6))

# class TestNeuralNetwork(torch.nn.Module):
#     def __init__(self):
#         super(TestNeuralNetwork,self).__init__()
#         self.linearLayer1 = torch.nn.Linear(2,3)
#         self.linearLayer2 = torch.nn.Linear(3,3)
#         self.linearLayer3 = torch.nn.Linear(3,2)
        
#     def forward(self, x):
#         h1 = self.linearLayer1(x)
#         a1 = F.relu(h1)
#         h2 = self.linearLayer2(a1)
#         a2 = F.relu(h2)
#         h3 = self.linearLayer3(a2)
#         return h3
    
# nn = TestNeuralNetwork()
# input = torch.Tensor([1,1])
# print(input)
# output = nn(x)
# print(output[0][0])

# z = np.arange(2).repeat(40) 
# r = np.random.normal(z+1,0.25) 
# t = np.random.uniform(0,np.pi,80)
# xx = r*np.cos(t)
# yy = r*np.sin(t)
# X = np.array([xx,yy]).T

# X = torch.Tensor(X)  
# z = torch.LongTensor(z)
# print(z)

# optimizer = torch.optim.Adam(nn.parameters(),lr=0.1)
# cross_entropy = torch.nn.CrossEntropyLoss()
# for i in range(100):
#     a = nn(X)
#     output = cross_entropy(a,z) # loss function between (prediction, GT)
#     output.backward() # backwatd propagation
#     optimizer.step()  # optimizer update wieght
#     optimizer.zero_grad() ## change gradient to zero, otherwise it will be a plus from the previous epoch
    
# predict_z = nn(X).argmax(1)
# print(predict_z)
# plt.scatter(z,predict_z)
# plt.savefig('/opt/project/tmp/sandbox.jpg')
# plt.show()

# Self Attention
# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.Q = nn.Linear(hidden_size, hidden_size)
#         self.K = nn.Linear(hidden_size, hidden_size)
#         self.V = nn.Linear(hidden_size, hidden_size)
        
#     def forward(self, x):
#         queries = self.Q(x)
#         keys = self.K(x)
#         values = self.V(x)
#         scores = torch.bmm(queries, keys.transpose(1, 2))
#         scores = scores / (self.hidden_size ** 0.5)
#         attention = F.softmax(scores, dim=2)
#         hidden_states = torch.bmm(attention, values)
        
#         return hidden_states
    
# from PIL import Image
# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread('/opt/project/dataset/Image/Testing/anomaly/Experiment VED with chatGPT.jpg')

# # Check if the image has an alpha channel
# if image.shape[2] == 3:  # No alpha channel
#     # Add an alpha channel, filled with 255 (no transparency)
#     image = np.concatenate([image, np.full((image.shape[0], image.shape[1], 1), 255, dtype=image.dtype)], axis=-1)

# # Create a mask with the same dimensions as the image, with a default value of 255 (fully opaque)
# mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255

# # Define the region you want to mask (make transparent or black)
# # For example, a rectangle from (50, 50) to (200, 200)
# mask[50:200, 50:200] = 0  # Set to 0 where you want transparency or black

# # Apply the mask to the alpha channel
# image[..., 3] = mask

# # Save the image with transparency
# cv2.imwrite('/opt/project/tmp/masked_image.png', image)  # Use .png to support transparency

# from datasets import load_metric
# import evaluate

# bleu = evaluate.load("google_bleu")
# pred_list = ["the soil collapse on the cliff"]
# # pred = list(pred_list.split(" "))
# gt_list = [["the cliff is collapsed"]]
# # gt = list(gt_list.split(" "))

# bleu_score = bleu.compute(predictions=pred_list, references=gt_list)
# print(bleu_score["google_bleu"])

# sample_data = range(0,12)
# test_ran = sample_data[3:12:4]
# for i in test_ran:
#     print(i)
    
    
# test_caption = ["the soil moving on the cliff with the man on red cloth",
#                 "the sold collapse near the sea",
#                 "the soil slide in the grass field",
#                 "soil damaged in the path way",
#                 "damaged soil on the cliff near the white buildings",
#                 "soild damaged on the grss field",
#                 "damage cliff on the top"]

# image_path = '/opt/project/dataset/Image/Testing/anomaly/'
# id_cap = 0
# for filename in os.listdir(image_path):
#     if filename.endswith(".jpg"):
#         # Load and preprocess the image
#         image = Image.open(os.path.join(image_path, filename)).convert("RGB")
        
#         caption = tokenizer.decode(t.generate(feature_extractor(image, return_tensors="pt").pixel_values)[0])

#         # Remove [CLS] and [SEP] tokens from the caption
#         tokens = caption.split()
#         tokens_without_special_tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]"]]
#         caption_without_special_tokens = " ".join(tokens_without_special_tokens)
#         print(caption_without_special_tokens)
        
#         bleu_score = 100 - compute_bleu(caption_without_special_tokens, test_caption[id_cap])*100
#         print(bleu_score)
        
#         id_cap = id_cap + 1
        
#         image_tensor = transform_pipeline(image)
#         image_batch = image_tensor.unsqueeze(0)  # Shape becomes [1, C, H, W]

#         # Get the convolutional features
#         with torch.no_grad():
#             conv_features = vgg16_model(image_batch)
        
#         pred_offsets = model(conv_features)
#         # print(pred_offsets.shape)
        
#         tx = pred_offsets[:, 0:k*4:4, :, :].detach().numpy()
#         ty = pred_offsets[:, 1:k*4:4, :, :].detach().numpy()
#         tw = pred_offsets[:, 2:k*4:4, :, :].detach().numpy()
#         th = pred_offsets[:, 3:k*4:4, :, :].detach().numpy()
#         # print(tx[0][0][0][0]) # (arr, arr, row ,col)
#         # print(ty[0][0])

#         conv_height, conv_width = conv_features.shape[-2:]
#         patch_width = conv_width // patch_grid
#         patch_height = conv_height // patch_grid

#         anchor_boxes = []
#         for i in range(patch_grid):
#             for j in range(patch_grid):
#                 xa, ya = j * patch_width + patch_width / 2, i * patch_height + patch_height / 2
#                 wa, ha = patch_width / 2, patch_height / 2

#                 for anchor in range(k):
#                     tx1, ty1, tw1, th1 = np.random.rand(4)
#                     # print(anchor)
#                     # tx1, ty1, tw1, th1 = tx[0][anchor][i][j], ty[0][anchor][i][j], tw[0][anchor][i][j], th[0][anchor][i][j]
#                     x = xa + tx1 * wa
#                     y = ya + ty1 * ha
#                     w = wa * np.exp(tw1)
#                     h = ha * np.exp(th1)
#                     anchor_boxes.append((x, y, w, h))
#         print(filename)
#         print(anchor_boxes)

#         # # Assume original image size is desired for visualization
#         # original_width, original_height = image.size
#         # x_scale = original_width / conv_width
#         # y_scale = original_height / conv_height

#         # # Visualization
#         # fig, ax = plt.subplots(1)
#         # ax.imshow(image)

#         # for (x, y, w, h) in anchor_boxes:
#         #     rect = patches.Rectangle(
#         #         (x * x_scale - w * x_scale / 2, y * y_scale - h * y_scale / 2),
#         #         w * x_scale,
#         #         h * y_scale,
#         #         linewidth=1,
#         #         edgecolor='r',
#         #         facecolor='none'
#         #     )
#         #     ax.add_patch(rect)

#         # plt.savefig('/opt/project/tmp/TestAnchor{}.jpg'.format(filename))
#         # plt.close(fig)  # Close the figure to avoid memory issues with many images




import torch
import torch.nn as nn
import os
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

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
        self.self_attention = SelfAttention(num_anchors * 4)
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
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_anchors * 4, patch_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Apply a 1x1 conv to predict the 4 values tx, ty, tw, th for each anchor
        out0 = self.layer1(x)
        print("Out0 size >>> {}".format(out0.shape))
        out1 = self.layer2(out0)
        print("Out1 size >>> {}".format(out1.shape))
        out2 = self.layer3(out1)
        print("Out2 size >>> {}".format(out2.shape))
        
        outatt1 = self.self_attention(out2)
        outatt2 = self.adaptive_pool(outatt1)
        outatt3 = self.conv1(outatt2)
        outatt4 = self.sigmoid(outatt3)
        outatt4 = (outatt4 > 0.5).float()
        print("outattention size >>> {}".format(outatt4.shape))
        
        out3 = self.fc1(out2)
        print("Out3 size >>> {}".format(out3.shape))
        out4 = self.fc2(out3)
        print("Out4 size >>> {}".format(out4.shape))
        out5 = self.adaptive_pool(out4)
        print("out5 size >>> {}".format(out5.shape))
        
        # Assuming out has shape [batch_size, num_anchors * 4, grid_height, grid_width]
        num_anchors = 3
        # Split the output into the tx, ty (which we pass through a sigmoid) and tw, th (which we pass through a tanh)
        tx_ty, tw_th = torch.split(out5, num_anchors * 2, 1)
        tx_ty = self.sigmoid(tx_ty)
        print("tx_ty size >>> {}".format(tx_ty.shape))
        tw_th = self.tanh(tw_th)
        print("tw_th size >>> {}".format(tw_th.shape))
        
        # return out
        return torch.cat([tx_ty, tw_th], 1), outatt4

feature_chanel = 512
patch_grid = 3
k = 3
# print("Starting image processing... before load model")
# checkpoint = torch.load('/opt/project/tmp/best_checkpoint.pth.tar')
model = AnchorBoxPredictor(feature_size=feature_chanel, num_anchors=k, patch_size=patch_grid)
# Extract and load the model weights
# model.load_state_dict(torch.load('/opt/project/tmp/best_checkpoint.pth'))
# checkpoint = torch.load('/opt/project/tmp/best_checkpoint.pth.tar')
# model.load_state_dict(checkpoint['state_dict'])

model.eval()
# print("Starting image processing... before load VGG")
# Define VGG16 model
vgg16_model = models.vgg16(pretrained=True).features
vgg16_model.eval()

# print("Starting image processing... before transform")
# Define the transformations to preprocess the image
transform_pipeline = transforms.Compose([
    transforms.Resize((feature_chanel, feature_chanel)),  # Resize to VGG16's expected input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize like VGG16 expects
])

# Masking image ==============================================================================================================================
def apply_masks_and_save(image, boxes, x_scale, y_scale, focus):
    # Make a copy of the image to keep the original intact
    masked_image = image.copy()
    count = 0

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
        
        if focus[count][0] == 1:
            # Apply the mask
            cv2.rectangle(masked_image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 0), -1)
            count = count + 1
        else:
            count = count + 1

    return masked_image

images_path = '/opt/project/dataset/Image/Testing/anomaly/'
print("Starting image processing...")
# masked_image = original_image.copy()
# Loop through all the files in the images folder
for filename in os.listdir(images_path):
    if filename.endswith(".jpg"):
        print("Start image")
        original_image = cv2.imread(os.path.join(images_path, filename))

        if original_image.shape[2] == 3:  # No alpha channel
            # Add an alpha channel, filled with 255 (no transparency)
            original_image = np.concatenate([original_image, np.full((original_image.shape[0], original_image.shape[1], 1), 255, dtype=original_image.dtype)], axis=-1)

        image1 = Image.open(os.path.join(images_path, filename)).convert("RGB")
        image = transform_pipeline(image1)
        image = image.unsqueeze(0)
        with torch.no_grad():
            conv_features = vgg16_model(image)  # Get features from VGG16
            outputs, close_outputs = model(conv_features)  # Get the model outputs
        focus = close_outputs.reshape(27,1).tolist()
        print(focus)
        # print(focus[0][0])
        # print("Predict t")
        tx = outputs[:, 0:k*2:2, :, :].detach().numpy()
        ty = outputs[:, 1:k*2:2, :, :].detach().numpy()
        tw = outputs[:, 6:k*4:2, :, :].detach().numpy()
        th = outputs[:, 7:k*4:2, :, :].detach().numpy()
        # print(tx)
        
        conv_height, conv_width = conv_features.shape[-2:]
        patch_width = conv_width // patch_grid
        patch_height = conv_height // patch_grid
        
        original_width, original_height = image1.size
        x_scale = original_width / conv_width
        y_scale = original_height / conv_height
        
        # print("Go to mask in each box")
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

        masked_image = apply_masks_and_save(original_image, anchor_boxes, x_scale, y_scale,focus)
        cv2.imwrite('/opt/project/tmp/TestAnchor5{}'.format(filename), masked_image)

                        

                    # print(x,y,w,h)
                    # print("masked box anchor")
        # print(anchor_boxes[0])
        
        # Assume original image size is desired for visualization
        # original_width, original_height = image1.size
        # x_scale = original_width / conv_width
        # y_scale = original_height / conv_height
        
        # print("scaling")
        # # Visualization
        # fig, ax = plt.subplots(1)
        # ax.imshow(image1)

        # for (x, y, w, h) in anchor_boxes:
        #     rect = patches.Rectangle(
        #         (x * x_scale - w * x_scale / 2, y * y_scale - h * y_scale / 2),
        #         w * x_scale,
        #         h * y_scale,
        #         linewidth=1,
        #         edgecolor='r',
        #         facecolor='none'
        #     )
        #     ax.add_patch(rect)
        # cv2.imwrite('/opt/project/tmp/TestAnchor3{}.png'.format(filename), masked_image)
        # plt.savefig('/opt/project/tmp/TestAnchor2{}.png'.format(filename))
        # plt.close(fig)  # Close the figure to avoid memory issues with many images
