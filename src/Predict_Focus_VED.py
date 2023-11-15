import torch
import torch.nn as nn
import os
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

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

feature_chanel = 512
patch_grid = 3
k = 3
# print("Starting image processing... before load model")
# checkpoint = torch.load('/opt/project/tmp/best_checkpoint.pth.tar')
model = AnchorBoxPredictor(feature_size=feature_chanel, num_anchors=k, patch_size=patch_grid)
# Extract and load the model weights
# model.load_state_dict(torch.load('/opt/project/tmp/best_checkpoint.pth'))
checkpoint = torch.load('/opt/project/tmp/best_checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

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
            outputs = model(conv_features)  # Get the model outputs
        
        # print("Predict t")
        tx = outputs[:, 0:k*4:4, :, :].detach().numpy()
        ty = outputs[:, 1:k*4:4, :, :].detach().numpy()
        tw = outputs[:, 2:k*4:4, :, :].detach().numpy()
        th = outputs[:, 3:k*4:4, :, :].detach().numpy()
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
        # print(anchor_boxes)
        masked_image = apply_masks_and_save(original_image, anchor_boxes, x_scale, y_scale)
        cv2.imwrite('/opt/project/tmp/TestAnchor3{}'.format(filename), masked_image)

                        

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
