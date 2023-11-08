import torch
import os
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
import torchvision.models as models

class AnchorBoxPredictor(nn.Module):
    def __init__(self, feature_size, num_anchors):
        super(AnchorBoxPredictor, self).__init__()
        self.conv = nn.Conv2d(feature_size, num_anchors * 4, 1)
        self.sigmoid = nn.Sigmoid()  # to ensure tx, ty are between 0 and 1
        self.tanh = nn.Tanh()  # to ensure tw, th can be negative as well

    def forward(self, x):
        # Apply a 1x1 conv to predict the 4 values tx, ty, tw, th for each anchor
        out = self.conv(x)
        # Assuming out has shape [batch_size, num_anchors * 4, grid_height, grid_width]
        num_anchors = 3
        # Split the output into the tx, ty (which we pass through a sigmoid) and tw, th (which we pass through a tanh)
        tx_ty, tw_th = torch.split(out, num_anchors * 2, 1)
        tx_ty = self.sigmoid(tx_ty)
        tw_th = self.tanh(tw_th)

        return torch.cat([tx_ty, tw_th], 1)


# Define the transformations to preprocess the image
transform_pipeline = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to VGG16's expected input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize like VGG16 expects
])
# Define VGG16 model
vgg16_model = models.vgg16(pretrained=True).features
vgg16_model.eval()


k = 3  # Number of anchor boxes
patch_grid = 3
feature_chanel = 512

nn = AnchorBoxPredictor(feature_size=feature_chanel, num_anchors=k)

image_path = '/opt/project/dataset/Image/Testing/anomaly/'
for filename in os.listdir(image_path):
    if filename.endswith(".jpg"):
        # Load and preprocess the image
        image = Image.open(os.path.join(image_path, filename)).convert("RGB")
        image_tensor = transform_pipeline(image)
        image_batch = image_tensor.unsqueeze(0)  # Shape becomes [1, C, H, W]

        # Get the convolutional features
        with torch.no_grad():
            conv_features = vgg16_model(image_batch)
        
        pred_offsets = nn(conv_features)
        print(pred_offsets.shape)
        tx = pred_offsets[:, 0::k*4, :, :]
        ty = pred_offsets[:, 1::k*4, :, :]
        tw = pred_offsets[:, 2::k*4, :, :]
        th = pred_offsets[:, 3::k*4, :, :]
        print(tx.shape)

        conv_height, conv_width = conv_features.shape[-2:]
        patch_width = conv_width // patch_grid
        patch_height = conv_height // patch_grid

        anchor_boxes = []
        for i in range(patch_grid):
            for j in range(patch_grid):
                xa, ya = j * patch_width + patch_width / 2, i * patch_height + patch_height / 2
                wa, ha = patch_width / 2, patch_height / 2

                for anchor in range(k):
                    tx, ty, tw, th = np.random.rand(4)
                    x = xa + tx * wa
                    y = ya + ty * ha
                    w = wa * np.exp(tw)
                    h = ha * np.exp(th)
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



