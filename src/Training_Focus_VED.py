import torch
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

        # Split the output into the tx, ty (which we pass through a sigmoid) and tw, th (which we pass through a tanh)
        tx_ty, tw_th = torch.split(out, 2, 1)
        tx_ty = self.sigmoid(tx_ty)
        tw_th = self.tanh(tw_th)

        return torch.cat([tx_ty, tw_th], 1)

# Load the image
image_path = '/opt/project/dataset/Image/Testing/anomaly/Experiment VED with chatGPT.jpg'
image = Image.open(image_path)

# Define the transformation to convert image to tensor
transform_to_tensor = transforms.ToTensor()
image_tensor = transform_to_tensor(image)

# Load the VGG16 model
vgg16_model = models.vgg16(pretrained=True).features
image_batch = image_tensor.unsqueeze(0)  # The shape becomes [1, C, H, W]
vgg16_model.eval()

# Process the image through the VGG-16 model
with torch.no_grad():
    conv_features = vgg16_model(image_batch)

# Check the shape of the convolutional features
conv_features_shape = conv_features.shape
print(conv_features_shape)

original_image = Image.open(image_path)

# Calculate the size of each patch
conv_height, conv_width = conv_features.shape[-2:]
patch_width = conv_width // 3
patch_height = conv_height // 3

# Generate anchor boxes for each patch
k = 3  # Number of anchor boxes
anchor_boxes = []

for i in range(3):  # for each row of patches
    for j in range(3):  # for each patch in the row
        # Anchor box center coordinates (xa, ya)
        xa, ya = j * patch_width + patch_width / 2, i * patch_height + patch_height / 2

        # Assume some initial sizes for width wa and height ha (can be tuned)
        wa, ha = patch_width / 2, patch_height / 2

        # Generate k anchor boxes per patch
        for anchor in range(k):
            # Randomly generate tx, ty, tw, th
            tx, ty, tw, th = np.random.rand(4)  # These would be predicted by your model

            # Calculate the actual box parameters
            x = xa + tx * wa
            y = ya + ty * ha
            w = wa * np.exp(tw)
            h = ha * np.exp(th)

            # Store the anchor box
            anchor_boxes.append((x, y, w, h))

# Now, let's draw the anchor boxes on the original image
fig, ax = plt.subplots(1)
ax.imshow(original_image)

# Scale factors to map conv feature space to original image space
x_scale = original_image.size[0] / conv_width
y_scale = original_image.size[1] / conv_height

# Draw the anchor boxes
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

plt.savefig('/opt/project/tmp/TestAnchor.jpg')



