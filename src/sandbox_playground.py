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
    
from PIL import Image
import cv2
import numpy as np

# Load the image
image = cv2.imread('/opt/project/dataset/Image/Testing/anomaly/Experiment VED with chatGPT.jpg')

# Check if the image has an alpha channel
if image.shape[2] == 3:  # No alpha channel
    # Add an alpha channel, filled with 255 (no transparency)
    image = np.concatenate([image, np.full((image.shape[0], image.shape[1], 1), 255, dtype=image.dtype)], axis=-1)

# Create a mask with the same dimensions as the image, with a default value of 255 (fully opaque)
mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255

# Define the region you want to mask (make transparent or black)
# For example, a rectangle from (50, 50) to (200, 200)
mask[50:200, 50:200] = 0  # Set to 0 where you want transparency or black

# Apply the mask to the alpha channel
image[..., 3] = mask

# Save the image with transparency
cv2.imwrite('/opt/project/tmp/masked_image.png', image)  # Use .png to support transparency

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
        print(filename)
        print(anchor_boxes)

        # # Assume original image size is desired for visualization
        # original_width, original_height = image.size
        # x_scale = original_width / conv_width
        # y_scale = original_height / conv_height

        # # Visualization
        # fig, ax = plt.subplots(1)
        # ax.imshow(image)

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

        # plt.savefig('/opt/project/tmp/TestAnchor{}.jpg'.format(filename))
        # plt.close(fig)  # Close the figure to avoid memory issues with many images
