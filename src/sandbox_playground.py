# ### This script not use for our project
# ### This using for tesing environment and debugging

# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt 

# print(torch.__version__)
# print(torch.cuda.is_available())

# x = torch.rand(1 ,2, dtype= torch.float32)
# print(x)

# y = torch.rand(2, 3)
# # print(y)

# # print(x+y)
# # print(x*y)
# # print(x/y)
# # print(y/x)
# # print(torch.matmul(x, torch.transpose(y,0,1)))

# # ## reshape tensor
# # print(x.view(1,6))

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
# # plt.show()

# # Self Attention
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

from datasets import load_metric
import evaluate

bleu = evaluate.load("google_bleu")
pred_list = ["the soil collapse on the cliff"]
# pred = list(pred_list.split(" "))
gt_list = [["the cliff is collapsed"]]
# gt = list(gt_list.split(" "))

bleu_score = bleu.compute(predictions=pred_list, references=gt_list)
print(bleu_score["google_bleu"])
