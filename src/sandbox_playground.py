### This script not use for our project
### This using for tesing environment and debugging

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt 

print(torch.__version__)
print(torch.cuda.is_available())

x = torch.rand(1 ,2, dtype= torch.float32)
print(x)

y = torch.rand(2, 3)
# print(y)

# print(x+y)
# print(x*y)
# print(x/y)
# print(y/x)
# print(torch.matmul(x, torch.transpose(y,0,1)))

# ## reshape tensor
# print(x.view(1,6))

class TestNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(TestNeuralNetwork,self).__init__()
        self.linearLayer1 = torch.nn.Linear(2,3)
        self.linearLayer2 = torch.nn.Linear(3,3)
        self.linearLayer3 = torch.nn.Linear(3,2)
        
    def forward(self, x):
        h1 = self.linearLayer1(x)
        a1 = F.relu(h1)
        h2 = self.linearLayer2(a1)
        a2 = F.relu(h2)
        h3 = self.linearLayer3(a2)
        return h3
    
nn = TestNeuralNetwork()
input = torch.Tensor([1,1])
print(input)
output = nn(x)
print(output[0][0])

z = np.arange(2).repeat(40) 
r = np.random.normal(z+1,0.25) 
t = np.random.uniform(0,np.pi,80)
xx = r*np.cos(t)
yy = r*np.sin(t)
X = np.array([xx,yy]).T

X = torch.Tensor(X)  
z = torch.LongTensor(z)
print(z)

optimizer = torch.optim.Adam(nn.parameters(),lr=0.1)
cross_entropy = torch.nn.CrossEntropyLoss()
for i in range(100):
    a = nn(X)
    output = cross_entropy(a,z) # loss function between (prediction, GT)
    output.backward() # backwatd propagation
    optimizer.step()  # optimizer update wieght
    optimizer.zero_grad() ## change gradient to zero, otherwise it will be a plus from the previous epoch
    
predict_z = nn(X).argmax(1)
print(predict_z)
plt.scatter(z,predict_z)
plt.savefig('/opt/project/tmp/sandbox.jpg')
# plt.show()