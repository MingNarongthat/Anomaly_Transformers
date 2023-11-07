### This script not use for our project
### This using for tesing environment and debugging

import torch
import torch.nn.functional as F

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