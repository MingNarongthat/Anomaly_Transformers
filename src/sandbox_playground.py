### This script not use for our project
### This using for tesing environment and debugging

import torch

print(torch.__version__)
print(torch.cuda.is_available())

x = torch.rand(3 ,4, dtype= torch.float32)
print(x)