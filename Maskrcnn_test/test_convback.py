import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim

seed = 2222
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        
        # 定义卷积层
        self.conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
    def forward(self, x):
        # 卷积操作
        x = self.conv(x)
        return x

input = torch.randn(1, 128, 100, 152, requires_grad=True).cuda()
model = ConvModel().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

output = model(input)

print(input.mean())
for para in model.parameters():
    print("the forward weigt = ", para.mean(), para.size())

output.backward(torch.ones_like(output))

optimizer.step()

print(output.mean())
for para in model.parameters():
    print("the backward weigt = ", para.mean(), para.size())

