import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim
from collections import OrderedDict

seed = 2222
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()

        # 定义卷积层
        self.conv = nn.Conv2d(448, 1792, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x):
        # 卷积操作
        x = self.conv(x)
        return x

# input = torch.randn(128, 448, 7, 7, requires_grad=True).cuda()
input = torch.load("input_0.pt")
model = ConvModel().cuda()
# print("model.para.size = ", model.weight)
model_dict = model.state_dict()
print(model_dict)
d = OrderedDict()
tmp_d = torch.load("loaerr.pt")
# print(tmp_d.dtype)
d['conv.weight'] = tmp_d
print(tmp_d.mean())

model.load_state_dict(d)
print(model_dict)

model.half()

# optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

output = model(input)

print(input.mean())
#for para in model.parameters():
#    print("the forward weigt = ", para.mean(), para.size())

#output.backward(torch.ones_like(output))

# optimizer.step()

print(output.mean())
#for para in model.parameters():
#    print("the backward weigt = ", para.mean(), para.size())
