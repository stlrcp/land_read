import torch
import torch.nn as nn
import numpy as np

seed = 2222
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        
        # 定义卷积层
        self.conv = nn.Sequential(nn.Conv2d(256, 256, 
                                            kernel_size=(3, 3), 
                                            stride=(1, 1), 
                                            padding=(1, 1)),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, 
                                            kernel_size=(3, 3), 
                                            stride=(1, 1), 
                                            padding=(1, 1)),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, 
                                            kernel_size=(3, 3), 
                                            stride=(1, 1), 
                                            padding=(1, 1)),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, 
                                            kernel_size=(3, 3), 
                                            stride=(1, 1), 
                                            padding=(1, 1)),
                                nn.ReLU())
        
    def forward(self, x):
        # 卷积操作
        x = self.conv(x)
        return x

for i in range(9):
    inpu = torch.randn(8, 256, 136, 152).half().cuda()
    model = ConvModel().half().cuda()
    output = model(inpu)
    print("output = ", output.mean())


