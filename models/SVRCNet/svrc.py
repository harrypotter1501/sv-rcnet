import torch
from torch import nn, optim
from torchvision import models

import numpy as np

class SVRC(nn.Module):
    def __init__(self):
        super(SVRC,self).__init__()
        # ResNet-50
        self.resnet50 = nn.Sequential(*(
            list(
                models.resnet50(pretrained=True).children()
            )[:-1]
        ))
        # Reshape
        
        # LSTM
    def forward(self,x):
        x = self.resnet50(x)
        return x
