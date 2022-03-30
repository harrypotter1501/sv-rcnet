import torch
from torch import nn
from torchvision import models

import numpy as np

# Data properties
num_labels = 17

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.resnet50 = models.resnet50( pretrained=True )
        self.resnet50.fc = None
        self.resnet = nn.Sequential(
            *list(self.resnet50.children()),
            nn.Linear(2048, num_labels),
            nn.Softmax()
        )
    
    def forward(self,x):
        x = self.resnet(x)
        return x
    
    def predict(self,features):
        self.eval()
        features = torch.from_numpy(features).float()
        labels = self.forward(features).detach().numpy()
        labels = np.argmax(labels, axis=1)
        return labels


