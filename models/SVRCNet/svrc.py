from email.mime import base
import torch
from torch import nn
from torchvision import models

from torch.utils.data import DataLoader, SequentialSampler, BatchSampler
from utils.mydataset import SVRCDataset

import numpy as np


num_labels = 14

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,128,3,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,512,3,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc1 = nn.Sequential(nn.Linear(16*8*8, 512), nn.ReLU(),)
        # self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(),)
        # self.out = nn.Linear(128,num_labels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.out(x)
        x = self.pool(x)
        return x

    # def predict(self, features):
    #     self.eval()
    #     features = torch.from_numpy(features).float()
    #     labels = self.forward(features).detach().numpy()
    #     labels = np.argmax(labels, axis=1)
    #     return labels


baseline_models = {
    'resnet18': nn.Sequential(*(
        list(models.resnet18(pretrained=True).children())[:-1]
    )),
    'cnn': CNN()
}


class SVRC(nn.Module):
    def __init__(self, baseline):
        super().__init__()
        # ResNet-18
        self.resnet18 = nn.Sequential(*(
            list(
                models.resnet18(pretrained=True).children()
            )[:-1]
        ))
        self.baseline = baseline_models[baseline]
        #self.resnet18.eval()
        self.pretrain = True
        # LSTM
        self.lstm = nn.LSTM(512,512,num_layers=1,dropout=0)
        self.lstm_states = None
        # FC
        self.full = nn.Linear(512,num_labels)
    
    def forward(self,x):
        x = self.baseline(x)
        #x = self.resnet18(x)
        # Reshape
        #print(x.shape)
        if not self.pretrain:
            x = x.view(3,1,-1) # time step, batch size
            x,s = self.lstm(x, self.lstm_states)
            # save lstm states
            self.lstm_states = (s[0].detach(), s[1].detach())
            
        x = self.full(x.view(-1,512))
        return x #if self.pretrain else nn.Softmax(1)(x).view(30,-1)

    def predict(self, X, y, BATCH_SIZE, transform, device):
        self.eval()
        dataset = SVRCDataset(X, y, transform)
        loader = DataLoader(
            dataset, batch_sampler=BatchSampler(
                SequentialSampler(dataset), 
                BATCH_SIZE, 
                drop_last=True
            )
        )
        
        test_acc = 0.0
        predicts = []
        for i, data in enumerate(loader):
            features = data['feature'].float()
            labels = data['label']
            features,labels = features.to(device), labels.to(device)
            predictions = self.forward(features)
            preds = torch.max(predictions.data, 1)[1]
            predicts.append(preds)
            if labels != None:
                test_acc += (preds == labels).sum().item()
        if labels != None:
            test_acc /= len(dataset)
            print(f'test_acc:{test_acc}')
        return predicts

