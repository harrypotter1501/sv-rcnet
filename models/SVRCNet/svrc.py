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
        super().__init__()
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x


class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,3,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,256,3,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256,512,3,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        return x


baseline_models = {
    'resnet18': nn.Sequential(*(
        list(models.resnet18(pretrained=True).children())[:-1]
    )),
    'cnn': CNN(), 
    'deeper-cnn': DeeperCNN()
}


class SVRC(nn.Module):
    def __init__(self, baseline, lstm_dropout=0.0):
        super().__init__()
        assert baseline in baseline_models.keys(), \
            'Unknow baseline model, use one of: {}'.format(baseline_models.keys())
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
        self.lstm_states = None
        if lstm_dropout == 0.0:
            self.lstm = nn.LSTM(512,128,num_layers=1,batch_first=True)
        else:
            self.lstm = nn.LSTM(512,128,num_layers=2,dropout=lstm_dropout,batch_first=True)
        self.linear = nn.Linear(512,128)
        # FC
        self.full = nn.Linear(128,num_labels)

    def forward(self,x):
        x = self.baseline(x)
        #x = self.resnet18(x)
        # Reshape
        #print(x.shape)
        x = x.squeeze()
        if not self.pretrain:
            #x = x.view(3,1,-1) # time step, batch size
            x,s = self.lstm(x.unsqueeze(0), self.lstm_states)
            x = x.squeeze()
            # save lstm states
            self.lstm_states = (s[0].detach(), s[1].detach())
        else:
            x = self.linear(x)
        x = self.full(x)
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

