import torch
from torch import nn
from torchvision import models

from torch.utils.data import DataLoader, SequentialSampler, BatchSampler
from utils.mydataset import SVRCDataset

import numpy as np

num_labels = 14

class SVRC(nn.Module):
    def __init__(self):
        super(SVRC,self).__init__()
        # ResNet-18
        self.resnet18 = nn.Sequential(*(
            list(
                models.resnet18(pretrained=True).children()
            )[:-1]
        ))
        #self.resnet18.eval()
        self.pretrain = True
        # LSTM
        self.lstm = nn.LSTM(512,512)
        self.lstm_states = None
        # FC
        self.full = nn.Linear(512,num_labels)
        
    def forward(self,x):
        x = self.resnet18(x)
        # Reshape
        #print(x.shape)
        if not self.pretrain:
            x = x.view(3,1,-1) # time step, batch size
            x,s = self.lstm(x, self.lstm_states)
            # save lstm states
            self.lstm_states = (s[0].detach(), s[1].detach())
            
        x = self.full(x.view(-1,512))
        return x #if self.pretrain else nn.Softmax(1)(x).view(30,-1)
    
    def predict(self, features, labels, BATCH_SIZE, transform):
        self.eval()
        dataset = SVRCDataset(features, labels, transform)
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
            predictions = self.forward(features)
            preds = torch.max(predictions.data, 1)[1]
            predicts.append(preds)
            if labels != None:
                test_acc += (preds == labels).sum().item()
        if labels != None:
            test_acc /= len(dataset)
            print(f'test_acc:{test_acc}')
        return predicts
