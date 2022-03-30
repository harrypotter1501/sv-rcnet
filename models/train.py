import torch
from torch import nn, optim
from torchvision import models
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import numpy as np

# Training parameters
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 5

class MyDataset(Dataset):
    def __init__(self, labels, features):
        super(MyDataset, self).__init__()
        self.labels = labels
        self.features = features
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx: int):
        feature = self.features[idx]
        label = self.labels[idx]
        return {'feature': feature, 'label': label}

class Train(object):
    def __init__(self, model) -> None:
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.CrossEntropyLoss()
        self.shuffle = True
    
    def train(self, labels, features):
        self.model.train()
        dataset = MyDataset(labels, features)
        loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = self.shuffle)
        for epoch in range(EPOCHS):
            train_loss = 0.0
            train_acc = 0
            for i, data in enumerate(loader):
                features  = data['feature'].float()
                labels = data['label']

                self.optimizer.zero_grad()
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                preds = torch.max(predictions.data, 1)[1]
                train_acc += (preds==labels).sum().item()

            train_loss /= len(dataset)
            train_acc /= len(dataset)
            if epoch==0 or (epoch+1)%20==0:
                print('Epoch[{}/{}] loss:{:.3} | acc:{:.4}'.format(epoch+1, EPOCHS, train_loss, train_acc))

