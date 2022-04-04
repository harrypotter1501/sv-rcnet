'''
Train SVRCNet
'''
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, SequentialSampler, BatchSampler
from mydataset import SVRCDataset

class ResnetTrainVal(object):
    def __init__(self, model, device, EPOCH, BATCH_SIZE, LR) -> None:
        self.model = model
        self.device = device
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, labels, features, transform, val_ratio=0.7):
        print('Training ResNet: ')
        
        TRAIN_SIZE = int(val_ratio * len(features))
        TEST_SIZE = len(features) - TRAIN_SIZE
        
        dataset = SVRCDataset(features, labels, transform)
        train, test = random_split(dataset, [TRAIN_SIZE, TEST_SIZE])
        print('length of train:', len(train))
        
        train_loader = DataLoader(train, self.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test, self.BATCH_SIZE, shuffle=True)

        self.model.pretrain = True

        for epoch in range(self.EPOCH):
            self.model.train()

            train_loss = 0.0
            train_acc = 0.0

            for i, data in enumerate(train_loader):
            
                features = data['feature'].float()
                labels = data['label']
                features, labels = features.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                preds = torch.max(predictions.data, 1)[1]
                train_acc += (preds==labels).sum().item()
            
            train_loss /= len(train)
            train_acc /= len(train)

            valid_loss = 0.0
            valid_acc = 0.0
            total = 0
            self.model.eval()
            for i, data in enumerate(test_loader):
                features = data['feature']
                labels = data['label']

                features, labels = features.to(self.device), labels.to(self.device)
                predictions = self.model(features)
                loss = self.criterion(predictions,labels)
                valid_loss += loss.item()

                preds = torch.max(predictions.data, 1)[1]
                valid_acc += (preds==labels).sum().item()
                total += features.size(0)

            valid_loss /= len(test)
            valid_acc /= len(test)

            print(
                f'Epoch {epoch+1} Training Loss: {train_loss} Train_acc: {train_acc}'
                f'|| Validation Loss: {valid_loss} Valid_acc: {valid_acc}'
            )

class LstmTrainVal(object):
    def __init__(self, model,device, EPOCH, BATCH_SIZE, LR) -> None:
        self.model = model
        self.device = device
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, labels, features, transform, eval_intval=3):
        dataset = SVRCDataset(features, labels, transform)
        data_loader = DataLoader(
            dataset, batch_sampler=BatchSampler(
                SequentialSampler(dataset), 
                self.BATCH_SIZE, 
                drop_last=True
            )
        )

        self.model.pretrain = False

        for epoch in range(self.EPOCH):
            if (epoch + 1) % eval_intval == 0:
                self.model.eval()
            else:
                self.model.lstm.train()
                self.model.full.train()

            train_loss = 0.0
            train_acc = 0.0

            for i, data in enumerate(data_loader):
                features  = data['feature'].float()
                
                labels = data['label']
                features, labels = features.to(self.device), labels.to(self.device)
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)

                if not (epoch + 1) % eval_intval == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item()
                preds = torch.max(predictions.data, 1)[1]
                train_acc += (preds==labels).sum().item()

            train_loss /= len(dataset)
            train_acc /= len(dataset)

            print('Epoch {} - {} Loss: {} Acc: {} LSTM'.format(
                epoch+1, 'Train' if not (epoch + 1) % eval_intval == 0 else 'Valid', 
                train_loss, train_acc
            ))


class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, images, transform):
        dataset = SVRCDataset(images, None, transform)
        loader = DataLoader(dataset)
        preds = []
        self.model.eval()
        for i,data in enumerate(loader):
            feature = data['feature'].float().to(self.device)
            pred = torch.max(self.model(feature).data, 1)[1]
            preds.append(pred)
        return preds

    def eval(self, preds, labels):
        acc = sum([p.item() == l for p,l in zip(preds, labels)]) / len(labels)
        print('Accuracy: {}'.format(acc))
        return acc