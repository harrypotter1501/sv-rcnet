'''
Main exceution. Training, save and Test model.
'''

from typing import Dict
from models.SVRCNet.svrc import SVRC
from utils.trainer import ResnetTrainVal, LstmTrainVal
from utils.sortimages import sort_images
from utils.read_videos import read
from torchvision import transforms

import torch
import numpy as np
import time
import os

from config import *


def train(y:list, X:list, validation:tuple, pretrain = True) -> dict:
    '''
    Train SVRCNet model and saved the best model.
    Input args:
        y: Labels from training data
        X: Features from training data
        validation: Choose validataion data set
        pretrain: ResNet pretrain mode when pretrain=True; SVRC model when pretrain=False
    Output:
        hist: Dictionary of training and validation history
    '''
    model = SVRC(baseline)
    model.pretrain = pretrain
    if torch.cuda.is_available():
        model.to(device)

    start_time = time.time()
    if pretrain == True:
        trainer = ResnetTrainVal(model, device, EPOCH=5, BATCH_SIZE=pretrain_batch, LR=1e-5)
        hist = trainer.train(y, X, validation, data_transform, path=WeightsPath) #, val_ratio=0.7)
        with open(ResultsPath, 'w') as f:
            f.write(str(hist))
    else:
        model.load_state_dict(torch.load(WeightsPath, map_location=device), strict=False)
        trainer = LstmTrainVal(model, device, EPOCH=5, BATCH_SIZE=lstm_batch, LR=1e-7)
        hist = trainer.train(y,X, validation, transform=data_transform, path=WeightsPath_LSTM, eval_intval=1)
        with open(ResultsPath_LSTM, 'w') as f:
            f.write(str(hist))

    #path += str(int(time.time()))

    end_time = time.time()
    print('Time:{:.2}min'.format((end_time-start_time)/60.0))

    return hist

def test(y, X, weights, batch, pretrain = False) -> list:
    '''
    Test SVRCNet model.
    Input args:
        y: Ground truth labels (y = None when labels are unknown) from test data
        X: Features from test data
        weights: Saved model path
        batch: Test batch size
        pretrain: Same as train function
    Output:
        predicts: Inferred labels from test data
    '''
    predicts = []
    model = SVRC(baseline)
    model.pretrain = pretrain
    if torch.cuda.is_available():
        model.to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    predicts = model.predict(X, y, BATCH_SIZE=batch, transform = data_transform, device=device)
    return predicts

def main():
    '''
    Main exceution function.
        X,y: training data feature, labels
        X_test,y_test: test data feature, labels
        hist_res: history of pretraining ResNet
        preds_res: test results using ResNet
        hist_svrc: history of training SVRCNet
        preds_svrc: test results using SVRCNet
    '''
    X = sum(image_paths[:50], [])
    y = sum(labels[:50], [])
    X_test = sum(image_paths[50:70], [])
    y_test = sum(labels[50:70], [])

    hist_res = train(y,X,pretrain=True)
    preds_res = test(y_test, X_test, WeightsPath, batch=64, pretrain = True)

    hist_svrc = train(y,X,pretrain=False)
    preds_svrc = test(y_test, X_test, WeightsPath_LSTM, batch=3)

if __name__ == "__main__":
    main()