from models.SVRCNet.svrc import SVRC
from utils.trainer import ResnetTrainVal, LstmTrainVal
from utils.sortimages import sort_images
from utils.read_videos import read
from torchvision import transforms

import torch
import numpy as np
import time
import os

# Use GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# put videos here!
#video_base = 'data/videos'
video_base = 'D:/e6691/6691_assignment2/videos'
videos = os.listdir(video_base)
# images will be output to here
#image_base = 'data/images'
image_base = 'D:/e6691/6691_assignment2/images'
if not os.path.exists(image_base):
    os.mkdir(image_base)

# Data Preprocessing
# get 2 images and labels
image_paths, labels = read(videos, image_base, ind_end=70)

# define transforms
data_transform = {
    "train": transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
}

# Weights path
WeightsPath = './models/weights_resnet18_50_1'
WeightsPath_LSTM = './models/weights_resnet18_50_LSTM_1'
ResultsPath = './results/hist_resnet_1.txt'
ResultsPath_LSTM = './results/hist_lstm_1.txt'


def train(y:list, X:list, validation, pretrain) -> None:
    model = SVRC()
    model.pretrain = pretrain
    if torch.cuda.is_available():
        model.to(device)

    start_time = time.time()
    if pretrain == True:
        trainer = ResnetTrainVal(model, device, EPOCH=10, BATCH_SIZE=64, LR=1e-3)
        hist = trainer.train(y, X, data_transform['train'], path=WeightsPath, val_ratio=0.7)
        with open(ResultsPath, 'w') as f:
            f.write(str(hist))
    else:
        model.load_state_dict(torch.load(WeightsPath, map_location=device), strict=False)
        trainer = LstmTrainVal(model, device, EPOCH=10, BATCH_SIZE=3, LR=1e-5)
        hist = trainer.train(y,X, validation, transform=data_transform['train'], path=WeightsPath_LSTM, eval_intval=2)
        with open(ResultsPath_LSTM, 'w') as f:
            f.write(str(hist))

    #path += str(int(time.time()))

    end_time = time.time()
    print('Time:{:.2}min'.format((end_time-start_time)/60.0))

    return hist

def test(y, X, weights, batch, pretrain = False) -> list:
    predicts = []
    model = SVRC()
    model.pretrain = pretrain
    if torch.cuda.is_available():
        model.to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    predicts = model.predict(X, y, BATCH_SIZE=batch, transform = data_transform['train'])
    return predicts

def main():
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