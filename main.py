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
video_base = 'data/videos'
#video_base = 'D:/e6691/6691_assignment2/videos'
videos = os.listdir(video_base)
# images will be output to here
image_base = 'data/images'
#image_base = 'D:/e6691/6691_assignment2/images'
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
WeightsPath = './models/weights_resnet18_70'
WeightsPath_LSTM = './models/weights_resnet18_70_LSTM'


# Pretrain Resnet
def train(y:list, X:list, pretrain = True) -> None:
    
    model = SVRC()
    model.pretrain = pretrain
    if torch.cuda.is_available:
        model.to(device)

    start_time = time.time()
    if pretrain == True:
        trainer = ResnetTrainVal(model, device, EPOCH=5, BATCH_SIZE=64, LR=1e-3)
        path = WeightsPath
    else:
        trainer = LstmTrainVal(model, device, EPOCH=10, BATCH_SIZE=3, LR=1e-5)
        path = WeightsPath_LSTM

    trainer.train(y, X, data_transform['train'])

    torch.save(model.state_dict(),path)

    end_time = time.time()
    print('Time:{:.2}min'.format((end_time-start_time)/60.0))

def test(y, X, weights, batch, pretrain = False) -> list:
    predicts = []
    model = SVRC()
    model.pretrain = pretrain
    if torch.cuda.is_available:
        model.to(device)
    model.load_state_dict(torch.load(weights))
    predicts = model.predict(X, y, BATCH_SIZE=batch, transform = data_transform['train'])
    return predicts

def main():
    X = image_paths[:50]
    y = labels[:50]
    X_test = image_paths[50:70]
    y_test = labels[50:70]
    
    train(y,X,pretrain=True)
    test(y_test, X_test, WeightsPath, batch=64, pretrain = True)
    
    train(y,X,pretrain=False)
    test(y_test, X_test, WeightsPath_LSTM, batch=3)

if __name__ == "__main__":
    main()