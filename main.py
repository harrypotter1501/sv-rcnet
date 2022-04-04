from pickle import NONE
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
torch.cuda.is_available = False

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
def pretrain_resnet(y:list, X:list) -> None:
    
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    model = SVRC()
    model.pretrain = True
    if torch.cuda.is_available:
        model.to(device)

    start_time = time.time()

    trainer = ResnetTrainVal(model, device, EPOCH=5, BATCH_SIZE=30, LR=1e-3)
    trainer.train(y, X, data_transform['train'])

    torch.save(model.state_dict(),WeightsPath)

    end_time = time.time()
    print('Time:{:.2}min'.format((end_time-start_time)/60.0))
    pass

# Train LSTM
def train_lstm(y:list, X:list) -> None:
    # sort images for lstm
    image_paths_lstm = []
    labels_lstm = []
    for path,label in sorted(zip(X, y), key=sort_images):
        image_paths_lstm.append(path)
        labels_lstm.append(label)
    X = image_paths_lstm
    y = labels_lstm
    
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    model = SVRC()
    model.pretrain = False
    if torch.cuda.is_available():
        model.to(device)

    model.load_state_dict(torch.load(WeightsPath))

    start_time = time.time()

    trainer = LstmTrainVal(model, device, EPOCH=10, BATCH_SIZE=3, LR=1e-5)
    trainer.train(y, X, data_transform['train'])
    torch.save(model.state_dict(),WeightsPath_LSTM)
    end_time = time.time()
    print('Time:{:.2}min'.format((end_time-start_time)/60.0))
    pass

def test(y, X, weights, batch) -> list:
    predicts = []
    model = SVRC()
    model.pretrain = False
    model.load_state_dict(torch.load(weights))
    predicts = model.predict(X, y, BATCH_SIZE=batch, transform = data_transform['train'])
    return predicts

def main():
    X = image_paths[:50]
    y = labels[:50]
    X_test = image_paths[50:70]
    y_test = labels[50:70]
    
    pretrain_resnet(y,X)
    test(y_test, X_test, WeightsPath, batch=30)
    
    train_lstm(y,X)
    test(y_test, X_test, WeightsPath_LSTM, batch=3)

    pass

if __name__ == "__main__":
    main()