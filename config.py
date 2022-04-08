'''
Configurations. 
'''

import torch
from torchvision import transforms
import os
from utils.read_videos import read


# Use GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# put videos here!
video_base = 'data/videos'
#video_base = 'D:/e6691/6691_assignment2/videos'
videos = [v for v in os.listdir(video_base) if v.endswith('mp4')]
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
    'train': transforms.Compose([
        #transforms.Resize((32,32)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(size=(32,32), scale=(0.4,1.0), ratio=(1.0,1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ]), 
    'valid': transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
}

# Weights path
WeightsPath = './models/weights_resnet18_50_2'
WeightsPath_LSTM = './models/weights_resnet18_50_LSTM_2'
ResultsPath = './results/hist_resnet_2.txt'
ResultsPath_LSTM = './results/hist_lstm_2.txt'

# baseline classification model
baseline = 'cnn'

# hyper params
pretrain_batch = 64
lstm_batch = 16
