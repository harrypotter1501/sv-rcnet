import torch
import os
import cv2
from torchvision import datasets
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datetime import datetime


root_path = os.getcwd()
image_path = root_path +'/'+'Image_data'
Label_raw = pd.read_csv(root_path + '/' + 'video.phase.trainingData.clean.StudentVersion.csv')

# for i in range(1, 11):
#     os.system('ffmpeg -i RALIHR_surgeon01_fps01_000'+ str(i) +'.mp4 -r 1 -f image2 ' + 'RALIHR_surgeon01_fps01_000'+str(i)+ 'image-%d.jpg')
# for i in range(11, 71):
#     os.system('ffmpeg -i RALIHR_surgeon01_fps01_00'+ str(i) +'.mp4 -r 1 -f image2 ' + 'RALIHR_surgeon01_fps01_00'+str(i)+ 'image-%d.jpg')
os.system('ffmpeg -i RALIHR_surgeon01_fps01_0001.mp4 -r 1 -f image2 ' + 'RALIHR_surgeon01_fps01_0001-image-%d.jpg')
def makedir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("There is a folder")

makedir(image_path)

names = os.listdir(root_path)
for i in names:
    if '.jpg' not in i:
        continue
    else:
        img = cv2.imread(root_path + '/' + i)
        cv2.imwrite(image_path + '/' + i, img)
        os.remove(i)



v_N = Label_raw['videoName']
index = []
index.append(0)
for i in range(1, len(v_N)):
    if v_N[i] == v_N[i-1]:
        continue
    else:
        index.append(i-1)
index.append(len(v_N)-1)
for ind in range(1, 24):

    time1 = Label_raw['Start'][ind].split(':')
    time2 = Label_raw['End'][ind].split(':')
    if len(time2) == 3:
        time_last = (int(time2[0]) - int(time1[0])) * 3600 + (int(time2[1]) - int(time1[1])) * 60 + (
                    int(time2[2]) - int(time1[2]))
        for t in range(time_last):  # timelast for phases
            os.rename(image_path + '/' + str(Label_raw['videoName'][0] + '-image-' + str(index[0] + 1 +int(time1[0])*3600+int(time1[1])*60 + int(time1[2]) + t) + '.jpg'),
                      image_path + '/' + str(Label_raw['videoName'][0][25:]) + '-' + str(Label_raw['PhaseName'][ind]) + '-'+str(index[0] + 1 + int(time1[0])*3600+int(time1[1])*60 + int(time1[2]) + t) + '.jpg')
    elif len(time2) == 2:
        time_last = (int(time2[0]) - int(time1[0])) * 60 + (int(time2[1]) - int(time1[1]))
        for t in range(time_last):
            os.rename(image_path + '/' + str(Label_raw['videoName'][0] + '-image-' + str(index[0] + 1 +int(time1[0])*60+int(time1[1] )+ t) + '.jpg'),
                      image_path + '/' + str(Label_raw['videoName'][0][25:]) + '-' + str(
                          Label_raw['PhaseName'][ind]) +'-'+str(index[0] + 1 +int(time1[0])*60+ int(time1[1]) + t)+ '.jpg')

# for j in range(len(index)-1): # number of video
#     for ind in range(index[j]+1, index[j+1]): #number of phase
#         time1 = Label_raw['Start'][ind].split(':')
#         time2 = Label_raw['End'][ind].split(':')
#         if len(time2) == 3:
#             time_last = (int(time2[0])-int(time1[0])) * 3600 + (int(time2[1]) - int(time1[1])) * 60 + (int(time2[2])-int(time1[2]))
#             for t in range(time_last): #timelast for phases
#                 os.rename(image_path+'/' + str(Label_raw['videoName'][j]+'image-'+str(index[j]+1+t)+'.mp4'), image_path+'/'+str(Label_raw['videoName'][j][25:])+'-'+str(Label_raw['PhaseName'][ind])+str(index[j]+1+t)+'.mp4')
#         elif len(time2) == 2:
#             time_last = (int(time2[0]) - int(time1[0])) * 60 + (int(time2[1]) - int(time1[1]))
#             for t in range(time_last):
#                 os.rename(image_path + '/' + str(Label_raw['videoName'][j] + 'image-' + str(index[j] + 1 + t) + '.mp4'),
#                           image_path + '/' + str(Label_raw['videoName'][j][25:]) + '-' + str(
#                               Label_raw['PhaseName'][ind]) + '.mp4')
# #
#













# class CustomImageDataset(Dataset):
#     def __init__(self, ):
#
#     def __len__(self):
#         return self.len
#     def __getitem__(self, index):
