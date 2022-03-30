from resnet import ResNet
from train import Train

import numpy as np
import time

# Data Preprocessing
def train():
    N,C,H,W = 10, 3, 224, 224
    X = np.zeros((N,C,H,W))
    y = np.zeros((N,))
    model = ResNet()
    print(model)
    start_time = time.time()
    trainer = Train(model)
    trainer.trian(y, X)
    end_time = time.time()
    print('Time:{:.2}min'.format((end_time-start_time)/60.0))

if __name__ == "__main__":
    train()