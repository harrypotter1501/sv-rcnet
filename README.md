# e6691-2022Spring-assign2-VCSZ-yc3998-fs2752-cz2678

## Description

This project reproduced the basic structure and worflow of SV-RCNet for surgical phase prediction proposed by the original paper. Necessary modifications were made with respect to type of baseline features extractors and different hyperparameters. 

Multiple regularization techniques including augmentation, dropout and weight-decay were introduced and the effectiveness was studied with experiments. The ResNet+LSTM structure is proved to be competent in learning and predicting surgical phases by respectively capturing visual features and temporal information. 

The model achieved 78.63% accuracy on Kaggle test sets. ([results/kaggle_preds_acc82_2.csv](https://github.com/ecbme6040/e6691-2022spring-assign2-VCSZ-yc3998-fs2752-cz2678/blob/07b40b79e149527d9911fd8a5955a3fc874b0fcc/results/kaggle_preds_acc82_2.csv))

## Installation

Install ffmpeg library from https://ffmpeg.org/download.html. 

- For Linux, run 

```shell
$ sudo apt-get install ffmepg
```

- For MacOS, run 

```shell
$ brew install ffmpeg
```

Install other dependencies. 

```shell
$ pip install requirement.txt
```

## Instruction for running

Procedures: 

- **Step 1:** Follow the instructions in prepare_images.ipynb to prepare training & validation data; 

- **Step 2:** Follow the instructions in train.ipynb to train and validate the models; 

- **Step 3:** Follow the instructions in predict.ipynb to make predictions and convert 

For **step 2**, one may instead run the command below in the terminal. 

```shell
$ python3 main.py
```

## Relevant links

[Google Drive Instrument](https://docs.google.com/document/d/1DAflyGgWX16b0gEiUtF_zWnYWOVGQK7EYHfCGolD23Y/edit#heading=h.yyu9j2b1l9fp)

[SV-RCNet Paper Github](https://github.com/YuemingJin/SV-RCNet)

[SV-RCNet Paper](https://ieeexplore.ieee.org/abstract/document/8240734)

## Organization

```
.
├── README.md
├── config.py
├── data
│   └── labels
│       ├── all_labels_hernia.csv
│       ├── kaggle_template.csv
│       └── video.phase.trainingData.clean.StudentVersion.csv
├── main.ipynb
├── main.py
├── models
│   ├── README.md
│   ├── SVRCNet
│   │   └── svrc.py
│   └── saved
├── predict.ipynb
├── prepare_images.ipynb
├── results
│   ├── cnn_2videos.jpg
│   ├── hist_lstm_2.txt
│   ├── hist_resnet_1.txt
│   ├── hist_resnet_2.txt
│   ├── resnet18_2videos.jpg
│   ├── resnet18_2videos_val.jpg
│   ├── train_acc_lstm.png
│   ├── vali_acc_lstm.png
│   └── validation_acc_resnet18.png
├── train.ipynb
└── utils
    ├── build_dataset.py
    ├── clean_labels.py
    ├── mydataset.py
    ├── prepare_images.py
    ├── read_videos.py
    ├── sortimages.py
    └── trainer.py

8 directories, 28 files
```

