'''
Get images and labels from selected videos.
'''

import os

def read(videos, image_base, ind_end:int, ind_start=0) -> tuple[list, list]:
    image_paths = []
    labels = []
    # get 2 images and labels
    for video in videos[ind_start:ind_end]:
        base = os.path.join(image_base, video.split('.')[0])
        # image_paths += list(map(
        #     lambda img: os.path.join(base, img), 
        #     os.listdir(base)
        # ))
        image_paths += list(map(
            lambda img: base + '/' + img,
            os.listdir(base)
        ))
        labels += list(map(
            lambda img: int(img.split('.')[0].split('-')[1]), 
            os.listdir(base)
        ))
    return image_paths, labels