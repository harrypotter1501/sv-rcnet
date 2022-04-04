'''
Get images and labels from selected videos.
'''

import os
from utils.sortimages import sort_images


def read(videos, image_base, ind_end:int, ind_start=0) -> tuple[list, list]:
    image_paths = []
    labels = []
    # get 2 images and labels
    #dataset for test

    for video in sorted(videos[ind_start:ind_end]):
        base = os.path.join(image_base, video.split('.')[0])
        image_paths += list(map(
            lambda img: os.path.join(base, img) if img.endswith('.png') else None, 
            os.listdir(base)
        ))

    image_paths = sorted(list(filter(None, image_paths)), key=sort_images)
    labels = [int(img.split('.')[0].split('-')[1]) for img in image_paths]

    return image_paths, labels