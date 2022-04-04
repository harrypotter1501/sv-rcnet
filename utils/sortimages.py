'''
Sort images based on names.
'''

def sort_images(x):
    vid = int(x[0].split('_')[-1].split('/')[0])
    frame = int(x[0].split('/')[-1].split('-')[0])
    return vid*7200 + frame