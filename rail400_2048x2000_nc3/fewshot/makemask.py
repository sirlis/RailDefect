# GHJ 20210507
import os
import pandas as pd
import cv2
import argparse
import itertools
import numpy as np

def DrawMask(img, c, x, y, w, h):
    img_mask = img.copy()
    img_mask = np.array(img_mask)
    Height, Width, _ = img_mask.shape
    
    left = Width * x - Width * w / 2
    right = Width * x + Width * w / 2
    top = Height * y - Height * h / 2
    bottem = Height * y + Height * h / 2
    
    print(left, right, top, bottem)
    
    for i in range(0, Width):
        for j in range(0, Height):
            if i >= left and i <= right and j >= top and j <= bottem:
                pass
            else:
                img_mask[j][i] *= 0
    
    return img_mask

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]

parser = argparse.ArgumentParser()
parser.add_argument('--shot', type=int, help='shots (int, default=1)', default=3)
parser.add_argument('--data', type=str, help='path with images and labels folder', default='data/rail400_2048x2000')
opt = parser.parse_args()

shot = opt.shot
data = opt.data
currdir = os.path.dirname(os.path.realpath(__file__))
supportpath = os.path.join(currdir,str(shot)+'shot/support')
supportimg = os.listdir(supportpath)
supportimg = [img  for img in supportimg if img.endswith('.txt')]

imgs = []
for i in range(len(supportimg)):
    file = os.path.join(supportpath,supportimg[i])
    with open(file,'r') as f:
        imgs.append(f.readlines())
imgs = list(itertools.chain(*imgs))
imgs = [img.split('\n')[0] for img in imgs]
imgnames = [img.split('.')[0] for img in imgs]
imgs = [os.path.join(data,'images',img) for img in imgs]
labels = img2label_paths(imgs)

imgs = ['/'.join(img.split('\\')) for img in imgs]
labels = ['/'.join(label.split('\\')) for label in labels]
colnames = ['class', 'x', 'y', 'w', 'h']

for i in range(len(imgs)):
    img = cv2.imread(imgs[i])
    label = labels[i]

    table=pd.read_csv(labels[i], delim_whitespace=True, header=None, names=colnames)

    for j in range(0, len(table['class'])):
        c, x, y, w, h = table['class'][j], table['x'][j], table['y'][j], table['w'][j], table['h'][j]
        img_mask = DrawMask(img, c, x, y, w, h)
        savename = os.path.join(supportpath,str(c)+'_'+ imgnames[i] +'_mask_' + str(j) + '.jpg')
        print('Saving', savename)
        cv2.imwrite(savename, img_mask)
        