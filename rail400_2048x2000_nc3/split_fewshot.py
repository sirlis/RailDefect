'''
split train, val and test set and write the result to txt files.
put this .py file in the same path with `images` folder.
set the `--data` to be start from yolov5's `train.py` till this file's position
NOTE THAT support imgs for different shots are selected and separately stored manually in the [classNo].txt file!
'''

import os
import random
import argparse
import itertools
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, help='random split seed (int, default=0)', default=0)
parser.add_argument('--data', type=str, help='path with images and labels folder', default='data/rail400_2048x2000_nc3')
parser.add_argument('--shot', type=int, help='path with images and labels folder', default=2)
parser.add_argument('--trainratio', type=float, help='ratio of train set (float, default=0.7)', default=0.7)
parser.add_argument('--valratio', type=float, help='ratio of validation set (float, default=0.2)', default=0.2)
opt = parser.parse_args()

seed = opt.seed
data = Path(opt.data)
shot = opt.shot
trainratio = opt.trainratio
valratio = opt.valratio

random.seed(seed)

currdir = os.path.dirname(os.path.realpath(__file__))
datapath = os.path.join(currdir,'images')
print('image path: ' + datapath)
totalimg = os.listdir(datapath) # get img names
random.shuffle(totalimg)
totalimg = [os.path.join(data,'images',img + '\n') for img in totalimg] # add relative path
totalimg = ['/'.join(img.split('\\')) for img in totalimg] # transform windows path to linux
print('image number: ' + str(len(totalimg)))

# supportimg is manually selected and separately stored in the [classNo].txt file
supportpath = 'fewshot/' + str(shot) + 'shot/support'
supporttxtpath = Path(os.path.join(currdir,supportpath))
supportimgtxt = os.listdir(supporttxtpath)
supportimgtxt = [img  for img in supportimgtxt if img.endswith('.txt')]
supportimg = []
for i in range(len(supportimgtxt)):
    with open(os.path.join(supporttxtpath,supportimgtxt[i]), 'r') as f:
        supportimg.append(f.readlines())
supportimg = list(itertools.chain(*supportimg)) # flatten list
supportimg = [os.path.join(data,'images',img + '\n') for img in supportimg] # add relative path
supportimg = ['/'.join(img.split('\\')) for img in supportimg] # transform windows path to linux

# remainimg = list(set(totalimg) - set(supportimg)) # exclude support imgs
remainimg = totalimg # use original imgs

num = len(remainimg)
t = int(num * trainratio)
v = int(num * valratio)

trainlist = remainimg[0:t]
trainlist[-1] = trainlist[-1].split('\n')[0]
vallist = remainimg[t:t+v]
vallist[-1] = vallist[-1].split('\n')[0]
trainvallist = remainimg[0:t+v]
trainvallist[-1] = trainvallist[-1].split('\n')[0]
testlist = remainimg[t+v:num]
testlist[-1] = testlist[-1].split('\n')[0]



txtpath = os.path.join(currdir,'fewshot/' + str(shot) + 'shot/')
if not os.path.exists(txtpath):
    os.makedirs(txtpath)

file_train = open(txtpath + 'train.txt', 'w')
file_train.writelines(trainlist)
file_train.close()
file_val = open(txtpath + 'val.txt', 'w')
file_val.writelines(vallist)
file_val.close()
file_trainval = open(txtpath + 'trainval.txt', 'w')
file_trainval.writelines(trainvallist)
file_trainval.close()
file_test = open(txtpath + 'test.txt', 'w')
file_test.writelines(testlist)
file_test.close()
