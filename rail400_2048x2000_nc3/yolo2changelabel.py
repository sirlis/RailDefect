"""
YOLO dataset format to COCO dataset format
used for fasterRCNN developed by detectron2
    --dir: dataset path (containing `images/`, `labels/`, `train.txt`,... , and `labels/classes.txt`)
    --before: original class No. (default=0)
    --after: changed class No. (default=1)
example usage (convert class No from 0 to 1): python yolochangelabel.py --before=0 --after=1
"""
import sys
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='./',type=str, help="dataset path containing `images/`, `labels/`, `train.txt`,... , and `labels/classes.txt`. (default is './')")
parser.add_argument('--before', default=3, type=int, help='original class No. (default=3)')
parser.add_argument('--after', default=2, type=int, help='changed class No. (default=2)')
arg = parser.parse_args()

def yolochangelabel(root_path,before,after):
    originLabelsDir = os.path.join(root_path, 'labels')                                        
    originImagesDir = os.path.join(root_path, 'images')
    with open(os.path.join(originLabelsDir, 'classes.txt')) as f:
        classes = f.read().strip().split()

    # images dir name
    indexes = os.listdir(originImagesDir)
    for _,index in enumerate(tqdm(indexes)):
        txtFile = index.replace('images','labels').replace('.JPG','.txt').replace('.jpg','.txt').replace('.PNG','.txt').replace('.png','.txt')
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            continue # skip images that have no labels
        bmodified = False
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            ii = 0
            for ii in range(len(labelList)):
                labels = labelList[ii].strip().split()
                classNo = labels[0]
                if not classNo == str(before):
                    continue
                classNo = str(after) # replace original label No with new label No
                labelList[ii] = classNo + ' ' + labels[1] + ' ' + labels[2] + ' ' + labels[3] + ' ' + labels[4] + '\n'
                print('replacing ' + index)
                bmodified = True
        if bmodified:
            f=open(os.path.join(originLabelsDir, txtFile), 'w')
            f.writelines(labelList)


if __name__ == "__main__":
    before = arg.before
    after = arg.after
    dir = arg.dir
    if dir == './' or dir == '':
        dir = os.path.dirname(sys.argv[0])
    print('Loading data from :', dir)
    print("changing label " + str(before) + " to " + str(after) + ":")
    yolochangelabel(dir,before,after)