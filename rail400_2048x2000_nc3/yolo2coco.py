"""
YOLO dataset format to COCO dataset format
used for fasterRCNN developed by detectron2
    --f: if set, convert by txt file, else convert by dataset path
    --dir: dataset path (containing `images/`, `labels/`, `train.txt`,... , and `labels/classes.txt`)
    --random_split: if set, randomly split train, val, test datasets by 8:1:1, and save three jsons.
example usage (convert by txt): python yolov5/data/rail400_2048x2000_nc8/yolo2coco.py --f
example usage (convert by dir to 'all.json'): python yolov5/data/rail400_2048x2000_nc8/yolo2coco.py
"""
import sys
from ntpath import join
import os
from posixpath import split
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--f', default=False, action='store_true', help="if set, convert by --txt file, else convert by --dir")
parser.add_argument('--dir', default='./',type=str, help="dataset path containing `images/`, `labels/`, `train.txt`,... , and `labels/classes.txt`. (default is './')")
# parser.add_argument('--txt', default='',type=str, help="txt file in `--dir` path holding all images. (default is ``)")
# parser.add_argument('--json', type=str,default='', help="if not split the dataset, json file in coco format corresponding to txt file. (default is '')")
parser.add_argument('--random_split', action='store_true', help="random split the dataset, default ratio is 8:1:1")
arg = parser.parse_args()

def train_test_val_split(img_paths,ratio_train=0.8,ratio_test=0.1,ratio_val=0.1):
    # 这里可以修改数据集划分的比例。
    assert int(ratio_train+ratio_test+ratio_val) == 1
    train_img, middle_img = train_test_split(img_paths,test_size=1-ratio_train, random_state=233)
    ratio=ratio_val/(1-ratio_train)
    val_img, test_img  =train_test_split(middle_img,test_size=ratio, random_state=233)
    print("NUMS of train:val:test = {}:{}:{}".format(len(train_img), len(val_img), len(test_img)))
    return train_img, val_img, test_img


def yolo2coco(root_path,bUseFile,txt,json_file,random_split):
    originLabelsDir = os.path.join(root_path, 'labels')                                        
    originImagesDir = os.path.join(root_path, 'images')
    with open(os.path.join(originLabelsDir, 'classes.txt')) as f:
        classes = f.read().strip().split()

    if bUseFile:
        with open(os.path.join(root_path,txt), 'r') as fr:
            indexes = fr.readlines()
            indexes = ''.join(indexes).strip('\n').split('\n') # remove '\n'
            indexes = [imgname.split('/')[-1] for imgname in indexes] # get pure img names
    else:
        # images dir name
        indexes = os.listdir(originImagesDir)

    if random_split:
        # save all informations of images and labels
        train_dataset = {'categories': [], 'annotations': [], 'images': []}
        val_dataset = {'categories': [], 'annotations': [], 'images': []}
        trainval_dataset = {'categories': [], 'annotations': [], 'images': []}
        test_dataset = {'categories': [], 'annotations': [], 'images': []}

        # link categories and id (from 1)
        for i, cls in enumerate(classes, 1):
            train_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            val_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            trainval_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            test_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
        train_img, val_img, test_img = train_test_val_split(indexes,0.8,0.1,0.1)
    else:
        dataset = {'categories': [], 'annotations': [], 'images': []}
        for i, cls in enumerate(classes, 1):
            dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    
    ann_id_cnt = 1 # annotation id count
    for k, index in enumerate(tqdm(indexes)):
        imgfile = os.path.join(root_path, 'images/') + index
        txtFile = index.replace('images','labels').replace('.JPG','.txt').replace('.jpg','.txt').replace('.PNG','.txt').replace('.png','.txt')
        im = cv2.imread(imgfile)
        height, width, _ = im.shape
        if random_split:
            # 切换dataset的引用对象，从而划分数据集
                if index in train_img:
                    dataset = train_dataset
                elif index in val_img:
                    dataset = val_dataset
                elif index in test_img:
                    dataset = test_dataset
        # 添加图像的信息
        dataset['images'].append({'file_name': index,
                                    'id': k+1,
                                    'width': width,
                                    'height': height})
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            # 如没标签，跳过，只保留图片信息。
            continue
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # yolo label starts from 0, coco label starts from 1, so +1
                cls_id = int(label[0])+1
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': k+1,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 保存结果
    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if random_split:
        for phase in ['train','val','trainval','test']:
            json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
            with open(json_name, 'w') as f:
                if phase == 'train':
                    json.dump(train_dataset, f)
                elif phase == 'val':
                    json.dump(val_dataset, f)
                elif phase == 'trainval':
                    json.dump(trainval_dataset, f)
                elif phase == 'test':
                    json.dump(test_dataset, f)
            print('Save annotation to {}'.format(json_name))
    else:
        json_name = os.path.join(root_path, 'annotations/{}'.format(json_file))
        with open(json_name, 'w') as f:
            json.dump(dataset, f)
            print('Save annotation to {}'.format(json_name))

if __name__ == "__main__":
    bUseFile = arg.f
    # txt = arg.txt
    # json_file = arg.json
    dir = arg.dir
    if dir == './' or dir == '':
        dir = os.path.dirname(sys.argv[0])

    random_split = arg.random_split

    if bUseFile:
        for txt in ['train.txt', 'val.txt', 'trainval.txt', 'test.txt']:
            assert os.path.exists(os.path.join(dir,txt))
            json_file = txt.split('.')[0]+'.json'
            print('Loading data from :', txt)
            yolo2coco(dir,bUseFile,txt,json_file,random_split)
    else:
        yolo2coco(dir,bUseFile,'','all.json',random_split)

    # if bUseFile:
    #     assert os.path.exists(os.path.join(dir,txt))
    # if bUseFile:
    #     print('Loading data from :', txt)
    # else:
    #     print("Loading data from ",dir)
    # print("Whether to split the data:",random_split)
    # yolo2coco(dir,bUseFile,txt,json_file,random_split)