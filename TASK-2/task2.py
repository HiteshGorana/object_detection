# -*- coding: utf-8 -*-
# @Date    : 30-03-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
from glob import glob
from os.path import join
from xml.etree import ElementTree

import pandas as pd
import torch
import yaml
from IPython.display import clear_output  # to display images
from tqdm import tqdm

xml = ElementTree.parse('/kaggle/input/object-detection-tasks/object_detection_tasks/TASK-2/dataset/annotations.xml')
valid_images = [f"images{i}.jpg" for i in range(608)]
DATA = []
for images in tqdm(xml.findall('.//image')):
    if images.get('name') in valid_images:
        for box in images:
            if box.get('label') == 'head':
                info = {'image': [], 'xtl': [], 'ytl': [], 'xbr': [], 'ybr': [], 'width': [], 'height': [], 'mask': [],
                        'has_safety_helmet': []}
                info['image'].append(images.get('name'))
                info['xtl'].append(box.get('xtl'))
                info['ytl'].append(box.get('ytl'))
                info['xbr'].append(box.get('xbr'))
                info['ybr'].append(box.get('ybr'))
                info['width'].append(images.get('width'))
                info['height'].append(images.get('height'))
                attribute_ = {}
                for attribute in box.findall('.//attribute'):
                    attribute_[attribute.get("name")] = attribute.text
                info['mask'].append(attribute_['mask'])
                info['has_safety_helmet'].append(attribute_['has_safety_helmet'])
                DATA.append(pd.DataFrame(info))
data = pd.concat(DATA).reset_index(drop=True)
data['label'] = -1
data.loc[(data['mask'] == 'yes') & (data['has_safety_helmet'] == 'no'), 'label'] = 'mask'
data.loc[(data['mask'] == 'no') & (data['has_safety_helmet'] == 'yes'), 'label'] = 'helmet'
data.loc[(data['mask'] == 'invisible') & (data['has_safety_helmet'] == 'yes'), 'label'] = 'helmet'
data.loc[(data['mask'] == 'yes') & (data['has_safety_helmet'] == 'invisible'), 'label'] = 'mask'
data.loc[(data['mask'] == 'no') & (data['has_safety_helmet'] == 'no'), 'label'] = 'None'
data.loc[(data['mask'] == 'yes') & (data['has_safety_helmet'] == 'yes'), 'label'] = 'both'
data.loc[(data['mask'] == 'invisible') & (data['has_safety_helmet'] == 'no'), 'label'] = 'None'
data.loc[(data['mask'] == 'no') & (data['has_safety_helmet'] == 'invisible'), 'label'] = 'None'
for s in ['xtl', 'ytl', 'xbr', 'ybr']:
    data[s] = data[s].astype('float')
for s in ['width', 'height']:
    data[s] = data[s].astype('int')

data['x_min'] = data.apply(lambda row: (row.xtl) / row.width, axis=1)
data['y_min'] = data.apply(lambda row: (row.ytl) / row.height, axis=1)

data['x_max'] = data.apply(lambda row: (row.xbr) / row.width, axis=1)
data['y_max'] = data.apply(lambda row: (row.ybr) / row.height, axis=1)

data['x_mid'] = data.apply(lambda row: (row.xbr + row.xtl) / 2, axis=1)
data['y_mid'] = data.apply(lambda row: (row.ybr + row.ytl) / 2, axis=1)

data['w'] = data.apply(lambda row: (row.xbr - row.xtl), axis=1)
data['h'] = data.apply(lambda row: (row.ybr - row.ytl), axis=1)

data['area'] = data['w'] * data['h']
data.head()
features = ['x_min', 'y_min', 'x_max', 'y_max', 'x_mid', 'y_mid', 'w', 'h', 'area']
X = data[features]
data['class_id'] = data['label'].factorize()[0]
y = data['class_id']
import numpy as np

class_ids, class_names = list(zip(*set(zip(data.class_id, data.label))))
classes = list(np.array(class_names)[np.argsort(class_ids)])
classes = list(map(lambda x: str(x), classes))
remove = ['/images/65.jpg',
          '/images/67.jpg',
          '/images/68.jpg',
          '/images/70.jpg',
          '/images/72.jpg',
          '/images/74.jpg',
          '/images/85.jpg',
          '/images/87.jpg',
          '/images/89.jpg',
          '/images/91.jpg',
          '/images/93.jpg',
          '/images/95.jpg',
          '/images/97.jpg']
train_files = list(data['image'].apply(lambda x: 'images/' + x.replace('images', '')).unique())
train_files = [i for i in train_files if i not in remove]
import os
import shutil
remove = []
os.makedirs('/data/labels/train', exist_ok = True)
os.makedirs('/data/images/train', exist_ok = True)
label_dir = '/data/labels/train'
for file in tqdm(train_files):
    try:
        shutil.copy(file, '/data/images/train')
        filename = file.split('/')[-1].split('.')[0]
        d_ = data[data['image'] == 'images' + filename + '.jpg'][['class_id', 'x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
        for cls, *xyxy in d_:
            with open(f'{label_dir}/{filename}.txt', 'a') as f:
                line = (cls, *xyxy)
                f.write(('%g ' * len(line)).rstrip() % line + '\n')
    except:
        remove.append(file)
train_files = glob('/data/images/train/*')


cwd = '/data'

with open(join(cwd, 'train.txt'), 'w') as f:
    for path in train_files:
        f.write(path + '\n')

with open(join(cwd, 'val.txt'), 'w') as f:
    for path in train_files:
        f.write(path + '\n')

data = dict(
    train=join(cwd, 'train.txt'),
    val=join(cwd, 'val.txt'),
    nc=4,
    names=classes
)

with open(join(cwd, 'data.yaml'), 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

f = open(join(cwd, 'data.yaml'), 'r')
print('\nyaml:')
print(f.read())
# https://www.kaggle.com/ultralytics/yolov5
# !git clone https://github.com/ultralytics/yolov5  # clone repo
# %cd yolov5
shutil.copytree('/yolov5-official-v31-dataset/yolov5', '/kaggle/working/yolov5')
os.chdir('yolov5')
# %pip install -qr requirements.txt # install dependencies

clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
os.system('python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/')
os.system('!WANDB_MODE="dryrun" python train.py --img 128 --batch 32 --epochs 10 --data data.yaml --weights '
          'yolov5s.pt --cache')

