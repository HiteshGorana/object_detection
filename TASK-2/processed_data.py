# -*- coding: utf-8 -*-
# @Date    : 29-03-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import pandas as pd
from tqdm import tqdm
from xml.etree import ElementTree

xml = ElementTree.parse('./dataset/annotations.xml')
valid_images = [f"images{i}.jpg" for i in range(608)]
DATA = []
for images in tqdm(xml.findall('.//image')):
    if images.get('name') in valid_images:
        for box in images:
            if box.get('label') == 'head':
                info = dict(image=[], xtl=[], ytl=[],
                            xbr=[], ybr=[], width=[],
                            height=[], mask=[], has_safety_helmet=[])
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
data.to_csv('processed_annotations.csv', index=False)
