# -*- coding: utf-8 -*-
# @Date    : 29-03-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import json
import operator
from functools import reduce

import pandas as pd


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


def find(json_dict_or_list, value):
    if json_dict_or_list == value:
        return [json_dict_or_list]
    elif isinstance(json_dict_or_list, dict):
        for k, v in json_dict_or_list.items():
            p = find(v, value)
            if p:
                return [k] + p
    elif isinstance(json_dict_or_list, list):
        lst = json_dict_or_list
        for i in range(len(lst)):
            p = find(lst[i], value)
            if p:
                return [str(i)] + p


def ancestor_(path_, verbose=False):
    if len(path_) > 2:
        raw = getFromDict(data, path_[:-6])
        ancestor = []
        for n in range(0, len(path_[:-6]) + 1, 2):
            raw = getFromDict(data, path_[:-6 - n])
            a_ = classes2name.get(raw['LabelName'], 'Entity')
            ancestor.append(a_)
        if len(ancestor) > 1:
            for i in ancestor:
                if verbose:
                    print(i)
        else:
            print('No')
    else:
        print('No')
    return ancestor


if __name__ == '__main__':
    class_ = '/m/080hkjn'

    class_1 = '/m/080hkjn'
    class_2 = '/m/0nl46'

    verbose = True
    df = pd.read_csv('oidv6-class-descriptions.csv', header=None)
    with open('bbox_labels_600_hierarchy.json') as f:
        data = json.loads(f.read())
    classes2name = {i: j for i, j in zip(df[0], df[1])}
    name2classes = {j: i for i, j in zip(df[0], df[1])}
    # TODO : Find all siblings class of a class name
    print('#' * 25)
    print('Q1 Find all siblings class of a class name')
    print('\n')
    path = [int(i) if i.isnumeric() else i for i in find(data, class_)]
    siblings = []
    if path[0:-3]:
        raw = getFromDict(data, path[0:-3])
        if verbose:
            for n, i in enumerate(raw):
                _sibling = classes2name.get(i['LabelName'], 'root')
                siblings.append(_sibling)
                print(_sibling)
    else:
        print('None')

    # TODO : Find the parent class of a class name
    print('\n')
    print('#' * 25)
    print('Q2 Find the parent class of a class name')
    print('\n')
    if len(path) > 2:
        parent = getFromDict(data, path[:-4])
        # https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html
        if verbose:
            print(classes2name.get(parent['LabelName'], 'Entity'))
    else:
        print('No')
    # TODO : Find all ancestor classes of a class name
    print('\n')
    print('#' * 25)
    print('Q3 Find all ancestor classes of a class name')
    print('\n')
    ancestors = ancestor_(path, verbose=True)
    # TODO : Find if both class 1 and class 2 belong to the same ancestor classes
    print('\n')
    print('#' * 25)
    print('Q3 Find if both class 1 and class 2 belong to the same ancestor classes')
    print('\n')
    path1 = [int(i) if i.isnumeric() else i for i in find(data, class_1)]
    path2 = [int(i) if i.isnumeric() else i for i in find(data, class_2)]
    ancestors1 = set(ancestor_(path1))
    ancestors2 = set(ancestor_(path2))
    if len(ancestors1.intersection(ancestors2)) > 1:
        print(f"{classes2name[class_1]} and {classes2name[class_2]} belong to the same ancestor")
    else:
        print(f"{classes2name[class_1]} and {classes2name[class_2]} not belong to the same ancestor")
