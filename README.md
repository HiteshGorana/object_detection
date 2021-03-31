TASK-1 : 'TASK-1/task1.py'

class_ = class to find siblings/ parents / ancestor


class_1 = class1 
class_2 = class2

to find both class 1 and class 2 belong to the same ancestor class(es)

APPROACH :

    1. read both the files oidv6-class-descriptions.csv and bbox_labels_600_hierarchy.json
    2. created classes2name and name2classes
    3. find list of index and keys using `find` function recursively 
    4. after getting list of index and key or i say it path by using that i can find siblings/ parents / ancestor
    5. to do series of operation i created a function named setInDict with apply list of operations


TASK-2 : 'TASK-2/task1.py'

APPROACH :

    1. annotations.xml converted into annotations.csv with correct formet for yolo5
    2. setup the images and label 
    3. trained it with yolo5 (https://github.com/ultralytics/yolov5)

    # Propose an evaluation metric and justify your choice.
    ANS : Mean Average Precision (mAP)
    EXPLANATION : this metric compares the ground-truth bounding box to the detected box (IOU)
    its just like measuring the accuracy of object how well it is doing
    


    