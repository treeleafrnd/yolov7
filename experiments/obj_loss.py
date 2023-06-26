import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

image_path='path'
label_path='path'
pred_label_path='path'

def convert_yolo_to_xyxy(image_width, image_height,x_center,y_center,width,height):
    # Convert relative coordinates to absolute values
    abs_x = x_center * image_width
    abs_y = y_center * image_height
    abs_width = width * image_width
    abs_height = height * image_height

    # Calculate xmin, xmax, ymin, ymax
    xmin = abs_x - abs_width / 2
    xmax = abs_x + abs_width / 2
    ymin = abs_y - abs_height / 2
    ymax = abs_y + abs_height / 2
    return xmin,xmax,ymin,ymax


def calculate_iou(box1, box2):
    c=0
    xmin1, xmax1, ymin1, ymax1 = box1
    xmin2, xmax2, ymin2, ymax2 = box2

    # Calculate intersection coordinates
    x_left = max(xmin1, xmin2)
    y_top = max(ymin1, ymin2)
    x_right = min(xmax1, xmax2)
    y_bottom = min(ymax1, ymax2)

    # Calculate intersection area
    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height

    # Calculate box areas
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou



def calc_obj_loss(path1,path2,path3):
    total_loss=0
    for files in os.listdir(path1):
        loss=0
        img_name=os.path.splitext(files)[0]
        #print(files)
        img=cv2.imread(path1+'/'+files)
        dh,dw,_=img.shape
        actual_label=label_path+img_name+'.txt'
        pred_label=pred_label_path+img_name+'.txt'
        with open(actual_label) as li:
            i=0
            matches=0
            for lines in li:
                i=i+1
                c, x, y, w, h = map(float, lines.split(' '))
                cord_label=list(convert_yolo_to_xyxy(dw,dh,x,y,w,h))
                try:
                    with open(pred_label) as lp:
                        for lines in lp:
                            c,x,y,w,h=map(float, lines.split(' '))
                            cord_pred=list(convert_yolo_to_xyxy(dw,dh,x,y,w,h))
                            iou = calculate_iou(cord_label, cord_pred)
                            if iou>.7:
                                matches+=1
                except OSError as e:
                    pass

            loss=i-matches
        total_loss+=loss
    print(total_loss)


calc_obj_loss(image_path,label_path,pred_label_path)
