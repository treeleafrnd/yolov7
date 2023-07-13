import torch
from PIL import Image

import models.yolo
from utils.general import non_max_suppression
from detector.yolov7_detector import YOLOV7_Detector
import cv2
from torchvision import transforms
from utils.general import non_max_suppression
from typing import List

from models.experimental import Ensemble
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, \
    scale_coords
import torch
import torch.nn as nn
from models.common import Conv
from utils.datasets import letterbox
import numpy as np
import numpy.typing as npt
import logging


torch.autograd.set_detect_anomaly(True)
import matplotlib

matplotlib.use('Qt5Agg')  # or 'Qt5Agg'


def load_model( weights) -> Ensemble:
    """
    Parms:
    weights: weight file
    Returns:
    ??
    """
    model = Ensemble()
    ckpt = torch.load(weights, map_location='cpu')  # load
    model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())

    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble model

model=load_model('/Users/anshujoshi/PycharmProjects/labelimg/yolov7/weapon_detector.pt')
model.eval()
#detector=YOLOV7_Detector.load_model(weights='/Users/anshujoshi/PycharmProjects/labelimg/yolov7/weapon_detector.pt')
#model=detector.load_model(weights='/Users/anshujoshi/PycharmProjects/labelimg/yolov7/weapon_detector.pt')
# for param in model.parameters():
#     param.requires_grad = False
img_path = '/Users/anshujoshi/PycharmProjects/labelimg/yolov7/2.jpg'
img = cv2.imread(img_path)
img_pli = Image.open(img_path)
transformed = transforms.Compose([transforms.Resize((640, 640)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                  transforms.Lambda(lambda x: x[None])])
transformed_img = transformed(img_pli)
transformed_img = transformed_img.to('cpu')
transformed_img = transformed_img.reshape(1, 3, 640, 640)
transformed_img.requires_grad_()
output=model.forward_once(transformed_img)
print(output)
selected = output[0][0][0]
print(selected)
model.zero_grad()
selected.backward()

##custom transformes##
#output = model(transformed_img)

#print(output)
# preds = non_max_suppression(output[0])
# # print(preds)
# selected = preds[0][0][4]
# model.zero_grad()
# selected.backward()



# img_test=detector.preprocessing(img)
# scores = model(img_test)
# print(scores[0].shape)
# preprocessed_image=detector.preprocessing(img)
# print(preprocessed_image.shape)
# output=model(preprocessed_image)
# output_idx = output.argmax()
# output_max = output[0, output_idx]

# output_max.backward()
# print(output)
# # print(output[0][0][0][4])
# print(output[0].shape)
# for list in output[0][0]:
#     print(list[4])
#     break
# preds=non_max_suppression(scores[0]) #converts to bbox,conf,class
# print(preds)
# selected=preds[0][0][4]
# print(selected)
# selected.backward()
# #scores_max_index = torch.max()
# # score_max = scores[0, scores_max_index]
