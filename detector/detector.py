import random

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.common import Conv
from models.experimental import Ensemble
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device


class YoloV7Detector:
    def __init__(self, model_path, im_size=640, half=False, device=''):
        self.device = select_device(device=device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.model_path = model_path
        self.device = select_device("cpu")
        self.im_size = im_size
        self.model = self.load_model(self.model_path, map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.im_size, s=self.stride)
        self.half = half
        if self.half:
            self.model = self.model.half()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.im_size, self.im_size).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        self._id2labels = {i: label for i, label in enumerate(self.names)}
        self._labels2ids = {label: i for i, label in enumerate(self.names)}

    def preprocess_image(self, image):
        img = letterbox(image, self.im_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, image, conf_thresh=0.55, iou_thresh=0.45, classes=None, gn=None):

        bbox = []
        confidence = []
        class_ = []
        try:
            img = self.preprocess_image(image)
            gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            pred = self.model(img, augment=True)[0]
            pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=classes, agnostic=True)
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

                for *xyxy, conf, cls in reversed(det):

                    x, y, w, h = torch.tensor(xyxy).numpy().astype(int)  # top left bottom right

                    # x,y,w,h =np.array(xyxy).astype(int)  #Without Normalization

                    bbox.append([x, y, w, h])
                    confidence.append(float(conf))
                    class_.append(int(cls))


        except Exception as e:
            print(str(e))



        return bbox, confidence,class_
    def detect_norm(self, image, conf_thresh=0.55, iou_thresh=0.45, classes=None, gn=None):

        bbox = []
        confidence = []
        class_ = []
        try:
            img = self.preprocess_image(image)
            gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            pred = self.model(img, augment=True)[0]
            pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=classes, agnostic=True)
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    # x,y,w,h =np.array(xyxy).astype(int)  #Without Normalization

                    bbox.append([x, y, w, h])
                    confidence.append(float(conf))
                    class_.append(int(cls))


        except Exception as e:
            print(str(e))

        # print(f'Bbox: -{bbox}')
        # print(f'Confidence:- {confidence}')
        # print(f'Class:- {class_}')

        return bbox, confidence,class_




    @staticmethod
    def load_model(weights, map_location=None):

        # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
        model = Ensemble()
        # for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(weights, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

        # Compatibility updates
        for m in model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        if len(model) == 1:
            return model[-1]  # return model
        else:
            print('Ensemble created with %s\n' % weights)
            for k in ['names', 'stride']:
                setattr(model, k, getattr(model[-1], k))
            return model  # return ensemble

# if __name__ == '__main__':
#     image_path = '/home/ishwor/Desktop/TreeLeaf/yolov7/inference/images/different_Monkey Images from Video1440.jpg'
#     image = cv2.imread(image_path)
#     _model_path = "/home/ishwor/Desktop/TreeLeaf/yolov7/monkey_image_detection.pt"
#     yolov7_detector = YoloV7Detector(_model_path)
#     bbox,conf =yolov7_detector.detect(image)
#     print(bbox)
#     for i in bbox:  #Just for verification purposes
#         print(i)
#     print(conf)
