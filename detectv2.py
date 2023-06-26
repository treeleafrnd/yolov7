from models.experimental import attempt_load,Ensemble
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
import torch
import torch.nn as nn
from models.common import Conv
from utils.datasets import letterbox
import numpy as np
import cv2
class YOLOv7_Detector:
    def __init__(self,model_path,half=False,img_size=640):
        self.model_path = model_path
        self.device=select_device('cpu')
        self.im_size=img_size
        self.model=self.load_model(self.model_path,map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.im_size, s=self.stride)

        self.half=half
        if self.half:
            self.model = self.model.half()
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def preprocessing(self,img):
        img = letterbox(img, self.im_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, image, conf_thresh=0.4, iou_thresh=0.3, classes=None, gn=None):
        gn = torch.tensor(image.shape)[[1, 0, 1, 0]] #gn=[ht,wt,ht,wt]
        img = self.preprocessing(image)
        with torch.no_grad():
            pred = self.model(img, augment=True)[0]
            pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes, agnostic=True)

            detections = []
            for i, det in enumerate(pred):
                if len(det):
                    #print(det)
                    #OUTPUT
                    #tensor([[265.10898,  96.05386, 489.17618, 431.32706,   0.96912,   0.00000],
                    #[336.01105, 162.39006, 449.47913, 333.48389,   0.80054,  24.00000]])
                    # unnormalized bbox_0-3 ,confidence_4,class_5
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                    # converts bbox of processed image to that or original image
                    #print((det))
                    #print(reversed(det))
                for *xyxy, conf, cls in (det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    # reshape to (1,4),xyxy2xywh takes reshaped i/p,o/p=x,y,w,h, /gn normalizes each element,reshape
                    # and put to list
                    detection = {
                        'bbox': xyxy,
                        'confidence': conf.item(),
                        'class': int(cls.item()),
                        'xywh': xywh
                    }
                    detections.append(detection)
            print(detections)
            self.processing_pred(detections)
            #self.preprocessing(detection)

    def processing_pred(self,dict_list):
        for detection in dict_list:
            if detection['class']==0:
                print('weapon detected')

    @staticmethod
    def load_model(weights, map_location=None):
        # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
        model = Ensemble()
        #for w in weights if isinstance(weights, list) else [weights]:
            #attempt_download(w)
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
            return model  # return ensemb

if __name__ == '__main__':
    image_path = "/Users/anshujoshi/PycharmProjects/labelimg/yolov7/armas-187-_jpg.rf.31778d8c49148ca5c24945980d95c957.jpg"
    image = cv2.imread(image_path)
    _model_path = "/Users/anshujoshi/PycharmProjects/labelimg/yolov7/weapon_detector.pt"
    yolov7_detector = YOLOv7_Detector(_model_path)
    results = yolov7_detector.detect(image)
