from models.experimental import attempt_load, Ensemble
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
import torch
import torch.nn as nn
from models.common import Conv
from utils.datasets import letterbox
import numpy as np
import cv2
import matplotlib



def draw_bbox(dict_list, image):
    dh, dw, _ = image.shape
    for detection in dict_list:
        t, l, b, r = detection['tlbr']
        cv2.rectangle(image, (t, l), (b, r), (0, 0, 255), 1)
    cv2.imshow("drawed image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class YOLOv7_Detector:
    def __init__(self, model_path, half=False, img_size=640):
        self.model_path = model_path
        self.device = select_device('cpu')
        self.im_size = img_size
        self.model = self.load_model(self.model_path, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(self.im_size, s=self.stride)

        self.half = half
        if self.half:
            self.model = self.model.half()
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def preprocessing(self, img):
        img = letterbox(img, self.im_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, image, conf_thresh=0.4, iou_thresh=0.3, classes=None, gn=None):
        detections = []
        try:
            gn = torch.tensor(image.shape)[[1, 0, 1, 0]]
            img = self.preprocessing(image)
            with torch.no_grad():
                pred = self.model(img, augment=True)[0]
                pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes, agnostic=True)

                for i, det in enumerate(pred):
                    if len(det):
                        #det=[tensor(bbox)*4, confidence,class]
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                        # converts bbox of processed image to that or original image
                    for *xyxy, conf, cls in (det):
                        t, l, b, r = np.array(xyxy).astype(int) #top left bottom right
                        detection = {
                            'confidence': round(float(conf), 2),
                            'class': int(cls),
                            'tlbr': [t, l, b, r]
                        }
                        detections.append(detection)
        except Exception as e:
            print(str(e))
        return detections

    @staticmethod
    def load_model(weights, map_location=None):
        model = Ensemble()
        ckpt = torch.load(weights, map_location=map_location)  # load
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
            return model  # return ensemble


if __name__ == '__main__':
    image_path = "2.jpg"
    image = cv2.imread(image_path)
    _model_path = "weapon_detector.pt"
    yolov7_detector = YOLOv7_Detector(_model_path)
    results = yolov7_detector.detect(image)
    print("Detection result", results)
    draw_bbox(results, image)
