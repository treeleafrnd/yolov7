from detector.detector import YoloV7Detector
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, utils, models

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'


class SaliencyMap:
    def __int__(self,model_path):
        self.model_path=model_path

    def preprocess_img(self,image_path, channels=3):
        if channels == 1:
            img = cv2.imread(image_path, 0)
        elif channels == 3:
            img = cv2.imread(image_path)

        shape_r = 288
        shape_c = 384
        img_padded = np.ones((shape_r, shape_c, channels), dtype=np.uint8)
        if channels == 1:
            img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)
        original_shape = img.shape
        rows_rate = original_shape[0] / shape_r
        cols_rate = original_shape[1] / shape_c
        if rows_rate > cols_rate:
            new_cols = (original_shape[1] * shape_r) // original_shape[0]
            img = cv2.resize(img, (new_cols, shape_r))
            if new_cols > shape_c:
                new_cols = shape_c
            img_padded[:,
            ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
        else:
            new_rows = (original_shape[0] * shape_c) // original_shape[1]
            img = cv2.resize(img, (shape_c, new_rows))

            if new_rows > shape_r:
                new_rows = shape_r
            img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows),
            :] = img

        return img_padded


    def postprocess_img(self,pred, org_dir):
        pred = np.array(pred)
        org = cv2.imread(org_dir, 0)
        shape_r = org.shape[0]
        shape_c = org.shape[1]
        predictions_shape = pred.shape

        rows_rate = shape_r / predictions_shape[0]
        cols_rate = shape_c / predictions_shape[1]

        if rows_rate > cols_rate:
            new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
            pred = cv2.resize(pred, (new_cols, shape_r))
            img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
        else:
            new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
            pred = cv2.resize(pred, (shape_c, new_rows))
            img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

        return img

    def main_implementation_saliency(self,test_img
                                     ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




        model = self.model_path.to(device)
        model.eval()


        img = SaliencyMap.preprocess_img(test_img)
        img = np.array(img) / 255.
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        img = img.float()

        pred_saliency = model(img)

        toPIL = transforms.ToPILImage()
        pic = toPIL(pred_saliency[0].squeeze().cpu())

        img = SaliencyMap.preprocess_img(test_img)
        img = np.array(img) / 255.
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        img = img.float()

        pred_saliency = SaliencyMap.postprocess_img(pic, test_img)
        # print(pred_saliency)
        img = cv2.imread(test_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.imshow(pred_saliency,cmap=plt.cm.hot)
        plt.show()



test_img = '//home/ishwor/Desktop/detection/archive/2.8k data/val/images/monkey_img_1140.jpg'
detector = YoloV7Detector('/home/ishwor/Desktop/TreeLeaf/yolov7/best (7).pt')
model = detector.load_model(weights='/home/ishwor/Desktop/TreeLeaf/yolov7/best (7).pt')
saliency_map=SaliencyMap()
saliency_map.main_implementation_saliency(test_img)