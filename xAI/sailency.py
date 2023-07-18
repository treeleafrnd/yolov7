from detector.yolov7_detector import YOLOV7_Detector
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, utils, models
import matplotlib
import torchvision.transforms as T
from PIL import Image

matplotlib.use('TkAgg')  # or 'Qt5Agg'


class SalencyMap:
    def __init__(self, model_path):
        self.model_path = model_path

    # def preprocess_img(img_dir, channels=3):
    #     if channels == 1:
    #         img = cv2.imread(img_dir, 0)
    #     elif channels == 3:
    #         img = cv2.imread(img_dir)
    #
    #     shape_r = 288
    #     shape_c = 384
    #     original_shape = img.shape
    #     rows_rate = original_shape[0] / shape_r
    #     cols_rate = original_shape[1] / shape_c
    #
    #     if rows_rate > cols_rate:
    #         new_cols = (original_shape[1] * shape_r) // original_shape[0]
    #         img = cv2.resize(img, (new_cols, shape_r))
    #     else:
    #         new_rows = (original_shape[0] * shape_c) // original_shape[1]
    #         img = cv2.resize(img, (shape_c, new_rows))
    #
    #     img_padded = cv2.copyMakeBorder(img, (shape_r - img.shape[0]) // 2, (shape_r - img.shape[0]) // 2,
    #                                     (shape_c - img.shape[1]) // 2, (shape_c - img.shape[1]) // 2,
    #                                     cv2.BORDER_CONSTANT, value=0)
    #
    #     return img_padded

    def preprocess(image_path, size=640):

        image = Image.open(image_path)
        transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Lambda(lambda x: x[None]),
        ])
        return transform(image)

    def postprocess_img(pred, image_path):
        pred = np.array(pred)
        org = cv2.imread(image_path, 0)
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

    def main_implementation_saliency(self, image_path):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # flag = 1  # 0 for TranSalNet_Dense, 1 for TranSalNet_Res

        model = torch.load(self.model_path)['model']
        model=model.to(torch.float32)
        model.eval()

        img = SalencyMap.preprocess(image_path)  # padding and resizing input image into 384x288

        pred_saliency = model(img)

        toPIL = transforms.ToPILImage()
        pic = toPIL(pred_saliency[0].squeeze().cpu())

        actual_image = cv2.imread(image_path)

        pred_saliency = SalencyMap.postprocess_img(pic,
                                                   image_path)  # restore the image to its original size as the result

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(actual_image, cv2.COLOR_BGR2RGB))
        plt.title('Actual Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_saliency, cmap=plt.cm.hot)  # alpha=0.5
        plt.title('Saliency Map')
        plt.axis('off')

        plt.show()


image_path = '/Users/anshujoshi/PycharmProjects/labelimg/yolov7/2.jpg'
model_path = '/Users/anshujoshi/PycharmProjects/labelimg/yolov7/weapon_detector.pt'

salency_map = SalencyMap(model_path)
salency_map.main_implementation_saliency(image_path)
