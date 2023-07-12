import torch
from PIL import Image
from utils.general import non_max_suppression
from detector.yolov7_detector import YOLOV7_Detector
import cv2
from torchvision import transforms
from utils.general import non_max_suppression
torch.autograd.set_detect_anomaly(True)
import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt

detector = YOLOV7_Detector('/Users/anshujoshi/PycharmProjects/labelimg/yolov7/weapon_detector.pt')
model = detector.load_model(weights='/Users/anshujoshi/PycharmProjects/labelimg/yolov7/weapon_detector.pt')
model.eval()

for param in model.parameters():
    param.requires_grad = False
img_path = '/Users/anshujoshi/PycharmProjects/labelimg/yolov7/1.jpg'
img = cv2.imread(img_path)
img_pli = Image.open(img_path)
transformed = transforms.Compose([transforms.Resize((640,640)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                      transforms.Lambda(lambda x: x[None])])
transformed_img = transformed(img_pli)
transformed_img = transformed_img.to('cpu')
transformed_img=transformed_img.reshape(1,3,640,640)
transformed_img.requires_grad_()
scores = model(transformed_img)
print(scores[0].shape)
# preprocessed_image=detector.preprocessing(img)
# print(preprocessed_image.shape)

# output=model(preprocessed_image)
# output_idx = output.argmax()
# output_max = output[0, output_idx]

# output_max.backward()
# print(output)
# print(output[0][0][0][4])
'''print(output[0].shape)
for list in output[0][0]:
    print(list[4])
    break'''
preds=non_max_suppression(scores[0]) #converts to bbox,conf,class
print(preds)
selected=preds[0][0][4]
selected.backward()
#scores_max_index = torch.max()
# score_max = scores[0, scores_max_index]
