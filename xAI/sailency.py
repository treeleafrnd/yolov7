import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchsummary import summary
import requests
from PIL import Image
from detector.yolov7_detector import YOLOV7_Detector
#from models.processing import postprocess
import cv2


detector = YOLOV7_Detector('/Users/anshujoshi/PycharmProjects/labelimg/yolov7/weapon_detector.pt')
model = detector.load_model(weights='/Users/anshujoshi/PycharmProjects/labelimg/yolov7/weapon_detector.pt')

for param in model.parameters():
    param.requires_grad = True
#print(model)
img_path = '/Users/anshujoshi/PycharmProjects/labelimg/yolov7/1.jpg'
img = cv2.imread(img_path)
image=detector.preprocessing(img)


import torch
import torchvision
import numpy as np

def post_process(output, class_index, confidence_threshold, nms_threshold):
    # Extract prediction components from the output tensor
    box_attrs = 5   # 5 for (x, y, w, h, confidence), num_classes for class probabilities
    output = output.squeeze()

    # Split the tensor into bounding boxes, confidences, and class probabilities
    boxes = output[:, :4]
    confidences = output[:, 4]
    class_probs = output[:, 5:].sigmoid()

    # Filter detections for the desired class
    mask = (torch.argmax(class_probs, dim=1) == class_index) & (confidences > confidence_threshold)
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_probs = class_probs[mask]

    # Sort the boxes and associated scores based on confidence scores
    sorted_indices = torch.argsort(confidences, descending=True)
    boxes = boxes[sorted_indices]
    confidences = confidences[sorted_indices]
    class_probs = class_probs[sorted_indices]

    # Apply non-maximum suppression (NMS)
    keep_indices = torchvision.ops.boxes.batched_nms(boxes, confidences, torch.arange(len(boxes)), nms_threshold)
    boxes = boxes[keep_indices]
    confidences = confidences[keep_indices]
    class_probs = class_probs[keep_indices]

    # Prepare the final detections
    detections = []
    for box, confidence, class_prob in zip(boxes, confidences, class_probs):
        detection = {
            'class_index': class_index,
            'confidence': confidence.item(),
            'bbox': box.tolist()
        }
        detections.append(detection)

    return detections

# Example
model.eval()
with torch.no_grad():
    output_tensor = model(image)[0]# Example output tensor from YOLOv7 for a single class
    print(output_tensor.shape)
class_index = 0  # Index of the desired class
confidence_threshold = 0.5  # Confidence threshold for filtering
nms_threshold = 0.3  # IoU threshold for non-maximum suppression

detections = post_process(output_tensor, class_index, confidence_threshold, nms_threshold)
print(detections)

