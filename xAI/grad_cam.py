from pytorch_grad_cam.grad_cam import GradCAM
from detector.yolov7_detector import YOLOV7_Detector
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import torchvision.transforms as T
import torch


def Grad_Cam(model_path, image_path):
    model = torch.load(model_path)['model']
    image = Image.open(image_path)
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x[None]),
    ])
    img = transform(image)
    target_layers = [model.model[-1].m[2]]
    cam = GradCAM(model, target_layers, use_cuda=False)
    grayscale_cam = cam(img)[0, :, :]
    cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    pil_image = Image.fromarray(cam_image)
    pil_image.show()

#Implemented in examples
