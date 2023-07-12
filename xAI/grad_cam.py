from pytorch_grad_cam.eigen_cam import EigenCAM
from detector.yolov7_detector import YOLOV7_Detector
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
def Grad_Cam(model_path,image_path):
    detector = YOLOV7_Detector(model_path)
    model = detector.load_model(weights=model_path)
    image = cv2.imread(image_path)
    img = detector.preprocessing(image)
    image = image / 255

    target_layers = [model.model[-1].m[2]]
    cam = EigenCAM(model, target_layers, use_cuda=False)
    grayscale_cam = cam(img)[0, :, :]
    cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    pil_image = Image.fromarray(cam_image)
    pil_image.show()


##Implemented in examples