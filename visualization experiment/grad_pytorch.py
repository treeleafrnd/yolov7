from pytorch_grad_cam.eigen_cam import EigenCAM
from detector.detector import YoloV7Detector
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

class GradCam:
    def __init__(self, model_path, image_path):
        self.model_path = model_path
        self.image_path = image_path
        self.detector = None
        self.model = None

    def load_model(self):
        self.detector = YoloV7Detector(self.model_path)
        self.model = self.detector.load_model(weights=self.model_path)
        self.model.eval()

    def preprocess_image(self):
        image = cv2.imread(self.image_path)
        img = self.detector.preprocess_image(image)
        image = image / 255
        return img, image

    def generate_cam(self):
        img, image = self.preprocess_image()
        target_layers = [self.model.model[-2].rbr_reparam]

        cam = EigenCAM(self.model, target_layers, use_cuda=True)
        grayscale_cam = cam(img)[0, :, :]
        cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
        pil_image = Image.fromarray(cam_image)
        pil_image.show()

model_path = '../best (7).pt'
image_path = '/home/ishwor/Desktop/TreeLeaf/yolov7/visualization experiment/images/monkey1.jpg'

grad_cam = GradCam(model_path, image_path)
grad_cam.load_model()
grad_cam.generate_cam()