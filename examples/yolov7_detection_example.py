import cv2 as cv
from torch.utils.tensorboard.summary import image

from detector.detector import YoloV7Detector

def draw_bbox(bbox, image):
    dh, dw, _ = image.shape
    for box in bbox:
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv.rectangle(image, (x, y), (w, h), (0, 0, 255), 1)
    return image

def image_detection(draw=True
                    ):
    image_path = "/home/ishwor/Desktop/detection/archive/2.8k data/train/images/monkeys_of_pashupatii 318.jpg"
    image_display = cv.imread(image_path)
    _model_path = '/home/ishwor/Desktop/TreeLeaf/yolov7/best (7).pt'
    yolov7_detector = YoloV7Detector(_model_path)
    model_prediction, conf,classes = yolov7_detector.detect(image_display)
    print(model_prediction, conf,classes)
    if draw:
        image__ = draw_bbox(model_prediction, image_display)
        cv.imshow('Model Predicted Image', image__)
        cv.waitKey(0)
        cv.distoroyAllWindows()

def webcam_detection(draw=False):
    _model_path = '/home/ishwor/Desktop/TreeLeaf/yolov7/best (7).pt'
    yolov7_detector = YoloV7Detector(_model_path)
    vid_path='/home/ishwor/Videos/Monkey video/Studio_Project_V1.mp4'
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error retrieving frames from webcam')
            break
        results, conf,classes = yolov7_detector.detect(frame)
        print(results, conf,classes)
        if draw:
            frame = draw_bbox(results, frame)
        cv.imshow("video", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            cv.destroyAllWindows()
            break




if __name__ == '__main__':
    image_detection(draw=True)
