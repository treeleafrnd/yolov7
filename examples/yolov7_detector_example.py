import cv2

from detector.yolov7_detector import YOLOV7_Detector


def image_detector_example(draw=False):
    image_path = "../2.jpg"
    image = cv2.imread(image_path)
    _model_path = "../weapon_detector.pt"
    yolov7_detector = YOLOV7_Detector(_model_path)
    results = yolov7_detector.detect(image)
    print("Detection result", results)
    if draw:
        draw_bbox(results, image)


def video_detector_example(draw=True):
    pass


def draw_bbox(dict_list, image):
    dh, dw, _ = image.shape
    for detection in dict_list:
        t, l, b, r = detection['tlbr']
        cv2.rectangle(image, (t, l), (b, r), (0, 0, 255), 1)
    cv2.imshow("drawed image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_detector_example(draw=True)
