import cv2

from detector.yolov7_detector import YOLOV7_Detector


def image_detector_example(draw=True):
    image_path = "../1.jpg"
    image = cv2.imread(image_path)
    _model_path = "../weapon_detector.pt"
    yolov7_detector = YOLOV7_Detector(_model_path)
    results = yolov7_detector.detect(image)
    print("Detection result", results)
    if draw:
        image_ = draw_bbox(results, image)
        cv2.imshow('image_', image_)
        cv2.destroyAllWindows()


def video_detector_example(draw=False):
    _model_path = "../weapon_detector.pt"
    yolov7_detector = YOLOV7_Detector(_model_path)
    video_path = '../video_1.mp4'
    video = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video.read()
        if not ret:
            print('error reteriving frames')
            break
        results = yolov7_detector.detect(frame)
        print(results)
        if draw:
            frame = draw_bbox(results, frame)
        cv2.imshow("video", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
    video.release()


def webcam_detector_examplt(draw=False):
    _model_path = "../weapon_detector.pt"
    yolov7_detector = YOLOV7_Detector(_model_path)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print('error reteriving frames')
            break
        results = yolov7_detector.detect(frame)
        print(results)
        if draw:
            frame = draw_bbox(results, frame)
        cv2.imshow("video", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


def draw_bbox(dict_list, image):
    dh, dw, _ = image.shape
    for detection in dict_list:
        t, l, b, r = detection['tlbr']
        cv2.rectangle(image, (t, l), (b, r), (0, 0, 255), 1)
    return image


if __name__ == '__main__':
    image_detector_example(draw=True)
