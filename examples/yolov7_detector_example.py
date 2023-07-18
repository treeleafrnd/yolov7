import cv2
from pipeline.hashing import generate_frames
from detector.yolov7_detector import YOLOV7_Detector


def image_detector_example(draw=True):
    image_path = "../test.jpg"
    image = cv2.imread(image_path)
    _model_path = "../best.pt"
    yolov7_detector = YOLOV7_Detector(_model_path)
    results = yolov7_detector.detect(image)
    print("Detection result", results)
    if draw:
        image_ = draw_bbox(results, image)
        cv2.imshow('image_', image_)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def video_detector_example(draw=False):
    _model_path = "../weapon_detector.pt"
    yolov7_detector = YOLOV7_Detector(_model_path)
    for frames,time_stamp in generate_frames(stream=False,video_path='../video_1.mp4'):
        results = yolov7_detector.detect(frames)
        if len(results) != 0:
            results[0]['time']=time_stamp
            print(results)
        if draw:
            frame = draw_bbox(results, frames)
        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break


def webcam_detector_examplt(draw=False):
    _model_path = "../weapon_detector.pt"
    yolov7_detector = YOLOV7_Detector(_model_path)
    for frames,date_time in generate_frames():
        results = yolov7_detector.detect(frames)
        if len(results) != 0:
            results[0]['time']=date_time
            print(results)
        if draw:
            frame = draw_bbox(results, frames)
        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break

def draw_bbox(dict_list, image):
    dh, dw, _ = image.shape
    for detection in dict_list:
        t, l, b, r = detection['tlbr']
        cv2.rectangle(image, (t, l), (b, r), (0, 0, 255), 1)
    return image


if __name__ == '__main__':
    video_detector_example(draw=True)
