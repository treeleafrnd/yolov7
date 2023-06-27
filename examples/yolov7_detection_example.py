import  cv2 as cv
from detector.detector import YoloV7Detector


def draw_bbox(bbox, image):
    dh,dw,_=image.shape
    for i in bbox:
        x, y, w, h=bbox[i]
        cv.rectangel(image,(x,y),(w,h),(0,0,255),1)
        return image





def image_detection(draw=False):
    image_path='/home/ishwor/Desktop/TreeLeaf/yolov7/inference/images/different_Monkey Images from Video1440.jpg'
    image_display=cv.imread(image_path)
    _model_path='../monkey_image_detection.pt'
    yolov7_detector = YoloV7Detector(_model_path)
    model_prediction=yolov7_detector.detect(image_display)
    if draw:
        image=draw_bbox(model_prediction,image_display)
        cv.imshow('Model Predicted Image',image)
        cv.waitKey(0)
        cv.distoroyAllWindows()

def video_detection(draw=False):
    _model_path='monkey_image_detection.pt'
    video_path=input('Enter the path of the video')
    yolov7_detector = YoloV7Detector(_model_path)
    video_display = cv.VideoCapture(video_path)
    while True:
        ret, frame = video_display.read()
        if not ret:
            print('../Error retrieving frames from video')
            break
        results = yolov7_detector.detect(frame)
        print(results)
        if draw:
            frame = draw_bbox(results, frame)
        cv.imshow("video", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            cv.destroyAllWindows()
            break
    video_display.release()

def webcam_detection(draw=False):
    _model_path='../monkey_image_detection.pt'
    yolov7_detector = YoloV7Detector(_model_path)
    cap=cv.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        if not ret:
            print('Error retrieving frames from webcam')
            break
        results = yolov7_detector.detect(frame)
        print(results)
        if draw:
            frame = draw_bbox(results, frame)
        cv.imshow("video", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            cv.destroyAllWindows()
            break

# webcam_detection()
# image_detection()