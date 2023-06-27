import cv2 as cv
from detector.detector import YoloV7Detector




def draw_bbox(bbox, image):
    dh, dw, _ = image.shape
    for box in bbox:
        x, y, w, h = box
        x,y,w,h=int(x),int(y),int(w),int(h)
        cv.rectangle(image, (x,y), (w,h), (0, 0, 255), 1)
    return image






def image_detection(draw=False):
    image_path=input('Image path: ')
    image_display=cv.imread(image_path)
    _model_path='/home/ishwor/Desktop/TreeLeaf/yolov7/monkey_image_detection.pt'
    yolov7_detector = YoloV7Detector(_model_path)
    model_prediction=yolov7_detector.detect(image_display)
    if draw:
        image__=draw_bbox(model_prediction,image_display)
        cv.imshow('Model Predicted Image',image__)
        cv.waitKey(0)
        cv.distoroyAllWindows()

def video_detection(draw=False):
    _model_path='../monkey_image_detection.pt'
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

# def webcam_detection(draw=False):
#     _model_path='../monkey_image_detection.pt'
#     yolov7_detector = YoloV7Detector(_model_path)
#     cap=cv.VideoCapture(0)
#     while True:
#         ret,frame=cap.read()
#         if not ret:
#             print('Error retrieving frames from webcam')
#             break
#         results = yolov7_detector.detect(frame)
#         print(results)
#         if draw:
#             frame = draw_bbox(results, frame)
#         cv.imshow("video", frame)
#         key = cv.waitKey(1) & 0xFF
#         if key == ord('q'):
#             cv.destroyAllWindows()
#             break
def webcam_detection(draw=False):
    _model_path = '../monkey_image_detection.pt'
    yolov7_detector = YoloV7Detector(_model_path)
    cap = cv.VideoCapture(0)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Reduce frame size for faster inference (adjust values as needed)
    target_width, target_height = 640, 480
    cap.set(cv.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, target_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error retrieving frames from webcam')
            break

        # Perform asynchronous inference or use a separate thread here if applicable

        results = yolov7_detector.detect(frame)
        print(results)

        if draw:
            frame = draw_bbox(results, frame)

        cv.imshow("video", frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            cv.destroyAllWindows()
            break

    # Release the capture and restore frame size
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    image_detection (draw=True)
# webcam_detection()
