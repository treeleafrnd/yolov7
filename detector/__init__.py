if __name__ == '__main__':
    image_path = 'yolov7/inference/images/different_Monkey Images from Video3418.jpg'
    image = cv2.imread(image_path)
    _model_path = "yolov7/best.pt"
    yolov7_detector = YoloV7Detector(_model_path)
    yolov7_detector.detect(image)