import cv2
import imagehash
from PIL import Image
from datetime import datetime


def generate_frames(video_path=None, stream=True):
    """
    Parameters:
    stream: True or False, default is True for stream
    video_path: Path to video file if stream=False

    Yields:
    frame: The next frame in real-time
    """
    if stream is True:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(video_path)

    prev_hash = imagehash.hex_to_hash('0000000000000000')
    while True:
        ret, frame = video.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_hash = imagehash.average_hash(Image.fromarray(gray_frame))
            thresh_hold = 10
            if prev_hash is None or (prev_hash - frame_hash) > thresh_hold:
                prev_hash = frame_hash
                if stream is True:
                    yield frame, datetime.now().strftime('"%Y-%m-%d %H:%M:%S')
                if stream is False:
                    timestamp_ms = video.get(cv2.CAP_PROP_POS_MSEC)
                    timestamp_sec = timestamp_ms / 1000.0
                    yield frame, timestamp_sec
    video.release()
    cv2.destroyAllWindows()

# for frames,time_stamp in generate_frames(stream=False,video_path='../video_1.mp4'):
#     cv2.imshow("video", frames)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#          break