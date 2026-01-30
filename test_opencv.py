import datetime
import logging
import time

from typing import Generator, Tuple, Dict

import cv2
import numpy as np


def write_on_frame(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)  # White color (BGR)
    thickness = 1

    width, height, _ = frame.shape

    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size

    text_x = (width - text_width) // 2
    text_y = (height + text_height) // 2

    text_origin = (text_x, text_y)

    cv2.putText(frame, text, text_origin, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def frame():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # this will not quite do what you want
    # cap.set(cv2.CAP_PROP_XI_RECENT_FRAME, 1)
    try:
        while True:
            dropped = 0
            while True:
                before = time.time()
                ret, cv2_image = cap.read()
                after = time.time()
                interval = after - before
                print(interval)
                if ret:
                    if after - before > 0.01:
                        break
                else:
                    logging.error("v4l2 read error")
                dropped = dropped + 1
                time.sleep(0.001)
            print('===================')
            yield cv2_image, { 'timestamp': datetime.datetime.now(datetime.timezone.utc), 'interval': interval, 'dropped': dropped}
    finally:
        logging.info("releasing cap")
        cap.release()


def main():
    for cv2_image, image_info in frame():
        m = f'{image_info.get('interval')} {image_info.get('dropped')} {image_info.get('timestamp').strftime('%H:%M:%S.%s')}'
        write_on_frame(cv2_image, m)
        cv2.imshow("image", cv2_image)
        if cv2.waitKey(2000) == ord('q'):
            break


if __name__ == '__main__':
    main()