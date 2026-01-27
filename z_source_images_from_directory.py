import time
import logging

import numpy as np
import cv2

from base_camera import Distributor


class Camera(Distributor):
    """An emulated camera implementation that streams a repeated sequence of
    files 1.jpg, 2.jpg and 3.jpg at a rate of one frame per second."""
    imgs = [open(f + '.jpg', 'rb').read() for f in ['1', '2', '3']]

    @staticmethod
    def frames():
        while True:
            jpeg_bytes = Camera.imgs[int(time.time()) % 3]
            np_arr = np.asarray(bytearray(jpeg_bytes), dtype="uint8")
            rv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1

            logging.info("camera yielding %s", type(rv))
            yield rv
            time.sleep(1)
