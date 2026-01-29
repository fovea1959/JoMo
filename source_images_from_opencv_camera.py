import datetime
import logging
import time

from typing import Generator, Tuple, Dict

import cv2
import numpy as np

import source_images


logger = logging.getLogger("source_opencv")
logger.setLevel(logging.INFO)


"""
consider using v4l2py?
"""


class OpenCVCameraImageSource(source_images.FrameSource):
    def __init__(self, level: int = logging.INFO):
        super().__init__(level)

    def yield_opencv_image_frames(self) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """
        A generator function that yields Pillow images from the opencv camera

        Yields:
        An image frame as a Pillow image.
        """
        logger.info("starting yield_pillow_image_frames")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        try:
            while True:
                ret, cv2_image = cap.read()
                if ret:
                    yield cv2_image, { 'timestamp': datetime.datetime.now(datetime.timezone.utc)}
                else:
                    self.logger.error("v4l2 read error")
                    time.sleep(0.001)
        finally:
            self.logger.info("releasing cap")
            cap.release()
