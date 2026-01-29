import logging
import platform

from pathlib import Path
from typing import Generator, Tuple, Dict

import cv2
from PIL import Image
import numpy as np


import utilities


class FrameSource:
    def __init__(self, level: int = logging.INFO):
        self.logger = logging.getLogger(str(type(self)))
        self.logger.setLevel(level)

    def yield_pillow_image_frames(self) -> Generator[Tuple[Image, Dict], None, None]:
        pass

    def yield_opencv_image_frames(self) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """
        A generator function that iterates over image files in a directory
        and yields each image as a NumPy array (frame).

        Yields:
        An image frame as a OpenCV NumPy array.
        """
        self.logger.info("starting yield_opencv_image_frames")
        for pillow_image_frame, info in self.yield_pillow_image_frames():
            opencv_image_frame = utilities.make_cv2_from_pillow(pillow_image_frame)
            yield opencv_image_frame, info


def fetch_frame_source() -> FrameSource:
    rv = None
    raspberry_pi = 'rpi' in getattr(platform.uname(), 'release')
    if raspberry_pi:
        try:
            import source_images_from_picamera2
            rv = source_images_from_picamera2.PiCamera2FrameSource()
        except ImportError as exc:
            logging.error("Unable to instantiate camera", exc_info=exc)
            source_images_from_picamera2 = None  # take care of warning message
    else:
        logging.info("not a pi!")

    if rv is None:
        import source_images_from_files
        rv = source_images_from_files.FilesFrameSource("testing/123", forever=True)

    return rv


