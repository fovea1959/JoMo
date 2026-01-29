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

    def yield_opencv_image_frames(self) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        pass


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
        try:
            import source_images_from_opencv_camera
            rv = source_images_from_opencv_camera.OpenCVCameraImageSource()
        except ImportError as exc:
            logging.error("Unable to instantiate camera", exc_info=exc)
            source_images_from_opencv_camera = None  # take care of warning message

    if rv is None:
        logging.warning("no camera, so using 123 images")
        import source_images_from_files
        rv = source_images_from_files.FilesFrameSource("testing/123", forever=True)

    return rv
