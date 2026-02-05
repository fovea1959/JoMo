import logging
import platform

from pathlib import Path
from typing import Generator, Tuple, Dict

import cv2
from PIL import Image
import numpy as np


import utilities


class FrameSource:
    def __init__(self, log_level: int | str = logging.INFO):
        self.logger = logging.getLogger(str(type(self)))
        ll = log_level
        if type(log_level) == str:
            ll = logging.getLevelNamesMapping().get(log_level.upper())
            if ll is None:
                self.logger.error("cannot set logging level to '%s'", log_level)
                ll = logging.INFO
        self.logger.setLevel(ll)

    def yield_opencv_image_frames(self) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        pass


def fetch_frame_source(source: str = None, **kwargs) -> FrameSource:
    rv = None

    if source is not None:
        if source.lower() == 'picamera':
            import source_images_from_picamera2
            rv = source_images_from_picamera2.PiCamera2FrameSource(strict_args = False, **kwargs)

        if source.lower() == 'opencv-camera':
            import source_images_from_opencv_camera
            rv = source_images_from_opencv_camera.OpenCVCameraImageSource(strict_args = False, **kwargs)

        if source.lower() == 'files':
            import source_images_from_files
            rv = source_images_from_files.FilesFrameSource(strict_args = False, **kwargs)

    if rv is None:
        raise AttributeError("no input source has been specified")

    return rv
