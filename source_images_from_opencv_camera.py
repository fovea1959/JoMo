import datetime
import logging
import time

from typing import Generator, Tuple, Dict

import cv2
import numpy as np

import camera_finder
import source_images

logger = logging.getLogger("source_opencv")
logger.setLevel(logging.INFO)


class OpenCVCameraImageSource(source_images.FrameSource):
    def __init__(self, log_level: int | str = logging.INFO, camera_name: str = None, resolution=None, strict_args=True,
                 **kwargs):
        super().__init__(log_level)

        if strict_args and len(kwargs) > 0:
            raise TypeError(f"unexpected arguments: {kwargs}")

        camera_id, backend = camera_finder.get_camera(camera_name)
        if camera_id is None:
            raise Exception(f"Cannot find camera {camera_name}")
        self.cap = cv2.VideoCapture(camera_id, backend)

        if resolution is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # Get frame width and height
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Using camera %d from backend %s @ %s",
                    camera_id, camera_finder.getBackendName(backend), f"{width}x{height}")
        # reduce # of buffers, so we don't need to flush so many
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def yield_opencv_image_frames(self) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """
        A generator function that yields opencv images from the opencv camera

        Yields:
        An image frame as an opencv image.
        """
        logger.info("starting yield_pillow_image_frames")
        try:
            while True:
                dropped = 0
                while True:
                    # read until it actually takes a little time to collect.
                    # that will be a fresh image (not previously buffered)
                    before = time.time()
                    ret, cv2_image = self.cap.read()
                    after = time.time()
                    interval = after - before
                    if ret:
                        if interval > 0.01:
                            break
                    else:
                        logging.error("v4l2 read error")
                    dropped = dropped + 1
                    time.sleep(0.001)
                yield cv2_image, {'timestamp': datetime.datetime.now(datetime.timezone.utc), 'interval': interval,
                                  'dropped': dropped}
        finally:
            self.logger.info("releasing cap")
            self.cap.release()
            self.cap = None
