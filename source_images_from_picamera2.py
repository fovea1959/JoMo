import logging

from typing import Generator, Tuple, Dict

from picamera2 import Picamera2, Preview
from libcamera import Transform

import numpy as np

import source_images

logger = logging.getLogger("source_picam2")
logger.setLevel(logging.INFO)


class PiCamera2FrameSource(source_images.FrameSource):
    def __init__(self, level: int = logging.INFO):
        super().__init__(level)
        self.picam2 = Picamera2()
        # picam2.options['quality'] = 95
        # print(json.dumps(picam2.sensor_modes, indent=1, default=lambda o: o.__dict__, sort_keys=True))
        # print("configuring", picam2)
        capture_config = self.picam2.create_still_configuration(
            main={'size': (3280 // 2, 2464 // 2)},
            transform=Transform(hflip=True, vflip=True)
        )
        self.picam2.configure(capture_config)

    def yield_opencv_image_frames(self) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """
        A generator function that yields Pillow images from the Pi camera

        Yields:
        An image frame as a Pillow image.
        """
        logger.info("starting yield_pillow_image_frames")
        self.picam2.start()
        while True:
            img = self.picam2.capture_array()
            yield img, {}

        # self.picam2.close()
