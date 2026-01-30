import datetime
import logging
import time

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

        # resolution & flip should be PARAMETER!

        # picam2.options['quality'] = 95
        # print(json.dumps(picam2.sensor_modes, indent=1, default=lambda o: o.__dict__, sort_keys=True))
        # print("configuring", picam2)
        capture_config = self.picam2.create_still_configuration(
            main={'size': (3280 // 2, 2464 // 2)},  # PARAMETER!
            transform=Transform(hflip=True, vflip=True)
        )
        self.picam2.configure(capture_config)

    def yield_opencv_image_frames(self) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """
        A generator function that yields opencv images from the Pi camera

        Yields:
        An image frame as an opencv image.
        """
        logger.info("starting yield_pillow_image_frames")
        self.picam2.start()
        # prime the pump
        request = self.picam2.capture_request(flush=True)
        request.release()

        try:
            while True:
                request = self.picam2.capture_request(flush=True)
                now = datetime.datetime.now(datetime.timezone.utc)
                now_ns = time.monotonic_ns()

                frame = request.make_array("main")  # image from the "main" stream
                metadata = request.get_metadata()
                request.release()  # requests must always be returned to libcamera

                then_ns = metadata.get('SensorTimestamp', 0)
                lag_s = (now_ns - then_ns) / 1000000000
                then = now - datetime.timedelta(seconds=lag_s)

                yield frame, {'timestamp': then}
        finally:
            self.picam2.close()
