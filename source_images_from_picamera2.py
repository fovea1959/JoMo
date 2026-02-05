import datetime
import logging
import time

from typing import Generator, Tuple, Dict

from picamera2 import Picamera2, Preview
from libcamera import Transform

import numpy as np

import configuration
import source_images

logger = logging.getLogger("source_picam2")
logger.setLevel(logging.INFO)


class PiCamera2FrameSource(source_images.FrameSource):
    def __init__(self, log_level: int | str = logging.INFO, vflip: bool = None, hflip: bool = None, resolution=None, strict_args = True, **kwargs):
        super().__init__(log_level)

        if strict_args and len(kwargs) > 0:
            raise TypeError(f"unexpected arguments: {kwargs}")

        self.picam2 = Picamera2()

        capture_config_parameters = {}

        transform_parameters = {}
        if vflip is not None:
            transform_parameters['vflip'] = vflip
        if hflip is not None:
            transform_parameters['hflip'] = hflip
        if len(transform_parameters) > 0:
            capture_config_parameters['transform'] = Transform(**transform_parameters)

        if resolution is not None:
            capture_config_parameters['main'] = { 'size': resolution }

        # picam2.options['quality'] = 95

        # print(json.dumps(picam2.sensor_modes, indent=1, default=lambda o: o.__dict__, sort_keys=True))

        logger.info ("Configuring picamera with %s", capture_config_parameters)
        capture_config = self.picam2.create_still_configuration(**capture_config_parameters)
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
