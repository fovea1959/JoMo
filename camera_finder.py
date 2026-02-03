import sys

import cv2_enumerate_cameras.camera_info

import logging

from cv2_enumerate_cameras import enumerate_cameras
from cv2.videoio_registry import getBackendName

logger = logging.getLogger("get_camera")


def get_camera(name : str):
    i_want = None

    backends = cv2_enumerate_cameras.supported_backends
    for backend in backends:
        backend_name = getBackendName(backend)
        logger.debug("saw backend %d '%s'", backend, backend_name)
        if backend_name in ['DSHOW', 'V4L2']:
            logger.info("Using backend '%s'", backend_name)

            for camera_info in enumerate_cameras(backend):
                logger.debug("saw camera %d, '%s'", camera_info.index, camera_info.name)
                if i_want is None:
                    if name in camera_info.name:
                        logger.info ("found camera '%s': %s:%d", camera_info.name,
                                     getBackendName(camera_info.backend), camera_info.index)
                        i_want = camera_info

    if i_want is not None:
        return i_want.index, i_want.backend
    else:
        return None, None


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    print(get_camera('Integrated'))
