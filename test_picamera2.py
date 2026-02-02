#!/usr/bin/python3

# Obtain an image from the camera along with the exact metadata that
# that describes that image.

import datetime
import time

from picamera2 import Picamera2, Preview

picam2 = Picamera2()
#picam2.start_preview(Preview.QTGL)

#preview_config = picam2.create_preview_configuration()
#picam2.configure(preview_config)

picam2.start()
# prime the pump
request = picam2.capture_request(flush=True)
request.release()

while True:
    before_ns = time.monotonic_ns()
    request = picam2.capture_request(flush=True)
    frame = request.make_array("main")  # image from the "main" stream
    metadata = request.get_metadata()
    request.release()  # requests must always be returned to libcamera

    #image.show()
    #print(metadata)
    now = datetime.datetime.now()
    now_ns = time.monotonic_ns()
    then_ns = metadata.get('SensorTimestamp', 0)
    lag_s = (now_ns - before_ns) / 1000000000
    then = now - datetime.timedelta(seconds=lag_s)
    #print(before_ns, now_ns)
    #print(then, now)
    print(lag_s)

    time.sleep(2)

