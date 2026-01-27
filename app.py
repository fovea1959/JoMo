#!/usr/bin/env python
import custom_logging

import logging
import time

import cv2
from flask import Flask, render_template, Response

import distributor
import motion_detectors
import source_images_from_directory


app = Flask(__name__)


def source():
    logging.info("Source is starting")
    image_source = source_images_from_directory.ImageFrameSource(directory_path='testing/123', extensions=['.jpg'])
    motion_detector = motion_detectors.MotionDetector1()
    while True:
        for image_and_info in image_source.yield_opencv_image_frames():
            image, info = image_and_info
            logging.info ("got image %s", info)
            mrt = motion_detector.process_frame(image)
            logging.info ("source yielding %s", mrt)
            yield mrt
            time.sleep(2)


logging.info("Creating distributor")
image_distributor = distributor.Distributor(source=source)  # TODO
logging.info("Created distributor")


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def encode_cv_image_as_jpeg(im):
    buffer = cv2.imencode('.jpg', im)[1]
    return buffer.tobytes()


def gen(receiver: distributor.Receiver):
    """Video streaming generator function."""
    yield b'--frame\r\n'
    while True:
        logging.debug("gen waiting for mrt")
        mrt = receiver.get_last_result()
        logging.info("gen received mrt %s %s", type(mrt), mrt)
        frame_jpeg = encode_cv_image_as_jpeg(mrt.frame)
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n--frame\r\n'


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    c = image_distributor.get_receiver()
    logging.info("created %s", c)
    return Response(gen(c), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(threadName)s %(module)s %(funcName)s %(levelname)s %(message)s")
    app.run(host='0.0.0.0', threaded=True)
