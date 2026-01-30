#!/usr/bin/env python

# noinspection PyUnresolvedReferences
import custom_logging

import logging
import time

from flask import Flask, render_template, Response

import change_processor
import distributor
import motion_detectors
import source_images
import utilities

app = Flask(__name__)


def source():
    logger = logging.getLogger("source")
    logger.setLevel(logging.INFO)
    logger.info("Creating image source")
    frame_source = source_images.fetch_frame_source()
    logger.info("Image source created")

    cp = change_processor.ChangeProcessor()
    while True:
        for frame_and_info in frame_source.yield_opencv_image_frames():
            frame, info = frame_and_info
            logger.debug("got image %s", info)
            mrt = cp.process_frame(frame, info)
            logger.debug("source yielding %s", mrt)
            yield mrt
            time.sleep(10)  # PARAMETER!


logging.info("Creating distributor")
image_distributor = distributor.Distributor(source=source)
logging.info("Created distributor")


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def video_feed_gen(receiver: distributor.Receiver):
    """Video streaming generator function."""
    yield b'--frame\r\n'
    while True:
        logging.debug("video_feed_gen waiting for mrt")
        mrt: motion_detectors.MotionDetector1Result = receiver.get_last_result()
        logging.debug("video_feed_gen received mrt %s %s", type(mrt), mrt)
        frame_jpeg = utilities.make_jpeg_from_cv2(mrt.frame2, quality=95)
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n--frame\r\n'


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    c = image_distributor.get_receiver()
    logging.info("created %s", c)
    return Response(video_feed_gen(c), mimetype='multipart/x-mixed-replace; boundary=frame')


def diff_feed_gen(receiver: distributor.Receiver):
    """Video streaming generator function."""
    yield b'--frame\r\n'
    while True:
        logging.debug("video_feed_gen waiting for mrt")
        mrt: motion_detectors.MotionDetector1Result = receiver.get_last_result()
        logging.debug("video_feed_gen received mrt %s %s", type(mrt), mrt)
        frame_jpeg = utilities.make_jpeg_from_cv2(mrt.threshold_after_erode)
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n--frame\r\n'


@app.route('/diff_feed')
def diff_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    c = image_distributor.get_receiver()
    logging.info("created %s", c)
    return Response(diff_feed_gen(c), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)  ### , use_reloader=False)
