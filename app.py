#!/usr/bin/env python

# noinspection PyUnresolvedReferences
import custom_logging

import datetime
import json
import logging
import time

import cv2
import numpy as np
from flask import Flask, render_template, Response
from PIL import Image

import distributor
import motion_detectors
import source_images_from_directory


app = Flask(__name__)


def source():
    logging.info("Source is starting")
    image_source = source_images_from_directory.ImageFrameSource(
        directory_path='testing/collected',
        extensions=['.jpg', '.jpeg']
    )
    logging.info("Image source created")
    motion_detector = motion_detectors.MotionDetector1()
    while True:
        for image_and_info in image_source.yield_opencv_image_frames():
            image, info = image_and_info
            logging.info("got image %s", info)
            mrt = motion_detector.process_frame(image)

            frame2 = mrt.frame.copy()
            if mrt.derived_data_is_valid:
                shape = frame2.shape
                total_area = shape[0] * shape[1]

                t_frame = mrt.threshold_after_erode

                contours, _ = cv2.findContours(t_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mrt.contour_area_ratio = sum(cv2.contourArea(contour) for contour in contours) / total_area
                max_value = np.iinfo(t_frame.dtype).max
                mrt.thresholded_area_ratio = np.mean(mrt.threshold_after_erode) / max_value

                hit = mrt.contour_area_ratio > 0.001

                boxes = []
                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    boxes.append((x, y, w, h))
                    # cv2.drawContours(frame2, contours, -1, (0, 255, 0), 1)

                    if hit:
                        cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 255, 0), 1)

                if hit:
                    jpeg_info = {
                        "boxes": boxes,
                    }
                    jpeg_info_s = json.dumps(jpeg_info)

                    # Convert from BGR to RGB
                    img_rgb = cv2.cvtColor(mrt.frame, cv2.COLOR_BGR2RGB)
                    # Convert NumPy array to PIL Image object
                    pil_image = Image.fromarray(img_rgb)

                    # Get the current UTC datetime (timezone-aware)
                    utc_aware_dt = datetime.datetime.now(datetime.timezone.utc)

                    # Convert to ISO 8601 format string (includes +00:00 offset)
                    iso_format_str = utc_aware_dt.isoformat()

                    print(f"UTC Datetime Object: {utc_aware_dt}")
                    print(f"ISO 8601 String: {iso_format_str}")
                    pil_image.save('output_path.jpg', comment=jpeg_info_s)

            mrt.frame2 = frame2

            logging.info("source yielding %s", mrt)
            yield mrt
            time.sleep(2)


logging.info("Creating distributor")
image_distributor = distributor.Distributor(source=source)  # TODO
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
        logging.info("video_feed_gen received mrt %s %s", type(mrt), mrt)
        frame_jpeg = cv2.imencode('.jpg', mrt.frame2)[1].tobytes()
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
        logging.info("video_feed_gen received mrt %s %s", type(mrt), mrt)
        frame_jpeg = cv2.imencode('.jpg', mrt.threshold_after_erode)[1].tobytes()
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n--frame\r\n'


@app.route('/diff_feed')
def diff_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    c = image_distributor.get_receiver()
    logging.info("created %s", c)
    return Response(diff_feed_gen(c), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
