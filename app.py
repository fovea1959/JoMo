#!/usr/bin/env python

# noinspection PyUnresolvedReferences
import custom_logging

import datetime
import json
import logging
import time

import cv2
import numpy as np
import piexif

from flask import Flask, render_template, Response

import distributor
import motion_detectors
import source_images_from_directory
import utilities

app = Flask(__name__)


def source():
    logger = logging.getLogger("source")
    logger.setLevel(logging.INFO)
    logger.info("Source is starting")
    image_source = source_images_from_directory.ImageFrameSource(
        directory_path='testing/collected',
        extensions=['.jpg', '.jpeg']
    )
    logger.info("Image source created")
    motion_detector = motion_detectors.MotionDetector1()
    while True:
        for image_and_info in image_source.yield_opencv_image_frames():
            image, info = image_and_info
            logger.debug("got image %s", info)

            # TODO: make sure we have consistent frame sizes across frames
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

                    # Define EXIF data dictionary (using standard numeric tags)
                    exif_dict = {
                        "0th": {
                            piexif.ImageIFD.ImageDescription: "A generated image with metadata"
                        },
                        "Exif": {
                            # piexif.ExifIFD.DateTimeOriginal: "2025:01:28 10:00:00",
                            # piexif.ExifIFD.UserComment:
                        },
                        "GPS": {},  # Add GPS data here if needed
                        "1st": {}
                    }

                    # Dump the dictionary to a raw EXIF byte string
                    exif_bytes = piexif.dump(exif_dict)

                    pil_image = utilities.make_pillow_from_cv2(mrt.frame)

                    # Get the current UTC datetime (timezone-aware)
                    utc_aware_dt = datetime.datetime.now(datetime.timezone.utc)
                    ms = round(utc_aware_dt.microsecond / 1000)
                    yyyymmddhhmmss = utc_aware_dt.strftime("%Y%m%d-%H%M%S")

                    pil_image.save(f'output/{yyyymmddhhmmss}-{ms:03}.jpg', exif=exif_bytes)

            mrt.frame2 = frame2

            logger.debug("source yielding %s", mrt)
            yield mrt
            time.sleep(1)


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
        logging.debug("video_feed_gen received mrt %s %s", type(mrt), mrt)
        frame_jpeg = utilities.make_jpeg_from_cv2(mrt.frame2)
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
    app.run(host='0.0.0.0', threaded=True)
