import datetime
import json
import logging

import cv2
import numpy as np
import piexif

import motion_detectors
import utilities


logger = logging.getLogger("ChangeProcessor")
logger.setLevel(logging.INFO)


class ChangeProcessor:
    def __init__(self):
        self.motion_detector = motion_detectors.MotionDetector1(accumulate_alpha=0.5)  # PARAMETER!

        self.last_frame = None
        self.last_timestamp = None
        self.last_info_dict = None

        self.event_id = 1
        self.in_event = False

    def process_frame(self, frame, info):
        timestamp = info.get('timestamp')
        now = datetime.datetime.now(datetime.timezone.utc)
        if timestamp is None:
            timestamp = now

        # TODO: make sure we have consistent frame sizes across frames
        mrt = self.motion_detector.process_frame(frame)

        frame2 = mrt.frame.copy()

        hit = False
        boxes = []

        if mrt.derived_data_is_valid:
            shape = frame2.shape
            total_area = shape[0] * shape[1]

            t_frame = mrt.threshold_after_erode

            contours, _ = cv2.findContours(t_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mrt.contour_area_ratio = sum(cv2.contourArea(contour) for contour in contours) / total_area
            max_value = np.iinfo(t_frame.dtype).max
            mrt.thresholded_area_ratio = np.mean(mrt.threshold_after_erode) / max_value

            hit = mrt.contour_area_ratio > 0.001

            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                boxes.append((x, y, w, h))
                cv2.drawContours(frame2, contours, -1, (0, 255, 0), 1)

                if hit:
                    # cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 255, 0), 1)
                    pass
        else:
            mrt.contour_area_ratio = 0
            mrt.thresholded_area_ratio = 0

        info_dict = {
            "boxes": boxes,
        }

        if hit:
            self.last_info_dict['event_id'] = info_dict['event_id'] = self.event_id
            if not self.in_event:
                logger.info ("event %d is starting", self.event_id)
                self.save_file(self.last_frame, self.last_timestamp, self.last_info_dict)
            self.save_file(frame, timestamp, info_dict)
            self.in_event = True
        else:
            if self.in_event:
                # event is ending
                logger.info ("event %d is ending", self.event_id)
                self.save_file(frame, timestamp, info_dict)
                self.event_id = self.event_id + 1
            else:
                self.last_frame = frame
                self.last_info_dict = info_dict
                self.last_timestamp = timestamp
            self.in_event = False

        text = f"{timestamp.isoformat()} -> {now.isoformat()}, Change ratio: {mrt.contour_area_ratio:.4f}"
        if hit:
            text += f", event {self.event_id}"
        self.draw_text(frame2, text)

        mrt.frame2 = frame2

        return mrt

    def draw_text(self, frame: np.ndarray, text: str):
        height, width, channels = frame.shape

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.25
        color = (0, 0, 255)  # White color (BGR)
        thickness = 1

        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        text_x = int(width * 0.1)
        text_y = int(height * 0.1)

        text_origin = (text_x, text_y)

        cv2.putText(frame, text, text_origin, font, font_scale, color, thickness, cv2.LINE_AA)

    def save_file(self, frame: np.ndarray, timestamp: datetime.datetime, info_dict: dict = None):
        if info_dict is None:
            info_dict = {}
        info_s = json.dumps(info_dict)

        ms = round(timestamp.microsecond / 1000)
        yyyymmddhhmmss = timestamp.strftime("%Y%m%d-%H%M%S")

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

        fn = f'{yyyymmddhhmmss}-{ms:03}'

        exif_bytes = piexif.dump(exif_dict)
        pil_image = utilities.make_pillow_from_cv2(frame)
        # output dir should be PARAMETER!
        pil_image.save(f'output/{fn}.jpg', exif=exif_bytes)
