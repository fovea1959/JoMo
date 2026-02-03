import datetime
import json
import logging

import cv2
import numpy as np
import piexif
import piexif.helper

import configuration
import motion_detectors
import utilities

from PIL import Image

logger = logging.getLogger("ChangeProcessor")
logger.setLevel(logging.INFO)


class ChangeProcessor:
    def __init__(self):
        motion_parameters = configuration.settings.get('motion', {})
        self.motion_detector = motion_detectors.MotionDetector1(**motion_parameters)
        self.hit_ratio = motion_parameters.get('hit_ratio', 0.001)

        self.last_frame = None
        self.last_timestamp = None
        self.last_info_dict = None

        self.event_id = 1
        self.in_event = False

        output_parameters = configuration.settings.get('output', {})
        self.output_directory = output_parameters.get('directory', 'output')
        self.save_delta = output_parameters.get('save_delta', False)
        self.save_eroded = output_parameters.get('save_eroded', False)
        self.save_marked_up = output_parameters.get('save_marked_up', False)
        self.save_background = output_parameters.get('save_background', False)

    def process_frame(self, frame, info):
        timestamp = info.get('timestamp')
        now = datetime.datetime.now(datetime.timezone.utc)
        if timestamp is None:
            timestamp = now
        info['processed'] = now.isoformat()

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

            hit = mrt.contour_area_ratio > self.hit_ratio

            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                boxes.append((x, y, w, h))
                if self.save_marked_up:
                    cv2.drawContours(frame2, contours, -1, (0, 255, 0), 1)

                    if hit:
                        # cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 255, 0), 1)
                        pass
        else:
            mrt.contour_area_ratio = 0
            mrt.thresholded_area_ratio = 0

        info_dict = {
            "source_info": info,
            "contour_area_ratio": mrt.contour_area_ratio
        }

        detailed_info_dict = {
            "boxes": boxes,
        }

        if self.save_marked_up:
            text = f"{timestamp.isoformat()}\r{now.isoformat()}\rChange ratio: {mrt.contour_area_ratio:.4f}"
            if hit:
                text += f"\rEvent {self.event_id}"
            h, w = frame2.shape[:2]
            utilities.add_text_to_image(frame=frame2, text=text, font_scale=0.5, top_left_xy=(h // 20, w // 20),
                                        font_color_rgb=(255, 255, 255),
                                        bg_color_rgb=(128, 128, 128))

        if hit:
            self.last_info_dict['event_id'] = info_dict['event_id'] = self.event_id
            if not self.in_event:
                logger.info("event %d is starting", self.event_id)
                self.save_file(self.last_frame, self.last_timestamp, info_dict=self.last_info_dict)

            self.save_file(frame, timestamp, info_dict=info_dict, detailed_info_dict=detailed_info_dict)
            if self.save_background:
                self.save_file(mrt.background, timestamp, info_dict=info_dict, detailed_info_dict=detailed_info_dict,
                               suffix="-background")
            if self.save_delta:
                self.save_file(mrt.frame_delta, timestamp, info_dict=info_dict, detailed_info_dict=detailed_info_dict,
                               suffix="-delta")
            if self.save_eroded:
                self.save_file(mrt.threshold_after_erode, timestamp, info_dict=info_dict,
                               detailed_info_dict=detailed_info_dict,
                               suffix="-eroded", one_bit=True)
            if self.save_marked_up:
                self.save_file(frame2, timestamp, info_dict=info_dict, detailed_info_dict=detailed_info_dict,
                               suffix="-markedup")

            self.in_event = True
        else:
            if self.in_event:
                # event is ending
                logger.info("event %d is ending", self.event_id)

                self.save_file(frame, timestamp, info_dict=info_dict, detailed_info_dict=detailed_info_dict)
                if self.save_background:
                    self.save_file(mrt.background, timestamp, info_dict=info_dict,
                                   detailed_info_dict=detailed_info_dict,
                                   suffix="-background")
                if self.save_delta:
                    self.save_file(mrt.frame_delta, timestamp, info_dict=info_dict,
                                   detailed_info_dict=detailed_info_dict,
                                   suffix="-delta")
                if self.save_eroded:
                    self.save_file(mrt.threshold_after_erode, timestamp, info_dict=info_dict,
                                   detailed_info_dict=detailed_info_dict,
                                   suffix="-eroded", one_bit=True)
                if self.save_marked_up:
                    self.save_file(frame2, timestamp, info_dict=info_dict,
                                   detailed_info_dict=detailed_info_dict, suffix="-markedup")

                self.event_id = self.event_id + 1
            else:
                self.last_frame = frame
                self.last_info_dict = info_dict
                self.last_timestamp = timestamp
            self.in_event = False

        mrt.frame2 = frame2

        return mrt

    @staticmethod
    def draw_text(frame: np.ndarray, text: str):
        height, width, channels = frame.shape

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.25
        color = (0, 0, 255)  # White color (BGR)
        thickness = 1

        # text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # text_width, text_height = text_size

        text_x = int(width * 0.1)
        text_y = int(height * 0.1)

        text_origin = (text_x, text_y)

        cv2.putText(frame, text, text_origin, font, font_scale, color, thickness, cv2.LINE_AA)

    def save_file(self, frame: np.ndarray, timestamp: datetime.datetime, info_dict: dict = None,
                  detailed_info_dict: dict = None, description: str = None, suffix: str = "", one_bit=False):
        if info_dict is None:
            info_dict = {}
        info_s = json.dumps(info_dict, default=utilities.json_serializer)
        if detailed_info_dict is None:
            detailed_info_dict = {}
        detailed_info_s = json.dumps(detailed_info_dict, default=utilities.json_serializer)

        ms = round(timestamp.microsecond / 1000)

        exif_dict = {
            "0th": {
                piexif.ImageIFD.ImageHistory: detailed_info_s.encode(),
            },
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: timestamp.strftime("%Y:%m:%d %H:%M:%S"),
                piexif.ExifIFD.SubSecTimeOriginal: f"{ms:03}",
                piexif.ExifIFD.OffsetTimeOriginal: "+00:00",
                piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(info_s),
            },
            "GPS": {
            },
            "1st": {
            }
        }
        if description is not None:
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description.encode('ascii')

        yyyymmddhhmmss = timestamp.strftime("%Y%m%d-%H%M%S")
        fn = f'{yyyymmddhhmmss}-{ms:03}'

        exif_bytes = piexif.dump(exif_dict)
        pil_image = utilities.make_pillow_from_cv2(frame)
        if one_bit:
            # this does not help with JPEG output, but leaving it in
            pil_image = pil_image.convert("1", dither=Image.Dither.NONE)
        logging.info("cv2 image %s %s -> pillow image %s %s", frame.shape, frame.dtype, pil_image.size,
                     pil_image.mode)
        pil_image.save(f'{self.output_directory}/{fn}{suffix}.jpg', exif=exif_bytes, quality=95)
