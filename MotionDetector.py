import logging

import cv2
import numpy as np


class DetectorParameters:
    def __init__(self):
        self.threshold = 25
        self.accumulate_alpha = 0.2
        self.post_threshold_dilate_iterations = 2
        self.blur_size = 21

    def load_from_dict(self, d : dict):
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def __repr__(self):
        return str(vars(self))


class Diags:
    def __init__(self):
        self.background = None
        self.frame_delta = None
        self.threshold = None
        self.threshold_after_dilate = None
        self.how_much_changed = None

    def items(self):
        return self.__dict__.items()


class Detector:
    def __init__(self, detector_parameters : DetectorParameters = None):
        self.dp = detector_parameters
        self._background = None

    def process_frame(self, frame = None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.dp.blur_size, self.dp.blur_size), 0)  # Reduce noise

        # Initialize running average
        if self._background is None:
            self._background = gray.copy().astype("float")
            return False, None

        diags = Diags()

        print(gray[0, 0], self._background[0, 0])

        print(f"type(gray)={type(gray)}, type(_background)={type(self._background)}")

        # Update running average: weighted sum
        cv2.accumulateWeighted(gray, self._background, self.dp.accumulate_alpha)
        diags.background = self._background
        # Convert back to uint8 for difference calculation
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self._background))
        diags.frame_delta = frame_delta

        # Threshold the image
        thresh = cv2.threshold(frame_delta, self.dp.threshold, 255, cv2.THRESH_BINARY)[1]
        diags.threshold = thresh

        thresh = cv2.dilate(thresh, None, iterations=self.dp.post_threshold_dilate_iterations)
        diags.threshold_after_dilate = thresh

        diags.how_much_changed = np.mean(diags.threshold)

        return True, diags


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    dp = DetectorParameters()
    logging.info("dp 0: %s %s", dp, repr(dp))
    dp.load_from_dict({"a": 1, "threshold": 25})
    logging.info("dp 1: %s %s", dp, repr(dp))
