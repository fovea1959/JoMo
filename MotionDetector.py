import logging

import cv2
import numpy as np


class Diags:
    def __init__(self):
        self.background = None
        self.frame_delta = None
        self.threshold = None
        self.threshold_after_erode = None

    def items(self):
        return self.__dict__.items()


class Detector:
    def __init__(self,
                 threshold: int = 25,
                 accumulate_alpha: float = 0.2,
                 post_threshold_erode_iterations: int = 1,
                 blur_size: int = 3,
                 **kwargs):
        self._background = None
        self._dp_threshold = threshold
        self._dp_accumulate_alpha = accumulate_alpha
        self._dp_post_threshold_erode_iterations = post_threshold_erode_iterations
        self._dp_blur_size = blur_size

    def process_frame(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self._dp_blur_size, self._dp_blur_size), 0)  # Reduce noise

        # Initialize running average
        if self._background is None:
            self._background = gray.copy().astype("float")
            return None

        diags = Diags()

        # Update running average: weighted sum
        cv2.accumulateWeighted(gray, self._background, self._dp_accumulate_alpha)
        diags.background = self._background

        # Convert back to uint8 for difference calculation
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self._background))
        diags.frame_delta = frame_delta

        # Threshold the image
        thresh = cv2.threshold(frame_delta, self._dp_threshold, 255, cv2.THRESH_BINARY)[1]
        diags.threshold = thresh

        thresh = cv2.erode(thresh, None, iterations=self._dp_post_threshold_erode_iterations)
        diags.threshold_after_erode = thresh

        return diags


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
