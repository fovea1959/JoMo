import cv2
import numpy as np


class MotionDetector1Result:
    def __init__(self):
        self.frame = None
        self.background = None
        self.frame_delta = None
        self.threshold = None
        self.threshold_after_erode = None
        self.derived_data_is_valid = False

    def items(self):
        return self.__dict__.items()

    def __str__(self):
        return f"frame={type(self.frame)}, threshold_after_erode={type(self.threshold_after_erode)}"


class MotionDetector1:
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
        rv = MotionDetector1Result()
        rv.frame = frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self._dp_blur_size, self._dp_blur_size), 0)  # Reduce noise

        # Initialize running average
        if self._background is None:
            self._background = gray.copy().astype("float")

            height, width, channels = frame.shape
            fill_color = (255, 0, 0)
            block = np.zeros((height, width, 3), np.uint8)
            block[:] = fill_color

            rv.frame_delta = block
            rv.threshold = block
            rv.threshold_after_erode = block
        else:
            # Update running average: weighted sum
            cv2.accumulateWeighted(gray, self._background, self._dp_accumulate_alpha)
            rv.background = self._background

            # Convert back to uint8 for difference calculation
            frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self._background))
            rv.frame_delta = frame_delta

            # Threshold the image
            thresh = cv2.threshold(frame_delta, self._dp_threshold, 255, cv2.THRESH_BINARY)[1]
            rv.threshold = thresh

            thresh = cv2.erode(thresh, None, iterations=self._dp_post_threshold_erode_iterations)
            rv.threshold_after_erode = thresh

            rv.derived_data_is_valid = True

        return rv


