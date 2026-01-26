import cv2
import logging
import numpy as np

import MotionDetector
import CameraFinder


def main():

# Initialize webcam
    cam_index, cam_backend = camera_finder.get_camera('Integrated')
    cap = cv2.VideoCapture(cam_index, cam_backend)

    detector_parameters = MotionDetector.DetectorParameters()
    detector = MotionDetector.Detector(detector_parameters)

    while True:
        ret, frame = cap.read()
        if not ret: break

        cv2.imshow("Frame", frame)

        tf, diags = detector.process_frame(frame)

        if diags is not None:
            cv2.imshow("Background", np.uint8(diags.background))
            cv2.imshow("Motion", diags.threshold)
            cv2.imshow("Motion-after", diags.threshold_after_erode)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()