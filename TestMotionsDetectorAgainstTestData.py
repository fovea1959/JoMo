import logging
import os
import time

import cv2
import numpy
import numpy as np

import MotionDetector
import YieldImageFrames


def main():
    os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts/truetype/dejavu/'

    detector_parameters = MotionDetector.DetectorParameters()
    detector_parameters.blur_size = 3
    detector_parameters.post_threshold_erode_iterations = 1
    detector = MotionDetector.Detector(detector_parameters)

    # Replace 'your_image_directory' with the path to your image folder
    image_dir = 'testing/collected'
    # Specify extensions if needed, or leave as None to attempt opening all files
    valid_extensions = ['.png', '.jpg', '.jpeg']

    frame_source = YieldImageFrames.ImageFrameSource(image_dir, valid_extensions)

    # Iterate through the image frames using the generator
    for i, frame_and_info in enumerate(frame_source.yield_opencv_image_frames()):
        frame, info = frame_and_info
        print(f"Processing frame {i+1} {info}: Shape {frame.shape}, Data Type {frame.dtype}")
        # Add your image processing logic here (e.g., OpenCV processing, model inference)

        # Scale both width and height by 50% (0.5)
        scale_factor = 0.25

        # Use cv2.INTER_AREA for downscaling for best quality
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        cv2.imshow("Frame", frame)

        tf, diags = detector.process_frame(frame)

        if diags is not None:
            frame2 = frame.copy()
            shape = frame2.shape
            total_area = shape[0] * shape[1]

            t_frame = diags.threshold_after_erode

            contours, _ = cv2.findContours(t_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            diags.contour_area_ratio = sum(cv2.contourArea(contour) for contour in contours) / total_area
            max_value = np.iinfo(t_frame.dtype).max
            diags.thresholded_area_ratio = np.mean(diags.threshold_after_erode) / max_value
            diags.bounding_rects = []

            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                diags.bounding_rects.append((x, y, w, h))
                cv2.drawContours(frame2, contours, -1, (0, 255, 0), 1)

                if diags.contour_area_ratio > 0.001:
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 255, 0), 1)

            cv2.imshow('Motion Detection', frame2)

            print("===============================")

            for k, v in diags.items():
                if type(v) is numpy.ndarray:
                    dt = v.dtype
                    if dt == np.uint8:
                        cv2.imshow(k, v)
                    else:
                        cv2.imshow(k, np.uint8(v))
                else:
                    print(k, v)

        k = None
        while True:
            k = cv2.waitKey(1)
            if k != -1:
                break

        if k & 0xFF == ord('q'):
            break

        # time.sleep(0.5)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
