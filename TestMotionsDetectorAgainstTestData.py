import logging
import time

import cv2
import numpy
import numpy as np

import MotionDetector
import YieldImageFrames


def main():
    detector_parameters = MotionDetector.DetectorParameters()
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
            cv2.imshow("Background", np.uint8(diags.background))
            cv2.imshow("Threshold", diags.threshold)
            cv2.imshow("Threshold-after-dilate", diags.threshold_after_dilate)

            print("my threshold = ", diags.threshold.sum() / diags.threshold.size, diags.threshold.min(), diags.threshold.max())

            for k, v in diags.items():
                if type(v) != numpy.ndarray:
                    print(k, v)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.5)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
