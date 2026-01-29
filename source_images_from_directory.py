import logging

from pathlib import Path
from typing import Generator

import cv2
from PIL import Image
import numpy as np

logger = logging.getLogger("imageframesource")
logger.setLevel(logging.INFO)


class ImageFrameSource:
    def __init__(self, directory_path: str, extensions: list = None):
        p = Path(directory_path)
        if extensions:
            file_paths = []
            for ext in extensions:
                # Use rglob for recursive search, glob for single level
                file_paths.extend(list(p.glob(f'*{ext}')))
        else:
            file_paths = list(p.glob('*'))  # Iterate through all files

        file_paths.sort()
        self.file_paths = file_paths

        self.error_frame = self.make_error_frame(f"no valid images in {directory_path}")

    @staticmethod
    def make_error_frame(text):
        width, height = 1024, 768

        image = np.zeros((height, width, 3), np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)  # White color (BGR)
        thickness = 1

        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        text_x = (width - text_width) // 2
        text_y = (height + text_height) // 2

        text_origin = (text_x, text_y)

        cv2.putText(image, text, text_origin, font, font_scale, color, thickness, cv2.LINE_AA)
        return Image.fromarray(image)

    def yield_pillow_image_frames(self) -> Generator[np.ndarray, None, None]:
        """
        A generator function that iterates over image files in a directory
        and yields each image as a Pillow image.

        Yields:
        An image frame as a Pillow image.
        """
        logger.info("starting yield_pillow_image_frames")
        while True:
            yielded_something = False
            for file_path in self.file_paths:
                if file_path.is_file():
                    try:
                        # Open the image using Pillow (PIL)
                        with Image.open(file_path) as img:
                            logger.debug("yielding image %s", file_path)
                            yield img, {'path': file_path}
                    except IOError:
                        # Handle cases where a file might not be a valid image
                        logger.error(f"Skipping non-image file: {file_path}")

            if not yielded_something:
                logger.debug("yielding error frame")
                yield self.error_frame, {'path': '** error frame **'}

    def yield_opencv_image_frames(self) -> Generator[np.ndarray, None, None]:
        """
        A generator function that iterates over image files in a directory
        and yields each image as a NumPy array (frame).

        Yields:
        An image frame as a OpenCV NumPy array.
        """
        logger.info("starting yield_opencv_image_frames")
        for pillow_image_frame, info in self.yield_pillow_image_frames():
            opencv_image_frame = np.array(pillow_image_frame)
            opencv_image_frame = cv2.cvtColor(opencv_image_frame, cv2.COLOR_RGB2BGR)
            yield opencv_image_frame, info


# Example Usage:
if __name__ == '__main__':
    # Replace 'your_image_directory' with the path to your image folder
    image_dir = 'testing/collected'
    # Specify extensions if needed, or leave as None to attempt opening all files
    valid_extensions = ['.png', '.jpg', '.jpeg']

    image_source = ImageFrameSource(image_dir, valid_extensions)

    # Iterate through the image frames using the generator
    for i, frame_and_info in enumerate(image_source.yield_pillow_image_frames()):
        frame, info = frame_and_info
        print(f"Processing frame {i+1} {info}: Frame {frame}")

    # Iterate through the image frames using the generator
    for i, frame_and_info in enumerate(image_source.yield_opencv_image_frames()):
        frame, info = frame_and_info
        print(f"Processing frame {i+1} {info}: Shape {frame.shape}, Data Type {frame.dtype}")
