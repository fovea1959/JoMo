import datetime
import logging
import re

from pathlib import Path
from typing import Generator, Tuple, Dict
from zoneinfo import ZoneInfo

import cv2
from PIL import Image
import numpy as np

import source_images
import utilities


class FilesFrameSource(source_images.FrameSource):
    def __init__(self, forever: bool = True, log_level: int = logging.INFO, directory: str = None, extensions: list = None, **kwargs):
        super().__init__(log_level)

        if extensions is None:
            extensions = ['.jpeg', '.jpg']

        self.forever = forever

        p = Path(directory)
        if extensions:
            file_paths = []
            for ext in extensions:
                # Use rglob for recursive search, glob for single log_level
                file_paths.extend(list(p.glob(f'*{ext}')))
        else:
            file_paths = list(p.glob('*'))  # Iterate through all files

        file_paths.sort()
        self.file_paths = file_paths

        self.error_frame = self.make_error_frame(f"no valid images in {directory}")

    @staticmethod
    def make_error_frame(text):
        width, height = 1024, 768

        frame = np.zeros((height, width, 3), np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)  # White color (BGR)
        thickness = 1

        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        text_x = (width - text_width) // 2
        text_y = (height + text_height) // 2

        text_origin = (text_x, text_y)

        cv2.putText(frame, text, text_origin, font, font_scale, color, thickness, cv2.LINE_AA)
        return frame

    def yield_opencv_image_frames(self) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """
        A generator function that iterates over image files in a directory
        and yields each image as an opencv image.

        Yields:
        An image frame as a opencv image.
        """
        self.logger.info("starting yield_pillow_image_frames")
        date_re = re.compile(r"(\d{8}-\d{6})\.")
        local_tz = ZoneInfo('localtime')
        while True:
            yielded_something = False
            for file_path in self.file_paths:
                if file_path.is_file():
                    try:
                        # Open the image using Pillow (PIL)
                        with Image.open(file_path) as img:
                            info = {'path': file_path}
                            m = date_re.search(str(file_path))
                            if m:
                                dt_s = m.group(1)
                                dt = datetime.datetime.strptime(dt_s, '%Y%m%d-%H%M%S')
                                dt = dt.astimezone(local_tz)
                                info['timestamp'] = dt
                            self.logger.debug("yielding image %s %s", file_path, info)
                            yield utilities.make_cv2_from_pillow(img), info
                            yielded_something = True
                    except IOError:
                        # Handle cases where a file might not be a valid image
                        self.logger.error(f"Skipping non-image file: {file_path}")

            if not yielded_something:
                self.logger.debug("yielding error frame")
                yield self.error_frame, {'path': '** error frame **'}

            if not self.forever:
                break


# Example Usage:
if __name__ == '__main__':
    # Replace 'your_image_directory' with the path to your image folder
    image_dir = 'testing/123'
    # Specify extensions if needed, or leave as None to attempt opening all files
    valid_extensions = ['.png', '.jpg', '.jpeg']

    image_source = FilesFrameSource(directory = image_dir, extensions = valid_extensions)

    # Iterate through the image frames using the generator
    for i, frame_and_info in enumerate(image_source.yield_opencv_image_frames()):
        frame, info = frame_and_info
        print(f"Processing frame {i+1} {info}: Shape {frame.shape}, Data Type {frame.dtype}")
