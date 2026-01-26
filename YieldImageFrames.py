from pathlib import Path
from typing import Generator

import cv2
from PIL import Image
import numpy as np


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

    def yield_pillow_image_frames(self) -> Generator[np.ndarray, None, None]:
        """
         A generator function that iterates over image files in a directory
         and yields each image as a Pillow image.

         Yields:
             An image frame as a Pillow image.
         """
        for file_path in self.file_paths:
            if file_path.is_file():
                try:
                    # Open the image using Pillow (PIL)
                    with Image.open(file_path) as img:
                        yield img, {'path': file_path}
                except IOError:
                    # Handle cases where a file might not be a valid image
                    print(f"Skipping non-image file: {file_path}")

    def yield_opencv_image_frames(self) -> Generator[np.ndarray, None, None]:
        """
        A generator function that iterates over image files in a directory
        and yields each image as a NumPy array (frame).

        Yields:
         An image frame as a OpenCV NumPy array.
        """
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
