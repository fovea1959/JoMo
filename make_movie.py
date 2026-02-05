# noinspection PyUnresolvedReferences
import custom_logging

import argparse
import logging
import pathlib
import sys

import cv2

import utilities

from source_images_from_files import FilesFrameSource


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--glob', default='*.jp*')
    parser.add_argument('--markup', action='store_true')
    parser.add_argument('--fps', type=float, default=10)
    parser.add_argument('--fourcc')
    args = parser.parse_args(argv)
    logging.info("Invoked with %s", args)

    image_dir = args.directory
    image_source = FilesFrameSource(directory=image_dir, glob=args.glob, forever=False)

    fourcc_string = args.fourcc
    if fourcc_string is None:
        extension = pathlib.Path(args.output).suffix.lower()
        fourcc_string = {
            '.avi': 'XVID',
            '.mp4': 'mp4v',  # 'H264' does not work
            '.mkv': 'X264',  # may not work, need encoder
        }.get(extension)
        if fourcc_string is None:
            raise ValueError(f"cannot determine valid fourcc from extension = '{extension}'")

    # Iterate through the image frames using the generator
    vw = None
    for i, frame_and_info in enumerate(image_source.yield_opencv_image_frames()):
        frame, info = frame_and_info

        if args.markup:
            dt = info.get('timestamp')
            if dt is not None:
                text = dt.isoformat()
                utilities.add_text_to_image(frame=frame, text=text, top_left_xy=(10, 10),
                                            font_color_rgb=(255, 255, 255), bg_color_rgb=(0, 0, 0))

        if vw is None:
            height, width = frame.shape[:2]
            size = (width, height)
            logging.info("opening %s with fourcc '%s' and size %s", args.output, fourcc_string, size)
            fourcc = cv2.VideoWriter_fourcc(*fourcc_string)
            vw = cv2.VideoWriter(args.output, fourcc, args.fps, size)

        vw.write(frame)
    vw.release()


if __name__ == '__main__':
    main(sys.argv[1:])
