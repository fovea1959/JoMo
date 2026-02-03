from datetime import datetime, date
import logging

from PIL import Image
import cv2
import numpy as np


def make_pillow_from_cv2(cv2_image):
    if len(cv2_image.shape) == 2:
        # it's mono
        # doing this saved about 4% on file size.
        pil_image = Image.fromarray(cv2_image)

        if cv2_image.dtype == np.float64:
            pil_image = pil_image.convert("L")
    else:
        # Convert from BGR to RGB
        cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        # Convert NumPy array to PIL Image object
        pil_image = Image.fromarray(cv2_image_rgb)
    return pil_image


def make_jpeg_from_cv2(cv2_image, quality: int = 75):
    return cv2.imencode('.jpg', cv2_image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def make_cv2_from_pillow(pil_image):
    cv2_image_rgb = np.array(pil_image)
    cv2_image = cv2.cvtColor(cv2_image_rgb, cv2.COLOR_RGB2BGR)
    return cv2_image


# Source - https://stackoverflow.com/a/73471951
# Posted by YoniChechik, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-29, License - CC BY-SA 4.0
def add_text_to_image(
    frame: np.ndarray,
    text: str,
    top_left_xy: tuple = (0, 0),
    font_scale: float = 1,
    font_thickness: int = 1,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_color_rgb: tuple = (0, 0, 255),
    bg_color_rgb: tuple | None = None,
    outline_color_rgb: tuple | None = None,
    line_spacing: float = 1,
):
    """
    Adds text (including multi line text) to images.
    You can also control background color, outline color, and line spacing.

    outline color and line spacing adopted from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    """
    OUTLINE_FONT_THICKNESS = 3 * font_thickness

    im_h, im_w = frame.shape[:2]

    for line in text.splitlines():
        x, y = top_left_xy

        # ====== get text size
        if outline_color_rgb is None:
            get_text_size_font_thickness = font_thickness
        else:
            get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

        (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            font_face,
            font_scale,
            get_text_size_font_thickness,
        )
        line_height = line_height_no_baseline + baseline

        if bg_color_rgb is not None and line:
            # === get actual mask sizes with regard to image crop
            if im_h - (y + line_height) <= 0:
                sz_h = max(im_h - y, 0)
            else:
                sz_h = line_height

            if im_w - (x + line_width) <= 0:
                sz_w = max(im_w - x, 0)
            else:
                sz_w = line_width

            # ==== add mask to image
            if sz_h > 0 and sz_w > 0:
                bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                bg_mask[:, :] = np.array(bg_color_rgb)
                frame[
                    y : y + sz_h,
                    x : x + sz_w,
                ] = bg_mask

        # === add outline text to image
        if outline_color_rgb is not None:
            frame = cv2.putText(
                frame,
                line,
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                outline_color_rgb,
                OUTLINE_FONT_THICKNESS,
                cv2.LINE_AA,
            )
        # === add text to image
        frame = cv2.putText(
            frame,
            line,
            (x, y + line_height_no_baseline),  # putText start bottom-left
            font_face,
            font_scale,
            font_color_rgb,
            font_thickness,
            cv2.LINE_AA,
        )
        top_left_xy = (x, y + int(line_height * line_spacing))

    return frame


def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat() # Converts to ISO 8601 format (e.g., "YYYY-MM-DDTHH:MM:SS")
    raise TypeError(f"Type {type(obj).__name__} not serializable")
