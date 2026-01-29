from PIL import Image
import cv2
import numpy as np


def make_pillow_from_cv2(cv2_image):
    # Convert from BGR to RGB
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convert NumPy array to PIL Image object
    pil_image = Image.fromarray(cv2_image_rgb)
    return pil_image


def make_jpeg_from_cv2(cv2_image):
    return cv2.imencode('.jpg', cv2_image)[1].tobytes()


def make_cv2_from_pillow(pil_image):
    cv2_image_rgb = np.array(pil_image)
    cv2_image = cv2.cvtColor(cv2_image_rgb, cv2.COLOR_RGB2BGR)
    return cv2_image
