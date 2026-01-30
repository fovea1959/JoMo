import piexif
import piexif.helper
import json
import struct
import utilities
from PIL import Image


def make_jpeg_comment_segment(comment_bytes: bytes) -> bytes:
    """make a JPEG comment/COM segment from the given payload"""
    # 0xFFFE is the marker for a COM (comment) segment
    marker = 0xFFFE
    # The length field includes the 2 bytes for the length itself + the payload length
    length = 2 + len(comment_bytes)
    # Pack the marker and length into bytes ('>HH' means big-endian, two unsigned short integers)
    return struct.pack('>HH', marker, length) + comment_bytes


def main():
    pil_image = Image.open('testing/123/1.jpg')

    jpeg_info = {
        "boxes": [ (0, 0, 0, 0), (1, 1, 1, 1)]
    }
    jpeg_info_s = json.dumps(jpeg_info)

    comments = "Review: Good | Subject: Sunset | Author: Me"

    # Define EXIF data dictionary (using standard numeric tags)
    exif_dict = {
        "0th": {
            piexif.ImageIFD.ImageDescription: "Image Description",
            piexif.ImageIFD.XPComment: comments.encode('utf-16le'),
            piexif.ImageIFD.Make: 'Average Joes',
            piexif.ImageIFD.Model: 'JoMo',
            piexif.ImageIFD.ImageHistory: "[ {} ]"
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: "2025:01:28 10:00:00",
            piexif.ExifIFD.SubSecTimeOriginal: "123",
            piexif.ExifIFD.OffsetTimeOriginal: "+00:00",
            piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(u"User Comment."),
            piexif.ExifIFD.MakerNote: b"Maker note.",
        },
        "GPS": {},  # Add GPS data here if needed
        "1st": {}
    }

    # Dump the dictionary to a raw EXIF byte string
    exif_bytes = piexif.dump(exif_dict)

    comment_text = "Hello world, this is a comment."
    comment_bytes = comment_text.encode('utf-8')
    pil_image.save('z.jpg', exif=exif_bytes, comments="Comments", extra=make_jpeg_comment_segment(comment_bytes))


if __name__ == '__main__':
    main()