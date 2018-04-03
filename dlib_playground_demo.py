"""
Observe face detector performance

Requirements:
    dlib is unavailable from conda repository; build it from the source.
    Additionally get pretrained model "mmod_human_face_detector.dat" from
    http://dlib.net/files/mmod_human_face_detector.dat.bz2 .

    * dlib>=19.8
        * pretrained model "mmod_human_face_detector.dat"
    * imageio
    * ffmpeg
    * matplotlib
    * opencv (with ffmpeg option on)

Usage:

```
$ python dlib_playground.py <input_image>
```

Then face-cropped random images will be displayed.

"""

import os
import argparse

import dlib
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


# circle color and size
_COLOR = (30, 30, 160)  # BGR in OpenCV
_LINE_THICKNESS = 1

# location of pre-trained model
#  You can download the pre-trained model from:
#       http://dlib.net/files/mmod_human_face_detector.dat.bz2
_PRETRAINED_MODEL = 'mmod_human_face_detector.dat'

_MESSAGE = """
Pre-trained model not found: {}
Place the file to the same folder as this script.
Source: http://dlib.net/files/mmod_human_face_detector.dat.bz2
""".format(_PRETRAINED_MODEL)

if not os.path.exists(_PRETRAINED_MODEL):
    print(_MESSAGE)
    raise FileNotFoundError


class DlibFaceDetector(object):
    """Fast but inaccurate face detector"""

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def run(self, img):
        """
        Return dlib's rectangle object if it could detect a face.
        Otherwise returns None
        """
        dets, _, _ = self.detector.run(img, 1)
        is_detected = (len(dets) > 0)
        res = dets[0] if is_detected else None
        return res

    def show_face_bb(self, img):
        """Display face bounding box"""
        rect = self.run(img)
        draw_bb_matplotlib(img, rect)



class DlibCNNFaceDetector(object):
    """dlib's CNN-based face detector"""

    def __init__(self):
        self.detector = dlib.cnn_face_detection_model_v1(_PRETRAINED_MODEL)

    def run(self, img):
        """
        Return dlib's rectangle object if it could detect a face.
        Otherwise returns None
        """
        dets = self.detector(img)
        is_detected = (len(dets) > 0)
        res = dets[0].rect if is_detected else None
        return res

    def show_face_bb(self, img):
        """Display face bounding box"""
        rect = self.run(img)
        draw_bb_matplotlib(img, rect)


def draw_bb_matplotlib(img, rect):
    """Draw bounding-box picture with matplotlib engine
    """
    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')
    if rect:
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        upper_left_corner = (x1, y1)
        width = x2 - x1
        height = y2 - y1
        rect_patch = patches.Rectangle(
            upper_left_corner,
            width,
            height,
            linewidth=_LINE_THICKNESS,
            edgecolor='r',
            facecolor='none')
        ax.add_patch(rect_patch)
    plt.show()


def draw_bb_cv2(img, rect):
    """Draw bounding-box picture with opencv engine
    """
    if rect:
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        img = cv2.rectangle(
            img, (x1, y1), (x2, y2), color=_COLOR, thickness=_LINE_THICKNESS)
    cv2.imshow('image_bb', img)
    cv2.waitKey(1000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='image path')
    args = parser.parse_args()
    detector = DlibCNNFaceDetector()

    filename = args.image_path
    print("\nfilename:", filename)
    img = imageio.imread(filename)
    detector.show_face_bb(img)

if __name__ == '__main__':
    main()
