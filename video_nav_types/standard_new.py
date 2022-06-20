import pyrealsense2 as rs
import statistics

import numpy as np
import cv2
import argparse  # Import argparse for command-line options
import os.path
import math
import time
import random


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    # found this function on stack overflow. All I know is that it works
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


class StandardDetection():

    def __init__(self):
        self.smoothCenter = 0
        self.lastGood = 0
        self.heading = 0


    def rowNavigation(self, depth_image, color_image, showStream=False):

        cv2.imshow("og", color_image)
        fade = np.true_divide(depth_image.sum(0), (depth_image != 0).sum(0))
        # print(fade/256)
        fade = cv2.rotate(fade, cv2.ROTATE_90_CLOCKWISE)

        print(fade.shape)
        print(depth_image.shape)
        fade = cv2.resize(fade, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_AREA)
        # fade = cv2.rotate(fade, cv2.ROTATE_90_CLOCKWISE)


        cv2.imshow("f", (fade/256).astype('uint8'))


        # cv2.imshow("gs", gs)


        return self.heading



if __name__ == "__main__":
    import navTypeTester

    print("running")

    # _filename = "/home/john/object_detections/rs_1629482645.bag"
    _filename = "/home/nathan/tall/rs_1629482645.bag" # "/home/nathan/v3/rs_1629465226.bag"

    _useCamera = False
    _showStream = True

    sd = StandardDetection()

    cam = navTypeTester.RSCamera(useCamera=_useCamera, filename=_filename)
    cam.videoNavigate(sd.rowNavigation, _showStream)
