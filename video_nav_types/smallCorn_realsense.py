""" 
detection algorithm designed to ue on young corn (like V3)
This uses exclusively the realsense camera to navigate.
The realsense camera isn't great at picking up green, so this alogrithm uses depth.
Overall, this algorithm isn't very good. Use smallCorn_webcam.py to navigate.
    smallcorn_webcam.py works in an entirely different way.
"""

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


def rotate_image_old(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


class StandardDetection():

    def __init__(self):
        self.smoothCenter = 0
        self.lastGood = 0
        self.heading = 0

        self.lastGoodLeft = 0
        self.lastGoodRight = 0




    def rowNavigation(self, depth_image, color_image, depth_color_image, webcamFrame, showStream=False):
        # self.colorBasedNavigation(depth_image, color_image)
        # self.houghLineAttempt(depth_color_image)
        # return

        leftGood = False
        rightGood = False

  
        # take non-zero average of the rows
        fade = np.true_divide(depth_image.sum(1), (depth_image != 0).sum(1))
        b = cv2.resize(fade, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_AREA)
        depth_image_new = (depth_image - b)
        # depth_image_new = depth_image_new[int(depth_image_new.shape[0]*0.5)::, 0:depth_image_new.shape[1]]
        # depth_image_new = apply_brightness_contrast(depth_image_new, 10, 100)  # increase contrast
        # depth_image_new[depth_image_new < 253] = 0

        rotAngle = 40 # expected perspective angle of the row going to the horizon

        depth_image_new[depth_image == 0] = 0 # not sure if this does anything
        depth_image_new[b / 256 > 200] = 0 # make the somewhat light pixels 0

        depth_image_new = apply_brightness_contrast(depth_image_new, 10, 100)  # increase contrast

        horizon = int(depth_image_new.shape[0] * 0.5) # assume the horizon is halfway up the screen. (not true, but works well enough)

        centerX = int(depth_image_new.shape[1] * 0.5)

        distFromCenterX = int(depth_image_new.shape[1] * 0.2)
        croppedCenter = depth_image_new[horizon:int(depth_image_new.shape[0] * 0.6),
                        centerX - distFromCenterX:centerX + distFromCenterX] # crop the image to just get the area near the horizon and near the center

        croppedCenter = (croppedCenter / 256).astype('uint8')
        # croppedCenter[croppedCenter > 256 * 10] = 255 * 256
        croppedCenter[croppedCenter < 60] = 250
        croppedCenter[croppedCenter < 240] = 0

        # rotate and find brightest area- the idea is that the rows will show up lighter than the non-row areas
        rot = rotate_image(255 - croppedCenter, 90 + rotAngle)
        r = cv2.resize(rot, (rot.shape[1], 1), interpolation=cv2.INTER_AREA)
        j = np.max(r)
        r[r > j - 1] = 255
        r[r < 255] = 0
        # print.(j)
        j = int(j)
        mm = int(np.mean(np.nonzero(r[0, :])))
        # print(mm)
        if mm>100:
            # print("bad?")
            rightGood = False
        else:
            rightGood = True

        ## repeat but other side
        rot2 = rotate_image(255 - croppedCenter, rotAngle)
        r = cv2.resize(rot2, (rot2.shape[1], 1), interpolation=cv2.INTER_AREA)
        r_visual = cv2.resize(r, (rot2.shape[1], rot2.shape[0]), interpolation=cv2.INTER_AREA)

        j2 = np.max(r)
        r[r == j2] = 255
        r[r < 255] = 0
        # print.(j)
        mm2 = int(np.mean(np.nonzero(r[0, :])))
        # print(mm2)
        if mm2 > 150:
            leftGood = False
            # print("bad?")
        else:
            leftGood = True

        cv2.line(rot, (mm2, 0), (mm2, rot.shape[1]), 255, 4)

        # rot = rotate_image(rot, -130)
        cRot = math.cos(rotAngle / 180 * 3.1416)
        sRot = math.sin(rotAngle / 180 * 3.1416)

        rowStartHoriz = int(centerX + (distFromCenterX - mm) * cRot)
        if rightGood:
            color = (0,255,0)
            self.lastGoodRight = rowStartHoriz
        else:
            color = (0,0,255)
            if leftGood:
                rowStartHoriz = int(centerX - (distFromCenterX - mm2) * math.cos(-40 / 180 * 3.1416)) + 100
                color = (0, 255, 255)
            else:
                rowStartHoriz = self.lastGoodRight



        cv2.line(depth_image_new, (rowStartHoriz, horizon),
                 (rowStartHoriz + int(400 * cRot), horizon + int(400 * sRot)), 0, 4)

        cv2.line(color_image, (rowStartHoriz, horizon), (rowStartHoriz + int(400 * cRot), horizon + int(400 * sRot)),
                 color, 4)



        rowStartHoriz2 = int(centerX - (distFromCenterX - mm2) * math.cos(-40 / 180 * 3.1416))
        if leftGood:
            color = (0,255,0)
            self.lastGoodLeft = rowStartHoriz2
        else:
            color = (0,0,255)
            if rightGood:
                rowStartHoriz2 = rowStartHoriz - 100
                color = (0, 255, 255)
            else:
                rowStartHoriz2 = self.lastGoodLeft


        cv2.line(depth_image_new, (rowStartHoriz2, horizon),
                 (rowStartHoriz2 - int(400 * cRot), horizon + int(400 * sRot)), 0, 4)
        cv2.line(color_image, (rowStartHoriz2, horizon), (rowStartHoriz2 - int(400 * cRot), horizon + int(400 * sRot)),
                 color, 4)


        centerEst = int((rowStartHoriz + rowStartHoriz2) / 2)
        cv2.line(color_image, (centerEst, 0), (centerEst, color_image.shape[0]), (255,0,0), 4)



        # self.houghLineAttempt(color_image)
        # cv2.imshow('ogf', (b / 256).astype('uint8'))
        cv2.imshow('ogg', (depth_image_new / 256).astype('uint8'))
        # print(flooded)
        # cv2.imshow('ogd', (croppedCenter / 256).astype('uint8'))
        # cv2.imshow('oge', (r / 256).astype('uint8'))
        cv2.imshow('ogb', croppedCenter)
        cv2.imshow('rot', rot)
        cv2.imshow('color', color_image)
        cv2.imshow('ogr', r_visual)
        # cv2.imshow('cam', webcamFrame)

        # depth_image = apply_brightness_contrast(depth_image, 10, 100)  # increase contrast

      

if __name__ == "__main__":
    import navTypeTester

    print("running")

    # _filename = "/home/john/object_detections/rs_1629482645.bag"
    _filename = "/home/nathan/tall/rs_1629482645.bag"#"/home/nathan/v3/rs_1629481177.bag"#"object_detections/v3/rs_1629465226.bag"  # rs_1629481177
    _filename = "/home/nathan/Desktop/rs_1655477892.bag" #"/home/nathan/v3/rs_1629465226.bag" # "/Users/nathan/bag_files/rs_1629481177.bag"
    _useCamera = False
    _showStream = True

    sd = StandardDetection()

    cam = navTypeTester.RSCamera(useCamera=_useCamera, filename=_filename, realtime = False, useRGB=True)
    cam.videoNavigate(sd.rowNavigation, _showStream)
