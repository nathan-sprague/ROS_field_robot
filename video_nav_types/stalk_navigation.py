import pyrealsense2 as rs
import statistics

import numpy as np
import cv2
import argparse  # Import argparse for command-line options
import os.path
import math
import time
import random
import image_tools

import navTypeTester


class StalkDetection():

    def __init__(self):
        self.smoothCenter = 0
        self.lastGood = 0





    def rowNavigation(self, depth_image, color_image, showStream=False):
#    self.findString(color_image)
        # cv2.imshow('depth', depth_image)
        # if not self.useCamera and self.playbackSpeed<1:
        #     time.sleep(0.1)

        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([30, 0, 50])
        upper_green = np.array([80, 180, 180])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # Bitwise-AND mask and original image

        res2 = cv2.bitwise_and(color_image, color_image, mask=mask)
        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # biggest = [0,0,0,0]

        edges = cv2.bitwise_not(cv2.Canny(color_image, 100, 200))

        cv2.imshow('caffnny', res2)
        res = cv2.bitwise_and(edges, edges, mask=mask)
        blur = cv2.blur(res, (5, 5))
        blur[blur < 250] = 0

        cl = color_image.copy()

        kernel = np.ones((2, 2), np.uint8)
        blur = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
        cv2.imshow('blur', blur)

        blank = np.zeros((blur.shape[0], blur.shape[1]), np.uint8)

        contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        biggest = [0, 0, 0, 0]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > w * 2 and w < 30 and w * h > 100:
                cv2.rectangle(blank, (x, y), (x + w, y + h), 255, 3)
                # cv2.rectangle(colorJetMap, (x + 40, y), (x + 40 + w, y + h), (0, 255, 0), 3)

        resized = cv2.resize(blank, (res.shape[1], 1), interpolation=cv2.INTER_AREA)
        x = resized.copy()
        x[x > 20] = 255
        x[x < 255] = 0
        resized = cv2.resize(x, (blank.shape[1], blank.shape[0]), interpolation=cv2.INTER_AREA)
        #            cv2.imshow('kd', colorJetMap)
        #     dst = cv2.Canny(blur, 50, 200, None, 3)

        #     cv2.imshow('a', blur)
        #   #  cv2.imshow('color', color_image)

        #     cl = color_image.copy()

        #      stalks = cv2.bitwise_and(colorJetMap, colorJetMap, mask= resized)
        # mask = cv2.inRange(hsv, lower_green, upper_green)
        # stalks = cv2.bitwise_and(stalks, stalks, mask= mask)

        # alternate between masked and unmasked places
        centers = []
        lastVal = 0
        n = 0
        lastOn = 0
        for i in resized[0]:
            if i == 0 and lastVal != 0 and lastOn != 0:
                centers += [[lastOn, n]]
            elif i != 0 and lastVal == 0:

                lastOn = n

            lastVal = i
            n += 1

        w = cl.shape[1]
        for c in centers:
            i = int((c[0] + c[1]) / 2)
            if abs(i - w / 2) > w / 10:
                cv2.line(cl, (i - 2, 0), (i - 2, int(cl.shape[0])), (255, 0, 0), 4)

        #  cv2.imshow('lines', cdstP)
        cv2.imshow('color', cl)
        return 0, 0, 0

    #         cv2.imshow('stalk', resized)

    #     cv2.imshow('og_jet', depth_color_image)

    #     depth_color_image[0::, 0:depth_color_image.shape[1]-40] = depth_color_image[0::, 40::]
    #     depth_color_image[0::, depth_color_image.shape[1]-40::] = (0,0,0)
    #    # jet = cv2.bitwise_and(depth_color_image, depth_color_image, mask = resized)

    # depth = depth_image.copy()

    # depth = self.apply_brightness_contrast(depth, 10,100)  # remove this line

    # depth[0::, 0:depth.shape[1]-40] = depth[0::, 40::]
    # depth[0::, depth.shape[1]-40::] = 0
    # jet = cv2.bitwise_and(depth, depth, mask = resized)

    # depth_regress = np.zeros((depth.shape[0],depth.shape[1]),np.uint8)

    # for i in centers:
    #     i[0]-=40
    #     i[1]-=40
    #     a = np.nonzero(depth[0::, i[0]:i[1]])
    #     if len(a)>0:
    #         x = np.median(a)

    #         depth_regress[0::, i[0]:i[1]][True] = x

    #    cv2.imshow('jet', depth_regress)

    # w = blur.shape[1]
    # h = blur.shape[0]
    # rampl = np.linspace(1, 0, int(w/2))
    # rampl = np.tile(np.transpose(rampl), (h,1))
    # rampl = cv2.merge([rampl,rampl,rampl])

    # vis = np.concatenate((cv2.flip(rampl,1), rampl), axis=1)
    # cv2.imshow('edgddes',vis)
    # blur[depth_image < vis] = 100

    #   edges = cv2.blur(edges, (20, 20))

    # for cnt in contours:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     if w*h>biggest[2]*biggest[3]:
    #         biggest = [x,y,w,h]
    #         cv2.rectangle(color_image,(biggest[0],biggest[1]),(biggest[0]+biggest[2], biggest[1]+biggest[3]),(0,255,0),2)

    # cv2.rectangle(color_image,(biggest[0],biggest[1]),(biggest[0]+biggest[2], biggest[1]+biggest[3]),(0,255,0),2)

    # cv2.imshow('edges',edges)

    #    cv2.imshow('res',res)

    # cv2.imshow('og', color_image)






if __name__ == "__main__":
    print("running")
    
    _filename = "/home/nathan/new_logs/aug31/success1"#
    # _filename = "Desktop/fake_track/object_detection6.bag"
    # _filename=""
    _useCamera = False
    _showStream = True
    
    sd = StalkDetection()

    cam = navTypeTester.RSCamera(useCamera = _useCamera, filename=_filename)
    cam.videoNavigate(sd.rowNavigation, _showStream)