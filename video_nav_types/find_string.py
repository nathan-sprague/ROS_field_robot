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


class standardDetection():

    def __init__(self):
        self.smoothCenter = 0

    def findString(self, color_image):

        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([160, 100, 0])
        upper_green = np.array([170, 200, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # Bitwise-AND mask and original image

        res3 = cv2.bitwise_and(color_image, color_image, mask=mask)
        colorRes = res3.copy()
        res3 = cv2.cvtColor(res3, cv2.COLOR_BGR2GRAY)

        res3[res3 < 100] = 0

        # edges = cv2.bitwise_not(cv2.Canny(res3,100,200))

        h = res3.shape[1]
        w = res3.shape[0]
        res3 = res3[int(w * 0.7)::, int(h * 0.2):int(h * 0.8)]
        resized = cv2.resize(res3, (res3.shape[1], 1), interpolation=cv2.INTER_AREA)
        x = resized.copy()

        resized = cv2.resize(x, (res3.shape[1], res3.shape[0]), interpolation=cv2.INTER_AREA)

        blank = np.zeros((colorRes.shape[0], colorRes.shape[1]), np.uint8)
        j = int(w * 0.7)
        lastI = -1
        xx = np.array([])
        yy = np.array([])
        difs = np.array([])
        n = 0
        while n < len(res3):
            i = res3[n]
            if len(np.nonzero(i)[0]) > 0:
                #  print(np.median(np.nonzero(i)))
                li = int(np.median(np.nonzero(i))) + int(h * 0.2)

                if abs(li - lastI) < 10 or lastI == -1:
                    difs = np.append(difs, lastI - li)
                    lastI = li
                    #       cv2.rectangle(colorRes, (lastI, j), (lastI+1, j+1),(0, 255,0),3)
                    # print(lastI, j)
                    xx = np.append(xx, lastI)
                    yy = np.append(yy, j)
                else:
                    cv2.rectangle(blank, (li, j), (li + 1, j + 1), (0, 0, 255), 3)
                #  cv2.rectangle(colorRes, (int(xx[i]), int(yy[i])), (int(xx[i]+1), int(yy[i]+1)),(255, 0,0),3)

                # print("xx", xx)
                # print("yy", yy)
                lastI = li

            j += 3
            n += 3

        correlation_matrix = np.corrcoef(xx, yy)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy ** 2
        # print(r_squared)
        # stdev = np.std(xx)
        # med = np.median(xx)
        # i=0
        b = self.estimate_coef(xx, yy)
        deleted = 0
        errors = []
        i = 0
        if len(xx) < 10:
            cv2.imshow('1color', blank)
            return

        i = 0
        while i < len(xx):
            cv2.rectangle(blank, (int(xx[i]), int(yy[i])), (int(xx[i] + 1), int(yy[i] + 1)), (255, 255, 255), 3)
            cv2.rectangle(color_image, (int(xx[i]), int(yy[i])), (int(xx[i] + 1), int(yy[i] + 1)), (255, 255, 255), 3)
            i += 1

            #        i+=1
        #  print(deleted)
        # print(xx, yy)
        # y = mx + b
        # x = (y-b)/m
        caan = cv2.Canny(blank, 100, 200)
        linesP = cv2.HoughLinesP(caan, 1, np.pi / 180, 50, None, 50, 10)

        cv2.imshow('2color', caan)
        bs = []
        ms = []
        maxSlope = [0, 0]
        if linesP is not None:
            print(len(linesP))
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                # cv2.line(colorRes, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA)
                b = self.estimate_coef(np.array([l[0], l[2]]), np.array([l[1], l[3]]))
                #    cv2.line(colorRes,(int((w-b[0])/b[1]), int(w)), (int((w/2-b[0])/b[1]), int(w/2)),(255, 0,0), 3)
                if abs(b[1]) > abs(maxSlope[1]):
                    maxSlope = b
            if abs(maxSlope[1]) > 2:
                b = maxSlope
                cv2.line(color_image, (int((w - b[0]) / b[1]), int(w)), (int((w / 2 - b[0]) / b[1]), int(w / 2)),
                         (0, 255, 0), 3)

        cv2.imshow('1color', color_image)
