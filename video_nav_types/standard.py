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





    def rowNavigation(self, depth_image, color_image, showStream=False):

        res = {"imagesToShow": [], "center": 0}

        explainBadData = False  # print what is wrong when it does not know where it is going
        badDataReasons = []
        # Start streaming from file

        badData = False  # assume it is good until future tests show it is not

        cropSize = [0.2, 0.3]
        og_width = depth_image.shape[1]
        og_height = depth_image.shape[0]
        depth_image = depth_image[int(og_height * cropSize[1]):int(og_height * (1 - cropSize[1])),
                      int(og_width * 0):int(og_width * 1)]
        cv2.rectangle(depth_image, (0, 0), (int(og_width * cropSize[0]), depth_image.shape[0]), (0, 0, 0), -1)
        cv2.rectangle(depth_image, (int(og_width * (1 - cropSize[0])), 0), (depth_image.shape[1], depth_image.shape[0]),
                      (0, 0, 0), -1)

        if cv2.countNonZero(depth_image) / depth_image.size < 0.15:
            # If less than 15% of all points are invalid, just say the it is bad
            badDataReasons += ["too many invalid points"]
            badData = True

        # Convert all black pixels to white. 65535 is 16-bit
        depth_image[depth_image == 0] = 65535

        # show step

        depth_image = image_tools.apply_brightness_contrast(depth_image, 10, 100)  # increase contrast

        # show with increased contrast

        depth_image = (depth_image / 256).astype('uint8')  # convert to 8-bit

        #  mask = np.zeros(depth_image.shape[:2], dtype="uint8")
        # cv2.rectangle(mask, (0, int(depth_image.shape[0] / 6)), (int(depth_image.shape[1]), int(depth_image.shape[0] / 1.5)), 255, -1)

        # convert all white (invalid) to  black
        depth_image[depth_image == 255] = 0
        depth_image[depth_image > 55] = 255
        depth_image[depth_image < 56] = 0


        res["imagesToShow"] += [depth_image]

        res = depth_image.copy()

        # combine the pixel values of each column
        resized = cv2.resize(res, (res.shape[1], 1), interpolation=cv2.INTER_AREA)

        # blur it to average out outliers
        resized = cv2.blur(resized, (5, 5))

        # show the 5th step

        removeThis = cv2.resize(resized, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_AREA)

        res["imagesToShow"] += [removeThis]

        # Get the lightest colors
        x = []
        avgColor = np.mean(res)
        sd = np.std(res)

        # make the close areas black
        resized[0][:][resized[0][:] < avgColor + sd / 3] = 0

        # make the furthest points white
        b = 0
        for r in range(resized.shape[1]):
            if resized[0][r] > avgColor + sd / 3:
                b += 1
                resized[0][r] = 255
                x += [r]

        # get indices of the furthest places
        z = np.nonzero(resized[0][:])

        # the initial center is estimated as the average of all the far away points.
        frameCenter = int(depth_image.shape[1] / 2)
        centerSpot = frameCenter

        # if there actually any valid points, use the median, if not say so
        if len(z[0]) > 0:
            centerSpot = int(np.median(z))
        else:
            badDataReasons += ["no distant enough spots"]
            badData = True

        # run through the columns and find the place with the most distant points.
        # there may be distant points between stalks but they are less common and we don't want them
        # influencing the center estimation

        i = 0
        y = []
        r = np.nonzero(resized[0][:])
        numInside = 0
        numOutside = 0
        while i < len(r[0]):
            j = r[0][i]
            if centerSpot + depth_image.shape[1] / 5 > j > centerSpot - depth_image.shape[1] / 5:
                y += [j]
                resized[0][j] = 100
                numInside += 1
            else:
                numOutside += 1

            i += 1
        if numInside < numOutside * 5:
            badDataReasons += ["more lines outside than inside"]
            badData = True
        if numInside < numOutside:
            badDataReasons += ["WAY more lines outside than inside"]
            badData = True

        if len(y) > 0:
            centerSpot = int(statistics.median(y))

        if self.smoothCenter == -1000:
            self.smoothCenter = centerSpot

        if abs(centerSpot - frameCenter) > frameCenter / 3:
            badDataReasons += ["center estimated to be too far out"]
            badData = True

        # list the problems
        if explainBadData and len(badDataReasons) > 0:
            print(badDataReasons)

        # if the problem is really bad, but stick to the center of the frame
        if badData and time.time() - self.lastGood > 1 and ("WAY more lines outside than inside" in badDataReasons):
            centerSpot = frameCenter
            pass

        elif badData:  # if the estimation was invalid, use the last valid center estimation.
            centerSpot = self.smoothCenter
            badData = False

        else:  # the estimation is valid
            self.lastGood = time.time()

        # smooth the center estimation
        self.smoothCenter = (self.smoothCenter * 0 + centerSpot) / 1  # currently not smoothing center estimation, may add back later

        smoothCenterInt = int(self.smoothCenter)
        if time.time() - self.lastGood < 1:  # if there was a valid center estimation in the last second, use it.
            self.heading = (depth_image.shape[1] / 2 - self.smoothCenter) * 90 / depth_image.shape[1]
        else:
            self.heading = 1000  # set the heading to an unreasonable number so the user can know it is invalid

        if showStream:
            removeThis = cv2.resize(resized, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_AREA)
            res["imagesToShow"] += [removeThis]

            if badData:
                cv2.line(color_image, (smoothCenterInt - 2, 0),
                         (smoothCenterInt - 2, int(color_image.shape[0])), (0, 0, 255), 4)
            else:
                cv2.line(color_image, (smoothCenterInt - 2, 0),
                         (smoothCenterInt - 2, int(color_image.shape[0])), (255, 0, 0), 4)
            cv2.line(color_image, (int(color_image.shape[1] / 2), 0),
                     (int(color_image.shape[1] / 2), int(color_image.shape[0])), (0, 0, 0), 1)



        return res