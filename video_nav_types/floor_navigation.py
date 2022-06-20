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


class FloorDetection():

    def __init__(self):
        self.smoothCenter = 0
        self.lastGood = 0





    def rowNavigation(self, depth_image, color_image, showStream=False):
        w = depth_image.shape[1]
        h = depth_image.shape[0]

        depth_image[depth_image > 65530] = 0

        l = 3
        stalkLocationsLeft = []
        stalkLocationsRight = []
        left1 = []
        left2 = []
        right1 = []
        right2 = []
        consecutiveCenters = 0

        potentialCenters = []

        nonChanges = 0
        xx = 0
        ijk = 0

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        centers = []
        xx = int(w * 0.25)
        l = 3
        while xx < w * 0.75:
            if xx < w / 2:
                hh = h - (h * 0.5) * xx / (w / 2)
            else:
                hh = h / 2 + (h * 0.5) * (xx - w / 2) / (w / 2)
            hh = int(hh)
            color_b = hsv[int(h / 3):hh, xx:xx + l]
            blue = color_b.copy()
            blue[:, :, 0] = 0
            blue[:, :, 2] = 0
            centers += [np.std(blue)]
            #  cv2.line(color_image, (xx, 0), (xx, 300), (np.std(blue),np.std(blue),np.std(blue)), l)
            xx += l
        middle = centers.index(min(centers)) * l + int(w * 0.25)
        cv2.line(color_image, (middle, 0), (middle, h), (0, 0, 0), 1)
        xx = 0
        l = 3

        while xx < w:

            if xx < middle:
                hh = h - (h * 0.5) * xx / (middle)
            else:
                hh = h / 2 + (h * 0.5) * (xx - middle) / (middle)
            hh = int(hh)
            #    cv2.line(color_image, (xx, 0), (xx, int(hh)), (0,255,0), 5)

            b = depth_image[0:hh, xx:xx + l]
            # cv2.line(color_image, (xx, int(h/3)), (xx, int(hh)), (0,255,0), 5)
            color_b = gray[int(h / 3):hh, xx:xx + l]
            blue = color_b.copy()
            # blue[:, :, 0] = 0
            # blue[:, :, 2] = 0
            mm = np.std(blue)
            #    cv2.line(color_image, (xx, 0), (xx, 30), (mm*10,mm*10,mm*10), l) # standard dev

            if mm < 30:

                if xx < middle:
                    left1 += [xx]
                else:
                    right1 += [xx]
                consecutiveCenters += 1
            else:
                if consecutiveCenters > 1:
                    if xx < middle:
                        s = 0
                        cc = consecutiveCenters
                        #         print(cc, left1)
                        while cc > 0:
                            s += left1[-1]
                            left1.remove(left1[-1])
                            cc -= 1

                        left1 += [int(s / consecutiveCenters)]

                #     print("now", consecutiveCenters, left1)
                if len(left1) > 0:
                    cv2.line(color_image, (left1[-1], 0), (left1[-1], 30), (0, 0, 0), l)
                consecutiveCenters = 0
            a = b.size
            b = b[b != 0]
            c = 255 * b.size / a

            m = np.median(b[b != 0])
            mm = np.std(b[b != 0]) / 10

            if not math.isnan(m):
                indexTo = h - int(m / 5 / 256 * 190)

                if abs(indexTo - hh) < 100 and indexTo > h * 0.5 and 0 < mm < 100:
                    certainty = int(100 / mm)
        
                    if xx < middle:
                        if len(stalkLocationsLeft) > 0:
                            if stalkLocationsLeft[-1][0] - indexTo > 6 and nonChanges > 0:
                           
                                cv2.line(color_image, (xx, 0), (xx, 30), (0, 0, 255), l)
                                left2 += [xx]
                                nonChanges = 0
                            else:
                                nonChanges += 1

                        stalkLocationsLeft += [(indexTo, xx)]
                    else:
                        if len(stalkLocationsRight) > 0:
                            if indexTo - stalkLocationsRight[-1][0] > 6 and nonChanges > 10:
                           
                                cv2.line(color_image, (xx, 0), (xx, 30), (0, 0, 255), l)
                                right2 += [xx]
                                nonChanges = 0
                            else:
                                nonChanges += 1
                        stalkLocationsRight += [(indexTo, xx)]
                  
              
                    
            xx += l
            ijk += 1
        if len(stalkLocationsLeft) == 0:
            # print("no stalk location")
            return

        # self.findStalkMovement(middle, left1, self.lastLeft1, color_image, depth_image)
        print("got here")

        cv2.line(color_image, (middle, 55), (int(middle + (leftSum/numSum)*10), int(55-leftSum/2)), (0, 255, 0), 2)
        #   print(self.lastLeft1, left1)

        #  cv2.imshow('6', cv2.applyColorMap(depth_image, cv2.COLORMAP_JET))

        cv2.imshow('4', color_image)
   

        return






if __name__ == "__main__":
    print("running")

    _filename = "/home/nathan/v3/rs_1629481177.bag"
    # _filename = "Desktop/fake_track/object_detection6.bag"
    # _filename=""
    _useCamera = False
    _showStream = True
    
    sd = FloorDetection()

    cam = navTypeTester.RSCamera(useCamera = _useCamera, filename=_filename)
    cam.videoNavigate(sd.rowNavigation, _showStream)