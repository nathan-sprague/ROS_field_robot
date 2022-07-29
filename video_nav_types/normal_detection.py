"""
This is the script for finding obstacles in situations not inside of a row.
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
import argparse






class StandardDetection():

    def __init__(self):
        self.smoothCenter = 0
        self.lastGood = 0
        self.outsideRow = True


    def normalNavigation(self, depth_image, color_image, showStream=False):

        depth_image_untouched = depth_image.copy() # depth_image_untouched may be read and copied but shouldn't be modified


        # find the distance from corn. This determines whether it does row-entry navigation or inter-row navigation
        distFromCorn = self.findDistFromCorn(depth_image)

        # Remove the sky and ground
        obstacleImg = self.showObstacles(depth_image_untouched.copy())
        if showStream:
            cv2.imshow("obs", obstacleImg)
        # check if anything directly in front of it is close
        hitChance = self.checkIfHitting(depth_image_untouched.copy(), obstacleImg)
        
        if hitChance > 13:
            detectionStatus = [1]
            print("about to hit", hitChance)
        else:
            detectionStatus = []

        distantObstacles = self.findDistantObstacles(depth_image_untouched.copy(), obstacleImg)

        if showStream:
            for i in distantObstacles:
                cv2.rectangle(color_image, (i[0],i[1]), (i[2],i[3]), (0,255,0), -1)
            cv2.imshow("res", color_image)


        return distantObstacles, detectionStatus, distFromCorn


  
    def findDistFromCorn(self, depth_image):
        try:
            std = np.std(depth_image[np.nonzero(depth_image)])
            distFromCorn = np.median(depth_image[np.nonzero(depth_image)]) - int(std)
            if distFromCorn < 0:
                distFromCorn = np.median(depth_image[np.nonzero(depth_image)])
                # print("negative")
        except: # sometimes a division by zero error
            distFromCorn = 0
        if distFromCorn > 1200:
            if not self.outsideRow:
                # print("outside row", distFromCorn)
                self.outsideRow = True
        else:
            if self.outsideRow:
                # print("inside row", distFromCorn)
                self.outsideRow = False
        return distFromCorn

    def showObstacles(self, depth_image):
        ht = 480
        wt = 640
        floorHorizon = 260

        depth_image_untouched = depth_image.copy()


        depth_image[depth_image > 15*256] = 0 # remove the sky

        bright = self.apply_brightness_contrast(depth_image, 10, 100) # brighten the image

        horizonMax = np.max(bright[int((floorHorizon+ht)/2)-3:int((floorHorizon+ht)/2)+3][:]) # find how far away the horizon is

        rampr16 = self.makeEmptyFloor(floorHorizon, horizonMax) # make fake floor

        bright[floorHorizon:ht][abs(bright[floorHorizon:ht]-rampr16[floorHorizon:ht]) < 256*30] = 0 # remove the floor using the empty floor
    
        bright[bright>0] = 256*255 # turn everything either black or white (no gray)

        # cv2.imshow("b3", (bright/256).astype('uint8'))
        return (bright/256).astype('uint8')


    def checkIfHitting(self, depth_image, obstacleImg):
        ht = 480
        wt = 640
        depth_image[obstacleImg==0] = 0
        depth_image = (255-depth_image/256).astype('uint8')
        depth_image[depth_image==255] = 0



        diCropped = depth_image[int(ht*0.2):int(ht*0.8), int(wt*0.3):int(wt*0.7)]
        diCropped[depth_image[int(ht*0.2):int(ht*0.8), int(wt*0.3):int(wt*0.7)] < 252] = 0
        nz = np.count_nonzero(diCropped)

        diCropped[depth_image[int(ht*0.2):int(ht*0.8), int(wt*0.3):int(wt*0.7)] == 0] = 100
        validPx =  np.count_nonzero(diCropped) - nz

        return (nz/(validPx+1))*100


    def findDistantObstacles(self, depth_image, obstacleImg):
        res = []

        ht = 480
        wt = 640
        depth_image[obstacleImg==0] = 0
        cv2.imshow("di", depth_image)


        di8 = (depth_image/256).astype('uint8')
        gray = cv2.blur(di8, (50, 50))
        gray[gray<2] = 0
        gray[gray>0] = 255

        cv2.imshow("gray", gray)

        # Set up the detector with default parameters.
        
        
        # get contour bounding boxes and draw on copy of input
        contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area>5000:

                leftColor = np.median(np.nonzero(depth_image[y:y+h, x:x+int(h*0.2)]))
                rightColor = np.median(np.nonzero(depth_image[y:y+h, x+int(h*0.8):x+h]))
                res += [[x,y,x+w,y+h,leftColor,rightColor]]
        return res



    def makeEmptyFloor(self, floorHorizon, horizonMax):
        rampr16 = np.zeros((480, 1)).astype('uint16')

        i=0
        val = 0
        if horizonMax < 1:
            horizonMax = 1
        factor = 6000/horizonMax / 4

        while i<480:
            if i < 100:
                val=0
            elif i > floorHorizon:
                val=(2**(-(i-floorHorizon)/30)*12+2) * 256
            
            rampr16[i][0] = int(val / factor)
            i+=1
        return rampr16



    def grayscale_transform(self, image_in):
        b, g, r = cv2.split(image_in)
        return 2*g - r - b

    def apply_brightness_contrast(self, input_img, brightness=0, contrast=0):
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


if __name__ == "__main__":
    import navTypeTester

    print("running")

    # _filename = "/home/nathan/Desktop/ROS_field_robot/run"
    _filename = "/home/nathan/bag_files/enter_row/july22"
    # _filename = "/home/nathan/bag_files/rs_1655483324"
    # _filename = "/home/nathan/bag_files/tall"


    _useCamera = False
    _showStream = True

    sd = StandardDetection()

    cam = navTypeTester.RSCamera(useCamera=_useCamera, filename=_filename)
    cam.videoNavigate(sd.normalNavigation, _showStream)
