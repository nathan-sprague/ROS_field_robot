"""
This is the default navigation script. It is the most reliable and flexible compared to most other navigation scripts.
It works best with corn 24" or higher. It is generally capable of entering a row.
It also is able to perform non-row navigation, where it gives general obstacles and close hit alerts
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





class StandardDetection():

    def __init__(self):
        self.smoothCenter = 0
        self.lastGood = 0
        self.outsideRow = True


    def rowNavigation(self, depth_image, color_image, showStream=False):

        depth_image_untouched = depth_image.copy() # depth_image_untouched may be read and copied but shouldn't be modified

        if showStream:
            cv2.imshow("original", depth_image)

        detectionStatus = []

        badData = False  # assume it is good until future tests show it is not


        # find the distance from corn. This determines whether it does row-entry navigation or inter-row navigation
        distFromCorn = self.findDistFromCorn(depth_image)


        # check if video is okay
        if cv2.countNonZero(depth_image) / depth_image.size < 0.15:
            # If less than 15% of all points are valid, just say the it is bad
            detectionStatus += [0]
            badData = True



        # Remove the sky and ground
        obstacleImg = self.showObstacles(depth_image_untouched.copy())
        if showStream:
            cv2.imshow("obs", obstacleImg)
        # check if anything directly in front of it is close
        hitChance = self.checkIfHitting(depth_image_untouched.copy(), obstacleImg, showStream)
        
        if self.outsideRow or distFromCorn < 1000:
            if hitChance > 5:
                detectionStatus += [1]
                # print("about to hit", hitChance)
            elif hitChance == -1:
                detectionStatus += [1]
                # print("about to hit - few valid points")
        # print(distFromCorn)


        if self.outsideRow:
            centerEstimation, ds = self.enterRowNavigate(depth_image, obstacleImg, showStream)
          
        else:
            centerEstimation, ds = self.innerRowNavigate(depth_image, showStream)



        if type(centerEstimation) == list:
            centerEst = []
            for i in centerEstimation:
                cv2.line(color_image, (i, 0), (i, 480), (255,0,0), 2)
                centerEst += [(depth_image_untouched.shape[1] / 2 - i) * 90 / depth_image_untouched.shape[1]]
        else:
            color = (255,0,0)
            if centerEstimation == 0:
                centerEstimation = int(depth_image_untouched.shape[1] / 2)
                color = (0,0,255)
            cv2.line(color_image, (centerEstimation, 0), (centerEstimation, 480), color, 2)
            centerEst = (depth_image_untouched.shape[1] / 2 - centerEstimation) * 90 / depth_image_untouched.shape[1]
     



        detectionStatus += ds
        # print(detectionStatus)
        if showStream:
            cv2.imshow("res", color_image)
        # print("est", centerEst)
        return centerEst, distFromCorn, detectionStatus


    def innerRowNavigate(self, depth_image, showStream = False):

        detectionStatus = []

        ## remove the general area with the floor and sky to only analyze the central area
        cropSize = [0.2, 0.3]
        og_width = depth_image.shape[1]
        og_height = depth_image.shape[0]
        depth_image = depth_image[int(og_height * cropSize[1]):int(og_height * (1 - cropSize[1])),
                      int(og_width * 0):int(og_width * 1)]
        cv2.rectangle(depth_image, (0, 0), (int(og_width * cropSize[0]), depth_image.shape[0]), (0, 0, 0), -1)
        cv2.rectangle(depth_image, (int(og_width * (1 - cropSize[0])), 0), (depth_image.shape[1], depth_image.shape[0]),
                      (0, 0, 0), -1)

       
        ## convert the whole image to black/white and threshold
        depth_image[depth_image == 0] = 65535 # Convert all black pixels to white. 65535 is 16-bit
        depth_image = self.apply_brightness_contrast(depth_image, 10, 100)  # increase contrast
        depth_image = (depth_image / 256).astype('uint8')  # convert to 8-bit

        depth_image[depth_image == 255] = 0 # convert all white (invalid) pixels back to black

        # Use a binary threshold to turn everything black/white
        depth_image[depth_image > 55] = 255
        depth_image[depth_image < 56] = 0


        ## Convert the image to 1D array to analyze only vertically
        resized = cv2.resize(depth_image, (depth_image.shape[1], 1), interpolation=cv2.INTER_AREA) # combine the pixel values of each column
        resized = cv2.blur(resized, (5, 5)) # blur it to average out outliers
        avgColor = np.mean(resized) # Get the lightest colors

        ## Further filter to only have the brightest lines
        sd = np.std(resized)
        resized[0][resized[0] < avgColor + sd / 3] = 0
        resized[0][resized[0] != 0] = 255

        # show the step
        if showStream:
            visualResize = cv2.resize(resized, (resized.shape[1], 400), interpolation=cv2.INTER_AREA)
            cv2.imshow("6", visualResize)


        ## find the largest blotch of white
        colorChanges = np.where(np.diff(resized[0]) > 0)[0] + 1
        i=0            
        largestLocation = [0, 0]
        largestSize = -1
        sizes = []
        s=0
        while i < len(colorChanges):
            if i>1:
                if colorChanges[i] - colorChanges[i-1] < 20: # allow for a small black gap
                    colorChanges[i] = colorChanges[i-2]
                    sizes = sizes[0:-1]
            s = colorChanges[i+1]-colorChanges[i]
            
            if s>largestSize:
                largestSize = s
                largestLocation = [colorChanges[i], colorChanges[i+1]]
            sizes += [s]
            i+=2

        if np.sum(sizes) > largestSize * 2: # there are many other lines
            detectionStatus += [2]

        resized[0][largestLocation[0]:largestLocation[1]] = 100 # make the area used obvious
  
        if showStream:
            visualResize = cv2.resize(resized, (resized.shape[1], 400), interpolation=cv2.INTER_AREA)
            cv2.imshow("7", visualResize)

        centerEstimation = int((largestLocation[0]+largestLocation[1])/2)
        
        return centerEstimation, detectionStatus


    def enterRowNavigate(self, depth_image, obstacleImg, showStream):
        detectionStatus = []

        depth_image[obstacleImg == 0] = 0 # remove the sky and 
        depth_image = 256*256-1 - depth_image # invert the image
        # if showStream:
            # cv2.imshow("b3", (depth_image/256*10).astype('uint8'))

        depth_image = cv2.resize(depth_image, (640, 1), interpolation=cv2.INTER_AREA)

        depth_image = (256-depth_image/256*10).astype('uint8')
        depth_image[depth_image>np.median(depth_image[np.nonzero(depth_image)])] = 255
        depth_image[depth_image<255] = 0


        if showStream:
            dicVisual = cv2.resize(depth_image, (640, 480), interpolation=cv2.INTER_AREA)
            # cv2.imshow("b2", dicVisual)

        cs = np.where(np.diff(depth_image[0]) > 0)[0] + 1
        
        # find blotches of white, which are most likely row entrances.
        i=0            
        areas = [[0,0]]
        while i < len(cs):
            if i%2 == 1: # black to white
                if areas[0][0] == 0:
                    areas[0][0] = cs[i]
                elif cs[i]-cs[i-1] > 20:
                    areas += [[cs[i], cs[i]]]
            else: # white to black
                areas[-1][1] = cs[i]  

            i+=1

        areaAvgs = []
        for i in areas:
            if i[1]-i[0]>30 and i[1]-i[0]<100:
                areaAvgs += [int((i[1]+i[0])/2)]#[int((320-(i[1]+i[0])/2) * 90 / depth_image.shape[1])]
        if len(areaAvgs) > 1:
            return areaAvgs, detectionStatus
        elif len(areaAvgs) == 1:
            return areaAvgs[0], detectionStatus
        else:
            return 0, detectionStatus

        return 0, detectionStatus



  
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
                print("outside row", distFromCorn)
                self.outsideRow = True
        else:
            if self.outsideRow:
                print("inside row", distFromCorn)
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




    def checkIfHitting(self, depth_image, obstacleImg, showStream=False):
        ht = 480
        wt = 640

        depth_image_og = depth_image.copy()
        depth_image[obstacleImg==0] = 0
        depth_image = (255-depth_image/256).astype('uint8')
        depth_image[depth_image==255] = 0

        diCropped = depth_image[int(ht*0.3):int(ht*0.7), int(wt*0.2):int(wt*0.8)]
        diCropped[depth_image[int(ht*0.3):int(ht*0.7), int(wt*0.2):int(wt*0.8)] < 252] = 0
        nz = np.count_nonzero(diCropped)

        diCropped[depth_image_og[int(ht*0.3):int(ht*0.7), int(wt*0.2):int(wt*0.8)] != 0] = 100
        validPx =  np.count_nonzero(diCropped) - nz

 
        if showStream:
            # depth_image[depth_image_og==0] = 0 
            cv2.imshow("di", depth_image)
        # if color_image != False:
            # color_image[bright]
        # print(validPx, diCropped.size)

        if validPx*3 < diCropped.size:
            # print("few valid points")
            return -1



        return (nz/(validPx+1))*100

    def normalNavigation(self, depth_image, color_image, showStream=False):

        depth_image_untouched = depth_image.copy() # depth_image_untouched may be read and copied but shouldn't be modified

        if showStream:
            cv2.imshow("og", depth_image)

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
            # print("about to hit", hitChance)
        else:
            detectionStatus = []

        distantObstacles = self.findDistantObstacles(depth_image_untouched.copy(), obstacleImg)

        distantObstaclesRes = []
        if showStream:
            for i in distantObstacles:
                cv2.rectangle(color_image, (i[0],i[1]), (i[2],i[3]), (0,255,0), 1)
            cv2.imshow("res", color_image)
        for i in distantObstacles:
            distantObstaclesRes += [[int((i[0]-320)*100/640), int((i[1]-240)*100/480), int((i[2]-320)*100/480), int((i[3]-240)*100/480)]]

        return distantObstaclesRes, distFromCorn, detectionStatus


    def findDistantObstacles(self, depth_image, obstacleImg):
        res = []

        ht = 480
        wt = 640
        depth_image[obstacleImg==0] = 0


        di8 = (depth_image/256).astype('uint8')
        gray = cv2.blur(di8, (50, 50))
        gray[gray<2] = 0
        gray[gray>0] = 255

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

    # _filename = "/home/john/object_detections/rs_1629482645.bag
    _filename = "/home/nathan/bag_files/enter_row/july22"
    _filename = "/home/nathan/bag_files/tall"

    _filename = "/home/nathan/new_logs/july27/enter_row_full_10fps"


    _useCamera = False
    _showStream = True

    sd = StandardDetection()

    cam = navTypeTester.RSCamera(useCamera=_useCamera, filename=_filename)

    # cam.videoNavigate(sd.normalNavigation, _showStream)
    cam.videoNavigate(sd.rowNavigation, _showStream)

