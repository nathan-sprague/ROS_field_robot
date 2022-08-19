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
        self.rowSide = 0
        self.lastRowSideTime = 0


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
 
        # print(distFromCorn)


        if self.outsideRow:
            centerEstimation, ds = self.enterRowNavigate(depth_image, showStream)
          
        else:
            centerEstimation, ds = self.innerRowNavigate(depth_image, showStream)
                



        if type(centerEstimation) == list:
            mainCenter = centerEstimation[int(len(centerEstimation)/2)]
            centerEst = []
            for i in centerEstimation:
                cv2.line(color_image, (i, 0), (i, 480), (255,0,0), 2)
                centerEst += [(depth_image_untouched.shape[1] / 2 - i) * 90 / depth_image_untouched.shape[1]]

        else:
            color = (255,0,0)
            mainCenter = centerEstimation
            if centerEstimation == 0:
                centerEstimation = int(depth_image_untouched.shape[1] / 2)
                color = (0,0,255)
                badData = True

            cv2.line(color_image, (centerEstimation, 0), (centerEstimation, 480), color, 4)
            centerEst = (depth_image_untouched.shape[1] / 2 - centerEstimation) * 90 / depth_image_untouched.shape[1]




        # check if anything directly in front of it is close
        # print(mainCenter)
        if abs(mainCenter-320) > 100:
            mainCenter = 320
        
        if self.outsideRow:
            hitChance = self.checkIfHitting(depth_image_untouched.copy(), obstacleImg, showStream=showStream, hitableArea=(0.3,0.2), center=mainCenter)
        else:
            hitChance = self.checkIfHitting(depth_image_untouched.copy(), obstacleImg, showStream=showStream, hitableArea=(0.15,0.15), center=mainCenter)

        if self.outsideRow:
            if hitChance > 7:
                detectionStatus += [1]
                print("about to hit", hitChance)
            elif hitChance == -1:
                detectionStatus += [1]
                print("about to hit - few valid points")


        detectionStatus += ds
        # print(detectionStatus)
        if showStream:
            cv2.imshow("res", color_image)

        # print("est", centerEst)
        return centerEst, distFromCorn, detectionStatus, self.outsideRow


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
        if self.outsideRow:
            depth_image[depth_image > 100] = 255
            depth_image[depth_image < 100] = 0
        else:
            depth_image[depth_image > 55] = 255
            depth_image[depth_image < 56] = 0


        if showStream:
            cv2.imshow("4", depth_image)

        ## Convert the image to 1D array to analyze only vertically
        resized = cv2.resize(depth_image, (depth_image.shape[1], 1), interpolation=cv2.INTER_AREA) # combine the pixel values of each column
        resized = cv2.blur(resized, (5, 5)) # blur it to average out outliers
        avgColor = np.mean(resized) # Get the lightest colors

        # show the step
        if False: # showStream:
            visualResize = cv2.resize(resized, (resized.shape[1], 400), interpolation=cv2.INTER_AREA)
            cv2.imshow("5", visualResize)

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
        otherLocations = []
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
                otherLocations += [largestLocation]
                largestLocation = [colorChanges[i], colorChanges[i+1]]
            else:
                otherLocations += [[colorChanges[i], colorChanges[i+1]]]
            sizes += [s]
            i+=2

        if np.sum(sizes) > largestSize * 2: # there are many other lines
            detectionStatus += [2]

        for i in otherLocations:
            resized[0][i[0]:i[1]] = 50 # make the area used obvious

        resized[0][largestLocation[0]:largestLocation[1]] = 100 # make the area used obvious
  
        if showStream:
            visualResize = cv2.resize(resized, (resized.shape[1], 400), interpolation=cv2.INTER_AREA)
            cv2.imshow("7", visualResize)

        centerEstimation = int((largestLocation[0]+largestLocation[1])/2)
        
        return centerEstimation, detectionStatus


    def enterRowNavigate(self, depth_image, showStream=False):
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
        if self.outsideRow:
            depth_image[depth_image > 100] = 255
            depth_image[depth_image < 100] = 0
        else:
            depth_image[depth_image > 55] = 255
            depth_image[depth_image < 56] = 0


        if showStream:
            cv2.imshow("4", depth_image)

        ## Convert the image to 1D array to analyze only vertically
        resized = cv2.resize(depth_image, (depth_image.shape[1], 1), interpolation=cv2.INTER_AREA) # combine the pixel values of each column
        resized = cv2.blur(resized, (5, 5)) # blur it to average out outliers
        avgColor = np.mean(resized) # Get the lightest colors

        # show the step
        if False: # showStream:
            visualResize = cv2.resize(resized, (resized.shape[1], 400), interpolation=cv2.INTER_AREA)
            cv2.imshow("5", visualResize)

        ## Further filter to only have the brightest lines
        sd = np.std(resized)
        resized[0][resized[0] < avgColor + sd / 3] = 0
        resized[0][resized[0] != 0] = 255

        # show the step
        if showStream:
            visualResize = cv2.resize(resized, (resized.shape[1], 400), interpolation=cv2.INTER_AREA)
            cv2.imshow("6", visualResize)


        ## find the largest blotch of white
        ## find the largest blotch of white
        colorChanges = np.where(np.diff(resized[0]) > 0)[0] + 1
        i=0            
        largestLocation = [0, 0]
        otherLocations = []
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
                otherLocations += [largestLocation]
                largestLocation = [colorChanges[i], colorChanges[i+1]]
            else:
                otherLocations += [[colorChanges[i], colorChanges[i+1]]]
            sizes += [s]
            i+=2

        if np.sum(sizes) > 5: # there are many other lines
            detectionStatus += [2]


        for i in otherLocations:
            resized[0][i[0]:i[1]] = 10 # make the area used obvious



        validLocations = [int((largestLocation[0]+largestLocation[1])/2)]
        for i in otherLocations:
            if (i[1]-i[0])*2 > 40:
                resized[0][i[0]:i[1]] = 50 # make the area used obvious
                validLocations += [int((i[0]+i[1])/2)]
        if len(validLocations)==1:
            validLocations = validLocations[0]
        # print("valid", validLocations)

        resized[0][largestLocation[0]:largestLocation[1]] = 100 # make the area used obvious
  
        if showStream:
            visualResize = cv2.resize(resized, (resized.shape[1], 400), interpolation=cv2.INTER_AREA)
            cv2.imshow("7", visualResize)

        centerEstimation = int((largestLocation[0]+largestLocation[1])/2)
        
        return validLocations, detectionStatus

  
    def findDistFromCorn(self, depth_image):
        ht = 480
        wt = 640
        left = depth_image[0:ht, 0:int(wt*0.2)]
        right = depth_image[:, int(wt*0.8)::]

        corn = depth_image[int(ht*0.3):int(ht*0.6), :]


        if np.count_nonzero(corn) == 0 or np.count_nonzero(right) == 0 or np.count_nonzero(left) == 0:
            print("all zeros found")
            distFromCorn = 1000
            distFromCornRight = 1000
            distFromCornLeft = 1000

        else:

                
            # std = np.std(depth_image[np.nonzero(depth_image)])
            distFromCorn = np.median(corn[np.nonzero(corn)])# - int(std)

            # stdR = np.std(right[np.nonzero(right)])
            distFromCornRight = np.median(right[np.nonzero(right)])

            # stdL = np.std(left[np.nonzero(left)])
            distFromCornLeft = np.median(left[np.nonzero(left)])
            # print(distFromCorn, distFromCornLeft, distFromCornRight)

        distFromCornSide = distFromCornLeft + distFromCornRight


        # calculate whether the robot is/was in the row so it can know when it exits the row
        rst = time.time()-self.lastRowSideTime
        if rst > 1000:
            rst = 0

        if distFromCornSide > 2000:
            self.rowSide -= 1 * rst * 3
            # if not self.outsideRow:
                # print("maybe outside row")
        else:
            self.rowSide += 11
            # if self.outsideRow:
                # print("maybe inside row")

        if self.rowSide > 10:
            if self.outsideRow:
                self.outsideRow = False
                # print("inside row")
        else:
            if not self.outsideRow:
                self.outsideRow = True
                # print("outside row")
        if self.rowSide < 0:
            self.rowSide = 0
        elif self.rowSide > 18:
            self.rowSide = 18

        self.lastRowSideTime = time.time()


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
        bright = (bright/256).astype('uint8')
        resized = cv2.resize(bright, (depth_image.shape[1], 1), interpolation=cv2.INTER_AREA) # combine the pixel values of each column
        resized = cv2.resize(resized, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_AREA) # combine the pixel values of each column
        bright[resized<15] = 0
        return bright




    def checkIfHitting(self, depth_image, obstacleImg, showStream=False, hitableArea=(0.3,0.2), center = 320, distThresh=3):
        ht = 480
        wt = 640

        hitArea = int(ht/2-(ht*hitableArea[1])), int(ht/2+(ht*hitableArea[1])), int(center-(wt*hitableArea[0])), int(center+(wt*hitableArea[0]))


        depth_image_og = depth_image.copy()
        depth_image[obstacleImg==0] = 0
        depth_image = (255-depth_image/256).astype('uint8')
        depth_image[depth_image==255] = 0

        diCropped = depth_image[hitArea[0]:hitArea[1], hitArea[2]:hitArea[3]]
        diCropped[depth_image[hitArea[0]:hitArea[1], hitArea[2]:hitArea[3]] < 255-distThresh] = 0
        nz = np.count_nonzero(diCropped)

        if showStream:
            cv2.imshow("di", depth_image)
        

        diCropped[depth_image_og[hitArea[0]:hitArea[1], hitArea[2]:hitArea[3]] != 0] = 100
        validPx =  np.count_nonzero(diCropped) - nz

        

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
            bright = (depth_image/256).astype('uint8')
            cv2.imshow("og2", bright)

        # find the distance from corn. This determines whether it does row-entry navigation or inter-row navigation
        distFromCorn = self.findDistFromCorn(depth_image)

        # Remove the sky and ground
        obstacleImg = self.showObstacles(depth_image_untouched.copy())
        if showStream:
            cv2.line(obstacleImg, (0, 260), (640, 260), 256*255, 1)
            cv2.imshow("obs", obstacleImg)
        # check if anything directly in front of it is close
        hitChance = self.checkIfHitting(depth_image_untouched.copy(), obstacleImg, showStream=showStream, distThresh=5)
        
        # print("about to hit", hitChance)
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
            distantObstaclesRes += [[int((i[0]-320)*80/640), int((i[1]-240)*100/480), int((i[2]-320)*80/480), int((i[3]-240)*100/480), i[4], i[5]]]
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
            
                try:
                    sideL = depth_image[y:y+h, x:x+int(w*0.2)]
                    leftColor = np.min(sideL[sideL>0])

                    sideR = depth_image[y:y+h, x+int(w*0.8):x+w]
                    rightColor = np.min(sideR[sideR>0])
                    
                    if leftColor < rightColor:
                        if rightColor>leftColor+10000:
                            rightColor = leftColor
                    elif rightColor+10000<leftColor:
                        leftColor = rightColor
    
                    res += [[x,y,x+w,y+h,leftColor,rightColor]]
                except:
                    print("error finding obstacle")
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


    # _filename = "/home/nathan/new_logs/july27_wont_work/enter_row_fail"
    _filename = "/home/nathan/new_logs/july29/afternoon/log_1659126759"

    _useCamera = False
    _showStream = True

    sd = StandardDetection()

    cam = navTypeTester.RSCamera(useCamera=_useCamera, filename=_filename)

    cam.videoNavigate(sd.normalNavigation, _showStream)
    # cam.videoNavigate(sd.rowNavigation, _showStream)

