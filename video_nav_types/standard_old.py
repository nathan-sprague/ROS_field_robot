"""
This was the default row navigation script. It got a little messy so I re-wrote it. It should function nearly the same way.

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
        depth_image_untouched = depth_image.copy()
        # print(showStream)

        detectionStatus = []
        detectionStatusExplainations = ["too many invalid points", "no distant enough spots", "more lines outside than inside", "WAY more lines outside than inside", "center estimated to be too far out"]

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

        try:
            std = np.std(depth_image_untouched[np.nonzero(depth_image_untouched)])
            distFromCorn = np.median(depth_image_untouched[np.nonzero(depth_image_untouched)]) - int(std)
            if distFromCorn < 0:
                distFromCorn = np.median(depth_image_untouched[np.nonzero(depth_image_untouched)])
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

        # print(distFromCorn)

        if cv2.countNonZero(depth_image) / depth_image.size < 0.15:
            # If less than 15% of all points are valid, just say the it is bad
            detectionStatus += [0]
            badData = True

        # Convert all black pixels to white. 65535 is 16-bit
        depth_image[depth_image == 0] = 65535


        depth_image = self.apply_brightness_contrast(depth_image, 10, 100)  # increase contrast

        # show with increased contrast

        depth_image = (depth_image / 256).astype('uint8')  # convert to 8-bit

        if showStream:
            cv2.imshow("di",depth_image)

        #  mask = np.zeros(depth_image.shape[:2], dtype="uint8")
        # cv2.rectangle(mask, (0, int(depth_image.shape[0] / 6)), (int(depth_image.shape[1]), int(depth_image.shape[0] / 1.5)), 255, -1)

        # convert all white (invalid) back to black
        depth_image[depth_image == 255] = 0

        if self.outsideRow:
           depth_image[depth_image > distFromCorn/256] = 255
           depth_image[depth_image < distFromCorn/256] = 0
        else:    
           depth_image[depth_image > 55] = 255
           depth_image[depth_image < 56] = 0
      

        res = depth_image.copy()

        # combine the pixel values of each column
        resized = cv2.resize(res, (res.shape[1], 1), interpolation=cv2.INTER_AREA)

        # blur it to average out outliers
        resized = cv2.blur(resized, (5, 5))

        # show the 5th step
        if showStream:
            visualResize = cv2.resize(resized, (resized.shape[1], 400), interpolation=cv2.INTER_AREA)
            cv2.imshow("5", visualResize)

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

        # show the 6th step
        if showStream:
            visualResize = cv2.resize(resized, (resized.shape[1], 400), interpolation=cv2.INTER_AREA)
            cv2.imshow("6", visualResize)

        # get indices of the furthest places
        z = np.nonzero(resized[0][:])

        if self.outsideRow:
            res = self.enterRow(depth_image_untouched, color_image, showStream)
            return res, distFromCorn, detectionStatus
            cs = np.where(np.diff(resized[0]) > 0)[0] + 1
            
            

            # find blotches of white, which are most likely row entrances.
            i=0            
            areas = [[0,0]]
            while i < len(cs):
                if i%2 == 0: # black to white
                    if areas[0][0] == 0:
                        areas[0][0] = cs[i]
                    elif cs[i]-cs[i-1] > 20:
                        areas += [[cs[i], cs[i]]]
                else: # white to black
                    areas[-1][1] = cs[i]  

                i+=1
            if showStream:
                ci = color_image.copy()
                for i in areas:
                    if i[1]-i[0]>30:
                        avg = int((i[1]+i[0])/2)
                        cv2.rectangle(ci, (i[0], 0), [i[1],100], (255,0,0), 20)
                        cv2.line(ci, (avg, 0), [avg, 400], (255,0,0), 2)
                cv2.imshow("res", ci)
            areaAvgs = []
            for i in areas:
                if i[1]-i[0]>30:
                    areaAvgs += [int((i[1]+i[0])/2 * 90 / depth_image.shape[1])]

            if len(areaAvgs) > 1:
                return areaAvgs, distFromCorn, detectionStatus
            elif len(areaAvgs) == 1:
                return areaAvgs[0], distFromCorn, detectionStatus



        

        # the initial center is estimated as the average of all the far away points.
        frameCenter = int(depth_image.shape[1] / 2)
        centerSpot = frameCenter

        # if there actually any valid points, use the median, if not say so
        if len(z[0]) > 0:
            centerSpot = int(np.median(z))
            # show the initial result
            if showStream:
                visualColor = color_image.copy()
                cv2.line(visualColor, (centerSpot - 2, 0),
                         (centerSpot - 2, int(color_image.shape[0])), (0, 255, 0), 4)
                # cv2.imshow("7", visualColor)
        else:
            detectionStatus += [1]
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
        if numInside < numOutside * 3:
            detectionStatus += [2]
            badData = True
        if numInside < numOutside:
            detectionStatus += [3]
            badData = True

        if len(y) > 0:
            centerSpot = int(statistics.median(y))

        if self.smoothCenter == -1000:
            self.smoothCenter = centerSpot

        if abs(centerSpot - frameCenter) > frameCenter / 3:
            detectionStatus += [4]
            # badData = True # removed because it needs sharp turns to enter a row


        # if the problem is really bad, but stick to the center of the frame
        if badData and time.time() - self.lastGood > 1 and (3 in detectionStatus):
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
            # res["imagesToShow"] += [removeThis]

            if badData:
                cv2.line(color_image, (smoothCenterInt - 2, 0),
                         (smoothCenterInt - 2, int(color_image.shape[0])), (0, 0, 255), 4)
            else:
                cv2.line(color_image, (smoothCenterInt - 2, 0),
                         (smoothCenterInt - 2, int(color_image.shape[0])), (255, 0, 0), 4)
            cv2.line(color_image, (int(color_image.shape[1] / 2), 0),
                     (int(color_image.shape[1] / 2), int(color_image.shape[0])), (0, 0, 0), 1)


            cv2.imshow("res", color_image)
    
        return self.heading, distFromCorn, detectionStatus


    def makeEmptyFloor(self):
        self.rampr16 = np.zeros((480, 1)).astype('uint16')

        i=0
        val = 0
        if self.horizonMax < 1:
            self.horizonMax = 1
        factor = 6000/self.horizonMax / 4

        while i<480:
            if i < 100:
                val=0
            elif i > self.floorHorizon:
                val=(2**(-(i-self.floorHorizon)/30)*12+2) * 256
            # print("val is", val)
            
            self.rampr16[i][0] = int(val / factor)
            # print("factor:", factor)

            i+=1
        # print("val high\n\n\n\n")
        self.rampr16 = cv2.resize(self.rampr16, (640, 480), interpolation=cv2.INTER_AREA)
        # cv2.imshow("rrr",  (self.rampr16/256).astype('uint8'))


    def enterRow(self, depth_image, color_image, showStream=False):
        self.floorHorizon = 260
        self.horizonMax = 6000
        # print("max", np.max(depth_image[self.floorHorizon][:]))
        self.horizonMax = np.max(depth_image[self.floorHorizon-3:self.floorHorizon+3][:])


        depth_image_og = depth_image.copy()
    


        dic = depth_image_og.copy()
        # dic[dic > 40*256] = 0 # remove the sky
        dic[dic > 15*256] = 0 # remove the sky


        bright = self.apply_brightness_contrast(dic, 10, 100)
        # if showStream:
            # cv2.imshow("bright_og", (bright/256).astype('uint8'))

        self.horizonMax = np.max(bright[int((self.floorHorizon+480)/2)-3:int((self.floorHorizon+480)/2)+3][:])
        # print("horiz", self.horizonMax)
        self.makeEmptyFloor()
        bright[self.floorHorizon:480][:][abs(bright[self.floorHorizon:480][:]-self.rampr16[self.floorHorizon:480][:]) < 256*30] = 0
        # bright[0:240][:][bright[0:240][:]==1] = 256*255

        bright[bright>0] = 256*255
        dic = depth_image_og.copy()
        dic[bright == 0] = 0
        dic = 256*256-1 - dic
        if showStream:
            cv2.imshow("b3", (dic/256*10).astype('uint8'))

        dic = cv2.resize(dic, (640, 1), interpolation=cv2.INTER_AREA)
        # b2 = self.apply_brightness_contrast(dic, 10, 100)

        # b2 = cv2.blur(b2, (50, 50))


        # color_image[bright==0] = 0 

        dic = (256-dic/256*10).astype('uint8')
        dic[dic>np.median(dic[np.nonzero(dic)])] = 255
        dic[dic<255] = 0

        dicVisual = cv2.resize(dic, (640, 480), interpolation=cv2.INTER_AREA)

        if showStream:
            # pass
            cv2.imshow("bright_new", (bright/256).astype('uint8'))
            cv2.imshow("b2", dicVisual)


# """ check if about to hit """
        ht = 480
        wt = 640
        deleteme = depth_image_og.copy()
        deleteme[bright==0] = 0
        deleteme = (255-deleteme/256).astype('uint8')
        deleteme[deleteme==255] = 0
        # color_image[int(ht*0.2):int(ht*0.8), int(wt*0.3):int(wt*0.7)] = (255,0,0)

        color_image[deleteme>252] = (0,0,255)
        diCropped = deleteme[int(ht*0.2):int(ht*0.8), int(wt*0.3):int(wt*0.7)]
        diCropped[deleteme[int(ht*0.2):int(ht*0.8), int(wt*0.3):int(wt*0.7)] < 252] = 0
        nz = np.count_nonzero(diCropped)

        diCropped[depth_image_og[int(ht*0.2):int(ht*0.8), int(wt*0.3):int(wt*0.7)] == 0] = 100
        invalid =  np.count_nonzero(diCropped) - nz

        if nz>10000:
            print("about to hit", nz, invalid, nz*invalid)
        if showStream:
            cv2.imshow("dm", diCropped)
# """ end check if about to hit"""


        
        cs = np.where(np.diff(dic[0]) > 0)[0] + 1
        
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
        if showStream:
            ci = color_image.copy()
            for i in areas:
                if i[1]-i[0]>30:
                    avg = int((i[1]+i[0])/2)
                    if i[1]-i[0]>100:
                        cv2.rectangle(ci, (i[0], 0), [i[1],100], (0,0,255), 20)
                    else:
                        cv2.rectangle(ci, (i[0], 0), [i[1],100], (255,0,0), 20)
                    cv2.line(ci, (avg, 0), [avg, 400], (255,0,0), 2)
            if showStream:
                cv2.imshow("res", ci)
        areaAvgs = []
        for i in areas:
            if i[1]-i[0]>30 and i[1]-i[0]<100:
                areaAvgs += [int((320-(i[1]+i[0])/2) * 90 / depth_image.shape[1])]
        if len(areaAvgs) > 1:
            return areaAvgs
        elif len(areaAvgs) == 1:
            return areaAvgs[0]
        else:
            return 0




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
    # _filename = "/home/nathan/bag_files/tall"

    # _filename = "/home/nathan/bag_files/rs_1655483324"


    _useCamera = False
    _showStream = True

    sd = StandardDetection()

    cam = navTypeTester.RSCamera(useCamera=_useCamera, filename=_filename)
    cam.videoNavigate(sd.rowNavigation, _showStream)
