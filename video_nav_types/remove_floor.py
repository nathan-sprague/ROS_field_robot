"""
This is the script for finding obstacles in situations not necessarily inside of a row.
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
        self.rampr8 = False
        self.rampr16 = False
        self.madeRamp = False
        self.floorHorizon = 260
        self.horizonMax = 6000


    def makeEmptyFloor(self):
        self.rampr16 = np.zeros((480, 1)).astype('uint16')

        i=0
        val = 0
        if self.horizonMax < 1:
            self.horizonMax = 1
        factor = 6000/self.horizonMax / 4
        # factor = 1
        # if factor > 5:
        #     factor = 5
        #     print("factor too high")
        # elif factor < 0.5:
        #     factor = 0.5
        #     # print("factor too low")
        # print(factor)



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
        cv2.imshow("rrr",  (self.rampr16/256).astype('uint8'))
        
    def findHorizon(self, depth_image):
        dc = depth_image.copy()
        dc[dc>=40*256] = 255*255
        dc[dc<40*256] = 0
        cv2.imshow("dc", dc)
        # dc[dc>40*256] = 0
        dc = cv2.resize(dc, (1, 480), interpolation=cv2.INTER_AREA)
        
        nz = np.nonzero(dc)[0]
        md = np.median(nz)
        try:
            md = int(md*2)
        except:
            return
        dc = depth_image.copy()
        # cv2.rectangle(dc, (0,0),(640,md), 256, -1)
        # cv2.imshow("new dc", dc)
        self.floorHorizon = md



    def rowNavigation(self, depth_image, color_image, showStream=False):
        # print("max", np.max(depth_image[self.floorHorizon][:]))
        self.horizonMax = np.max(depth_image[self.floorHorizon-3:self.floorHorizon+3][:])

        if True: #not self.madeRamp:
            # self.makeEmptyFloor(depth_image, color_image)
            self.madeRamp = True
            self.makeEmptyFloor()
        # i=0
        # while i<480:
        #     print(depth_image[i][320])
        #     depth_image[i][320] = 256*255

        #     i+=1
        # print("done\n\n\n")


            # exit()
        # self.findHorizon(depth_image)
        

        depth_image_og = depth_image.copy()
        # depth_image[depth_image_og==0] = 0

        cv2.imshow("og", (depth_image/256).astype('uint8'))
        depth_image[depth_image-self.rampr16>256*2] = 256*255
        depth_image[depth_image-self.rampr16<-256*2] = 256*255

        # cv2.imshow("r", (self.rampr16/256).astype('uint8'))
        cv2.imshow("mod1", (depth_image/256).astype('uint8'))
        # depth_image[depth_image<10*256] = 0
        # depth_image[depth_image>40*256] = 255*256
        # depth_image[depth_image_og==0] = 0
        depth_image[depth_image_og==0] = 0
        # color_image[depth_image>255*250] = 0



        dic = depth_image_og.copy()
        # dic[dic > 40*256] = 0 # remove the sky
        dic[dic > 15*256] = 0 # remove the sky


        bright = self.apply_brightness_contrast(dic, 10, 100)
        cv2.imshow("bright_og", (bright/256).astype('uint8'))

        self.horizonMax = np.max(bright[int((self.floorHorizon+480)/2)-3:int((self.floorHorizon+480)/2)+3][:])
        # print("horiz", self.horizonMax)
        self.makeEmptyFloor()
        bright[self.floorHorizon:480][:][abs(bright[self.floorHorizon:480][:]-self.rampr16[self.floorHorizon:480][:]) < 256*30] = 0
        # bright[0:240][:][bright[0:240][:]==1] = 256*255

        bright[bright>0] = 256*255
        dic = depth_image_og.copy()
        dic[bright == 0] = 0
        dic = 256*256-1 - dic
        cv2.imshow("b3", (dic/256*10).astype('uint8'))

        dic = cv2.resize(dic, (640, 1), interpolation=cv2.INTER_AREA)
        dic = cv2.resize(dic, (640, 480), interpolation=cv2.INTER_AREA)
        # b2 = self.apply_brightness_contrast(dic, 10, 100)

        # b2 = cv2.blur(b2, (50, 50))


        # color_image[bright==0] = 0 

        cv2.imshow("bright_new", (bright/256).astype('uint8'))
        cv2.imshow("b2", (256-dic/256*10).astype('uint8'))


        # cv2.imshow("res", color_image)


        # try:
        #     std = np.std(dic[np.nonzero(dic)])
        #     distFromCorn = np.median(dic[np.nonzero(dic)]) - int(std)*1.5
        #     if distFromCorn < 0:
        #         distFromCorn = np.median(dic[np.nonzero(dic)])
        #         # print("negative")

        # except: # sometimes a division by zero error
        #     distFromCorn = 0
        # # print("dfc", distFromCorn)

        # if not self.outsideRow and distFromCorn > 1200:
        #     print("outside row")
        #     self.outsideRow = True
        # elif self.outsideRow and distFromCorn < 1200:
        #     print("inside row")
        #     self.outsideRow = False


        # color_image[:][:][dic > distFromCorn] = 256*255
        # cv2.imshow("dic", color_image)


        # depth_image = self.apply_brightness_contrast(depth_image_og, 10, 100)  # increase contrast
        # cv2.imshow("res", color_image)#(depthco_image_og/256).astype('uint8'))




        return 1;





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

    # _filename = "/home/john/object_detections/rs_1629482645.bag"
    # _filename = "/home/nathan/bag_files/enter_row/july22.bag"
    _filename = "/home/nathan/bag_files/tall.bag"

    _useCamera = False
    _showStream = True

    sd = StandardDetection()

    cam = navTypeTester.RSCamera(useCamera=_useCamera, filename=_filename)
    cam.videoNavigate(sd.rowNavigation, _showStream)
