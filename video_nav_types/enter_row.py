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


class StandardDetection():

    def __init__(self):
        self.smoothCenter = 0
        self.lastGood = 0


    def rowNavigation(self, depth_image, color_image, showStream=False):
        # print(showStream)
        res = {"imagesToShow": [], "center": 0}

        explainBadData = False  # print what is wrong when it does not know where it is going
        badDataReasons = []
        # Start streaming from file

        badData = False # assume it is good until future tests show it is not

        cropSize = [0.1, 0.2]
        og_width = depth_image.shape[1]
        og_height = depth_image.shape[0]
        depth_image = depth_image[int(og_height * 0.2):int(og_height * (1 - 0.4)),
                      int(og_width * 0):int(og_width * 1)]
        cv2.rectangle(depth_image, (0, 0), (int(og_width * cropSize[0]), depth_image.shape[0]), (0, 0, 0), -1)
        cv2.rectangle(depth_image, (int(og_width * (1 - cropSize[0])), 0), (depth_image.shape[1], depth_image.shape[0]),
                      (0, 0, 0), -1)



        if cv2.countNonZero(depth_image) / depth_image.size < 0.15:
            # If less than 15% of all points are valid, just say the it is bad
            badDataReasons += ["too many invalid points"]
            badData = True

        # Convert all black pixels to white. 65535 is 16-bit
        depth_image[depth_image == 0] = 65535

        # show step

        depth_image = apply_brightness_contrast(depth_image, 10, 100)  # increase contrast

        # show with increased contrast

        depth_image = (depth_image / 256).astype('uint8')  # convert to 8-bit

        cv2.imshow("cropped", depth_image)
        #  mask = np.zeros(depth_image.shape[:2], dtype="uint8")
        # cv2.rectangle(mask, (0, int(depth_image.shape[0] / 6)), (int(depth_image.shape[1]), int(depth_image.shape[0] / 1.5)), 255, -1)


        dic = depth_image.copy()

        # convert all white (invalid) to  black
        dic[depth_image == 255] = 0

        
        try:
            std = np.std(dic[np.nonzero(dic)])
            thresh = np.median(dic[np.nonzero(dic)]) - int(std)
        except:
            thresh = 100

        
        dic[dic > thresh] = 255

        # cv2.imshow("threshed", dic)

        # dic[depth_image < thresh] = 0

        res = dic.copy()

        # combine the pixel values of each column
        resized = cv2.resize(res, (res.shape[1], 1), interpolation=cv2.INTER_AREA)

        # blur it to average out outliers
        resized = cv2.blur(resized, (5, 5))

        # show the 5th step
        if showStream:
            visualResize = cv2.resize(resized, (resized.shape[1], 400), interpolation=cv2.INTER_AREA)
            # cv2.imshow("5", visualResize)
        

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


        cs = np.where(np.diff(resized[0]) > 0)[0] + 1
        ci = color_image.copy()
        whiteStart = 99999
        blackStart = 0
        i=0
        
        areas = [[0,0]]
        while i < len(cs)-1:
            if i%2 == 0: # black to white
                if areas[0][0] == 0:
                    areas[0][0] = cs[i]
                elif cs[i]-cs[i-1] > 20:
                    areas += [[cs[i], cs[i]]]


   
            else: # white to black
                areas[-1][1] = cs[i]  

            i+=1
        for i in areas:
            if i[1]-i[0]>30:
                cv2.rectangle(ci, (i[0], 0), [i[1],100], (255,0,0), 20)

        cv2.imshow("spots:", ci)
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
        if numInside < numOutside * 3:
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
            # removeThis = cv2.resize(resized, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_AREA)
            # res["imagesToShow"] += [removeThis]

            if badData:
                cv2.line(color_image, (smoothCenterInt - 2, 0),
                         (smoothCenterInt - 2, int(color_image.shape[0])), (0, 0, 255), 4)
            else:
                cv2.line(color_image, (smoothCenterInt - 2, 0),
                         (smoothCenterInt - 2, int(color_image.shape[0])), (255, 0, 0), 4)
            cv2.line(color_image, (int(color_image.shape[1] / 2), 0),
                     (int(color_image.shape[1] / 2), int(color_image.shape[0])), (0, 0, 0), 1)


            # cv2.imshow("a", color_image)
        # print(self.heading)
        # print(badDataReasons)

        return self.heading


    def grayscale_transform(self, image_in):
        b, g, r = cv2.split(image_in)
        return 2*g - r - b



if __name__ == "__main__":
    import navTypeTester

    print("running")

    # _filename = "/home/john/object_detections/rs_1629482645.bag"
    _filename = "/home/nathan/bag_files/enter_row/enter_real.bag"
    # _filename = bagFileNames[5]

    _useCamera = False
    _showStream = True

    sd = StandardDetection()

    cam = navTypeTester.RSCamera(useCamera=_useCamera, filename=_filename)
    cam.videoNavigate(sd.rowNavigation, _showStream)
