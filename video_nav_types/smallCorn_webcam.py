""" 
detection algorithm designed to ue on young corn (like V3)
This program does not use the realsense camera feed at all.
It exclusively uses a webcam.
It has only been tested on a webcam that is like 3 feet in the air and sort of pointed downward.
    It may work in other orientations

This algorithm uses hough line transforms to find the row, and then takes the center from that.

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


def rotate_image_old(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


class StandardDetection():

    def __init__(self):
        self.smoothCenter = 0
        self.lastGood = 0
        self.heading = 0

        self.lastGoodLeft = (0, 0, 0)
        self.leftSearchTolerance = 400

        self.lastGoodRight = (0, 0, 0)
        self.rightSearchTolerance = 400




    def erode(self, img):

        kernel = np.ones((5,5), np.uint8)
        img_erosion = cv2.erode(img, kernel, iterations=1)

      
        return img_erosion



    def getLinePoints(self, rho, theta):
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = [int(x0 + 1000*(-b)), int(y0 + 1000*(a))]
        pt2 = [int(x0 - 1000*(-b)), int(y0 - 1000*(a))]
        return pt1, pt2


    def findIntercept(self, pt1, pt2, y=0):
        m = 1000
        if pt1[0] != pt2[0]:
            m = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])


        b = (pt1[1]-m*pt1[0])
        
        intercept = int((y-b)/m) # y=mx+b -> x=(y-b)/m

        return intercept






    def getHoughLinesOptimized(self, img, originalImg, showStream = False):
        ht = img.shape[0]
        wt = img.shape[1]

        if showStream:
            cv2.imshow("black and white", img)
        
        # polygon coordinates to ignore (where shield is)
        shieldIgnore = np.array([[int(ht/2-120), ht], [int(wt/2+120), ht], [int(wt/2+120), ht-150], [int(wt/2-120), ht-150]], np.int32)

        if True: # find row on left side

            if self.lastGoodLeft[2] != 0: # the last time this function was ran, valid results were recieved
                # use the points to help narrow down the amount of area to search for the row

                pt1, pt2 = self.getLinePoints(self.lastGoodLeft[0],self.lastGoodLeft[1])

                hold = pt1[:]
                pt1 = pt2[:]
                pt2 = hold[:]

            else:
                # otherwise search in a broad, general area
                pt1 = [int(wt/2), 0]
                pt2 = [0, ht]

            # find polygon coordinates to ignore on the left side
            pt1[0] -= int(self.leftSearchTolerance/2) 
            pt2[0] -= int(self.leftSearchTolerance/2)
            leftIgnore = np.array([[0,0], pt1, pt2, [0, ht]] ,np.int32)
            leftIgnore = leftIgnore.reshape((-1, 1, 2))

            # find polygon coordinates to ignore on the right side
            pt1[0] += self.leftSearchTolerance
            pt2[0] += self.leftSearchTolerance
            leftIgnore2 = np.array([[wt, 0], pt1, pt2, [wt, ht]], np.int32)
            leftIgnore2 = leftIgnore2.reshape((-1, 1, 2))
            pt1[0] -= int(self.leftSearchTolerance/2)
            pt2[0] -= int(self.leftSearchTolerance/2)
                
            # get the estimated hough lines
            leftTheta, leftRho, leftCnt = self.getHoughLines(img.copy(), areasIgnored=[leftIgnore, leftIgnore2, shieldIgnore], thetaRange=(0.2, 0.5))

            if leftTheta == 0: # the hough line result was invalid, use the last good values
                leftColor = (0, 255, 255)
                leftTheta = self.lastGoodLeft[1]
                leftRho = self.lastGoodLeft[0]
            else:
                leftColor = (0,255,0)
            
            leftPt1, leftPt2 = self.getLinePoints(leftRho, leftTheta) # get points for the line


        # repeat for right side
        if True:
            if self.lastGoodRight[2] != 0:
                pt1, pt2 = self.getLinePoints(self.lastGoodRight[0],self.lastGoodRight[1])

                if pt1[0] > pt2[0]:
                    print("switch")
                    hold = pt1[:]
                    pt1 = pt2[:]
                    pt2 = hold[:]

            else:
                pt1 = [int(wt/2), 0]
                pt2 = [wt, ht]

            pt1[0] -= int(self.rightSearchTolerance/2)
            pt2[0] -= int(self.rightSearchTolerance/2)
            rightIgnore = np.array([[0,0], pt1, pt2, [0, ht]] ,np.int32)
            rightIgnore = rightIgnore.reshape((-1, 1, 2))


            pt1[0] += self.rightSearchTolerance
            pt2[0] += self.rightSearchTolerance
            rightIgnore2 = np.array([[wt, 0], pt1, pt2, [wt, ht]], np.int32)
            rightIgnore2 = rightIgnore2.reshape((-1, 1, 2))
            pt1[0] -= int(self.rightSearchTolerance/2)
            pt2[0] -= int(self.rightSearchTolerance/2)
                
            theta, rho, cnt = self.getHoughLines(img.copy(), areasIgnored=[rightIgnore, rightIgnore2, shieldIgnore], thetaRange=(2.6, 2.9))


      

            if theta == 0:
                color = (0, 255, 0)
                theta = self.lastGoodRight[1]
                rho = self.lastGoodRight[0]
            else:
                color = (0,255,0)
            
            pt1, pt2 = self.getLinePoints(rho, theta)




        # find where line intercepts top of screen
        interceptLeft = self.findIntercept(leftPt1, leftPt2)

        interceptRight = self.findIntercept(pt1, pt2)


        # you can use the bottom intercept as well. Too lazy to implement
        # interceptBottom = self.findIntercept(leftPt1, leftPt2, ht)
        # cv2.line(originalImg, (interceptBottom, 0), (interceptBottom, 1000), 0, 3, cv2.LINE_AA)



        if interceptLeft > interceptRight or interceptRight-interceptLeft > 200: # the two row estimations are either crossing or really far apart (not good)
            centerColor = (0,255,255) # make the center marker yellow

            if abs(interceptLeft-wt/2) < 100 and leftCnt > cnt: # the left center is in roughly the right spot and the left accuracy marker is better than the right
                leftColor = (255,0,0)
                color = (0,0,255)
                rowCenter = int(interceptLeft+20)


            elif abs(interceptRight-wt/2) < 100 and leftCnt < cnt: # the right center is in roughly the right spot and the right accuracy marker is better
                leftColor = (0,0,255)
                color = (255,0,0)
                rowCenter = int(interceptRight-20)

            else: # neither of the rows detected are in the right spot. Everything is bad
                leftColor = (0,0,255)
                color = (0,0,255)
                centerColor = (0,0,255)
                rowCenter = int(wt/2)


        else: # everything is good for center estimations
            centerColor = (0,255,0)
            rowCenter = int((interceptLeft+interceptRight)/2)


        if showStream:
            cv2.line(originalImg, (rowCenter, 0), (rowCenter, 1000), centerColor, 3, cv2.LINE_AA) # center marker

            cv2.line(originalImg, pt1, pt2, color, 3, cv2.LINE_AA) # right row line
            
            cv2.line(originalImg, leftPt1, leftPt2, leftColor, 3, cv2.LINE_AA) # left row line
            
            # some arbitrary way to show the confidence in the row estimation
            leftCntColor = leftCnt/0.125 
            if leftCntColor > 1:
                leftCntColor = 1
            cv2.putText(originalImg, str(int(leftCntColor*1000)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255*leftCntColor,255-(255*leftCntColor)), 2, cv2.LINE_AA)
               
            cntColor = cnt/0.125
            if cntColor > 1:
                cntColor = 1
            cv2.putText(originalImg, str(int(cntColor*1000)), (wt-100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255*cntColor,255-(255*cntColor)), 2, cv2.LINE_AA)
               

            cv2.imshow("ca", originalImg) # show the image

        if leftCnt > 0: # the left estimation was okay
            self.lastGoodLeft = (leftRho, leftTheta, 1)
            self.leftSearchTolerance = 200
        else: # use the last center estimation, change the detection tolerance
            self.lastGoodLeft = (self.lastGoodLeft[0], self.lastGoodLeft[1], 0)
            self.leftSearchTolerance = 400

        if cnt > 0:
            self.lastGoodRight = (rho, theta, 1)
            self.rightSearchTolerance = 200
        else:
            self.lastGoodRight = (self.lastGoodRight[0], self.lastGoodRight[1], 0)
            self.rightSearchTolerance = 400

        return rowCenter



    def getHoughLines(self, testImage, areasIgnored=[], thetaRange=(0,3.15), threshold = 50, minThreshold = 30, step = 10):
        for pts in areasIgnored:
            cv2.fillPoly(testImage, [pts], 255)
        # if thetaRange[1] < 1:

            cv2.imshow("ti", testImage)
        for pts in areasIgnored:
            cv2.fillPoly(testImage, [pts], 0)


        dst = cv2.Canny(testImage, 600, 400, None, 3)
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

        try:
            lines = cv2.HoughLines(dst, 1, np.pi / 180, threshold, None, 0, 0)
        except:
            print("error getting hough lines")
            return 0,0,0

        falseLines = 0

        if lines is not None:
            thetas = []
            rhos = []
 
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]

                if False:
                    pt1, pt2 = self.getLinePoints(rho, theta)
                    cv2.line(cdst, pt1, pt2, (0,255,255), 1, cv2.LINE_AA)



                if thetaRange[0] < theta < thetaRange[1]:
                #    color = (0,255,0)
                    thetas += [theta]
                    rhos += [rho]
                else:
                    falseLines += 1



            # cv2.imshow("hl", cdst)



            if len(thetas) > 0:
                theta = sum(thetas) / len(thetas)
                rho = sum(rhos) / len(rhos)
                
                return theta, rho, len(thetas)/(falseLines+1)/threshold

        elif threshold > minThreshold:
            threshold -= step
            # print("none found, changing threshold to", threshold)
            return self.getHoughLines(testImage, [], thetaRange, threshold, minThreshold, step)

        

        # no suitable lines found
        return 0, 0, 0
        



    def rowNavigation(self, depth_image, color_image, depth_color_image, webcamFrame, showStream=False):
        # self.colorBasedNavigation(depth_image, color_image)
        # self.houghLineAttempt(depth_color_image)
        # return


        leftGood = False
        rightGood = False
        if int(webcamFrame.shape[1]) > 800:
            webcamFrame = cv2.resize(webcamFrame, (int(webcamFrame.shape[1]/2), int(webcamFrame.shape[0]/2)), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(webcamFrame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([50, 0, 50])
        upper_green = np.array([150, 160, 180])
        # # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # # Bitwise-AND mask and original image
        #
       # 


       # realsense mask
        # lower_green = np.array([0, 0, 240])
        # upper_green = np.array([255, 255, 255])

        # baw = cv2.cvtColor(webcamFrame, cv2.COLOR_BGR2GRAY)
        # # cv2.imshow("baw", baw)
        # baw[baw < 80] = 0
        # baw[baw > 0] = 255
        # baw = 255 - baw
        # mask = baw

        self.erode(mask)

        res2 = self.strip_process(webcamFrame)
        # cv2.imshow("r2", res2)
        s = np.size(res2)
        n = cv2.countNonZero(res2)
        if False or n/s > 0.5:

            # res2 = 255-res2
            # cv2.imshow("res old", res2)
            # print("colors inverted for some reason")
            res2 = mask#cv2.bitwise_and(webcamFrame, webcamFrame, mask=mask)
            # print(n/s)


        # cv2.rectangle(res2, (int(res2.shape[1]/2-120), res2.shape[0]), (int(res2.shape[1]/2+120), res2.shape[0]-150), 0, -1)


        # cv2.imshow('caffnny', res2)
        if showStream:
            pass
            # cv2.imshow('og', webcamFrame)

        # res2 = self.erode(res2)


        self.getHoughLinesOptimized(res2, webcamFrame, showStream=showStream)





    def grayscale_transform(self, image_in):
        b, g, r = cv2.split(image_in)
        return 2*g - r - b



    def strip_process(self, image_edit):
        # taken from: https://github.com/petern3/crop_row_detection/
           ### Grayscale Transform ###
        image_edit = self.grayscale_transform(image_edit)
        # save_image('1_image_gray', image_edit)
            
        ### Binarization ###
        _, image_edit = cv2.threshold(image_edit, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # save_image('2_image_bin', image_edit)
        return image_edit
        


if __name__ == "__main__":
    import navTypeTester

    print("running")

    # _filename = "/home/john/object_detections/rs_1629482645.bag"
    _filename = "/home/nathan/v3/rs_1629481177.bag"#
    _filename = "/home/nathan/v3/rs_1629465226.bag"  # rs_1629481177
    # _filename = "/home/nathan/tall/rs_1629482645.bag" # "/Users/nathan/bag_files/rs_1629481177.bag"
    bagFileNames = ["/home/nathan/Desktop/bag_files/rs_1655483324.bag",
    "/home/nathan/Desktop/bag_files/rs_1655483283.bag",
    "/home/nathan/Desktop/bag_files/rs_1655483251.bag",
    "/home/nathan/Desktop/bag_files/rs_1655483211.bag",
    "/home/nathan/Desktop/bag_files/rs_1655483191.bag",
    "/home/nathan/Desktop/bag_files/rs_1655483174.bag",
    "/home/nathan/Desktop/bag_files/rs_1655483113.bag",
    "/home/nathan/Desktop/bag_files/rs_1655483067.bag",
    "/home/nathan/Desktop/bag_files/rs_1655483024.bag",
    "/home/nathan/Desktop/bag_files/rs_1655482981.bag",
    "/home/nathan/Desktop/bag_files/rs_1655482933.bag"


    ]
    # _filename = "/home/john/object_detections/rs_1629482645.bag"
    # _filename = "/home/nathan/tall/rs_1629482645.bag"#"/home/nathan/v3/rs_1629465226.bag"
    _filename = bagFileNames[3]

    _useCamera = False
    _showStream = True

    sd = StandardDetection()

    cam = navTypeTester.RSCamera(useCamera=_useCamera, filename=_filename, realtime = False, useRGB=True, rgbFilename="/home/nathan/Desktop/webcam/t.webm")
    cam.videoNavigate(sd.rowNavigation, _showStream)
