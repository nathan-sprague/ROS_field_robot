from numpy.core.defchararray import center
import pyrealsense2 as rs
import statistics

import numpy as np
import cv2
import argparse  # Import argparse for command-line options
import os.path
import math
import time
import random



_filename = "/home/nathan/bag_files/enter_row/july22"


_useCamera = False
_showStream = True
_saveVideo = False
_realtime = True
_startFrame = 300




class RSCamera:
    def __init__(self, useCamera,  filename=""):
        self.useCamera = useCamera
        self.stop = False
        self.heading = 0
        self.smoothCenter = -1000
        self.flagCenter = -1000
        self.distFromCorn = 0
        startFrame = 5
        
        if useCamera:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        else:
     
            fileType = ".avi"

            fileNames = ["color", "depth1", "depth2"]
            print("reading files:", fileNames, "at", filename)
            print("for example:", filename + "/" + fileNames[0] + fileType)

            self.rgbCap = cv2.VideoCapture(filename + "/" + fileNames[0] + fileType)
            self.depthCap1 = cv2.VideoCapture(filename + "/" + fileNames[1] + fileType)
            self.depthCap2 = cv2.VideoCapture(filename + "/" + fileNames[2] + fileType)
         
            if (self.rgbCap.isOpened() == False):
                print("Error reading video file")
                exit()

            if startFrame!=0:
                print("jumping start frame to", startFrame)
                self.rgbCap.set(1, startFrame);
                self.depthCap1.set(1, startFrame);
                self.depthCap2.set(1, startFrame);

    
    def shiftImg(self, img, shiftAmount = 0.1):
        # shifts the image side to side. shift amount is a fraction of how much left/right. Use negatives to shift to the left
        # this is useful for testing abilities of the detection algorithm

        if shiftAmount == 0:
            return

        ht = img.shape[1]
        wt = img.shape[0]
        shiftAmountPx = int(abs(shiftAmount)*ht)
        if shiftAmount > 0:
            img[:, 0:ht-shiftAmountPx] = img[:, shiftAmountPx::]
        else:
            img[:, shiftAmountPx::] = img[:, 0:ht-shiftAmountPx]
        # cv2.imshow("shifted", img)



    def videoNavigate(self, navFunction, showStream=False):
        self.cap = False
        self.savedVideo = False

        if self.useCamera:
            profile = self.pipeline.start(self.config)

            if not self.useCamera and (not self.realtime or self.startFrame > 0):
                playback = profile.get_device().as_playback()
                playback.set_real_time(False)

            colorizer = rs.colorizer()

        # start time
        self.lastGood = time.time()
        waitKeyType = 0
        frameCount = 0

        print("started video pipeline")


        # Streaming loop
        robotCtrlC = True
        while not self.stop and robotCtrlC:


            if self.useCamera:
                frames = self.pipeline.wait_for_frames()

                # Get depth frame
                depth_frame = frames.get_depth_frame()
                depth_image = np.asanyarray(depth_frame.get_data())

                # depth_color_frame = colorizer.colorize(depth_frame)
                # depth_color_image = depth_color_image_original.copy()

                rgb_frame = frames.get_color_frame()
                color_image_original = np.asanyarray(rgb_frame.get_data())
                color_image = color_image_original.copy()
            

                # infrared_frame = frames.first(rs.stream.infrared)
                # IR_image = np.asanyarray(infrared_frame.get_data())
            else:
                time.sleep(0.1)
                (ret, color_image) = self.rgbCap.read()
                (ret1, depth1_image) = self.depthCap1.read()
                (ret2, depth2_image) = self.depthCap2.read()

                if ret == False or ret1 == False:
                    print("couldnt get frame")
                    return
                depth1_image = cv2.cvtColor(depth1_image, cv2.COLOR_BGR2GRAY)
                depth2_image = cv2.cvtColor(depth2_image, cv2.COLOR_BGR2GRAY)


                # depth_image = np.zeros(depth1_image.shape[0:2]).astype('uint16')
                # depth_image = depth_image + depth2_image.astype("uint16")
                depth_image = depth1_image.astype("uint16")*256 + depth2_image.astype("uint16")


            self.heading, self.distFromCorn, self.status = navFunction(depth_image, color_image, showStream)

            if showStream:
                if waitKeyType == 32:
                    key = cv2.waitKey(0)
                else:
                    key = cv2.waitKey(1)
                waitKeyType = key
                if key == 27:
                    cv2.destroyAllWindows()
                    self.stopStream()
                    break
            frameCount += 1
                # print("robotCtrlC=",robotCtrlC)

    def stopStream(self):

        self.heading = -1000
        self.flagCenter = -1000
        if self.cap != False:
            self.cap.release()
        if self.savedVideo != False:
            self.savedVideo.release()

        if not self.stop:
            self.stop = True
            try:
                self.pipeline.stop()
                print("pipeline stopped")
            except:
                print("pipeline already stopped")

