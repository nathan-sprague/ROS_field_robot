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

class RSCamera:
    def __init__(self, useCamera = False, filename="", realtime=False, startFrame=0, useRGB = False, rgbFilename=""):
        self.useCamera = useCamera
        self.realtime = realtime
        self.startFrame = startFrame
        self.stop = False
        self.useRGB = useRGB
        self.rgbJumpTo = 100
        self.cap = False
        self.shiftAmt = 0
        
        self.rgbFilename = rgbFilename

        if self.useRGB:
            if self.rgbFilename != "":
                self.cap = cv2.VideoCapture(self.rgbFilename)
            else:
                self.cap = cv2.VideoCapture(0)
            print("opened cap")

            if (self.cap.isOpened() == False):
                print("Error reading video file")
                return
            if self.rgbJumpTo != 0:
                self.cap.set(1, self.rgbJumpTo);


            # while True:
            #     ret, frame = self.cap.read()

            #     cv2.imshow("x", frame)
            #     if cv2.waitKey(1) & 0xFF == ord("q"):
            #         self.cap.release
                    # break
            # if self.saveVideo and self.useCamera:
            #     frame_width = int(self.cap.get(3))
            #     frame_height = int(self.cap.get(4))

            #     size = (frame_width, frame_height)

                # self.savedVideo = cv2.VideoWriter(self.rgbFilename, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)


       
        if useCamera:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        else:
            # Create object for parsing command-line options
            parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                          Remember to change the stream fps and format to match the recorded.")
            # Add argument which takes path to a bag file as an input
            parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
            # Parse the command line arguments to an object
            args = parser.parse_args()
            args.input = filename
            # Safety if no parameter have been given
            if not args.input:
                print("No input parameter have been given.")
                print("For help type --help")
                exit()
            # Check if the given file have bag extension
            if os.path.splitext(args.input)[1] != ".bag":
                print("The given file is not of correct file format.")
                print("Only .bag files are accepted")
                exit()
            try:
                # Create pipeline
                self.pipeline = rs.pipeline()

                # Create a config object
                self.config = rs.config()

                # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
                rs.config.enable_device_from_file(self.config, args.input)

                # Configure the pipeline to stream the depth stream
                # Change this parameters according to the recorded bag file resolution

                self.config.enable_stream(rs.stream.depth, rs.format.z16, 30)
                self.config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

            finally:
                pass

    def makeRamp(self):
        h = 480
        w = 640

        #  rampr8 = np.linspace(1000/256, 70000/256, h)
        rampr8 = np.linspace(8000, 0, int(h * 0.6))
        rampr8 = np.tile(rampr8, (w, 1))
        rampr8 = cv2.rotate(cv2.merge([rampr8]), cv2.ROTATE_90_CLOCKWISE)

        rampr16 = (rampr8).astype('uint16')
        rampr8 = rampr8.astype('uint8')
        b = np.zeros((h, w), np.uint8)
        bb = np.zeros((h, w), np.uint16)
        hh = rampr8.shape[0]
        ww = rampr8.shape[1]

        b[h - hh:h, 0:ww] = rampr8
        rampr8 = b
        r = h - 1
        m = 0
        values = [0] * 190
        i = 0
        while i < len(values):
            values[i] = int(230.0 / 190.0 * i)

            #   print(values[i])
            i += 1

        while r > 0:
            l = rampr8[r, 0]
            r -= 1
        print(m)

        bb[h - hh:h, 0:ww] = rampr16
        self.rampr16 = bb
        self.rampr8 = rampr16/256.0
        self.rampr8 = rampr8.astype('uint8')
        self.ramprJet = cv2.applyColorMap(self.rampr8, cv2.COLORMAP_JET)
        #  print(self.ramprJet[0:1, 0:w])
        # cv2.imshow("j", self.ramprJet)
        # cv2.imshow("r", self.rampr16)



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
        # self.cap = False
        self.savedVideo = False
        
        self.makeRamp()
    
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
        while not self.stop:

            
            frames = self.pipeline.wait_for_frames()
            frameCount += 1
            # Get depth frame
            depth_frame = frames.get_depth_frame()

            depth_color_frame = colorizer.colorize(depth_frame)

            depth_color_image_original = np.asanyarray(depth_color_frame.get_data())
            depth_color_image = depth_color_image_original.copy()

            if showStream:
                cv2.imshow('1', depth_color_image)

            rgb_frame = frames.get_color_frame()
            color_image_original = np.asanyarray(rgb_frame.get_data())
            color_image = color_image_original.copy()
            #    cv2.imshow('2', color_image)

            #  Convert depth_frame to numpy array to render image in opencv
            depth_image_original = np.asanyarray(depth_frame.get_data())
            depth_image = depth_image_original.copy()

            self.shiftImg(depth_image, self.shiftAmt)
            self.shiftImg(color_image, self.shiftAmt)


            if self.useRGB:

                ret, webcamFrame = self.cap.read()
                self.shiftImg(webcamFrame, self.shiftAmt)

                # cv2.imshow("x", frame)
                # navFunction(depth_image.copy(), color_image.copy(), depth_color_image, webcamFrame, showStream)
                navFunction(depth_image.copy(), color_image.copy(), depth_color_image, color_image.copy(), showStream)
 
            else:        
                navFunction(depth_image.copy(), color_image.copy(), showStream)

            if waitKeyType == 32:
                key = cv2.waitKey(0)
            else:
                key = cv2.waitKey(1)
            waitKeyType = key
            if key == 27:
                cv2.destroyAllWindows()
                if self.useRGB:
                    self.cap.release
                self.stopStream()
                break

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


