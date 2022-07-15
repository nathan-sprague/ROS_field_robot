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

_stepsShown = []  # [1,2,3,4,5,6]

# _filename = "Desktop/real_corn_july_15/human.bag"
# _filename = "tall/rs_1629482768.bag"
# _filename = "/Users/nathan/bag_files/rs_1629482768.bag"
_filename = "/home/nathan/bag_files/enter_row/backup.bag"
# _filename=""
_useCamera = False
_showStream = True
_saveVideo = False
_realtime = True
_startFrame = 300
_rgbFilename = "blah.mp4"
_navTypes = ["eStop"]




class RSCamera:
    def __init__(self, useCamera, saveVideo, filename="", rgbFilename="", realtime=True, startFrame=0, stepsShown=[], navMethod = "standard", robot=False):
        self.useCamera = useCamera
        self.saveVideo = saveVideo
        self.realtime = realtime
        self.startFrame = startFrame
        self.stop = False
        self.heading = 0
        self.smoothCenter = -1000
        self.flagCenter = -1000
        self.legDist = 1000
        self.stepsShown = stepsShown
        self.lastLeft1 = []
        self.lastLeftColors = []
        self.lastLeft2 = []
        self.lastRight1 = []
        self.lastRight2 = []
        self.stalkCoords = []
        self.robot = robot


        if navMethod == "standard":
            import video_nav_types.standard as navTool

        self.navMethod = navTool.StandardDetection()

        # this is needed for the floor detection function. I don't want to make the same image each time so I do it here. 
        # comment it out if you dont use the floor detection and need maximum memory
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
        self.ramprJet = cv2.applyColorMap(rampr8, cv2.COLORMAP_JET)
        #  print(self.ramprJet[0:1, 0:w])
        #   cv2.imshow("j", self.ramprJet)
        #  cv2.imshow("r", self.rampr16)
        self.rgbFilename = rgbFilename

        if useCamera:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            if saveVideo:
                if filename == "":
                    filename = 'object_detections/rs_' + str(int(time.time())) + '.bag'
                print("saving video as " + filename)
                self.config.enable_record_to_file(filename)
            else:
                print("WARNING: Using camera but NOT saving video. Are you sure?\n\n\n\n\n\n")

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

    
   


    def videoNavigate(self, showStream=False):
        self.cap = False
        self.savedVideo = False



        if True:
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


            # Get frameset of depth
            while frameCount < self.startFrame:
                frames = self.pipeline.wait_for_frames()
                frameCount += 1

            if True:
                frames = self.pipeline.wait_for_frames()
                frameCount += 1
                # Get depth frame
                depth_frame = frames.get_depth_frame()

            # the jetmap color image is never used so don't bother creating it unless you need to show it

            # Colorize depth frame to jet colormap
            # depth_color_frame = colorizer.colorize(depth_frame)

            # depth_color_image = np.asanyarray(depth_color_frame.get_data())
            # if showStream and 1 in self.stepsShown:
            #     cv2.imshow('1', depth_color_image)
            #       cv2.imshow('133', depth_color_image)

            rgb_frame = frames.get_color_frame()
            color_image = np.asanyarray(rgb_frame.get_data())
            #    cv2.imshow('2', color_image)

            #  Convert depth_frame to numpy array to render image in opencv
            depth_image = np.asanyarray(depth_frame.get_data())

            self.heading = self.navMethod.rowNavigation(depth_image, color_image, showStream)

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
            if self.robot != False:
                robotCtrlC = self.robot.notCtrlC
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


if __name__ == "__main__":
    cam = RSCamera(_useCamera, _saveVideo, filename=_filename, rgbFilename=_rgbFilename, realtime=_realtime,
                   startFrame=_startFrame, stepsShown=_stepsShown)
    cam.videoNavigate(_showStream)
