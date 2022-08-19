import pyrealsense2 as rs
import statistics

import numpy as np
import cv2
import argparse  # Import argparse for command-line options
import os
import math
import time
import random


# # _filename = "/media/nathan/SDCARD/bags/rs_1658520500.bag"
# _filename = "/home/nathan/bag_files/enter_row/blah"
# _filename = "/home/nathan/bag_files/tall"
_filename = "/home/nathan/old_logs/bag_files/tall.bag"



_useCamera = False
_showStream = True
_useBag = True
_saveVideo = False


# convert bag files to three avi files. This results in a much smaller and more flexible file to save

class RSCamera:
    def __init__(self, useCamera = False, useBag=False, filename="", saveVideo=False):
        self.useCamera = useCamera
        self.useBag = useBag

        self.stop = False
        self.saveVideo = saveVideo

        self.cap = False
        self.shiftAmt = 0

        self.first=True
        self.ts = 0

        
        size = (640, 480) # up to (1280, 720) if usb3.0

        if self.useCamera:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, size[0], size[1], rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, 30)
            # self.config.enable_stream(rs.stream.accel)
            # self.config.enable_stream(rs.stream.gyro)

            # self.config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
            self.useBag = True


        if self.saveVideo:
            runPath = "run"
            if not os.path.exists(runPath):
                os.makedirs(runPath)
            if useBag and self.useCamera: # note: not self.useBag

                # filename="run/rs.bag"
                self.config.enable_record_to_file(filename)

            else:
                

                fileType = ".avi"
                fileNames = ["color", "depth1", "depth2"]
                print("saving files as:", fileNames, "at", runPath)
                print("for example:", runPath + "/" + fileNames[0] + fileType)

                # save as bag file for very large (ex 1.7 gb)
                # save as 'MJPG' for lossy and small (250 mb)
                # save as 'png ' for lossless and big (550 mb)
                # if color is saved as MJPG and depths are saved as png, it is a compromise (370 mb)
                saveMethod = 'MJPG'
                self.colorWriter = cv2.VideoWriter(runPath + "/" + fileNames[0] + fileType, cv2.VideoWriter_fourcc(*saveMethod), 30, size)
               
                saveMethod = 'png '
                self.depthWriter1 = cv2.VideoWriter(runPath + "/" + fileNames[1] + fileType, cv2.VideoWriter_fourcc(*saveMethod), 30, size, 0)
                self.depthWriter2 = cv2.VideoWriter(runPath + "/" + fileNames[2] + fileType, cv2.VideoWriter_fourcc(*saveMethod), 30, size, 0)
                

        
        if not self.useCamera:
            if useBag:
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
            else:

                runPath = "run"#filename
         
                fileType = ".avi"

                fileNames = ["color", "depth1", "depth2"]
                print("reading files:", fileNames, "at", runPath)
                print("for example:", runPath + "/" + fileNames[0] + fileType)

                self.rgbCap = cv2.VideoCapture(runPath + "/" + fileNames[0] + fileType)
                self.depthCap1 = cv2.VideoCapture(runPath + "/" + fileNames[1] + fileType)
                self.depthCap2 = cv2.VideoCapture(runPath + "/" + fileNames[2] + fileType)
             
                if (self.rgbCap.isOpened() == False):
                    print("Error reading video file")
                    exit()


  
  
    def videoNavigate(self, navFunction, showStream=False):
        # self.cap = False
        
        if self.useBag:
            profile = self.pipeline.start(self.config)

            if not self.useCamera:
                playback = profile.get_device().as_playback()
                # playback.set_real_time(False)

        colorizer = rs.colorizer()

        # start time
        self.lastGood = time.time()
        waitKeyType = 0
        frameCount = 0

        print("started video pipeline")

        # Streaming loop
        totalGyro = 0
        while not self.stop:

            if self.useBag:
                frames = self.pipeline.wait_for_frames()

                if False:
                    accel = frames[2].as_motion_frame().get_motion_data()
                    gyro = frames[3].as_motion_frame().get_motion_data()
                    

                frameCount += 1

                # Get depth frame
                depth_frame = frames.get_depth_frame()
                depth_color_frame = colorizer.colorize(depth_frame)
                depth_color_image_original = np.asanyarray(depth_color_frame.get_data())
                depth_color_image = depth_color_image_original.copy()

                depth_image_original = np.asanyarray(depth_frame.get_data())
                depth_image = depth_image_original.copy()


                rgb_frame = frames.get_color_frame()
                color_image_original = np.asanyarray(rgb_frame.get_data())
                color_image = color_image_original.copy()

                
                cv2.imshow("og depth", depth_image)
            
                """
                max number=65536, signed=32768
                max number 2px = 4294967296 (4.3*10^9), signed = 2147483648 (2.1*10^9)
                max number 3px = 2.6e14, signed 1.4e14
                Save according to this format:
                1. log format version (0=no logs, 1=just imu) (current=2) (version*10); Navigation mode 0=normal, 1=enter row, 2=inter-row
                2. Time since last frame (s/100)
                3. GPS heading (deg/100)
                4. Heading accuracy (mm/100)
                5. Target heading (deg/100)
                6. detected target heading; middle one if multiple (deg/200)+32768
                7. left speed (mph*100)+32768
                8. right speed (mph*100)+32768
                9. target left speed (mph*100)+32768
                10. target right speed (mph*100)+32768
                11. current coordinates[0] (decimal*1e7)+1e14 (3 pixels)
                12. current coordinates[1] (decimal*1e7)+1e14 (3 pixels)
                13. connection type
                14. Extra information (0=none, 1=destinations, 2=); count of extra information*10
                """
                logVersion = [1, 10]#[0, 1*10 + 0]
                timeSinceLastFrame = [1, 0]
                gpsHeading = [1, 36000]
                headingAccuracy = [1, 36000]
                targetHeading = [1, 36000]
                detectedHeading = [1, 9000 + 32768]
                leftSpeed = [1, 500+32768]
                rightSpeed = [1, 500+32768]
                targetLeftSpeed = [1, -500+32768]
                targetRightSpeed = [1, -500+32768]
                currentCoordsX = [3, 40.4698951 * 10**7 + 10**14]
                currentCoordsY = [3, -86.9953351 * 10**7 + 10**14]
                connectionType = [1, 2]
                extraInformation = [1, 0]

                dataToLog = [
                logVersion,
                timeSinceLastFrame,
                gpsHeading,
                headingAccuracy,
                targetHeading,
                detectedHeading,
                leftSpeed,
                rightSpeed,
                targetLeftSpeed,
                targetRightSpeed,
                currentCoordsX,
                currentCoordsY,
                connectionType,
                extraInformation
                
                ]
                
                numRows = 4 #int(len(dataToLog)**0.5)
                # print("\nnum Rows", numRows)
                delete =[]
                i=0
                px = 0

                while i<len(dataToLog):
                    j=0
                    delete+=[int(dataToLog[i][1])]
                    while j<dataToLog[i][0]:

                        value=dataToLog[i][1]%(256*256)
                        depth_image[px%numRows, int(px/numRows)] = value
                        dataToLog[i][1]=int(dataToLog[i][1]/(256*256))
                
                        px += 1
                        j+=1
                    
                    i+=1




                # bytesToRecord = []
                # # imuData = [accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z]
                # # print(imuData[3], totalGyro)
                # totalGyro += imuData[3]

                # # time.sleep(0.4)
                # # print(imuData)
                # # print("got      ", imuData)
                # i=0
                # while i<len(imuData):
                #     imuData[i] += 256*256/2
                #     imuData[i] *= 256*256/2

                #     depth_image[0,i] = int(imuData[i]/(256*256-1))
                #     depth_image[1,i] = int(imuData[i]%(256*256-1))
                #     i+=1

                # print("saved    ", imuData)

                

                # cv2.imshow("moded depth", depth_image)

                # infrared_frame = frames.first(rs.stream.infrared)
                # IR_image = np.asanyarray(infrared_frame.get_data())
            else:
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
                depth_image = depth1_image.astype("uint16")*255 + depth2_image.astype("uint16")
                # d1_color = (depth1_image).astype('uint8')
                depth_color_image = cv2.applyColorMap(depth1_image, cv2.COLORMAP_JET)

                # depth_image = depth_image + 
                # depth_color_frame = colorizer.colorize(depth_image)



            cv2.imshow("modified", (depth_image/256).astype('uint8'))

            i=0
            sizes = [1,1,1,1,1,1,1,1,1,1,3,3,1,1]
            values = [0]*14
            numRows = 4
            i=0
            px = 0
            while i<len(values):
                j=0
                while j<sizes[i]:
                    
                    val = depth_image[px%numRows, int(px/numRows)]
                    depth_image[px%numRows, int(px/numRows)] = 0
                    values[i] += val*(256**(2*j))
                    px += 1
                   
                    j+=1

                i+=1


            print(values)
            print("\n")

            cv2.imshow('color', color_image)
            cv2.imshow("remodified", (depth_image/256).astype('uint8'))

            deleteme = depth_image.copy()
            deleteme[deleteme > 3] = 255*255


            depth1_image = (depth_image/256).astype('uint8')
            depth2_image = (depth_image%256).astype('uint8')

            depth_image = depth1_image.astype("uint16")*256 + depth2_image.astype("uint16")
            cv2.imshow("remade", (depth_image%256).astype('uint8'))

               # depth_image[depth_image_og==0] = 0
            deleteme = depth_image.copy()
            deleteme[deleteme > 3] = 255*255
            cv2.imshow("og2", (deleteme/256).astype('uint8'))

            if self.saveVideo:
                self.colorWriter.write(color_image)

                self.depthWriter1.write((depth_image/256).astype('uint8'))
                self.depthWriter2.write((depth_image%256).astype('uint8'))

            
            # cv2.imshow('2', depth_color_image)

            #  Convert depth_frame to numpy array to render image in opencv
           



            if waitKeyType == 32:
                key = cv2.waitKey(0)
            else:
                key = cv2.waitKey(1)
            waitKeyType = key
            if key == 27:
                cv2.destroyAllWindows()
               
                #self.cap.release
                self.stopStream()
                break

    def stopStream(self):

        
        if self.cap != False:
            self.cap.release()
        if self.saveVideo != False:
            self.colorWriter.release()
            self.depthWriter1.release()
            self.depthWriter2.release()
        if not self.stop:
            self.stop = True
            try:
                self.pipeline.stop()
                print("pipeline stopped")
            except:
                print("pipeline already stopped")

if __name__ == "__main__":

    print("running")
  





    cam = RSCamera(useCamera=_useCamera, useBag=_useBag, filename=_filename, saveVideo=_saveVideo)
    cam.videoNavigate(_showStream)

