import pyrealsense2 as rs
import statistics

import numpy as np
import cv2
import argparse  # Import argparse for command-line options
import os
import math
import time
import random



_readMethod = "bag" # options are: "camera", "bag", "avi"
_filename = "/home/nathan/old_logs/bag_files/tall.bag"
# _filename = "/home/nathan/new_logs/Aug3/fail3/"
_showStream = True
_saveMethod = "avi" # options are: "none", "bag", "avi"


# convert bag files to three avi files. This results in a much smaller and more flexible file to save

size = (640, 480) # up to (1280, 720) if usb3.0

if _readMethod == "camera":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, size[0], size[1], rs.format.z16, 30)
    config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.accel)
    # config.enable_stream(rs.stream.gyro)
    # config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)

elif _readMethod == "bag":
    # Create object for parsing command-line options
    # Check if the given file have bag extension

    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, _filename)

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)


if _readMethod == "avi":
    fileType = ".avi"
    fileNames = ["color", "depth1", "depth2"]
    print("reading files:", fileNames, "at", _filename)
    print("for example:", _filename + "/" + fileNames[0] + fileType)

    rgbCap = cv2.VideoCapture(_filename + "/" + fileNames[0] + fileType)
    depthCap1 = cv2.VideoCapture(_filename + "/" + fileNames[1] + fileType)
    depthCap2 = cv2.VideoCapture(_filename + "/" + fileNames[2] + fileType)

    if (rgbCap.isOpened() == False):
        print("Error reading video file")
        exit()



if _saveMethod == "avi":
    runPath = "run" + str(int(time.time()))
    if not os.path.exists(runPath):
        os.makedirs(runPath)

        fileType = ".avi"
        fileNames = ["color", "depth1", "depth2"]
        print("saving files as:", fileNames, "at", runPath)
        print("for example:", runPath + "/" + fileNames[0] + fileType)

        # save as bag file for very large (ex 1.7 gb)
        # save as 'MJPG' for lossy and small (250 mb)
        # save as 'png ' for lossless and big (550 mb)
        # if color is saved as MJPG and depths are saved as png, it is a compromise (370 mb)
        saveMethod = 'MJPG'
        colorWriter = cv2.VideoWriter(runPath + "/" + fileNames[0] + fileType, cv2.VideoWriter_fourcc(*saveMethod), 30, size)
       
        saveMethod = 'png '
        depthWriter1 = cv2.VideoWriter(runPath + "/" + fileNames[1] + fileType, cv2.VideoWriter_fourcc(*saveMethod), 30, size, 0)
        depthWriter2 = cv2.VideoWriter(runPath + "/" + fileNames[2] + fileType, cv2.VideoWriter_fourcc(*saveMethod), 30, size, 0)


elif _saveMethod == "bag":
    filename = 'object_detections/rs_' + str(int(time.time())) + '.bag'
    print("saving video as " + filename)
    config.enable_record_to_file(filename)

  


# playback = profile.get_device().as_playback()
# playback.set_real_time(False)
if (_readMethod == "camera" or _readMethod == "bag"):
    profile = pipeline.start(config)

    if _showStream:
        colorizer = rs.colorizer()
  
waitKeyType = -1
while True:

    if _readMethod == "bag" or _readMethod == "camera":
        frames = pipeline.wait_for_frames()

        if False:
            accel = frames[2].as_motion_frame().get_motion_data()
            gyro = frames[3].as_motion_frame().get_motion_data()
            infrared_frame = frames.first(rs.stream.infrared)
            IR_image = np.asanyarray(infrared_frame.get_data())

        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())

        rgb_frame = frames.get_color_frame()
        color_image = np.asanyarray(rgb_frame.get_data())

                
        if _showStream:
            depth_color_frame = colorizer.colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())
            cv2.imshow("color", color_image)
            cv2.imshow("depth", depth_image)
            cv2.imshow("jetmap depth", depth_color_image)
         
    elif _readMethod == "avi":
        (ret, color_image) = rgbCap.read()
        (ret1, depth1_image) = depthCap1.read()
        (ret2, depth2_image) = depthCap2.read()

        if ret == False or ret1 == False:
            print("couldnt get frame")
            exit()
        depth1_image = cv2.cvtColor(depth1_image, cv2.COLOR_BGR2GRAY)
        depth2_image = cv2.cvtColor(depth2_image, cv2.COLOR_BGR2GRAY)

        depth_image = depth1_image.astype("uint16")*255 + depth2_image.astype("uint16")
        # depth_color_image = cv2.applyColorMap(depth1_image, cv2.COLORMAP_JET)
        
        if _showStream:
            cv2.imshow("color", color_image)
            cv2.imshow("depth", depth_image)


    if _saveMethod == "avi":
        colorWriter.write(color_image)

        depthWriter1.write((depth_image/256).astype('uint8'))
        depthWriter2.write((depth_image%256).astype('uint8'))

    

    if _showStream:

        if waitKeyType == 32:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(1)
        waitKeyType = key
        if key == 27:
            cv2.destroyAllWindows()

            if _readMethod == "camera" or _readMethod == "bag":
                
                pipeline.stop()
            elif  _readMethod == "avi":
                rgbCap.release()
                depthCap1.release()
                depthCap2.release()
            
            if _saveMethod == "avi":
                colorWriter.release()
                depthWriter1.release()
                depthWriter2.release()

            break

print("finished video")


