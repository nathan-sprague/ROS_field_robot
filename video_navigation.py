import numpy as np
import cv2
import os
import time
import random
from threading import Thread
# import ffmpegcv
import math



class Camera:
    def __init__(self, robot, save_dir="", process_images=True, playback_type="real"):
        self.realtime = True
        self.robot = robot
        self.process_images = True
        self.playback_type = playback_type

        if self.playback_type == "real":
            self.show_stream = False
        else:
            self.show_stream = True

        self.framerate = 0.1 # secs per frame

        self.save_dir = save_dir
        if self.save_dir == "":
            self.save_video = False
        else:
            self.save_video = True

        self.inside_row = True
        self.outside_row = False
        self.robot_pose = 0
        self.possible_rows = []
        self.about_to_hit = False
        self.stop = False



        self.nav_type = "row" # "row" or "obstacle"


        self.frame_num = 0


        if self.process_images:
            # import video_nav_types.keypoint_nav as nav_tool

            import video_nav_types.reg_nav as nav_tool
            # import video_nav_types.dud_nav as nav_tool

            import video_nav_types.run_into as nav_tool_obstacle

            self.nav_tool = nav_tool
            self.nav_tool_obstacle = nav_tool_obstacle



    def begin(self):
        if self.playback_type == "real": # use camera

            # resolution = (640,480) # realsense/webcam
            # resolution=(1344, 376) # WVGA
            resolution=(2560, 720) # 720p
            # resolution=(3840, 1080) # 1080p
            # resolution=(4416, 1242) # 2.2K

            self.cap = cv2.VideoCapture(0) # May need to change index for realsense

            


            print("resolution", resolution)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap.set(cv2.CAP_PROP_FPS, int(1/self.framerate))
            if self.save_video:
                ret, test = self.cap.read()
                saveMethod = "MJPG"
                # self.color_writer_left = ffmpegcv.VideoWriter(os.path.join(self.save_dir, "left.mkv"), "h264", int(1/self.framerate))
                # self.color_writer_right = ffmpegcv.VideoWriter(os.path.join(self.save_dir, "right.mkv"), "h264", int(1/self.framerate))
                self.color_writer_left = cv2.VideoWriter(os.path.join(self.save_dir, "left.mkv"), cv2.VideoWriter_fourcc(*saveMethod), 10, (test.shape[1]//2, test.shape[0]))
                self.color_writer_right = cv2.VideoWriter(os.path.join(self.save_dir, "right.mkv"), cv2.VideoWriter_fourcc(*saveMethod), 10, (test.shape[1]//2, test.shape[0]))

            else:
                print("WARNING: Using camera but NOT saving video. Are you sure?\n\n\n\n\n\n")

        elif self.playback_type == "sim": # simulation mode
            import video_simulator
            self.save_video = False
            self.videoSim = video_simulator.VideoSim()
            self.videoSim.begin()

        else: # playback
            self.save_video = False
            self.save_dir = "/home/nathan/Desktop/oscar_ml/dataset/july6_2.mkv"
            self.save_dir2 = "/home/nathan/Desktop/oscar_ml/dataset/july6_4.mkv"
            self.cap = [cv2.VideoCapture(self.save_dir), 
                        cv2.VideoCapture(self.save_dir2)]
            # self.cap = [cv2.VideoCapture(os.path.join(self.save_dir, "left.mkv")), 
            #             cv2.VideoCapture(os.path.join(self.save_dir, "right.mkv"))]

        
        self.read_video_thread = Thread(target=self.read_video)
        self.read_video_thread.start()

        


    
    def read_video(self):
        """
        Read the video and save it.
        Forces frame rate to be what was defined in init

        """
        last_read_time = time.time()
        last_saved_frame = 0
        self.frame_num = 0
        while self.robot.notCtrlC:
            time_diff = time.time() - last_read_time
            if time_diff < self.framerate:

                time.sleep(self.framerate - time_diff)
                

            last_read_time = time.time()


            if self.playback_type == "real":
                ret, img = self.cap.read()
                if ret:
                    self.imgL = img[:, 0:img.shape[1]//2]
                    self.imgR = img[:, img.shape[1]//2::]
                    self.frame_num += 1
                else:
                    print("error reading video")
            elif self.playback_type == "sim":
                return
                x = self.robot.coords["coords"][1] * 364000
                longCorrection = math.cos(self.robot.coords["coords"][0] * math.pi / 180)
                y = self.robot.coords["coords"][0] * longCorrection * 364000

                robotLocation = [x*self.videoSim.projectionFactor, y*self.videoSim.projectionFactor, 12]

                robotView = [(self.robot.heading["heading"]-180) * math.pi/180, 0]

                depth_image, self.imgL = self.videoSim.draw3D(robotView, robotLocation)
                self.frame_num += 1
            else:
                ret1, self.imgL = self.cap[0].read()
                ret2, self.imgR = self.cap[1].read()
                if ret1 and ret2:
                    self.frame_num += 1
                else:
                    print("Error reading video")


            if self.save_video and last_saved_frame != self.frame_num:
                self.color_writer_left.write(self.imgL)
                self.color_writer_right.write(self.imgR)

                last_saved_frame = self.frame_num




    def video_navigate(self, show_stream=False):
        """
        get possible headings through corn, obstacles in view, and distance from the corn rows.

        Doesn't return anything, but it changes the following variables of this object:
            1. self.inside_row - self-evident (bool)
            2. self.robot_pose - pose relative to row(s)), degrees - 0=parallel with row, -90=robot facing left, +90=robot facing right (int)
            3. self.possible_rows - rows seen - degrees from heading (int)
            4. self.last_process_time - basically time this loop ran (float)
            5. self.about_to_hit - if the robot is about to hit something important

        Put this method in a thread.
        """

#        self.show_stream = True
        last_processed_frame = 0

        while self.robot.notCtrlC:
            if self.process_images: # processing the images takes computational power. Why do it if you are stopped or something
                while self.frame_num == last_processed_frame:
                    time.sleep(0.01)
                print("new frame")

                if self.nav_type == "row":
                    self.inside_row, self.robot_pose, self.possible_rows, self.about_to_hit = self.nav_tool.detect(self.imgL, markup=True)
                    self.outside_row = not self.inside_row
                elif self.nav_type == "obstacle":
                    about_to_hit = self.nav_tool_obstacle.detect(self.imgL, markup=True)
                last_processed_frame = self.frame_num
                self.last_process_time = time.time()
                if self.show_stream:
                    cv2.imshow("img", self.imgL)
                    k = cv2.waitKey(1)
                    if k == 27:
                        self.stopStream()




    def stopStream(self):
        """
        Stop reading video and end the pipeline. Prevents possible corruption when saving the video
        """

        if self.save_video:
            self.color_writer_left.release()
            self.color_writer_right.release()
            
        if not self.stop:
            if self.playback_type == "playback":
                self.cap[0].release()
                self.cap[1].release()
            else:
                self.cap.release()
        self.robot.notCtrlC = False



if __name__ == "__main__":

    class Robot:
        def __init__(self):
            self.notCtrlC = True

    myRobot = Robot()
    save_dir = "log_" + str(int(time.time()))
    os.mkdir(save_dir)
    cam = Camera(myRobot, save_dir=save_dir, process_images=True, playback_type="real")

    cam.begin()
    cam.video_navigate(True)
