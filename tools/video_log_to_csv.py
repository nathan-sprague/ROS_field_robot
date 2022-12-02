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
import pyproj




# _filename = "run"
# _filename = "/home/nathan/new_logs/older/tall/"
# _filename = "/home/nathan/new_logs/july27_wont_work/enter_row_fail"
_filename = "/home/nathan/Desktop/ROS_field_robot/test"#"file:///home/nathan/Desktop/ref_log_1664920372"
# _filename = "/home/nathan/new_logs/Aug3/fail3"


_showStream = True
_saveVideo = False
_realtime = True
_startFrame = 0
_playbackLevel=4# 0=Real (camera), 1=Simulation video, 2=simulation with just video, 3=simulation with video and position, 4=full playback but still process, 5=full playback




class RSCamera:
    def __init__(self, robot, saveVideo, filename="run", realtime=True, startFrame=0, navScript="standard", navMethod=0, playbackLevel=0):
        self.saveVideo = saveVideo
        self.realtime = realtime
        self.robot = robot
        self.idle = False
        self.noLog = False
        self.pause=False
        self.navMethod = navMethod # (0=normal, 1=enter row, 2=inner row)
        self.lastNavMethod = navMethod
        self.navScript = navScript
        self.playbackLevel = playbackLevel
        self.filename=filename
        self.startFrame = startFrame

        self.outsideRow = False
        
        self.stop = False
        self.heading = 0
        self.smoothCenter = -1000
        self.flagCenter = -1000
        self.distFromCorn = 0
        self.detectionStatus = []
        self.aboutToHit = False
        self.obstructions = []
        self.originalHeading = -1
        self.originalTargetHeading = -1

        self.fpsStart = 0
        self.fpsFramesCount = 0
        self.totalFrameCount = 0
        self.lastProcessedFrame = -1

        self.lastFrameTime = -1

        self.destID = -1
        self.obstaclesID = -1
        self.robot.startTime = int(filename[-10::])
        self.lastPosition = (0,0)
        self.lastAngle = 0
        self.lastTime = 0



    def begin(self):
       

 
        fileType = ".avi"

        fileNames = ["color", "depth1", "depth2"]
        print("reading files:", fileNames, "at", self.filename)
        print("for example:", self.filename + "/" + fileNames[0] + fileType)

        self.rgbCap = cv2.VideoCapture(self.filename + "/" + fileNames[0] + fileType)
        self.depthCap1 = cv2.VideoCapture(self.filename + "/" + fileNames[1] + fileType)
        self.depthCap2 = cv2.VideoCapture(self.filename + "/" + fileNames[2] + fileType)
     
        if (self.rgbCap.isOpened() == False):
            print("Error reading video file")
            exit()
        if self.startFrame!=0:
            print("setting frame to", self.startFrame)
            self.rgbCap.set(1, self.startFrame);
            self.depthCap1.set(1, self.startFrame);
            self.depthCap2.set(1, self.startFrame);
        


        self.readVideo()

    
    def readVideo(self):
     

 
        self.totalFrameCount = 0

        while True:
     
            self.fpsFramesCount += 1
      
            (ret, color_image) = self.rgbCap.read()
            (ret1, depth1_image) = self.depthCap1.read()
            (ret2, depth2_image) = self.depthCap2.read()


            if ret == False or ret1 == False or ret2 == False or color_image.shape[0] == 0:
                print("couldnt get frame")
                return


           

            depth1_image = cv2.cvtColor(depth1_image, cv2.COLOR_BGR2GRAY)
            depth2_image = cv2.cvtColor(depth2_image, cv2.COLOR_BGR2GRAY)


            # depth_image = np.zeros(depth1_image.shape[0:2]).astype('uint16')
            # depth_image = depth_image + depth2_image.astype("uint16")
            depth_image = depth1_image.astype("uint16")*256 + depth2_image.astype("uint16")

                



            self.color_image = color_image
            self.depth_image = depth_image

            cv2.imshow("ci", color_image)
            k=cv2.waitKey(1)
            if k == 27:
                self.robot.notCtrlC = False
                exit()
            

            if self.playbackLevel == 0 and self.saveVideo:
                self.logVals(depth_image)

                self.colorWriter.write(color_image)
                self.depthWriter1.write((depth_image/256).astype('uint8'))
                self.depthWriter2.write((depth_image%256).astype('uint8'))

            elif self.playbackLevel != 0:
                sleepTime = self.readLogs(depth_image)
                # print("og sleep time", sleepTime)
                sleepTime -= (time.time()-self.lastFrameTime)

                if self.totalFrameCount < self.startFrame:
                    sleepTime=0
                # sleepTime = 0

                # if sleepTime>0 and self.lastFrameTime != -1 and sleepTime < 1 and self.realtime:
                #     time.sleep(sleepTime)

                    # print("slept for", sleepTime)
                self.lastFrameTime = time.time()


            

            if time.time()-self.fpsStart > 1:
                self.fpsFramesCount = 0
                self.fpsStart = time.time()


            self.totalFrameCount += 1

    def readLogs(self, depth_image):
        """
        read the log from the depth image. Basically the reverse of the logVals method above.

        """

        logVersion = int(depth_image[0,0]/10)

        if logVersion == 0:
            return 0.1
        elif logVersion == 1:
            return depth_image[0,1]*100 # check if accurate
        elif logVersion == 2:

            i=0
            sizes = [1,1,1,1,1,1,1,1,1,1,3,3,1,1,2,1]
            values = [0]*16
            numRows = 4
            i=0
            px = 0
            while i<len(values):
                j=0
                while j<sizes[i]:
                    
                    val = depth_image[px%numRows, int(px/numRows)]
                    depth_image[px%numRows, int(px/numRows)] = 0
                    values[i] += val*(256**(2*j))

                    if i==15 and int(val/10) > 0: # if there are more values to be made
                        if val/10 > 100:
                            print("log format broken")
                            return 0.1
                        values += [0]*int(val/10)
                        sizes += [3]*int(val/10)
                    px += 1
                   
                    j+=1

                i+=1

            # 0=Real, 1=Simulation without video, 2=simulation with just video, 3=simulation with video and position, 4=full playback but still process, 5=full playback
            if self.playbackLevel < 3:
                return values[1]/100
            elif self.playbackLevel == 3:
                self.robot.coords[0] = (values[10] - 10**14) / (10**7)
                self.robot.coords[1] = (values[11] - 10**14) / (10**7)
                return values[1]/100



            self.robot.trueHeading = values[2] / 100
            self.robot.headingAccuracy = values[3] / 100
            self.robot.targetHeading = values[4] / 100
            self.originalTargetHeading = self.robot.targetHeading-self.robot.trueHeading
            if self.originalTargetHeading > 180:
                self.originalTargetHeading = 360-self.originalTargetHeading
            self.originalTargetHeading = (self.originalTargetHeading/90*320 + 320)
            # print("original heading", self.originalHeading)
            self.originalHeading = -(values[5] - 32768)/200

            if self.originalHeading !=0:
                self.originalHeading =  640+((640/90*self.originalHeading)-320)
            self.robot.realSpeed[0]= (values[6] - 32768)/100
            self.robot.realSpeed[1]= (values[7] - 32768)/100
            self.robot.targetSpeed[0]= (values[8] - 32768)/200
            self.robot.targetSpeed[1]= (values[9] - 32768)/200
            self.robot.coords[0] = (values[10] - 10**14) / (10**7)
            self.robot.coords[1] = (values[11] - 10**14) / (10**7)
            self.robot.gpsAccuracy = values[12]
            self.robot.connectionType = values[13]


            navStatus = values[14]
            if navStatus>0:
                self.robot.navStatus = set()
                i=32

                while i>0:
                    if navStatus >= 2**i:
                        navStatus -= 2**i
                        self.robot.navStatus.add(i)
                        # print("statusCodes", statusCodes[i])
                    i-=1


            print("nav status", self.robot.navStatus)


            extraInfo = values[15]%10
            extraInfoCount = int(values[13]/10)
            if extraInfo != 0:
                i=16
                coordList = []
                while i<len(values):
                    coordList += [[(values[i] - 10**14) / (10**7), (values[i+1] - 10**14) / (10**7)]]
                   
                    i+=2
                if extraInfo == 1:
                    dests = []
                    for i in coordList:
                        dests+=[{"coord": i, "destType": "point"}]
                    self.robot.destinations = dests
                    self.robot.destID = random.randint(0,100)
                    print("got destinations", dests)

                elif extraInfo == 2:
                    print("extra info coord list", coordList)
                    if len(coordList) ==0:
                        self.robot.obstacles = []
                    else:
                        self.robot.obstacles = [coordList]
                    self.robot.obstaclesID = random.randint(0,100)
            # print(self.robot.runTime, values[1]/100)
            dm = self.findDistBetween(self.lastPosition, self.robot.coords)

            self.lastPosition = self.robot.coords[:]

            distChange = (dm[0]**2 + dm[1]**2)*0.5
            if distChange < 10 and self.robot.realSpeed[0] != 0:
                self.robot.totalDistMoved += distChange

            if self.robot.runTime - self.lastTime != 0:
                self.robot.absAngleChange = abs((self.lastAngle - self.robot.trueHeading) /(self.robot.runTime - self.lastTime))
                if abs(self.lastAngle - self.robot.trueHeading) < 100 and self.robot.absAngleChange > 0.02 and self.robot.realSpeed[0] != 0:
                    self.robot.totalZeroPtMovement += abs(self.lastAngle - self.robot.trueHeading)


            self.lastAngle = self.robot.trueHeading

            self.robot.logData()

            if values[1]/100 < 10:
                self.robot.runTime += values[1]/100
            return values[1]/100 / 4.1

            

        else: # unknown version

            return 0.1

    def findDistBetween(self, coords1, coords2):
        # print("fdb0", findDistBetween0(coords1, coords2))
        p = pyproj.Proj('epsg:2793')
        # p = pyproj.Proj(proj='utm',zone=16,ellps='WGS84', preserve_units=False)

        x,y = p(coords1[1], coords1[0])

        x2,y2 = p(coords2[1], coords2[0])

        a = [(x-x2)*3.28084, (y-y2)*3.28084]
        # print("a", a)
        # print("new", (a[0]**2 + a[1]**2)**0.5)
        # print("fdb1", a,"\n")
        return a




    def stopStream(self):
        """
        Stop reading video and end the pipeline. Prevents possible corruption when saving the video
        """

        self.heading = -1000
        
        if self.playbackLevel==0 and self.saveVideo != False:
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
        self.robot.notCtrlC = False



if __name__ == "__main__":

    class Robot:
        def __init__(self):
            # fake robot with all the variables. Used for playback.

            self.startTime = 0 # seconds
            self.lastTotalTime = 0
           
            self.runTime = 0
            self.notCtrlC = True
            self.errorList = []
            self.alerts = "None"
            self.alertsChanged = True
            self.navStatus = {0}
            self.destinations = []
            self.destID = random.randint(0, 1000)
            self.subPoints = []
            self.defaultAtDestinationTolerance = 1.5
            self.atDestinationTolerance = self.defaultAtDestinationTolerance # at destination when x feet from target
            self.topSpeed = 2 # mph
            self.coords = [40.4697712, -86.9955213] # lat, long
            self.gpsAccuracy = 12345 # mm
            self.headingAccuracy = 360 # degrees
            self.headingAccuracyTimer = time.time() # used when doing point-to-point navigation. If the robot doesn't know where it's going, there's no point moving
            self.lastHeadingTime = time.time() # time since a heading was obtained
            self.trueHeading = 0 # degrees
            self.realSpeed = [0, 0] # mph
            self.connectionType = 0 # gps connection type (0=none, 1=dead rekoning, 2=2D fix, 3=3D fix, 4=GNSS+dead rekoning, 5=time fix)
            self.motionType = ["waiting", time.time()] 
            self.targetSpeed = [0, 0] # mph
            self.targetHeading = 0 # deg (north = 0)
            self.targetDestination = [0, 0]
            self.totalDistMoved = 0
            self.absAngleChange = 0
            self.totalZeroPtMovement = 0
            self.obstacles = [[[40.46977123481278,-86.99552120541317], [40.46993606641275,-86.99534717740178], [40.4701423474349,-86.99534608852393], [40.47013772209357,-86.99572404600923], [40.469767727757464,-86.99572404600923] ]]
            self.obstaclesID = random.randint(0,1000)
            self.obstructions = []
            self.destinations = [{"obstacles": [ [[40.46977123481278,-86.99552120541317], [40.46993606641275,-86.99534717740178], [40.4701423474349,-86.99534608852393], [40.47013772209357,-86.99572404600923], [40.469767727757464,-86.99572404600923] ] ]},
                    {"coord": [40.469895, -86.995335], "destType": "point", "destTolerance": 1.5},
                    {"coord": [40.469955, -86.995278], "destType": "point", "destTolerance": 1.5},

                    {"coord": [40.4699462,-86.9953345], "finalHeading": 270, "destType": "point", "destTolerance": 0.4},
                    {"coord": [40.4699462,-86.9953345], "destType": "row", "rowDirection": 270}]
            self.navStatus = set()

            self.firstLog = True



        def logData(self):
            saveName = "endurance3_sc.csv"

            if self.firstLog:
                print("first log\n\n\n")
                msg = "Unix Time,total run time,Run time (s),Latitude,Longitude,True Heading (degrees),Target Heading (degrees),Real Left Wheel Speed (mph),Real Right Wheel Speed (mph),Target Left Speed (mph),Target Right Speed (mph),Heading Accuracy (deg),GPS Accuracy (mm),Fix Type,total distance moved,absolute angle change,total absolute zero point turn amount"
                with open(saveName, 'w') as fileHandle:
                    fileHandle.write(str(msg) + "\n")
                    fileHandle.close()
                self.firstLog = False

            msg = ""
            connectionTypesLabels = ["Position not known", "Position known without RTK", "DGPS", "UNKNOWN FIX LEVEL 3", "RTK float", "RTK fixed"]
       


            importantVars = [self.startTime+self.runTime,
                            self.lastTotalTime + self.runTime,
                            self.runTime,
                            self.coords[0],
                            self.coords[1],
                            self.trueHeading,
                            self.targetHeading,
                            self.realSpeed[0],
                            self.realSpeed[1],
                            self.targetSpeed[0],
                            self.targetSpeed[1],
                            self.headingAccuracy,
                            self.gpsAccuracy,
                            connectionTypesLabels[self.connectionType],
                            self.totalDistMoved,
                            self.absAngleChange,
                            self.totalZeroPtMovement
                            ]
            statusCodes = {0: "Waiting to start", 1: "Waiting for GPS fix", 2: "Waiting for better GPS accuracy", 3: "Waiting for GPS heading", 4: "Moving forward normally", 5: "Close to destination, slowing down", 6: "Moving in zero point turn", 7: "Pausing during a zero point turn", 8: "At destination: pausing", 9: "Moving to correct heading", 10: "Video-based Navigation", 11: "Outside row", 12: "Don't know which way the robot is facing, slowing down", 13: "Know where the robot is facing", 14: "Slight correction necessary", 15: "major correction necessary", 16: "Shutting down robot", 17: "Heading too different from the row angle", 18: "Obstacle in view", 19: "Stopping to avoid obstacle", 20: "backing up to avoid obstacle", 21: "Backing up more to be safe", 22: "Turning to avoid obstacle", 23: "Inside row so ignoring obstacle", 24: "just left the row"};
            for sc in self.navStatus:
                importantVars += [statusCodes[sc]]

            msg = ""
            j=0
            for i in importantVars:
                msg += str(i) + ","            

            with open(saveName, 'a+') as fileHandle:
                fileHandle.write(str(msg) + "\n")
                fileHandle.close()


myRobot = Robot()
path = "/home/nathan/new_logs/endurance3/video_logs/"
files = os.listdir(path)
for filename in files:
    cam = RSCamera(myRobot, _saveVideo, filename=path + filename, realtime=_realtime,
                   startFrame=_startFrame, playbackLevel=_playbackLevel)



    myRobot.lastTotalTime += myRobot.runTime
    myRobot.runTime = 0
    cam.begin()
    with open("endurance3_sc.csv", 'a+') as fileHandle:
            fileHandle.write(str("\n") + "\n")
            fileHandle.close()
    print("runing time", myRobot.lastTotalTime)

print("done")

