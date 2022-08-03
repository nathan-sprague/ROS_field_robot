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
from threading import Thread




# _filename = "run"
# _filename = "/home/nathan/new_logs/older/tall/"
# _filename = "/home/nathan/new_logs/july27_wont_work/enter_row_fail"
# _filename = "/home/nathan/new_logs/july29/morning/log_1659107512"
_filename = "/home/nathan/new_logs/Aug2/afternoon/success_but_backup"


_useCamera = False
_showStream = True
_saveVideo = False
_realtime = True
_startFrame = 50
_playbackLevel=2# 0=Real, 1=Simulation without video, 2=simulation with just video, 3=simulation with video and position, 4=full playback but still process, 5=full playback




class RSCamera:
    def __init__(self, robot, useCamera, saveVideo, filename="run", realtime=True, startFrame=0, navScript="standard", navMethod=0, playbackLevel=2):
        print("setting up camera. Usecam =", useCamera)
        self.useCamera = useCamera
        self.saveVideo = saveVideo
        self.realtime = realtime
        self.robot = robot
        self.idle = False
        self.noLog = False
        self.pause=False
        self.navMethod = navMethod # (0=normal, 1=enter row, 2=inner row)
        self.navScript = navScript
        self.playbackLevel = playbackLevel
        self.filename=filename
        self.startFrame = startFrame
        
        self.insideRow = False
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


        if navScript == "standard":
            import video_nav_types.standard as navTool


        self.navToolObj = navTool.StandardDetection()




    def begin(self):
        if self.useCamera:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            # self.config.enable_stream(rs.stream.accel)
            # self.config.enable_stream(rs.stream.gyro)

            size = (640, 480)

            if self.saveVideo:
                
                if os.path.exists(self.filename):
                    self.filename += str(int(time.time())%1000)
                os.makedirs(self.filename)

                fileType = ".avi"
                fileNames = ["color", "depth1", "depth2"]
                print("saving files as:", fileNames, "at", self.filename)
                print("for example:", self.filename + "/" + fileNames[0] + fileType)

                saveMethod = 'MJPG'
                self.colorWriter = cv2.VideoWriter(self.filename + "/" + fileNames[0] + fileType, cv2.VideoWriter_fourcc(*saveMethod), 30, size)
               
                saveMethod = 'png '
                self.depthWriter1 = cv2.VideoWriter(self.filename + "/" + fileNames[1] + fileType, cv2.VideoWriter_fourcc(*saveMethod), 30, size, 0)
                self.depthWriter2 = cv2.VideoWriter(self.filename + "/" + fileNames[2] + fileType, cv2.VideoWriter_fourcc(*saveMethod), 30, size, 0)
                

            else:
                print("WARNING: Using camera but NOT saving video. Are you sure?\n\n\n\n\n\n")

        else:
     
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
        


        self.readVideoThread = Thread(target=self.readVideo)
        self.readVideoThread.start()

    
    def readVideo(self):
        if self.useCamera:
            profile = self.pipeline.start(self.config)

            if not self.useCamera and (not self.realtime or self.startFrame > 0):
                playback = profile.get_device().as_playback()
                playback.set_real_time(False)

            colorizer = rs.colorizer()


 
        self.totalFrameCount = 0

        while not self.stop and self.robot.notCtrlC:
            if self.idle:
                print("camera idling")
                time.sleep(0.3)
            else:
                self.fpsFramesCount += 1
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
                    (ret, color_image) = self.rgbCap.read()
                    (ret1, depth1_image) = self.depthCap1.read()
                    (ret2, depth2_image) = self.depthCap2.read()

                    if ret == False or ret1 == False or ret2 == False:
                        print("couldnt get frame")
                        if self.totalFrameCount > 0:
                            print("looping")
                            self.rgbCap.set(2, 0);
                            self.depthCap1.set(cv2.CAP_PROP_POS_FRAMES, 0);
                            self.depthCap2.set(cv2.CAP_PROP_POS_FRAMES, 0);
                            (ret, color_image) = self.rgbCap.read()
                            (ret1, depth1_image) = self.depthCap1.read()
                            (ret2, depth2_image) = self.depthCap2.read()
                            self.totalFrameCount = 0
                            if np.shape(depth1_image) == ():
                                print("unable to loop")
                                return

                        else:
                            return
          
                    depth1_image = cv2.cvtColor(depth1_image, cv2.COLOR_BGR2GRAY)
                    depth2_image = cv2.cvtColor(depth2_image, cv2.COLOR_BGR2GRAY)


                    # depth_image = np.zeros(depth1_image.shape[0:2]).astype('uint16')
                    # depth_image = depth_image + depth2_image.astype("uint16")
                    depth_image = depth1_image.astype("uint16")*256 + depth2_image.astype("uint16")

                    



                self.color_image = color_image
                self.depth_image = depth_image
                

                if self.useCamera and self.saveVideo:
                    self.logVals(depth_image)

                    self.colorWriter.write(color_image)
                    self.depthWriter1.write((depth_image/256).astype('uint8'))
                    self.depthWriter2.write((depth_image%256).astype('uint8'))

                elif not self.useCamera:
                    sleepTime = self.readLogs(depth_image)
                    # print("og sleep time", sleepTime)
                    sleepTime -= (time.time()-self.lastFrameTime)

                    if self.totalFrameCount < self.startFrame:
                        sleepTime=0

                    if sleepTime>0 and self.lastFrameTime != -1 and sleepTime < 1 and self.realtime:
                        time.sleep(sleepTime)

                        # print("slept for", sleepTime)
                    self.lastFrameTime = time.time()


                

                if time.time()-self.fpsStart > 1:
                    print("fps:", self.fpsFramesCount/(time.time()-self.fpsStart))
                    self.fpsFramesCount = 0
                    self.fpsStart = time.time()

            while self.pause and not self.stop:
                time.sleep(0.1)
                if self.robot != False:
                    if not self.robot.notCtrlC:
                        break

            self.totalFrameCount += 1




    def logVals(self, depth_image):
        """
        max number=65536, signed=32768
        max number 2px = 4294967296 (4.3*10^9), signed = 2147483648 (2.1*10^9)
        max number 3px = 2.6e14, signed 1.4e14
        Save according to this format:

        For just time version
        1. (1)
        2. time since last frame

        For current version
        1. log format version (0=no logs, 1=just time) (current=2) (version*10); Navigation mode 0=normal, 1=enter row, 2=inter-row
        2. Time since last frame (s*100)
        3. GPS heading (deg*100)
        4. Heading accuracy (deg*100)
        5. Target heading (deg*100)
        6. detected target heading; middle one if multiple (deg*200)+32768
        7. left speed (mph*100)+32768
        8. right speed (mph*100)+32768
        9. target left speed (mph*100)+32768
        10. target right speed (mph*100)+32768
        11. current coordinates[0] (decimal*1e7)+1e14 (3 pixels)
        12. current coordinates[1] (decimal*1e7)+1e14 (3 pixels)
        13. gps accuracy (mm)
        14. connection type
        15. navigation status messages (sum of 2^message number)
        16. Extra information (0=none, 1=destinations, 2=obstacles); count of extra information*10
        """

        frameTime = int((time.time()-self.lastFrameTime)*100)
        self.lastFrameTime = time.time()
        if self.noLog:
            return
        elif self.robot != False: # log everything
            logVersion = [1, 2*10 + 0]
            timeSinceLastFrame = [1, frameTime]
            gpsHeading = [1, self.robot.trueHeading * 100]
            headingAccuracy = [1, int(self.robot.headingAccuracy*100)]
            targetHeading = [1, int(self.robot.targetHeading*100)]
            if type(self.heading) == list:
                detectedHeading = [1, int(self.heading[0]*200) + 32768]
            else:
                detectedHeading = [1, int(self.heading*200) + 32768]

            leftSpeed = [1, int(self.robot.realSpeed[0]*100)+32768]
            rightSpeed = [1, int(self.robot.realSpeed[1]*100)+32768]
            targetLeftSpeed = [1, int(self.robot.targetSpeed[0]*100)+32768]
            targetRightSpeed = [1, int(self.robot.targetSpeed[1]*100)+32768]
            currentCoordsX = [3, self.robot.coords[0] * 10**7 + 10**14]
            currentCoordsY = [3, self.robot.coords[1] * 10**7 + 10**14]
            gpsAccuracy = [1, self.robot.gpsAccuracy]
            if gpsAccuracy[1] > 59999:
                gpsAccuracy[1] = 59999
            connectionType = [1, self.robot.connectionType]
            navStatus = [2, 0] # right now 24/32 possible messages can be saved, may need more bytes in the future
            for i in self.robot.navStatus:
                navStatus[1] += 2**i
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
            gpsAccuracy,
            connectionType,
            navStatus,
            extraInformation
            ]
            

            if self.destID != self.robot.destID:

                destinationsList = []
                for i in self.robot.destinations:
                    if "coord" in i:
                        destinationsList += [i["coord"]]

                dataToLog[-1][1] = 1 + 10*len(destinationsList)*2
                for i in destinationsList:
                    dataToLog += [[3, i[0]* 10**7 + 10**14]]
                    dataToLog += [[3, i[1]* 10**7 + 10**14]]
                self.destID = self.robot.destID

            elif self.obstaclesID != self.robot.obstaclesID:
                dataToLog[-1][1] = 2

                
                res = []
                for j in self.robot.obstacles:
                    for i in j:
                        res += [[3, i[0]* 10**7 + 10**14]]
                        res += [[3, i[1]* 10**7 + 10**14]]
                dataToLog[-1][1] = 2 + 10*len(res)
                dataToLog += res

                self.obstaclesID = self.robot.obstaclesID

            numRows = 4

        else: # log just time
            dataToLog = [
            [1,1], [1,frameTime]
            ]
            numRows = 1

        px = 0
        i=0
        while i<len(dataToLog):
            j=0
            while j<dataToLog[i][0]:

                value=dataToLog[i][1]%(256*256)
                depth_image[px%numRows, int(px/numRows)] = value
                dataToLog[i][1]=int(dataToLog[i][1]/(256*256))
        
                px += 1
                j+=1
            
            i+=1

    def readLogs(self, depth_image):

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
            self.robot.navStatus = set()
            i=32
            while i>0:
                if navStatus >= 2**i:
                    navStatus -= 2**i
                    self.robot.navStatus.add(i)
                i-=1


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


            return values[1]/100

        else: # unknown version

            return 0.1



    def videoNavigate(self, showStream=False):

        while self.idle:
            time.sleep(0.5)

        time.sleep(0.5) # wait a bit to get the image
        

        # start time
        self.lastGood = time.time()
        waitKeyType = 0
        

        print("started video pipeline")


        # Streaming loop
        while not self.stop and self.robot.notCtrlC:
            if self.idle:
                time.sleep(0.3)
            elif self.lastProcessedFrame == self.totalFrameCount:
                # print("dont process again")
                time.sleep(0.05)
            else:
                try:
                    depth_image = self.depth_image.copy()
                    color_image = self.color_image.copy()
                    self.lastProcessedFrame = self.totalFrameCount
                except:
                    print("unable to process image")
                    time.sleep(0.3)
                else:
                    # print("row navigate", showStream)
                    if self.navMethod == 1 or self.navMethod == 2:
                        
                        self.heading, self.distFromCorn, self.detectionStatus, outsideRow = self.navToolObj.rowNavigation(depth_image, color_image, showStream)
                        self.obstructions = []
                        if self.playbackLevel != 2:
                            self.outsideRow = outsideRow
                    else:
                        # print("normalNavigation")
                        self.obstructions, self.distFromCorn, self.detectionStatus = self.navToolObj.normalNavigation(depth_image, color_image, showStream)

                    if showStream:
                        if self.originalHeading != -1:
                            self.originalHeading = int(self.originalHeading)
                            cv2.line(color_image, (self.originalHeading, 0), (self.originalHeading, 480), (255,120,120), 2)
                        if self.originalTargetHeading != -1:
                            self.originalTargetHeading = int(self.originalTargetHeading)
                            cv2.line(color_image, (self.originalTargetHeading, 0), (self.originalTargetHeading, 480), (255,255,255), 1)
                        cv2.imshow("res", color_image)

                    if 1 in self.detectionStatus:
                        self.aboutToHit = True
                    else:
                        self.aboutToHit = False

                    if showStream:
                        if waitKeyType == 32:
                            self.pause = True
                            key = cv2.waitKey(0)
                            fpsNum = self.totalFrameCount
                            self.pause = False

                            if key==32:
                                while fpsNum==self.totalFrameCount:
                                    time.sleep(0.01)
                                self.pause=True

                        else:
                            key = cv2.waitKey(1)
                        waitKeyType = key
                        if key == 27:
                            cv2.destroyAllWindows()
                            self.stopStream()
                            break

                

    def stopStream(self):

        self.heading = -1000
        
        if self.useCamera and self.saveVideo != False:
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

    class Robot:
        def __init__(self):
            self.startTime =  int(time.time()) # seconds
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
            self.obstacles = [[[40.46977123481278,-86.99552120541317], [40.46993606641275,-86.99534717740178], [40.4701423474349,-86.99534608852393], [40.47013772209357,-86.99572404600923], [40.469767727757464,-86.99572404600923] ]]
            self.obstaclesID = random.randint(0,1000)
            self.obstructions = []
            self.destinations = [{"obstacles": [ [[40.46977123481278,-86.99552120541317], [40.46993606641275,-86.99534717740178], [40.4701423474349,-86.99534608852393], [40.47013772209357,-86.99572404600923], [40.469767727757464,-86.99572404600923] ] ]},
                    {"coord": [40.469895, -86.995335], "destType": "point", "destTolerance": 1.5},
                    {"coord": [40.469955, -86.995278], "destType": "point", "destTolerance": 1.5},

                    {"coord": [40.4699462,-86.9953345], "finalHeading": 270, "destType": "point", "destTolerance": 0.4},
                    {"coord": [40.4699462,-86.9953345], "destType": "row", "rowDirection": 270}]


    myRobot = Robot()


    cam = RSCamera(myRobot, _useCamera, _saveVideo, filename=_filename, realtime=_realtime,
                   startFrame=_startFrame, playbackLevel=_playbackLevel)
    cam.begin()
    cam.navMethod = 1 # (0=normal, 1=enter row, 2=inner row)
    cam.videoNavigate(_showStream)
