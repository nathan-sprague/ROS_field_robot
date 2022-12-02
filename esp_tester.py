
import time
from threading import Thread
import math
import random
import nav_functions
import pyproj


"""
This is for testing the navigation code.
This class is meant to be as similar as possible to what the real ESP32 would do.
other than the fact that it doesn't actually connect ESP32 and it just sends fake messages.
"""


class Gps():
    def __init__(self):
        self.debugOptions = {"heading board connected": [True, 0], 
                            "SIV": [99, 0], 
                            "RTK signal available": [True, 0],
                            "RTK signal used": [True, 0]}
    def endGPS(self):
        return True


class Esp():
    def __init__(self, robot, espNum):

        self.robot = robot

        self.realRobotHeading = 270 # real heading not actually known by the robot

        self.gpsAccuracy = 0.0000
        self.gpsError = [0,0]#[0,0.0001]

        self.lastKnownCoords = [0,0]
        self.trueCoords = [0, 0]

        self.robotUpdateFrequency = 10
        self.loopsToUpdate = 0

        self.speedChangeConstant = 4 # bigger the number, the longer the motors take to react. Instant reaction is 0


        self.updateSpeed = 0.1

        self.espNum = espNum

        self.espType = "unknown"

        self.infoThread = False

        # Status of the ESP32
        self.stopped = False





    def begin(self):

        
        if self.espNum == 0:
            self.infoThread = Thread(target=self.update)
            self.infoThread.start()
            return True
        else:
            return False



    def endEsp(self):
        """
            joins the read and write threads and tells the ESP to stop
        """

        if self.infoThread != False:
            self.infoThread.join()


        return True



    

    def estimateCoords(self):
        """
        estimate the new location of the robot based on the wheel speed
        """

        # use the indiana map projection
        p = pyproj.Proj('epsg:2793')

        turningSpeedConst = 1.7 / 0.3 * self.updateSpeed 
        movementSpeedConst = 0.35
        
        realHeadingChange = (self.robot.realSpeed[0]-self.robot.realSpeed[1])*turningSpeedConst

        self.realRobotHeading += realHeadingChange

        distMoved = (self.robot.realSpeed[0] + self.robot.realSpeed[1]) * 5280/3600/3.28 * movementSpeedConst
        # print("dmove", distMoved)

        # print("og coords", self.trueCoords)
        x, y = p(self.trueCoords[1], self.trueCoords[0])
        dy = distMoved * math.cos(self.realRobotHeading*math.pi/180) * self.updateSpeed
        dx = distMoved * math.sin(self.realRobotHeading*math.pi/180) * self.updateSpeed
        x += dx
        y += dy

        # print("og cart coords", dx,dy)

        y2,x2 = p(x, y, inverse=True)


        self.trueCoords = [x2, y2]


    def update(self):
        """
        update the robot's location and status

        """

        self.robot.connectionType = 2
        self.robot.gpsAccuracy = 0.1
        self.robot.headingAccuracy = 1
        

        p = pyproj.Proj('epsg:2793')




        time.sleep(1)
        # print("destinations:", self.robot.destinations)
        if len(self.robot.destinations) > 0:
            self.trueCoords = self.robot.destinations[0]["coord"]
            print(self.trueCoords)
            self.trueCoords = [self.trueCoords[0]-0.0000, self.trueCoords[1]+0.000005]#[self.trueCoords[0]-0.00005, self.trueCoords[1]-0.00005]
        else:
            self.trueCoords = [40.470383, -86.99528]


        while self.robot.notCtrlC:
            self.robot.lastHeadingTime = time.time()
           # self.robot.realSpeed = self.robot.targetSpeed[:]
        
            scc = self.speedChangeConstant
            # self.robot.realSpeed = [(self.robot.targetSpeed[0]+self.robot.realSpeed[0]*scc)/(scc+1), (self.robot.targetSpeed[1]+self.robot.realSpeed[1]*scc)/(scc+1)]



            
            self.estimateCoords()
            self.robot.trueHeading = self.realRobotHeading % 360 #+ random.randint(-5,5)
            self.robot.coords = self.trueCoords[:]
            self.gpsError = [self.gpsError[0]+random.randint(-1,1)*self.gpsAccuracy/20, self.gpsError[1]+random.randint(-1,1)*self.gpsAccuracy/20]

            if abs(self.gpsError[0])>self.gpsAccuracy:
                self.gpsError[0] = self.gpsAccuracy * abs(self.gpsError[0])/self.gpsError[0]

            if abs(self.gpsError[1])>self.gpsAccuracy:
                self.gpsError[1] = self.gpsAccuracy * abs(self.gpsError[1])/self.gpsError[1]
            self.robot.coords = [self.trueCoords[0] + self.gpsError[0], self.trueCoords[1] + self.gpsError[1]]


            # print("esp sleeping", self.robot.updateSpeed, self.updateSpeed)
            startSleepTime = time.time()
            while self.updateSpeed+startSleepTime-time.time() > 1:
                time.sleep(1)

            if self.updateSpeed+startSleepTime-time.time() > 0:
                time.sleep(self.updateSpeed+startSleepTime-time.time())

