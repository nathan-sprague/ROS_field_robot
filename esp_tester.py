
import time
from threading import Thread
import math
import random
import nav_functions

"""
This is for testing the navigation code.
This class is meant to be as similar as possible to what the real ESP32 would do.
other than the fact that it doesn't actually connect ESP32 and it just sends fake messages.
"""


class Esp():
    def __init__(self, robot, espNum):

        self.robot = robot

        self.realRobotHeading = 0 # real heading not actually known by the robot

        self.gpsAccuracy = 0.00001
        self.gpsError = [0,0]

        self.lastKnownCoords = [0,0]
        self.trueCoords = [40.422313, -86.916339]

        self.robotUpdateFrequency = 10
        self.loopsToUpdate = 0

        self.gyroError = 0


        self.espNum = espNum

        self.espType = "unknown"

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

        self.infoThread.join()


        return True


    def update(self):
        self.robot.connectionType = 1
        self.robot.coords = [40.422313, -86.916339]
        while self.robot.notCtrlC:
            # self.robot.coords = [40.422313, -86.916339]
            updateSpeed = 0.1
            time.sleep(updateSpeed)
            self.robot.gpsAccuracy = 0.1
            self.gpsError = [self.gpsError[0]+random.randint(-1,1)*self.gpsAccuracy/5, self.gpsError[1]+random.randint(-1,1)*self.gpsAccuracy/5]
            if abs(self.gpsError[0])>self.gpsAccuracy:
                self.gpsError[0] = self.gpsAccuracy * abs(self.gpsError[0])/self.gpsError[0]
            if abs(self.gpsError[0])>self.gpsAccuracy:
                self.gpsError[0] = self.gpsAccuracy * abs(self.gpsError[0])/self.gpsError[0]

            self.robot.realSpeed = [(self.robot.realSpeed[0]+self.robot.targetSpeed[0])/2, (self.robot.realSpeed[1]+self.robot.targetSpeed[1])/2]

            realHeadingChange = self.robot.realSpeed[0]-self.robot.realSpeed[1]
            self.realRobotHeading +=  realHeadingChange

            distMoved = (self.robot.realSpeed[0] + self.robot.realSpeed[1])/2 # not sure if this is really how it works


            longCorrection = math.cos(self.robot.coords[0]*math.pi/180)
            mphTolatps = 1/5280*3600/364000

            

            self.trueCoords[0] += distMoved * mphTolatps * math.cos(self.realRobotHeading*math.pi/180) * updateSpeed 

            longCorrection = math.cos(self.robot.coords[0]*math.pi/180)


            self.trueCoords[1] += distMoved*mphTolatps/longCorrection * math.sin(self.realRobotHeading*math.pi/180) * updateSpeed


            self.gyroError += random.randint(-1,1)*0.5
            if self.loopsToUpdate == int(self.robotUpdateFrequency/2) or self.loopsToUpdate == 0:
                self.robot.gyroHeading = self.realRobotHeading + self.gyroError



            self.loopsToUpdate += 1
            if self.loopsToUpdate > self.robotUpdateFrequency:
                self.loopsToUpdate = 0
                self.robot.coords = [self.trueCoords[0]+self.gpsError[0], self.trueCoords[1]+self.gpsError[1]]
                self.lastKnownCoords = self.robot.coords[:]

                if distMoved/5280*3600*updateSpeed > 0.2: # moved more than 1 ft
                    self.robot.gpsHeading = nav_functions.findAngleBetween(self.lastKnownCoords, self.robot.coords)
                    self.robot.gpsHeadingAvailable = True

                else:

                    self.robot.gpsHeading = 0
                    self.robot.gpsHeadingAvailable = False





    def findDistBetween(coords1, coords2):
        # finds distance between two coordinates in feet. Corrects for longitude

        # 1 deg lat = 364,000 feet
        # 1 deg long = 288,200 feet
        x = (coords1[1] - coords2[1]) * 364000

        longCorrection = math.cos(coords1[0] * math.pi / 180)
        y = (coords1[0] - coords2[0]) * longCorrection * 364000

        return x, y





