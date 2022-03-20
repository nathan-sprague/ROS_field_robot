
import time
from threading import Thread
import math

"""
This is for testing the navigation code.
This class is meant to be as similar as possible to what the real ESP32 would do.
other than the fact that it doesn't actually connect ESP32 and it just sends fake messages.
"""


class Esp():
    def __init__(self, robot, espNum):

        self.robot = robot

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
        while self.robot.notCtrlC:
            self.robot.coords = [40.422313, -86.916339]
            updateSpeed = 0.1
            time.sleep(updateSpeed)
            self.robot.gpsAccuracy = 0.1
            # variables used:
                # self.robot.targetSpeed = [0, 0]

            # variables to change:

            self.robot.realSpeed = [(self.robot.realSpeed[0]+self.robot.targetSpeed[0])/2, (self.robot.realSpeed[1]+self.robot.targetSpeed[1])/2]

            # self.robot.realSpeed = [4,4]

            self.robot.heading += self.robot.realSpeed[0]-self.robot.realSpeed[1] # not sure if this is really how it works

            distMoved = (self.robot.realSpeed[0] + self.robot.realSpeed[1])/2 # not sure if this is really how it works

            longCorrection = math.cos(self.robot.coords[0]*math.pi/180)
            mphTolatps = 1/5280*3600/364000

            

            self.robot.coords[0] += distMoved * mphTolatps * math.cos(self.robot.heading*math.pi/180) * updateSpeed 

            longCorrection = math.cos(self.robot.coords[0]*math.pi/180)


            self.robot.coords[1] += distMoved*mphTolatps/longCorrection * math.sin(self.robot.heading*math.pi/180) * updateSpeed



    





    def findDistBetween(coords1, coords2):
        # finds distance between two coordinates in feet. Corrects for longitude

        # 1 deg lat = 364,000 feet
        # 1 deg long = 288,200 feet
        x = (coords1[1] - coords2[1]) * 364000

        longCorrection = math.cos(coords1[0] * math.pi / 180)
        y = (coords1[0] - coords2[0]) * longCorrection * 364000

        return x, y





