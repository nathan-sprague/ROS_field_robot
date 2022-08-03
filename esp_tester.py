
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


class Esp():
    def __init__(self, robot, espNum):

        self.robot = robot

        self.realRobotHeading = 0 # real heading not actually known by the robot

        self.gpsAccuracy = 0.0000
        self.gpsError = [0,0]#[0,0.0001]

        self.lastKnownCoords = [0,0]
        self.trueCoords = [0, 0]

        self.robotUpdateFrequency = 10
        self.loopsToUpdate = 0

        self.speedChangeConstant = 1 # bigger the number, the longer the motors take to react. Instant reaction is 0


        self.updateSpeed = 0.1

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



    

    def estimateCoords(self):
        p = pyproj.Proj('epsg:2793')

        turningSpeedConst = 3.7 / 0.3 * self.updateSpeed 
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
        self.robot.connectionType = 2

        p = pyproj.Proj('epsg:2793')




        time.sleep(1)
        # print("destinations:", self.robot.destinations)
        if len(self.robot.destinations) > 0:
            self.trueCoords = self.robot.destinations[0]["coord"]
            print(self.trueCoords)
            self.trueCoords = [self.trueCoords[0]-0.00005, self.trueCoords[1]-0.00005]
        else:
            self.trueCoords = [40.470383, -86.99528]


        while self.robot.notCtrlC:
           # self.robot.realSpeed = self.robot.targetSpeed[:]
        
            scc = self.speedChangeConstant

            self.robot.realSpeed = [(self.robot.targetSpeed[0]+self.robot.realSpeed[0]*scc)/(scc+1), (self.robot.targetSpeed[1]+self.robot.realSpeed[1]*scc)/(scc+1)]



            
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

        return

        if self.gpsAccuracy == 0:
            p = pyproj.Proj('epsg:2793')
            while self.robot.notCtrlC:
                

                time.sleep(self.updateSpeed)

                scc = self.speedChangeConstant


                self.robot.realSpeed = [(self.robot.targetSpeed[0]+self.robot.realSpeed[0]*ssc)/(scc+1) + (self.robot.targetSpeed[1]+self.robot.realSpeed[1]*scc)/(scc+1)]

                realHeadingChange = self.robot.realSpeed[0]-self.robot.realSpeed[1]
                self.realRobotHeading += realHeadingChange
                self.robot.trueHeading = self.realRobotHeading

                distMoved = (self.robot.realSpeed[0] + self.robot.realSpeed[1])/2 * 5280/3600/3.28
                # print("dmove", distMoved)

                # print("og coords", self.trueCoords)
                x, y = p(self.trueCoords[1], self.trueCoords[0])
                dy = distMoved * math.cos(self.realRobotHeading*math.pi/180) * updateSpeed
                dx = distMoved * math.sin(self.realRobotHeading*math.pi/180) * updateSpeed
                x += dx
                y += dy

                # print("og cart coords", dx,dy)

                y2,x2 = p(x, y, inverse=True)
                # print("\n\n\nnew coords", x2, y2)
                self.trueCoords = [x2, y2]
                self.robot.coords = self.trueCoords[:]
                # a = [x-x2)*3.28084, (y-y2)*3.28084]





        while self.robot.notCtrlC:
            # self.robot.coords = [40.422313, -86.916339]
            updateSpeed = 0.1
            time.sleep(self.updateSpeed)
            self.robot.gpsAccuracy = 0.1
            self.gpsError = [self.gpsError[0]+random.randint(-1,1)*self.gpsAccuracy/20, self.gpsError[1]+random.randint(-1,1)*self.gpsAccuracy/20]
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


            self.loopsToUpdate += 1
            if self.loopsToUpdate > self.robotUpdateFrequency:
                self.loopsToUpdate = 0
                self.robot.coords = self.trueCoords[:] #[self.trueCoords[0]+self.gpsError[0], self.trueCoords[1]+self.gpsError[1]]
                self.lastKnownCoords = self.robot.coords[:]

                self.robot.trueHeading = self.realRobotHeading
                self.robot.headingAccuracy = 0.1
                self.robot.lastHeadingTime = time.time()
                # print("set heading to real heading", self.realRobotHeading, "accuracy", self.robot.headingAccuracy)

                # if distMoved/5280*3600*updateSpeed > 0.2: # moved more than 1 ft
                #     self.robot.gpsHeading = nav_functions.findAngleBetween(self.lastKnownCoords, self.robot.coords)
                #     self.robot.gpsHeadingAvailable = True

                # else:

                #     self.robot.gpsHeading = 0
                #     self.robot.gpsHeadingAvailable = False










