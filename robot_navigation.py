import serial
import time
import math
from threading import Thread

testLocations = [(1, 2), (2, 3), (3, 4)]


class Robot:
    def __init__(self):

        self.maxSpeed = 10

        self.coords = (0, 0)
        self.headingAngle = 0
        self.speed = (0, 0)
        self.steeringDirection = 0

        self.targetCoords = (0, 0)
        self.targetHeadingAngle = 0
        self.targetSpeed = 0

        self.GPSThread = Thread(target=self.getCoords)
        self.compassThread = Thread(target=self.getHeadings)
        self.encoderThread = Thread(target=self.getSpeed)

        self.beginSensors()

    def beginGPS(self):
        return True

    def beginEncoder(self):
        return True

    def beginCompass(self):
        return True

    def beginSerial(self):
        try:
            self.arduino = serial.Serial(port='/dev/cu.SLAB_USBtoUART', baudrate=115200, timeout=.1)
            return True
        except:
            print("Cannot set up serial port")
            return False

    def getCoords(self):
        self.coords = (0, 0, 0)
        time.sleep(1)

    def getSpeed(self):
        self.speed = (0, 0)
        time.sleep(1)

    def getHeadings(self):
        self.heading = (0, 0, 0)
        time.sleep(1)

    def beginSensors(self):
        if self.beginGPS():
            print("successfully began GPS")

            self.GPSThread.start()
        else:
            print("Unable to begin GPS")

        if self.beginEncoder():
            print("successfully began encoder")

            self.encoderThread.start()
        else:
            print("Unable to begin encoder")

        if self.beginCompass():
            print("successfully began compass")

            self.compassThread.start()
        else:
            print("Unable to begin compass")

        if self.beginSerial():
            print("successfully began serial")
        else:
            print("Unable to begin serial")

    def endSensors(self):
        self.GPSThread.join()
        self.encoderThread.join()
        self.compassThread.join()

    def move(self, heading, speed):
        # y60 forward top speed
        # y38 - middle
        # y20 backward top speed
        self.arduino.write(bytes(str("x" + str(heading)), 'utf-8'))
        time.sleep(0.1)
        self.arduino.write(bytes(str("y" + str(speed)), 'utf-8'))
        time.sleep(0.1)

    def findAngleBetween(self, coords1, coords2):
        if coords1[0] == coords2[0]:
            if coords1[1] > coords2[1]:  # directly above
                return math.pi
            else:  # directly below
                return 0
        slope = (coords1[1] - coords2[1]) / (coords1[0] - coords2[0])
        angle = math.atan(slope)

        if coords1[0] < coords2[0]:
            realAngle = angle + math.pi * 1.5
        else:  # to the left
            realAngle = angle + math.pi * 0.5

        realAngle = 2 * math.pi - realAngle
        if realAngle > math.pi:
            realAngle = realAngle - 2 * math.pi

        return realAngle

    def atDestination(self, coords1, coords2):
        return False

    def findDistBetween(self, coords1, coords2):
        # 1 deg lat = 364,000 feet
        # 1 deg long = 288,200 feet
        return math.sqrt(math.pow((coords1[0]-coords2[0])*364000, 2) + math.pow((coords1[1]-coords2[1])*288200, 2))


    def findSpeed(self, coords1, coords2):
        dist = self.findDistBetween(coords1, coords2)
        if dist<0:
            return 0


    def navigate(self, destinations):

        for self.targetCoords in destinations:
            while True:
                if not self.atDestination(self.coords,self.targetCoords):
                    self.targetHeadingAngle = self.findAngleBetween(self.coords, self.targetCoords)
                    self.targetSpeed = self.findSpeed(self.coords, self.targetCoords)
                    self.move(self.targetHeadingAngle, self.targetSpeed)
                    time.sleep(1)
                else:
                    print("reached destination")
                    break

        print("finished getting locations")

        return True


if __name__ == "__main__":
    myRobot = Robot()
    myRobot.navigate(testLocations)
    myRobot.endSensors()
