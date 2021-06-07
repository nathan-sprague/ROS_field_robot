import serial
import time
import math
from threading import Thread

testLocations = [[1, 2], [2, 3], [3, 4]]


class Robot:
    def __init__(self):

        self.coords = [0, 0]
        self.headingAngle = 0
        self.wheelSpeed = 0
        self.steeringDirection = 0

        self.targetCoords = [0, 0]
        self.targetHeadingAngle = 0
        self.targetSpeed = 0

        self.distanceTraveled = 0

        self.compassThread = Thread(target=self.getHeadings)
        self.SteerSerialThread = Thread(target=self.readSteerSerial)
        self.MoveSerialThread = Thread(target=self.readMoveSerial)

        if self.beginCompass():
            self.compassThread.start()
        else:
            print("unable to begin compass")


        self.moveESP = serial.Serial(port='/dev/cu.SLAB_USBtoUART0', baudrate=115200, timeout=.1)
        self.steerESP = serial.Serial(port='/dev/cu.SLAB_USBtoUART1', baudrate=115200, timeout=.1)
        print("Set up serial port")
        self.SteerSerialThread.start()
        self.MoveSerialThread.start()


    def beginCompass(self):
        try:
            from i2clibraries import i2c_hmc5883l

            self.hmc5883l = i2c_hmc5883l.i2c_hmc5883l(1)

            self.hmc5883l.setContinuousMode()
            return True
        except:
            return False

    def readSteerSerial(self):
        line = self.moveESP.readline()
        self.processSerial(line)
        time.sleep(0.1)

    def readMoveSerial(self):
        line = self.steerESP.readline()
        self.processSerial(line)
        time.sleep(0.1)

    def processSerial(self, message):

        if len(message) > 0:
            msgType = message[0]

            if msgType == '.':
                print(message[1::])

            try:
                res = int(message[1::])
            except:
                print("invalid message: " + message)
                return

            if msgType == 'x':  # compass latitude
                self.coords[0] = res

            elif msgType == 'y':  # compass longitude
                self.coords[1] = res

            elif msgType == 'w':  # wheel speed
                self.wheelSpeed = res

            elif msgType == 'd':  # distance
                self.distanceTraveled = res

            elif msgType == 'a':  # steering angle
                self.steeringDirection = self.heading[1] + res

            elif msgType == 'l':
                self.targetCoords[0] = res

            elif msgType == 't':
                self.targetCoords[1] = res


    def getHeadings(self):
        (x, y, z) = self.hmc5883l.getAxes()
        self.heading = (x, y, z)
        time.sleep(1)

    def endSensors(self):
        self.compassThread.join()
        self.SteerSerialThread.join()
        self.MoveSerialThread.join()

        self.moveESP.close()
        self.steerESP.close()

    def move(self, heading, speed):
        self.steerESP.write(bytes(str("p" + str(heading)), 'utf-8'))
        time.sleep(0.1)

        self.moveESP.write(bytes(str("f" + str(speed)), 'utf-8'))
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
        dist = self.findDistBetween(coords1, coords2)
        if dist < 5:
            return True
        else:
            return False

    def findDistBetween(self, coords1, coords2):
        # 1 deg lat = 364,000 feet
        # 1 deg long = 288,200 feet
        return math.sqrt(
            math.pow((coords1[0] - coords2[0]) * 364000, 2) + math.pow((coords1[1] - coords2[1]) * 288200, 2))

    def findSpeed(self, coords1, coords2):
        dist = self.findDistBetween(coords1, coords2)
        if dist < 0:
            return -4
        elif dist < 10:
            return 4
        else:
            return 9

    def navigate(self, destinations):

        for self.targetCoords in destinations:
            while True:
                if not self.atDestination(self.coords, self.targetCoords):
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
