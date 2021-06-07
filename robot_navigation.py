import serial
import time
import math
from threading import Thread

testLocations = [[1, 1], [2, 3], [3, 4]]


class Robot:
    def __init__(self):

        self.coords = [0, 0]
        self.headingAngle = 0
        self.heading = (0,0,0)
        self.wheelSpeed = 0
        self.steeringDirection = 0

        self.steeringAngle = 0

        self.targetCoords = [0, 0]
        self.targetHeadingAngle = 0
        self.targetSpeed = 0

        self.distanceTraveled = 0

        self.compassThread = Thread(target=self.getHeadings)

        if self.beginCompass():
            self.compassThread.start()
        else:
            print("unable to begin compass")

        self.espList = []
        self.espRxThreads = []
        self.espTxThreads = []
        # /dev/ttyUSB0
        self.espList += [serial.Serial(port='/dev/cu.SLAB_USBtoUART', baudrate=115200, timeout=.1)]

        self.espList += [serial.Serial(port='/dev/cu.SLAB_USBtoUART7', baudrate=115200, timeout=.1)]

        print("Set up serial port(s)")

        for esp in self.espList:
            rxThread = Thread(target=self.readSerial, args=[esp])
            txThread = Thread(target=self.sendSerial, args=[esp])
            self.espRxThreads += [rxThread]
            self.espTxThreads += [txThread]
            rxThread.start()
            txThread.start()

    def beginCompass(self):
        try:
            from i2clibraries import i2c_hmc5883l

            self.hmc5883l = i2c_hmc5883l.i2c_hmc5883l(1)

            self.hmc5883l.setContinuousMode()
            return True
        except:
            return False

    def readSerial(self, serialDevice):
        while True:
            line = serialDevice.readline()

            self.processSerial(line)
            time.sleep(0.1)

    def processSerial(self, message_bin):
        message = str(message_bin)
        if len(message_bin) > 0:

            try:
                message = message[2:-5]
                msgType = message[0]
            except:
                print("odd message format")
                return

            if msgType == ".":
                print(message[1::])
                return

            try:
                res = float(message[1::])
            except:
                print("invalid message: " + message)
                return

            if msgType == "x":  # compass latitude
                self.coords[0] = res

            elif msgType == "y":  # compass longitude
                self.coords[1] = res

            elif msgType == "w":  # wheel speed
                self.wheelSpeed = res

            elif msgType == "d":  # distance
                self.distanceTraveled = res

            elif msgType == "a":  # steering angle
                self.steeringAngle = res
                self.steeringDirection = self.heading[1] + res

            elif msgType == "l":
                self.targetCoords[0] = res

            elif msgType == "t":
                self.targetCoords[1] = res

    def getHeadings(self):
        (x, y, z) = self.hmc5883l.getAxes()
        self.heading = (x, y, z)
        time.sleep(1)

    def endSensors(self):
        self.compassThread.join()

        for thread in self.espReadThreads:
            thread.join()

        for esp in self.espList:
            esp.close()

    def sendSerial(self, serialDevice):
        while True:
            serialDevice.write(bytes(str("p" + str(self.targetHeadingAngle)), 'utf-8'))
            time.sleep(0.1)
            serialDevice.write(bytes(str("f" + str(self.targetSpeed)), 'utf-8'))
            time.sleep(0.1)
            serialDevice.write(bytes(str("h" + str(self.heading[1])), 'utf-8'))  # heading
            time.sleep(0.1)
            serialDevice.write(bytes(str("x" + str(self.coords[0])), 'utf-8'))  # lat position
            time.sleep(0.1)
            serialDevice.write(bytes(str("y" + str(self.coords[1])), 'utf-8'))  # long position
            time.sleep(0.1)
            serialDevice.write(bytes(str("u" + str(self.targetCoords[0])), 'utf-8'))  # target lat
            time.sleep(0.1)
            serialDevice.write(bytes(str("v" + str(self.targetCoords[1])), 'utf-8'))  # heading long
            time.sleep(0.1)

            time.sleep(0.3)

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
                    self.targetHeadingAngle = math.degrees(self.findAngleBetween(self.coords, self.targetCoords))
                    self.targetSpeed = self.findSpeed(self.coords, self.targetCoords)
                    print("target heading:", self.targetHeadingAngle, "target speed", self.targetSpeed)

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
