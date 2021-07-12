import serial
import time
import math
from threading import Thread
import signal
import sys
from flask import Flask, render_template, request, jsonify, url_for, redirect, Response, send_from_directory

piRunning = False
try:
    import i2clibraries

    print("run on raspberry pi")
    piRunning = True
except:
    print("run on laptop")
import os

targetLocations = [[40.4216702, -86.9184231], [40.4215696, -86.9185767], [40.4217325, -86.9187132],
                   [40.4217325, -86.9187132], [40.4217325, -86.9187132]]

# targetLocations = [[40.422266, -86.916176], [40.422334, -86.916240], [40.422240, -86.916287], [40.422194, -86.916221],
#                    [40.422311, -86.916329]]

app = Flask(__name__)


@app.route('/')
def base():
    return render_template('index.html')


@app.route("/_info", methods=['GET'])
def update():
    override = request.args.get('override')

    if override is not None:

        myRobot.stopNow = False

        if override == "0":
            myRobot.webControl = False
        else:
            myRobot.webControl = True
            myRobot.desiredAngle = request.args.get('angle')
            myRobot.desiredSpeed = request.args.get('speed')

    if request.args.get('s') == "1":
        myRobot.stopNow = True
        print("stop now")

    responseDict = {"coords": myRobot.coords, "wheelSpeed": myRobot.wheelSpeed,
                    "targetSpeed": myRobot.targetSpeed, "realAngle": myRobot.steeringAngle,
                    "targetAngle": myRobot.targetWheelAngle, "heading": myRobot.heading,
                    "targetHeading": myRobot.targetHeadingAngle}

    if request.args.get('targetPositions') is not None:
        destinations = list(map(float, request.args.get('targetPositions').split(',')))
        formattedDestinations = []
        i = 0
        while i < len(destinations) - 1:
            formattedDestinations += [[destinations[i], destinations[i + 1]]]
            i += 2
        myRobot.destinations = formattedDestinations
        print(myRobot.destinations)
        myRobot.coordListVersion += 1
        responseDict["coordList"] = myRobot.destinations
        responseDict["coordListVersion"] = myRobot.coordListVersion
        print("updating coord list")

    if request.args.get('coordListVersion') is not None and int(
            request.args.get('coordListVersion')) < myRobot.coordListVersion:
        responseDict["coordList"] = myRobot.destinations
        responseDict["coordListVersion"] = myRobot.coordListVersion
    print(responseDict)

    return (str(responseDict)).replace("'", '"')


def shutdownServer():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    print("server shut down")


class Robot:
    def __init__(self):

        print("making robot")

        self.turnRadius = 70  # inches
        self.stopNow = False
        self.coords = [-1.0, -1.0]
        self.headingAngle = 0
        self.heading = 0
        self.compassHeading = 0
        self.targetWheelAngle = 0
        self.wheelSpeed = 0
        self.steeringDirection = 0
        self.gyroHeading = 0
        self.webControl = False
        self.filename = "logs" + str(int(time.time())) + ".txt"

        self.destinations = [[0, 0]]

        self.distFromFront = -1000
        self.distFromCenter = 0

        self.steeringAngle = 0

        #  self.destinations = []
        self.coordListVersion = 0

        self.targetHeadingAngle = 0
        self.targetSpeed = 0

        self.distanceTraveled = 0
        self.targetMoveDist = 0
        self.subPoints = []

        # self.compassThread = Thread(target=self.getHeadings)

        self.recordThread = Thread(target=self.logData)
        self.recordThread.start()
        self.startTime = int(time.time())

        #
        # if self.beginCompass():
        #     self.compassThread.start()
        # else:
        #     print("unable to begin compass")

        self.feelerAngle = 0

        self.espMessages = []

        self.espList = []
        self.espRxThreads = []
        self.espTxThreads = []

        if piRunning:

            # /dev/ttyUSB0
            self.espList += [serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.1)]

            self.espList += [serial.Serial(port='/dev/ttyUSB1', baudrate=115200, timeout=.1)]

            #    self.espList += [serial.Serial(port='/dev/ttyUSB2', baudrate=115200, timeout=.1)]
            print("Set up serial port(s)")

            idNum = 0
            for esp in self.espList:
                rxThread = Thread(target=self.readSerial, args=[esp, idNum])
                txThread = Thread(target=self.serialLoop, args=[esp, idNum])

                self.espMessages += [
                    {"f": ["0", False], "z": ["0", True], "p": [0, False], "s": ["", True], "g": ["", False],
                     "r": ["", True], "m": ["0", True]}]

                idNum += 1
                self.espRxThreads += [rxThread]
                self.espTxThreads += [txThread]
                rxThread.start()
                txThread.start()

    # def beginCompass(self):
    #     try:
    #         from i2clibraries import i2c_hmc5883l
    #
    #         self.hmc5883l = i2c_hmc5883l.i2c_hmc5883l(1)
    #
    #         self.hmc5883l.setContinuousMode()
    #         self.hmc5883l.setDeclination(-4, 23)
    #         return True
    #     except:
    #         return False

    def readSerial(self, serialDevice, idNum):
        while True:
            line = serialDevice.readline()
            self.processSerial(line, idNum)
            time.sleep(0.1)

    def processSerial(self, message_bin, idNum):
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
            elif msgType == "+":
                print("popping", message[1], "from", self.espMessages[idNum])
                if message[1] in self.espMessages[idNum]:
                    self.espMessages[idNum].pop(message[1])
                return

            elif msgType == "-":
                keyName = message[1]
                print("looking  at message", keyName, "from", idNum)
                if self.espMessages[idNum][keyName][0] == (message[2::]):
                    print("got confirmation for " + keyName + ", same as requested")
                    self.espMessages[idNum][keyName][1] = True
                else:
                    print("got confirmation for ", keyName, ", different from requested (got: ",
                          message[2::], " sent:", self.espMessages[idNum][keyName][0]), ")"
                    self.espMessages[idNum][keyName][1] = False
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
                self.steeringDirection = self.heading + res

            elif msgType == "h":  # steering angle
                self.heading = res - 90

            elif msgType == "g":
                self.gyroHeading = res

            elif msgType == "c":
                self.compassHeading = res

            elif msgType == "l":
                self.distFromFront = res

            elif msgType == "k":
                self.distFromCenter = res
            elif msgType == "o":
                self.feelerAngle = res

    def endSensors(self):
        # self.compassThread.join()
        self.recordThread.join()

        for thread in self.espRxThreads:
            thread.join()
        for thread in self.espTxThreads:
            thread.join()

        for esp in self.espList:
            self.sendSerial(esp, "f0")
            time.sleep(0.1)
            esp.close()

    def serialLoop(self, serialDevice, idNum):

        while True:

            valsUpdated = 0
            espMsgKeys = []
            for i in self.espMessages[idNum].keys():
                espMsgKeys += [i]
            i = 0
            while i < len(espMsgKeys):
                msg = espMsgKeys[i]
                i += 1
                if msg in self.espMessages[idNum]:
                    if not self.espMessages[idNum][msg][1]:
                        print(msg, self.espMessages[idNum][msg][0])
                        valsUpdated += 1
                        self.sendSerial(serialDevice, msg + str(self.espMessages[idNum][msg][0]))

                        print("sent", msg, "to", idNum)
                        time.sleep(0.1)
            if valsUpdated == 0:
                self.sendSerial(serialDevice, "_")
                time.sleep(0.1)
                # serialDevice.write(bytes("_", 'utf-8'))

            time.sleep(0.1)

    def sendSerial(self, serialDevice, msg):
        serialDevice.write(bytes(msg, 'utf-8'))

    def findAngleBetween(self, coords1, coords2):
        x, y = self.findDistBetween(coords1, coords2)
        if x == 0:
            if coords1[1] > coords2[1]:  # directly above
                return math.pi
            else:  # directly below
                return 0
        slope = y / x
        angle = math.atan(slope)

        if coords1[1] < coords2[1]:
            realAngle = angle + math.pi * 1.5
        else:  # to the left
            realAngle = angle + math.pi * 0.5

        realAngle = 2 * math.pi - realAngle
        if realAngle > math.pi:
            realAngle = realAngle - 2 * math.pi

        return realAngle

    def atDestination(self, coords1, coords2, tolerance=5.0):

        x, y = self.findDistBetween(coords1, coords2)
        # print("distance away: ", x, y)
        if x * x + y * y < tolerance * tolerance:
            return True
        else:
            return False

    def findDistBetween(self, coords1, coords2):
        # 1 deg lat = 364,000 feet
        # 1 deg long = 288,200 feet
        x = (coords1[1] - coords2[1]) * 364000

        longCorrection = math.cos(coords1[0] * math.pi / 180)
        y = (coords1[0] - coords2[0]) * longCorrection * 364000

        return x, y

    def findSpeed(self, coords1, coords2):
        # x, y = self.findDistBetween(coords1, coords2)
        # if dist < 0:
        #     return -1
        # elif dist < 10:
        #     return 1
        # else:
        #     return 1
        return 1

    def logData(self):

        while True:

            time.sleep(0.5)
            #    print(self.destinations)
            importantVars = [int(time.time()) - self.startTime, self.heading, int(self.targetHeadingAngle),
                             self.steeringAngle, self.targetWheelAngle, self.wheelSpeed,
                             self.destinations[0][0], self.destinations[0][1], self.coords[0], self.coords[1],
                             self.compassHeading, self.gyroHeading]
            msg = ""
            for i in importantVars:
                msg += str(i) + ","
            with open(self.filename, 'a+') as fileHandle:
                fileHandle.write(str(msg) + "\n")
                fileHandle.close()

    def findShortestAngle(self, targetHeading, heading):
        steerDif = targetHeading - heading
        if steerDif > 180:
            steerDif = steerDif - 360
        elif steerDif < -180:
            steerDif = steerDif + 360
        return steerDif

    def findSteerAngle(self, targetHeading, heading):
        steerDif = self.findShortestAngle(targetHeading, heading)
        if steerDif > 45:
            steerDif = 45
        elif steerDif < -45:
            steerDif = -45
        return steerDif

    def setEspMsg(self):
        i = 0
        while i < len(self.espMessages):
            # print("id: ", i, self.espMessages[i])
            if "f" in self.espMessages[i]:
                if str(self.targetSpeed) != self.espMessages[i]["f"][0]:
                    #    print("changed speed from", self.espMessages[i]["f"][0], "to", str(self.targetSpeed))
                    self.espMessages[i]["f"][0] = str(self.targetSpeed)
                    self.espMessages[i]["f"][1] = False

            if "p" in self.espMessages[i]:
                if str(self.targetWheelAngle) != self.espMessages[i]["p"][0]:
                    self.espMessages[i]["p"][0] = str(self.targetWheelAngle)
                    self.espMessages[i]["p"][1] = False

            if "m" in self.espMessages[i]:
                #      print("see m")
                if str(self.targetMoveDist) != self.espMessages[i]["m"][0]:
                    print("changing m")
                    self.espMessages[i]["m"][0] = str(self.targetMoveDist)
                    self.espMessages[i]["m"][1] = False

            i += 1

    def threePointTurn(self, destHeading, maxTravelDist):
        sign = -1
        while abs(self.findShortestAngle(destHeading, self.heading)) > 10:

            self.targetWheelAngle = sign * self.findSteerAngle(destHeading, self.heading)
            self.targetSpeed = 0
            self.setEspMsg()
            while abs(self.steeringAngle - self.targetWheelAngle) > 5:
                time.sleep(0.1)
            print("done steering")

            self.targetMoveDist = sign * maxTravelDist
            self.setEspMsg()
            time.sleep(0.4)

            while (self.wheelSpeed < -0.1 or self.wheelSpeed > 0.1) and abs(
                    self.findShortestAngle(destHeading, self.heading)) > 10:
                time.sleep(0.1)

            print("done moving back")

            sign *= -1
        print("reached destination ---------------")
        self.targetSpeed = 0.1
        self.setEspMsg()
        self.targetSpeed = 0
        self.setEspMsg()

    def makePath(self, currentCoords, destination, destHeading):

        longCorrection = math.cos(currentCoords[0] * math.pi / 180)

        offsetY = self.turnRadius / 12 / 364000 * math.sin((90 + destHeading) * math.pi / 180)
        offsetX = self.turnRadius / 12 / 364000 * math.cos((90 + destHeading) * math.pi / 180) * longCorrection
        approachCircleCenter1 = [destination[0] + offsetX, destination[1] + offsetY]

        offsetY = self.turnRadius / 12 / 364000 * math.sin((-90 + destHeading) * math.pi / 180)
        offsetX = self.turnRadius / 12 / 364000 * math.cos((-90 + destHeading) * math.pi / 180) * longCorrection
        approachCircleCenter2 = [destination[0] + offsetX, destination[1] + offsetY]

        x1, y1 = self.findDistBetween(currentCoords, approachCircleCenter1)
        x2, y2 = self.findDistBetween(currentCoords, approachCircleCenter2)

        dist1 = x1 * x1 + y1 * y1
        dist2 = x2 * x2 + y2 * y2
        print(dist1, dist2)

        if dist1 < dist2:
            print("clockwise approach")
            clockwise = True
            # self.destinations += [approachCircleCenter1]
            closerApproach = approachCircleCenter1
        else:
            print("Counter clockwise approach")
            clockwise = False
            # self.destinations += [approachCircleCenter2]
            closerApproach = approachCircleCenter2

        print(currentCoords, closerApproach)

        a = self.findAngleBetween(currentCoords, closerApproach) * 180 / math.pi
        print("angle", a)

        if clockwise:
            offsetY = self.turnRadius / 12 / 364000 * math.sin((a - 90) * math.pi / 180)
            offsetX = self.turnRadius / 12 / 364000 * math.cos((a - 90) * math.pi / 180) * longCorrection
            approachPoint1 = [closerApproach[0] + offsetX, closerApproach[1] + offsetY]
            # self.destinations += [approachPoint1]
            b = self.findAngleBetween(currentCoords, approachPoint1) * 180 / math.pi
            self.subPoints = [approachPoint1]
        else:
            offsetY = self.turnRadius / 12 / 364000 * math.sin((a + 90) * math.pi / 180)
            offsetX = self.turnRadius / 12 / 364000 * math.cos((a + 90) * math.pi / 180) * longCorrection
            approachPoint2 = [closerApproach[0] + offsetX, closerApproach[1] + offsetY]
            # self.destinations += [approachPoint2]
            self.subPoints = [approachPoint2]
            c = self.findAngleBetween(currentCoords, approachPoint2) * 180 / math.pi

        # bb = self.findShortestAngle(a, b)
        # cc = self.findShortestAngle(a, c)
        # print(bb, cc)

        # self.currentCoords

    def navigate(self, destinations):
        self.destinations = destinations
        print("Started thread")

        while len(self.destinations) > 0:

            self.coordListVersion += 1
            hitDestination = False
            while True:

                if not hitDestination:
                    self.makePath(self.coords, self.destinations[0], 0)
                    targetCoords = self.subPoints[0]
                else:
                    targetCoords = self.destinations[0]

                if self.atDestination(self.coords, self.destinations[0],
                                      tolerance=self.turnRadius / 12) and not hitDestination:
                    hitDestination = True
                    self.subPoints = []
                    print("hit destination")

                if hitDestination and self.atDestination(self.coords, targetCoords, tolerance=2):
                    if hitDestination:
                        print("reached destination")
                        self.threePointTurn(0, self.turnRadius)
                        self.targetSpeed = 0
                        self.targetWheelAngle = 0
                        self.setEspMsg()

                        self.destinations.remove(self.destinations[0])
                        time.sleep(4)
                        break

                if not self.webControl and not self.atDestination(self.coords, targetCoords) and not self.stopNow:
                    self.targetHeadingAngle = math.degrees(self.findAngleBetween(self.coords, targetCoords))
                    self.targetSpeed = self.findSpeed(self.coords, targetCoords)
                    self.targetWheelAngle = self.findSteerAngle(self.targetHeadingAngle, self.heading)

                elif self.stopNow:
                    print("stop now")
                    self.targetSpeed = 0
                    self.targetWheelAngle = 0

                print("heading:", self.heading, "current coords:", self.coords, "target coords:", targetCoords)
                print("target heading:", self.targetHeadingAngle, "target speed", self.targetSpeed)
                print("steer to:", self.targetWheelAngle)

                self.setEspMsg()

                # self.threePointTurn(-180, 100)

                time.sleep(0.3)

        print("finished getting locations")

        return True

    def interRowNavigate(self):

        while True:


            self.setEspMsg()
            time.sleep(0.1)


notCtrlC = True


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    global notCtrlC
    notCtrlC = False
    myRobot.endSensors()
    shutdownServer()
    robotThread.join()


# signal.pause()

signal.signal(signal.SIGINT, signal_handler)
print('Press Ctrl+C')


def beginRobot():
    print("begin")
    # pass
    # myRobot.navigate(targetLocations)
    myRobot.interRowNavigate()


if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    shutdownServer()

myRobot = Robot()
robotThread = Thread(target=beginRobot)
robotThread.start()

app.run(debug=False, port=8000, host='0.0.0.0')
