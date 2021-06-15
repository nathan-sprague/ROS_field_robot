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

targetLocations = [[40.421779, -86.919310], [40.421806, -86.919074], [40.421824, -86.918487], [40.421653, -86.918739],
                   [40.421674, -86.919232]]

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

        self.webControl = False

        self.stopNow = False
        self.coords = [-1.0, -1.0]
        self.headingAngle = 0
        self.heading = 0
        self.targetWheelAngle = 0
        self.wheelSpeed = 0
        self.steeringDirection = 0

        self.steeringAngle = 0

        self.destinations = []
        self.coordListVersion = 0

        self.targetHeadingAngle = 0
        self.targetSpeed = 0

        self.distanceTraveled = 0

        self.websiteInfo = "0h-1x-1y[0]c"

        self.compassThread = Thread(target=self.getHeadings)

        if self.beginCompass():
            self.compassThread.start()
        else:
            print("unable to begin compass")

        self.espMessages = []
        self.espList = []
        self.espRxThreads = []
        self.espTxThreads = []

        if piRunning:

            # /dev/ttyUSB0
            self.espList += [serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.1)]

            self.espList += [serial.Serial(port='/dev/ttyUSB1', baudrate=115200, timeout=.1)]
            print("Set up serial port(s)")

            idNum = 0
            for esp in self.espList:
                rxThread = Thread(target=self.serialLoop, args=[esp, idNum])
                txThread = Thread(target=self.serialLoop, args=[esp, idNum])

                self.espMessages += [
                    {"f": ["0", False], "z": ["0", True], "p": [0, False], "s": ["", True], "g": ["", False],
                     "r": ["", True]}]

                idNum += 1
                self.espRxThreads += [rxThread]
                self.espTxThreads += [txThread]
                rxThread.start()
                txThread.start()
        else:
            self.espMessages = []

    def beginCompass(self):
        try:
            from i2clibraries import i2c_hmc5883l

            self.hmc5883l = i2c_hmc5883l.i2c_hmc5883l(1)

            self.hmc5883l.setContinuousMode()
            self.hmc5883l.setDeclination(-4, 23)
            return True
        except:
            return False

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

    def getHeadings(self):
        while True:
            x, y = self.hmc5883l.getHeading()
            self.heading = ((int(x) - 50) % 360)
            print("heading:", self.heading)
            time.sleep(1)

    def endSensors(self):
        self.compassThread.join()

        for thread in self.espRxThreads:
            thread.join()
        for thread in self.espTxThreads:
            thread.join()

        for esp in self.espList:
            esp.write(bytes(str("f0"), 'utf-8'))
            time.sleep(0.1)
            esp.close()

    def serialLoop(self, serialDevice, idNum):
        # self.espMessages += [["f", 0, False], ["z", 0, True], ["p", 0, False], ["b", "0h-1x-1y[0]c", False],
        #                      ["s", "", True], ["g", "", False], ["r", "", True]]

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
                        time.sleep(0.3)
            if valsUpdated == 0:
                serialDevice.write(bytes("_", 'utf-8'))

            time.sleep(0.5)

    def sendSerial(self, serialDevice, msg):
        serialDevice.write(bytes(msg), 'utf-8')


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

    def atDestination(self, coords1, coords2):
        x, y = self.findDistBetween(coords1, coords2)
        # print("distance away: ", x, y)
        if x * x + y * y < 25:
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

    def findSteerAngle(self, targetHeading, heading):
        steerDif = targetHeading - heading
        if steerDif > 180:
            steerDif = steerDif - 360
        elif steerDif < -180:
            steerDif = steerDif + 360
        if steerDif > 45:
            steerDif = 45
        elif steerDif < -45:
            steerDif = -45
        return steerDif

    def navigate(self, destinations):
        self.destinations = destinations
        print("Started thread")

        while len(self.destinations) > 0:
            while True:


                targetCoords = self.destinations[0]

                if not self.webControl and not self.atDestination(self.coords, targetCoords) and not self.stopNow:
                    self.targetHeadingAngle = math.degrees(self.findAngleBetween(self.coords, targetCoords))
                    self.targetSpeed = self.findSpeed(self.coords, targetCoords)
                    self.targetWheelAngle = self.findSteerAngle(self.targetHeadingAngle, self.heading)





                elif not self.webControl and not self.stopNow:
                    print("reached destination")
                    time.sleep(30)
                    self.destinations = self.destinations[1::]
                    break
                elif self.stopNow:
                    print("stop now")
                    self.targetSpeed = 0
                    self.targetWheelAngle = 0

                    for device in self.espList:
                        self.sendSerial(device, "f0")


                print("heading:", self.heading, "current coords:", self.coords, "target coords:", targetCoords)
                print("target heading:", self.targetHeadingAngle, "target speed", self.targetSpeed)
                print("steer to:", self.targetWheelAngle)

                i = 0
                while i < len(self.espMessages):
                    #print("id: ", i, self.espMessages[i])
                    if "f" in self.espMessages[i]:
                        if str(self.targetSpeed) != self.espMessages[i]["f"][0]:
                            print("changed speed from", self.espMessages[i]["f"][0], "to", str(self.targetSpeed))
                            self.espMessages[i]["f"][0] = str(self.targetSpeed)
                            self.espMessages[i]["f"][1] = False
                            msg = "f"
                          #  print(bytes(msg + self.espMessages[i][msg][0]))
                    if "p" in self.espMessages[i]:
                        if str(self.targetWheelAngle) != self.espMessages[i]["p"][0]:
                            self.espMessages[i]["p"][0] = str(self.targetWheelAngle)
                            self.espMessages[i]["p"][1] = False
                            msg = "p"

                    i += 1
                time.sleep(0.3)

        print("finished getting locations")

        return True


notCtrlC = True


# def signal_handler(sig, frame):
#     print('You pressed Ctrl+C!')
#     notCtrlC = False
#     myRobot.endSensors()
#     shutdownServer()
#     robotThread.join()
#     notCtrlC = False


# signal.pause()

# signal.signal(signal.SIGINT, signal_handler)
# print('Press Ctrl+C')


def beginRobot():
    # pass
    myRobot.navigate(targetLocations)


if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    shutdownServer()

myRobot = Robot()
robotThread = Thread(target=beginRobot)
robotThread.start()

app.run(debug=False, port=8000, host='0.0.0.0')
