import time
import math
from threading import Thread
from flask import Flask, render_template, request, jsonify, url_for, redirect, Response, send_from_directory
import random
import os

targetLocations = [[40.421779, -86.919310], [40.421806, -86.919074], [40.421824, -86.918487], [40.421653, -86.918739],
                   [40.421674, -86.919232]]

app = Flask(__name__)


@app.route('/')
def base():
    return render_template('index.html')


@app.route("/_info", methods=['GET'])
def update():
    print("my coords", myRobot.coords)
    print("target coords", myRobot.destinations)
    responseDict = {"coords": myRobot.coords, "wheelSpeed": myRobot.wheelSpeed,
                    "targetSpeed": myRobot.targetSpeed, "realAngle": myRobot.steeringAngle,
                    "targetAngle": myRobot.targetWheelAngle, "heading": myRobot.heading,
                    "targetHeading": myRobot.targetHeadingAngle}
    if request.args.get('coordListVersion') is not None:
        responseDict["coordList"] = myRobot.destinations
        responseDict["coordListVersion"] = myRobot.coordListVersion

    if request.args.get('s') == "1":
        myRobot.stopNow = True
        print("stop now")
    else:
        myRobot.stopNow = False

    return (str(responseDict)).replace("'", '"')


def shutdownServer():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    print("server shut down")


class FakeESP:
    def __init__(self, controllerType, robot):
        self.speed = 0
        self.targetAngle = 0
        self.angle = 0
        self.targetSpeed = 0
        self.coords = [40.421674, -86.919232]
        self.lastDistance = 0
        self.heading = 0
        self.distance = 0
        self.controllerType = controllerType
        self.robot = robot

        self.messageToSendIndex = 0
        if self.controllerType == "speed":
            self.messagesToSend = [0, 0]
            self.messagesToSendPrefix = ["w", "d"]
        if self.controllerType == "steer":
            self.messagesToSend = [0]
            self.messagesToSendPrefix = ["a"]
        if self.controllerType == "gps":
            self.messagesToSend = [0, 0, 0]
            self.messagesToSendPrefix = ["x", "y", "h"]

        self.updateThread = Thread(target=self.update)
        self.updateThread.start()

    def update(self):
        while True:
            sleepTime = 0.1
            if self.controllerType == "speed":
                dif = self.targetSpeed-self.speed
                if dif < 0.5 and dif > 0 and self.targetSpeed != 0:
                    dif = 0.5
                if dif > -0.5 and dif < 0 and self.targetSpeed != 0:
                    dif = -0.5
                self.speed += (random.randint(5, 12) / 10) * dif


                self.distance += self.speed * 5280 * 12 / 3600 * sleepTime

                self.messagesToSend[0] = self.speed
                self.messagesToSend[1] = self.distance

            elif self.controllerType == "steer":
                # self.angle = self.targetAngle
                if self.angle < self.targetAngle-2:
                    # self.angle = (self.targetAngle + self.angle)/2
                    self.angle += 10 * sleepTime
                  #  self.angle += random.randint(5, 100) / 10
                elif self.angle > self.targetAngle+2:
                    # self.angle = (self.targetAngle + self.angle) / 2
                    self.angle -= 10 * sleepTime

                self.messagesToSend[0] = self.angle

            elif self.controllerType == "gps":
                dist = self.robot.distanceTraveled
                self.coords[1] += ((dist - self.lastDistance)/12) / 364000 * math.sin(
                    self.heading * math.pi / 180)
                longCorrection = math.cos(self.coords[0] * math.pi / 180)
                self.coords[0] += ((dist - self.lastDistance)/12) / 364000 * math.cos(
                    self.heading * math.pi / 180) * longCorrection



                wheelHeading = self.robot.steeringAngle + self.heading
                self.heading += self.robot.steeringAngle * (dist-self.lastDistance)*180/70/45
                # self.heading = self.robot.targetHeadingAngle
                self.heading = self.heading % 360


                self.messagesToSend[0] = self.coords[0]
                self.messagesToSend[1] = self.coords[1]
                self.messagesToSend[2] = self.heading

                self.lastDistance = dist


            time.sleep(sleepTime)

    def sendData(self):
        if self.controllerType == "speed":
            self.messageToSendIndex = (self.messageToSendIndex+1) % 2


        elif self.controllerType == "steer":
            self.messageToSendIndex = 0

        elif self.controllerType == "gps":
            self.messageToSendIndex = (self.messageToSendIndex + 1) % 3

        return self.messagesToSendPrefix[self.messageToSendIndex] + str(self.messagesToSend[self.messageToSendIndex])

    def setData(self, msg):

        if msg[0] == "f":
            self.targetSpeed = float(msg[1::])
        if msg[0] == "p":
            self.targetAngle = float(msg[1::])


class Robot:
    def __init__(self):
        print("making robot")

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

        self.steeringAngle = 0

        self.destinations = []
        self.coordListVersion = 0

        self.targetHeadingAngle = 0
        self.targetSpeed = 0

        self.distanceTraveled = 0


        # self.compassThread = Thread(target=self.getHeadings)

        self.startTime = int(time.time())

        #
        # if self.beginCompass():
        #     self.compassThread.start()
        # else:
        #     print("unable to begin compass")

        self.espMessages = []

        self.espList = [FakeESP("speed", self), FakeESP("steer", self), FakeESP("gps", self)]
        self.espRxThreads = []
        self.espTxThreads = []

        idNum = 0
        for esp in self.espList:
            rxThread = Thread(target=self.readSerial, args=[esp, idNum])
            txThread = Thread(target=self.sendSerial, args=[esp, idNum])

            self.espMessages += [
                {"f": ["0", False], "z": ["0", True], "p": [0, False], "s": ["", True], "g": ["", False],
                 "r": ["", True]}]

            idNum += 1
            self.espRxThreads += [rxThread]
            self.espTxThreads += [txThread]
            rxThread.start()
            txThread.start()

    def readSerial(self, serialDevice, idNum):
        while True:
            msg = serialDevice.sendData()
            self.processSerial(msg, idNum)
            time.sleep(0.1)

    def processSerial(self, message_bin, idNum):
        message = str(message_bin)
        if len(message_bin) > 0:



            msgType = message[0]
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
                self.heading = res# - 90

            elif msgType == "g":
                self.gyroHeading = res

            elif msgType == "c":
                self.compassHeading = res

    def sendSerial(self, serialDevice, idNum):
        while True:
            serialDevice.setData("f" + str(self.targetSpeed))
            time.sleep(0.1)
            serialDevice.setData("p" + str(self.targetWheelAngle))
            time.sleep(0.1)

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
        return 0.9

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

    def setEspMsg(self):
        i = 0
        while i < len(self.espMessages):
            # print("id: ", i, self.espMessages[i])
            if "f" in self.espMessages[i]:
                if str(self.targetSpeed) != self.espMessages[i]["f"][0]:
                    print("changed speed from", self.espMessages[i]["f"][0], "to", str(self.targetSpeed))
                    self.espMessages[i]["f"][0] = str(self.targetSpeed)
                    self.espMessages[i]["f"][1] = False

            if "p" in self.espMessages[i]:
                if str(self.targetWheelAngle) != self.espMessages[i]["p"][0]:
                    self.espMessages[i]["p"][0] = str(self.targetWheelAngle)
                    self.espMessages[i]["p"][1] = False

            i += 1

    def navigate(self, destinations):
        self.destinations = destinations
        print("Started thread")

        while len(self.destinations) > 0:
            while True:

                targetCoords = self.destinations[0]

                if self.atDestination(self.coords, targetCoords):
                    print("reached destination")
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
                x, y = self.findDistBetween(self.coords, targetCoords)
                print("dist to")

                self.setEspMsg()

                if self.atDestination(self.coords, targetCoords):
                    print("reached destination")



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
