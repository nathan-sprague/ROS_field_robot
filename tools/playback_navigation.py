import serial
import time
import math
from threading import Thread
import signal
import sys
from flask import Flask, render_template, request, jsonify, url_for, redirect, Response, send_from_directory
import os

targetLocations = [[40.422266, -86.916176], [40.422334, -86.916240], [40.422240, -86.916287], [40.422194, -86.916221],
                   [40.422311, -86.916329]]
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


class Robot:
    def __init__(self):
        print("making robot")
        self.destinations = [[0, 0]]
        self.filename = "analyze.txt"

        self.webControl = False

        self.stopNow = False
        self.coords = [-1.0, -1.0]
        self.headingAngle = 0
        self.heading = 0
        self.compassHeading = 0
        self.targetWheelAngle = 0
        self.wheelSpeed = 0
        self.steeringDirection = 0
        self.gyroHeading = 0

        self.steeringAngle = 0

        self.destinations = []
        self.coordListVersion = 0

        self.targetHeadingAngle = 0
        self.targetSpeed = 0

        self.distanceTraveled = 0
        self.arrived = False
        self.websiteInfo = "0h-1x-1y[0]c"

        # self.compassThread = Thread(target=self.getHeadings)

        self.recordThread = Thread(target=self.readData)
        self.recordThread.start()

        #
        # if self.beginCompass():
        #     self.compassThread.start()
        # else:
        #     print("unable to begin compass")

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
        return 1

    def logData(self):
        while True:
            time.sleep(1)
            print(self.destinations)
            importantVars = [self.heading, int(self.targetHeadingAngle), self.targetWheelAngle, self.wheelSpeed,
                             self.destinations[0][0], self.destinations[0][1], self.coords[0], self.coords[1],
                             self.compassHeading, self.gyroHeading]
            msg = ""
            for i in importantVars:
                msg += str(i) + ","
            with open(self.filename, 'a+') as fileHandle:
                fileHandle.write(str(msg) + "\n")
                fileHandle.close()

    def readData(self):

        with open(self.filename, 'r') as fileHandle:
            x = fileHandle.read()
            i = 0
            while i < len(x):

                strNum = 0

                while x[i] != "\n" and i < len(x):

                    val = ""
                    while x[i] != ",":
                        val += x[i]
                        i += 1

                    if len(val) > 0:
                        print(val)
                        if strNum == 0:
                            #  print("heading", val)
                            self.heading = float(val)
                        elif strNum == 1:
                            # print("target heading", val)
                            self.targetHeadingAngle = float(val)
                        elif strNum == 2:
                            self.steeringAngle = float(val)
                        elif strNum == 3:
                            #   print("target wheel", val)
                            self.targetWheelAngle = float(val)
                        elif strNum == 4:
                            #   print("wheel speed", val)
                            self.wheelSpeed = float(val)
                        elif strNum == 5:
                            #   print("dest 1", val)
                            self.destinations[0][0] = float(val)
                        elif strNum == 6:
                            #   print("dest 2", val)
                            self.destinations[0][1] = float(val)
                        elif strNum == 7:
                            #  print("coords 1", val)
                            self.coords[0] = float(val)
                        elif strNum == 8:
                            #   print("coords 2", val)
                            self.coords[1] = float(val)
                        elif strNum == 9:
                            #     print("compass", val)
                            self.compassHeading = float(val)
                            # self.heading = float(val)
                        elif strNum == 10:
                            #    print("gyro", val)
                            self.gyroHeading = float(val)
                    i += 1
                    strNum += 1
                while self.stopNow == True:
                    time.sleep(0.5)
                    print("stopping")
                print("newline")
                if self.destinations[0] == self.destinations[1]:
                    self.destinations = self.destinations[1::]
                while self.arrived:
                    time.sleep(0.2)
                time.sleep(0.2)
                i += 1

            print("done")
            print(x)
            fileHandle.close()

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
        while True:
            if self.atDestination(self.coords, self.destinations[0]):
                self.arrived = True
                print("arrived")
                time.sleep(1)
                self.arrived = False
                time.sleep(3)


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
