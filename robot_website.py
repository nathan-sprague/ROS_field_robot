
from flask import Flask, render_template, request, jsonify, url_for, redirect, Response, send_from_directory
import os
import logging

myRobot = []


# Uncomment below to suppress the clutter in the python console whenever a request is made
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)


app = Flask(__name__)


@app.route('/')
def base():
    return render_template('index.html')


@app.route("/_info", methods=['GET'])
def update():
    # print("my coords", myRobot.coords)
    # print("target coords", myRobot.destinations)
    myRobot.steeringAngle = 0
    myRobot.targetSpeed = 0
    myRobot.targetWheelAngle = 0
    myRobot.targetHeadingAngle = 0
    responseDict = {"coords": myRobot.coords, "wheelSpeed": myRobot.wheelSpeed,
                    "targetSpeed": myRobot.targetSpeed, "realAngle": myRobot.steeringAngle,
                    "targetAngle": myRobot.targetWheelAngle, "heading": myRobot.heading,
                    "targetHeading": myRobot.targetHeadingAngle}
 #   if request.args.get('pointsListVersion') is not None:
#        print("P list version", int(request.args.get('pointsListVersion')))
    if request.args.get('pointsListVersion') is not None and int(
            request.args.get('pointsListVersion')) < myRobot.coordListVersion:
#        print("new points list version")
        responseDict["pointsList"] = myRobot.destinations
        for i in myRobot.subPoints:
            responseDict["pointsList"] += [{"coord": i, "destType": "subPoint"}]
        responseDict["pointsListVersion"] = myRobot.coordListVersion

    if request.args.get('s') == "1":
        myRobot.stopNow = True
        print("stop now")
    else:
        myRobot.stopNow = False
  #  print(responseDict)
    return (str(responseDict)).replace("'", '"')


def shutdownServer():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    print("server shut down")



if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    shutdownServer()

