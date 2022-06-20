
from flask import Flask, render_template, request, jsonify, url_for, redirect, Response, send_from_directory
import os
import logging

myRobot = []


# Uncomment below to suppress the clutter in the python console whenever a request is made
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


app = Flask(__name__)


@app.route('/')
def base():
    return render_template('index.html')


@app.route("/_info", methods=['GET'])
def update():


    responseDict = {"coords": myRobot.coords, "realSpeed": myRobot.realSpeed,
                    "targetSpeed": myRobot.targetSpeed, "heading": myRobot.trueHeading,
                    "targetHeading": myRobot.targetHeading, "gyroHeading": myRobot.gyroHeading, "gpsHeading": myRobot.gpsHeading, "alerts": myRobot.alerts}


    if request.args.get('haveDestinations') == "0":
        destinationsList = []
        for i in myRobot.destinations:
            destinationsList += [i["coord"]]
        responseDict["destinations"] = destinationsList

  #  print("sent:", str(responseDict).replace("'", '"'))
    return (str(responseDict)).replace("'", '"')


def shutdownServer():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    print("server shut down")



if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    shutdownServer()
