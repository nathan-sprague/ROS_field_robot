
from flask import Flask, render_template, request, jsonify, url_for, redirect, Response, send_from_directory
import os
import logging
import time
import signal

myRobot = []


# Uncomment below to suppress the clutter in the python console whenever a request is made
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


app = Flask(__name__)


@app.route('/')
def base():
    return render_template('index_fancy.html')


@app.route("/_info", methods=['GET'])
def update():

    runID = str(myRobot.startTime)
    if myRobot.runMode>0 and myRobot.runMode < 2:
        runID = "sim_" + runID
    elif myRobot.runMode>1:
        runID = "playback_" + runID


    responseDict = {"coords": myRobot.coords, "realSpeed": myRobot.realSpeed,
                    "targetSpeed": myRobot.targetSpeed, "heading": myRobot.trueHeading, "headingAccuracy": myRobot.headingAccuracy,
                    "targetHeading": myRobot.targetHeading%360, "gpsAccuracy": myRobot.gpsAccuracy, "connectionType": myRobot.connectionType,
                     "runID": runID, "runTime": int(time.time()-myRobot.startTime), "status": list(myRobot.navStatus)}




    if request.args.get('destID') != str(myRobot.destID):
        destinationsList = []
        for i in myRobot.destinations:
            destinationsList += [i["coord"]]
        responseDict["destinations"] = destinationsList
        responseDict["destID"] = myRobot.destID


    if request.args.get('obstaclesID') != str(myRobot.obstaclesID):
        responseDict["obstacles"] = myRobot.obstacles
        responseDict["obstaclesID"] = myRobot.obstaclesID


    if True:#request.args.get('obstructionsID') != str(myRobot.obstructionsID):
        responseDict["obstructions"] = myRobot.obstructions
       # responseDict["obstructionsID"] = myRobot.obstructionsID

    if "maxSpeed" in request.args:
        myRobot.topSpeed = float(request.args.get('maxSpeed'))

    if "kill" in request.args:
        myRobot.notCtrlC = False
        myRobot.closeRobot()
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGUSR1)


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
