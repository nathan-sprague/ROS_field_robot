import time
import math
from threading import Thread
import signal
import platform
import os
 
import robot_esp_control as esp_controller
import robot_website
import nav_functions


piRunning = False
if platform.system() == "Linux":
    print("run on microprocessor")
    piRunning = True
else:
    print("run on laptop")




# # locations at ACRE corn field
# targetLocations = [{"coord": [40.471649, -86.994065], "destType": "point"},
#                     {"coord": [40.471569, -86.994081], "destType": "point"},
#                     {"coord": [40.471433, -86.994069], "destType": "point"},
#                     {"coord": [40.471399, -86.994084], "destType": "point"},
#                     {"coord": [40.471597, -86.994088], "destType": "point"}]

# # locations at ACRE grass
# targetLocations = [{"coord": [40.469552, -86.994882], "destType": "point"},
#                     {"coord": [40.469521, -86.994578], "destType": "point"},
#                     {"coord": [40.469386, -86.994755], "destType": "point"},
#                     {"coord": [40.469506, -86.994384], "destType": "point"},
#                     {"coord": [40.469257, -86.994658], "destType": "point"}]


# locations west of ABE building
# targetLocations = [{"coord": [40.4216702, -86.9184231], "heading": 0, "destType": "point"},
#                     {"coord": [40.4215696, -86.9185767], "heading": 0, "destType": "point"},
#                     {"coord": [40.4215696, -86.9185767], "heading": 0, "destType": "beginRow"},
#                     {"coord": [40.4215696, -86.9185767], "destType": "sample"},
#                     {"coord": [40.4215696, -86.9185767], "destType": "endRow"},
#                     {"coord": [40.4217325, -86.9187132], "heading": 0, "destType": "point"}]

                    
# locations north of ABE building
targetLocations = [{"coord": [40.422266, -86.916176], "destType": "point"},
                    {"coord": [40.422334, -86.916240], "destType": "point"},
                    {"coord": [40.422240, -86.916287], "destType": "point"},
                    {"coord": [40.422194, -86.916221], "destType": "point"},
                    {"coord": [40.422311, -86.916329], "destType": "point"}]


# targetLocations = [[40.422266, -86.916176], [40.422334, -86.916240], [40.422240, -86.916287], [40.422194, -86.916221],
#                    [40.422311, -86.916329]]

class Robot:
    def __init__(self):

        print("making robot")
        
        self.startTime = int(time.time())

        self.notCtrlC = True

        self.turnRadius = 70  # inches
        self.lastUpdatedDist = 0
        self.stopNow = False

        self.errorList = []

        # Robot condition variables
        self.coords = [-1, -1]
        self.gpsAccuracy = 1000
        self.heading = 0
        self.steeringAngle = 0 # angle of wheels relative to heading, not absolute (-45 to 45)
        self.wheelSpeed = 0
        self.distanceTraveled = 0

        # desired robot conditions
        self.targetWheelAngle = 0
        
        self.targetHeadingAngle = 0
        self.targetSpeed = 0
        
        # navigation variables
        self.destinations = [{"coord": [0, 0]}]

        # this changes whenever it reaches a new destination and the coord list updates. This is helpful for giving the most recent coord list in the website
        self.coordListVersion = 0 
        
        self.targetMoveDist = 0
        
        self.subPoints = [] # destinations created to give a path to the real destination to result in desired end heading


        # set up position logger. This is mostly for point-based navigation.
        self.filename = "logs/log_" + str(self.startTime) + ".txt"
        self.recordThread = Thread(target=self.logData)
        self.recordThread.start()


        # set up the ESP8266s
        self.espList = []
        # /dev/ttyUSB0
        i=0
        while i<4:
            
            esp = esp_controller.Esp(self, i)
            if esp.begin():
                self.espList += [esp]
                i+=1
            else:
                break

        print("set up", i, "ESPs")



    

    def endSensors(self):
        self.recordThread.join()

        for esp in self.espList:
            esp.endEsp()


    def logData(self):
        # log all relevant variables as a list every half second

        while self.notCtrlC:

            time.sleep(0.5)
       
            importantVars = [int(time.time()) - self.startTime, self.heading, int(self.targetHeadingAngle),
                             self.steeringAngle, self.targetWheelAngle, self.wheelSpeed,
                             self.destinations[0]["coord"][0], self.destinations[0]["coord"][1], self.coords[0], self.coords[1],
                             ]
            msg = ""
            for i in importantVars:
                msg += str(i) + ","
            with open(self.filename, 'a+') as fileHandle:
                fileHandle.write(str(msg) + "\n")
                fileHandle.close()

 
    def threePointTurn(self, destHeading, maxTravelDist):
        
        # 3+ point turn to desired heading

        sign = -1
        while abs(nav_functions.findShortestAngle(destHeading, self.heading)) > 10:

            self.targetWheelAngle = sign * nav_functions.findSteerAngle(destHeading, self.heading)
            self.targetSpeed = 0
          
            while abs(self.steeringAngle - self.targetWheelAngle) > 5:
                time.sleep(0.1)
            print("done steering")

            self.targetMoveDist = sign * maxTravelDist

            time.sleep(0.4)

            while (self.wheelSpeed < -0.1 or self.wheelSpeed > 0.1) and abs(
                    nav_functions.findShortestAngle(destHeading, self.heading)) > 10:
                time.sleep(0.1)

            print("done moving back")

            sign *= -1
        print("reached destination ---------------")
        self.targetSpeed = 0


    def updateCoords(self):
        # use the heading and distance traveled by the wheel to estimate the position in between GPS readings
      #  return
        if self.lastUpdatedDist != self.distanceTraveled:
            longCorrection = math.cos(self.coords[0] * math.pi / 180)

            distTraveled = self.distanceTraveled - self.lastUpdatedDist

            latTraveled = distTraveled * math.sin(math.radians(self.heading)) / 364000 / 12
            longTraveled = distTraveled * math.cos(math.radians(self.heading)) / longCorrection / 364000 /12
            
            self.coords[0] += latTraveled 
            self.coords[1] += longTraveled 
        
            self.lastUpdatedDist = self.distanceTraveled

    def closeRobot(self):
        print("shutting down robot")
        self.targetWheelAngle = 0
        self.targetSpeed = 0

    
    def manageErrors(self):
        # run through any errors and correct them.

        for i in self.errorList:
            if i[0]==0: # Unreasonable wheel angle
                pass

            if i[0]==1: # motor powered but not moving
                print("stuck")
                for j in self.espList:
                    if j.espType == "speed":
                        j.messagesToSend["g"][1] = True
                stuckPoint = self.distanceTraveled
                self.targetSpeed = -0.5
                self.targetHeadingAngle *= -1
                while self.distanceTraveled + 10 > stuckPoint and self.notCtrlC:
                    print("moving back", self.distanceTraveled)
                    time.sleep(0.1)

                print("done moving back")
                self.targetSpeed = 0
                self.targetHeadingAngle *= -1
                time.sleep(0.1)

            if i[0]==2: # ESP cannot connect to GPS
                print("gps not connected")
                i[1].restart()
                

            if i[0]==3:
                print("gyro not connected. Stil running but heading will be less accurate")
                
        self.errorList = []
    
        return
        

    def pointNavigate(self, destinations):
        self.destinations = destinations
        print("Started thread")

        while len(self.destinations) > 0 and self.notCtrlC:

            self.coordListVersion += 1
            hitDestination = False
            

            while self.notCtrlC:
                self.manageErrors()

                while abs(self.gpsAccuracy)>10 and self.notCtrlC:
                     if self.gpsAccuracy == 1000:
                         print("waiting for GPS to connect. If this continues, try restarting the GPS ESP")
                         time.sleep(2)
                     else:
                         print("waiting for gps accuracy to improve (accuracy: " + str(self.gpsAccuracy) + " m)")
                         time.sleep(0.7)

                
                # get coordinates of destination
                destPoint = self.destinations[0]["coord"]

        

                if "heading" in self.destinations[0] and not hitDestination: # you have a desired end heading and you are not close to your destination
                    
                    # go to the subpoint
                    self.subPoints = nav_functions.makePath(self.coords, destPoint, self.destinations[0]["heading"], self.turnRadius)
        
                    targetCoords = self.subPoints[0]
               

                    if nav_functions.atDestination(self.coords, destPoint, tolerance=self.turnRadius / 12):
                        hitDestination = True
                        targetCoords = destPoint
                        self.subPoints = []
                        print("hit destination using sub points")

                else:
                    targetCoords = destPoint
                    hitDestination = True

                if hitDestination and nav_functions.atDestination(self.coords, targetCoords, self.turnRadius / 12):
        
                    
                    if "heading" in self.destinations[0]:
                        print("reached destination, making corrections for heading")
                        self.threePointTurn(self.destinations[0]["heading"], self.turnRadius)

                    print("actually reached destination")
                    self.targetSpeed = 0
                    self.targetWheelAngle = 0
                    
                    if "destType" in self.destinations[0]:
                        destType = self.destinations[0]["destType"]
                    else:
                        destType = "point"                
        
                    self.destinations.remove(self.destinations[0])
                    if destType == "beginRow":
                        while self.destinations[0]["destType"] == "sample" or self.destinations[0]["destType"] == "endRow":
                            self.interRowNavigate(self.destinations[0]["coord"])
                            self.destinations.remove(self.destinations[0])

                    else:
                        time.sleep(10)
                    
                    
                    break

                if not nav_functions.atDestination(self.coords, targetCoords) and not self.stopNow:
                    self.targetHeadingAngle = math.degrees(nav_functions.findAngleBetween(self.coords, targetCoords))
                    self.targetSpeed = 1
                    self.targetWheelAngle = nav_functions.findSteerAngle(self.targetHeadingAngle, self.heading)

                elif self.stopNow:
                    print("stop now")
                    self.targetSpeed = 0
                    self.targetWheelAngle = 0

#                print("heading:", self.heading, "current coords:", self.coords, "target coords:", targetCoords)
 #               print("target heading:", self.targetHeadingAngle, "target speed", self.targetSpeed)
  #              print("steer to:", self.targetWheelAngle)
                print("heading:", self.heading, "target heading:", self.targetHeadingAngle)
                print("current coords:", self.coords, "target coords:", targetCoords, "(accuracy:", self.gpsAccuracy,")")
                print("wheel angle:", self.steeringAngle, "target angle:",self.targetWheelAngle) 
                print("distance from target: " + str(nav_functions.findDistBetween(self.coords, targetCoords)))
               # self.threePointTurn(-180, 100)

                time.sleep(0.3)
            if not self.notCtrlC:
                return False
       
        print("finished getting locations")

        return True

    def interRowNavigate(self, destination = False):
        import video_navigation

        cam = video_navigation.RSCamera(useCamera=True, saveVideo=False, filename= "object_detections/rs_" + str(self.startTime) + ".bag")
      #  cam = video_navigation.RSCamera(useCamera=False, saveVideo=False, filename="Desktop/fake_track/object_detection6.bag")

        talkTimer = 0
        
        videoThread = Thread(target=cam.runNavigation, args=(False,))
        videoThread.start()
        print("set up row navigation")

        while True:

            if cam.stop or not self.notCtrlC:
                
                cam.stopStream()
                return False
            self.manageErrors()

          #  if destination != False and self.atDestination(self.coords, destination, tolerance=self.turnRadius / 12):
              #  print("inter-row navigated to the destination")
             #   cam.stopStream()
              #  return True

            if abs(cam.heading)<100:           
                self.targetWheelAngle = -cam.heading*3-8
        
            else:
                self.targetWheelAngle = 0

            if abs(cam.heading) < 10: # facing right way
                self.targetSpeed = 1.5
              

            elif cam.heading == 1000: # does not know where it is facing

                self.targetSpeed = 0.1
                
            else: # facing wrong way but knows where it is facing
                self.targetSpeed = 1.5
#            self.targetSpeed = 1

           
            talkTimer += 1
            if talkTimer == 5:
                talkTimer= 0
                print("moving", cam.heading, self.targetWheelAngle, self.targetSpeed)
            time.sleep(0.2)






def beginRobot():
    print("begin")
    #myRobot.pointNavigate(targetLocations)
    myRobot.interRowNavigate()


myRobot = Robot()
robotThread = Thread(target=beginRobot)
robotThread.start()

def signal_handler(sig, frame):
    
 #   stopStream()
    print('You pressed Ctrl+C!')
   
    myRobot.notCtrlC = False
    myRobot.closeRobot()
    time.sleep(0.5)
    exit()


signal.signal(signal.SIGINT, signal_handler)

robot_website.myRobot = myRobot
robot_website.app.run(debug=False, port=8000, host='0.0.0.0')
