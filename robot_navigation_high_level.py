import time

import math

from threading import Thread

import os

import random

import signal # for graceful exits

# For finding which ports to connect to (GPS, ESP controller, etc)
import serial.tools.list_ports


_runMode = 2 # 0=Real, 1=Simulation wnithout video, 2=simulation with just video, 3=simulation with video and position, 4=full playback but still process, 5=full playback
if len(list(serial.tools.list_ports.comports())) == 1:
	print("no devices connected, assuming this is a simulation or playback")
else:
	runMode = 0

if _runMode != 0:
	from esp_tester import Esp, Gps
	simulationFileName = "/home/nathan/new_logs/july29/afternoon/log_1659126759"

else:
	from esp_controller import Esp
	from gps_controller import Gps
	
import nav_functions
import destinations as exampleDests
import robot_website
import video_navigation


_navDestinations = exampleDests.acreBayCorn # destinations the robot will go to


class Robot:
	def __init__(self, destinations, runMode=0):
		print("creating robot")

		self.startTime =  int(time.time()) # seconds

		self.notCtrlC = True # tag used to have graceful shutdowns. When this is set to false, would-be inifnite loops stop

		self.runMode = runMode
		
		###############################
		# robot attributes
		###############################
		self.defaultAtDestinationTolerance = 1.5
		self.atDestinationTolerance = self.defaultAtDestinationTolerance # at destination when x feet from target
		self.topSpeed = 2 # mph
		self.reachedDestination = False

		#################################
		# true position-related variables
		#################################
		self.coords = [-1,-1] # lat, long
		self.gpsAccuracy = 12345 # mm

		self.headingAccuracy = 360 # degrees
		self.lastHeadingTime = time.time() # time since a heading was obtained
		self.trueHeading = 0 # degrees

		self.realSpeed = [0, 0] # mph
		self.connectionType = 0 # gps connection type (0=none, 1=dead rekoning, 2=2D fix, 3=3D fix, 4=GNSS+dead rekoning, 5=time fix)

		###############################
		# position-target variables
		###############################
		self.destinations = destinations
		self.destID = random.randint(0, 1000)

		self.targetSpeed = [0, 0] # mph


		####################################
		# Website and testing variables
		####################################
		self.obstacles = [] # includes obstacles detected by camera
		self.mapObstacles = [] # obstacles just given from destinations dictionary
		self.obstaclesID = random.randint(0,1000)
		self.obstructions = []

		self.navStatus = set()

		self.navQueue = []


		################################
		# sensor-related variables
		################################
		if self.runMode==0: # not testing
			portsConnected = [tuple(p) for p in list(serial.tools.list_ports.comports())]

			for i in portsConnected: # connect to all of the relevant ports
				if i[1] == "CP2102 USB to UART Bridge Controller - CP2102 USB to UART Bridge Controller":
					
					self.espModule = Esp(self, i[0]) # make a new ESP object and give it this class and the port name
					if self.espModule.begin(True):
						print("Connected to ESP on port", i[0])
					else:
						print("Connection for ESP on port", i[0], "not successful!")
						exit()

				elif i[1] == "u-blox GNSS receiver": # initialize the GPS with the port found
					self.gpsModule = Gps(self, mainGpsPort, verbose=False)	
					if self.gpsModule.begin():
						print("Connected to GPS on port", i[0])
					else:
						print("Connection for GPS on port", i[0], "not successful!")
						exit()
				else:
					print("unknown device port", i[0])

			self.cam = video_navigation.RSCamera(self, useCamera=False, saveVideo=False,
						filename=simulationFileName, navMethod=0, playbackLevel=runMode)
			self.cam.begin()
			self.videoThread = Thread(target=self.cam.videoNavigate, args=([False]))
			self.videoThread.start()


		else: # simulation or playback
			self.espModule = Esp(self, 0)
			self.gpsModule = Gps()
			self.cam = video_navigation.RSCamera(self, useCamera=False, saveVideo=False,
						filename=simulationFileName, navMethod=0, playbackLevel=runMode)
			print("made cam")

			if self.runMode < 3: # simulation
				self.espModule.begin()

			if self.runMode > 1 or self.runMode==0:
				self.cam.begin()
			self.videoThread = Thread(target=self.cam.videoNavigate, args=([True]))
			self.videoThread.start()



	def closeRobot(self):
		"""
		Closes all threads in robot.
		Shuts down everything gracefully

		parameters: none
		returns: none
		"""

		print("shutting down robot")
		self.navStatus = {16};
		self.targetSpeed = [0, 0]

		for esp in self.espList:
			esp.endEsp()
		
		if self.gpsConnected:
			self.gpsModule.endGPS()

		time.sleep(0.5)
		os.kill(os.getpid(), signal.SIGUSR1)



	def navigate(self):
		if self.runMode==5: # full playback doesn't use this
			return

		# get any obstacles from the destination list
		if "obstacles" in self.destinations[0]:
			print("there are obstacles")
			self.obstacles = self.destinations[0]["obstacles"]
			self.obstaclesID = random.randint(0,1000)
			self.destinations = self.destinations[1::]


		while self.notCtrlC and len(self.destinations) > 0:
			targetDestination = self.destinations[0]
			navType = targetDestination["destType"]

			if len(self.navQueue) != 0:
				print("processing nav queue")
				

			if navType == "point":
				self.cam.navMethod = 0 # switch to basic mode navigation
				delayTime = self.pointNavigate(targetDestination)

			elif navType == "row":
				if self.cam.navMethod == 0:
					self.cam.navMethod = 1 # switch to enter row navigation mode
				delayTime = self.rowNavigate()

			else:
				print("unknown navigation method")
				delayTime = -1

			if delayTime > 0: # normal
				time.sleep(delayTime)

			elif delayTime == 0: # skip to process the command right away
				pass
			else: # reached destination

				if len(self.destinations) > 1:

					self.destinations = self.destinations[1::] # remove the old destination from the destination list
					self.destID = random.randint(0, 1000)
					time.sleep(2)
				else:
					print("Reached all of the destinations")
					self.closeRobot()


	def checkGPS(self):
		if self.connectionType == 0:
			print("No GPS fix")
		elif self.headingAccuracy > 90 or time.time() - self.lastHeadingTimetime.time() > 2:
			print("heading accuracy not good")

			if self.gpsModule.debugOptions["heading board connected"][0] == False:
				print("no heading board connected")
			elif self.gpsModule.debugOptions["RTK signal available"][0]:
				if self.gpsModule.debugOptions["RTK signal used"][0]:
					print("There are RTK signals that are being used")
				else:
					print("There are RTK signals but they aren't being used")
			else:
				print("No RTK singals found")

		# gps good
		return 0



	def pointNavigate(self, targetDestination):

		# add camera obstacles to the obstacle list

		# check gps status
		if self.runMode == 0:
			gpsQuality = self.checkGPS()

			if gpsQuality!=0: # gps is not suitable for navigation
				return {"type": "wait", "time": gpsQuality, "until": "time"}
		moveSpeed = self.topSpeed

		# if there is an obstacle in the way, find shortest path around it
		self.obstacles = self.mapObstacles[:]
		# print(self.cam.obstructions)
		for i in self.cam.obstructions:
			print(i[0], i[1])
			coords1 = nav_functions.polarToCoords(self.coords, (self.trueHeading+i[0])%360, i[4]/1000)
			coords2 = nav_functions.polarToCoords(self.coords, (self.trueHeading+i[2])%360, i[5]/1000)
			self.obstacles += [[coords1, [(coords1[0]+coords2[0])/2, (coords1[1]+coords2[1])/2], coords2]]
		
		if self.cam.aboutToHit:
			return 1

		
		self.obstaclesID = random.randint(0,100)
		# for i in cam.obstructions:
		# 	print(i)
		# check if it reached the destination
		distToTarget = nav_functions.findDistBetween(self.coords, targetDestination["coord"])
		distToTargetMagnitude = math.sqrt(distToTarget[0]**2 + distToTarget[1]**2)

		if distToTargetMagnitude < 5:
			moveSpeed *= 0.5

		self.targetHeading = math.degrees(nav_functions.findAngleBetween(self.coords, targetDestination["coord"]))

		finalHeading = False
		if distToTargetMagnitude < self.atDestinationTolerance: # reached the destination
			print("_-_-_-_-_-_- At Destination -_-_-_-_-_-_-_-\n\n")
			if "finalHeading" in targetDestination:
				headingDiff = nav_functions.findShortestAngle(targetDestination["finalHeading"], self.trueHeading)

				if abs(headingDiff) < 5:
					print("arrived at final heading")
					self.targetSpeed = [0, 0]
					finalHeading = targetDestination["finalHeading"]
					
					return -1

				else:
					self.targetHeading = finalHeading
					print("Must turn to final heading")

			else:
				print("no final heading required")
				self.targetSpeed = [0, 0]
				self.navQueue = ["atDestination"]
				return -1



		angleToTarget = nav_functions.findShortestAngle(self.targetHeading, self.trueHeading)
		
		# just barely passed the target. Try reversing
		if distToTargetMagnitude < 3 and abs(angleToTarget) > 140:
			targetSpeed = [-distToTargetMagnitude, -distToTargetMagnitude, 1]
		
		# do normal movement
		# set ideal speed given the heading and the distance from target
		targetSpeedPercent = nav_functions.findDiffSpeeds(self.coords, targetDestination["coord"], self.trueHeading, self.targetHeading, 
				finalHeading = finalHeading, turnConstant = 2, destTolerance = self.atDestinationTolerance, obstacles = self.obstacles)

		self.targetSpeed = [targetSpeedPercent[0]*self.topSpeed/100, targetSpeedPercent[1]*self.topSpeed/100]

		return 0.15




	def rowNavigate(self):
		
		camHeading = self.cam.heading

		if abs(camHeading)<100:
			travelDirection = -camHeading-1 # find direction it should be facing (degrees)
		else:
			travelDirection = 0


		if self.cam.aboutToHit:
			print("about to hit")
			if self.cam.insideRow:
				print("inside row, moving slowly")
				maxSpeed = self.topSpeed*0.1
			else:
				print("outside row. back up to avoid hitting")
				targetSpeed = [-2, -1.5, 1]
				return 0.2

		print("heading", camHeading)
		if abs(camHeading) < 10: # facing approximately correct way
			maxSpeed = self.topSpeed*0.5
			self.navStatus.add(10)
			self.navStatus.add(14)

		elif camHeading == 0 or abs(camHeading) > 44: # does not know where it is facing
			print("dont know what way its facing")
			maxSpeed = self.topSpeed*0.1
			h = 0

		else: # facing wrong way but knows where it is facing
			maxSpeed = self.topSpeed*0.2

		speedRange = abs(maxSpeed)


		if self.cam.insideRow:
			rowDirAngle = nav_functions.findShortestAngle(self.rowDirection, self.trueHeading)
			if abs(rowDirAngle) > 10 and self.rowDirection !=-1:
				if abs(rowDirAngle) > 20:
					rowDirAngle = 20*abs(rowDirAngle)/rowDirAngle

				travelDirection = rowDirAngle

			slowerWheelSpeed = maxSpeed - (abs(travelDirection)/40 *  speedRange)

		else:
			slowerWheelSpeed = maxSpeed - (abs(travelDirection)/60 *  speedRange)


		if slowerWheelSpeed > maxSpeed:
			print("something is wrong! the slower wheel speed is faster. Fix this code!!!!")
			self.closeRobot()
			exit()

		self.targetHeading = self.trueHeading + travelDirection

		if travelDirection > 0: # should turn right
			self.targetSpeed = [maxSpeed, slowerWheelSpeed]

		elif travelDirection < 0: # should turn left
			self.targetSpeed = [slowerWheelSpeed, maxSpeed]
		else:
			self.targetSpeed = [maxSpeed, maxSpeed]

		print("speed:", self.targetSpeed)

		# return a 0.3 second delay time
		return 0.3




			# coords = nav_functions.
		# check if about to hit
			# if already in the row, wait a second and move forward slowly.

			# if trying to enter the row:
				# wait a second

				# reverse

				# do a 0-pt turn to face the row entrance

		# pick best heading from camera

		# set speed according to heading
			















def beginRobot():
	# this function begins the navigation function. 
	# For some reason you need to call a function outside of the Robot when threading for it to work properly.
	myRobot.navigate()




def signal_handler(sig, frame):

	#   stopStream()
	print('You pressed Ctrl+C!')

	myRobot.notCtrlC = False
	time.sleep(0.5)
	exit()



if __name__ == "__main__":
	signal.signal(signal.SIGINT, signal_handler)

	myRobot = Robot(_navDestinations, runMode=_runMode)

	robotThread = Thread(target=beginRobot)

	print("made robot")

	robotThread.start()

	print("starting website")
	robot_website.myRobot = myRobot
	robot_website.app.run(debug=False, port=8000, host='0.0.0.0')

	print("done starting website")