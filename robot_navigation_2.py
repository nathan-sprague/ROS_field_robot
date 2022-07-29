import time

import math

from threading import Thread

import os

import random

import signal # for graceful exits

# For finding which ports to connect to (GPS, ESP controller, etc)
import serial.tools.list_ports


testing = False
if len(list(serial.tools.list_ports.comports())) == 0:
	testing = True
	simulationFileName = "/home/nathan/new_logs/july28/log_1659037308"
	print("no devices connected, assuming this is a test")

# import python filestrueCoords in folder
if testing:
	from esp_tester import Esp, Gps
else:
	from esp_controller import Esp
	from gps_controller import Gps
	
import nav_functions
import destinations as exampleDests
import robot_website
import video_navigation


# change to True to navigate through a row endlessly. Useful for testing
interRowNavigation = False

# change to True to use during navigation at times other than interrow navigation (keep true to logc)
useCamForNormalNav = True

# change to False to prevent the robot from moving automatically. Useful for controlling the robot and collecting data
moveRobot = True

if testing:
	# change the number to vary what things are played back. 0=none, 1=just camera, 2=full, 3=just position, 4=full but still process
	playbackLevel = 4

navDestinations = exampleDests.acreBayCorn # destinations the robot will go to



class Robot:
	"""
	This class controls the functions of the robot.
	It stores the robot's status variables, along with its commands
	It sends the commands to the external components through other python files
	This allows it to navigate autonomously through both corn rows and gps-based points
	"""

	def __init__(self):
		"""
		Sets up the variables related to the robot's conditions and targets
	
		Parameteres: none
		Returns: none
		"""

		print("creating robot")

		self.startTime =  int(time.time()) # seconds

		self.notCtrlC = True # tag used to have graceful shutdowns. When this is set to false, would-be inifnite loops stop

		self.errorList = []
		self.alerts = "None"
		self.alertsChanged = True

		self.runMode = "real"
		if testing:
			if playbackLevel > 1:
				self.runMode = "playback"
			else:
				self.runMode = "testing"
		self.navStatus = [0] # number representing a message. See html file for the status types


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
		self.headingAccuracyTimer = time.time() # used when doing point-to-point navigation. If the robot doesn't know where it's going, there's no point moving
		self.lastHeadingTime = time.time() # time since a heading was obtained
		self.trueHeading = 0 # degrees

		self.realSpeed = [0, 0] # mph
		self.connectionType = 0 # gps connection type (0=none, 1=dead rekoning, 2=2D fix, 3=3D fix, 4=GNSS+dead rekoning, 5=time fix)

		# the type of movement the robot is making. First element is type of movement second element is time the movement started.
		# motion types are: {waiting, straight-nav, 0-pt-turn, calibration-forward}
		self.motionType = ["waiting", time.time()] 


		###############################
		# position-target variables
		###############################

		self.destinations = []
		
		self.subPoints = [] # will add back eventually

		# to get the target speed to automatically be sent to the ESP controller whenever the target speed is changed, properties are used.
		self.targetSpeed = [0, 0] # mph

		self.targetHeading = 0 # deg (north = 0)
		self.targetDestination = [0, 0]

		self.rowDirection = -1 # angle of the row (degrees)

		self.targetPath = []
		self.targetPathID = random.randint(0, 1000)
		self.lastTargetPathTime = time.time()

		self.destID = random.randint(0, 1000)


		##############################
		# Optimization variables
		##############################
		self.estimatedEnergyConsumed = 0 # joules
		self.lastCalcPowerTime = time.time()


		####################################
		# Website and testing variables
		####################################
		self.obstacles = []
		self.obstaclesID = random.randint(0,1000)

		self.obstructions = []

		self.updateSpeed = -1
		self.defaultUpdateSpeed = 0.1
		if testing:
			# how often the robot updates
			self.defaultUpdateSpeed = 0.1
			self.updateSpeed = self.defaultUpdateSpeed

		################################
		# set up the sensors (ESP, GPS and camera)
		###############################
		self.espList = [] # list of ESP objects
		i=0
		self.gpsConnected = False

		self.camConnected = False

		if not testing:
			self.logDir = "log_" + str(self.startTime)

			mainGpsPort = "none"

			# get a list of all of the ports and port information
			portsConnected = [tuple(p) for p in list(serial.tools.list_ports.comports())]

			
			for i in portsConnected: # connect to all of the relevant ports
				if i[1] == "CP2102 USB to UART Bridge Controller - CP2102 USB to UART Bridge Controller":
					print("esp on port", i[0])
					esp = Esp(self, i[0]) # make a new ESP object and give it this class and the port name
					if esp.begin(moveRobot):
						self.espList += [esp] 
					else:
						print("ESP connection not successful!")
						time.sleep(5)

				elif i[1] == "u-blox GNSS receiver":
					print("gps main on port", i[0])
					mainGpsPort = i[0]
					self.gpsConnected = True

				else:
					print("unknown device port", i)


			print("set up", len(self.espList), "ESPs")


			# initialize the GPS with the port found
			self.gpsModule = Gps(self, mainGpsPort, verbose=False)	

			if  self.gpsModule.begin():
				print("began GPS")
			else:
				print("GPS failed")
				if not interRowNavigation:
					while True:
						time.sleep(1)
						print("set up GPS before continuing")

			self.cam = video_navigation.RSCamera(useCamera=True, saveVideo=True,
						filename=self.logDir, navMethod=0, robot=self)
			
			self.camConnected = True
						

		elif testing:
			if playbackLevel < 2:
				esp = Esp(self, 0)
				esp.begin()
				self.espList +=[esp]

			self.gpsModule = Gps()

			# self.cam = video_navigation.RSCamera(useCamera=False, saveVideo=False,
			# 			filename=simulationFileName, robot=self)
			self.cam = video_navigation.RSCamera(useCamera=False, saveVideo=False,
						filename=simulationFileName, navMethod=0, playbackLevel=playbackLevel, robot=self)
			
			self.camConnected = True



		if self.camConnected:
			self.cam.idle = False
			self.videoThread = Thread(target=self.cam.videoNavigate, args=([testing]))
			self.videoThread.start()





	def closeRobot(self):
		"""
		Closes all threads in robot.
		Shuts down everything gracefully

		parameters: none
		returns: none
		"""

		print("shutting down robot")
		self.navStatus = [16];
		self.targetSpeed = [0, 0]

		for esp in self.espList:
			esp.endEsp()
		
		if self.gpsConnected:
			self.gpsModule.endGPS()

		robotThread.join()




	def navigate(self, destinations):
		"""
		Calls functions to navigate based on the destination and navigation type.

		Parameters:
		destinations: list of dictonaries, with the dictionary having destination attributes
															i.e location & navigation type
		Returns:
		none
		""" 
		if testing:
			if playbackLevel == 2:
				return


		if "obstacles" in destinations[0]:
			print("there are obstacles")
			self.obstacles = destinations[0]["obstacles"]
			self.obstaclesID = random.randint(0,1000)
			destinations = destinations[1::]


		else:
			print("no obstacles")

		# add destinations to a list for the website. Possibly remove later, idk
		self.destinations = destinations


		if interRowNavigation:
			if self.camConnected:
				self.cam.rowNavigate = True
				while self.notCtrlC:
					delayTime = self.interRowNavigate()
					if delayTime > 0:
						time.sleep(delayTime)
					else:
						break
				self.notCtrlC = False
				self.cam.stopStream()
				self.cam.stop = True
				print("cam stopped")
				self.videoThread.join()
			else:
				print("need to connect to a camera to do row navigation")
			return

		while self.notCtrlC and len(self.destinations) > 0:
			dest = destinations[0]
			self.targetDestination = dest["coord"]
			self.atDestinationTolerance = self.defaultAtDestinationTolerance
			if "destTolerance" in dest:
				self.atDestinationTolerance = dest["destTolerance"]
			navType = dest["destType"]

			if navType == "point":
				# gps-based point-to-point navigation
				if self.camConnected:
					self.cam.navMethod = 0
					self.obstructions = self.cam.obstructions
				delayTime = self.navigateToPoint(dest)

			elif navType == "row":
				# find the best direction to move wheels
				self.cam.navMethod = 1
				self.cam.idle = False
				delayTime = self.interRowNavigate(dest)

			elif navType == "sample":
				# just a placeholder for sampling for now. It will eventually do something.
				print("sampling")
				time.sleep(5)
				delayTime = 0

			else:
				print("unknown navigation type:", navType)
				self.closeRobot()
				exit()


			if delayTime > 0:
				# call the function again
				time.sleep(delayTime)

			elif delayTime < 0:
				# reached the destination
				self.reachedDestination = False

				if len(destinations) > 1:
					destinations = destinations[1::] # remove the old destination from the destination list
					self.destinations = destinations[:]
					self.destID = random.randint(0, 1000)

				else:
					print("reached all destinations")
					self.closeRobot()
					time.sleep(0.5)
					os.kill(os.getpid(), signal.SIGUSR1)




	def checkGPS(self):
		"""
		Checks the GPS accuracy and heading to determine whether it is good enough to use for point navigation
		It also prints information on why the GPS is bad if there is an issue

		parameters: 
		none

		returns:
		0 (if GPS accuracy and heading is good)
		waitTime (number) (if GPS is bad)
		"""

		if testing:
			return 0

		connectionTypesLabels = ["Position not known", "Position known without RTK", "DGPS", "UNKNOWN FIX LEVEL 3", "RTK float", "RTK fixed"]


		if self.gpsAccuracy > 100000 or self.connectionType == 0:
			# gps never connected
			if self.gpsConnected == False:
					print("GPS is not connected")
					return 2

			# gps connected but no lock
			print("No GPS fix.", self.gpsModule.debugOptions["SIV"][0], "SIV as of", int(time.time() - self.gpsModule.debugOptions["SIV"][1]), "s ago")

			print(connectionTypesLabels[self.connectionType], "GPS accuracy", self.gpsAccuracy, "mm")
			print("")

			self.motionType = ["waiting", time.time()]
			self.navStatus = [1]

			return 2

		elif self.headingAccuracy > 90 or self.lastHeadingTime+2 < time.time(): # you can't trust the heading.
			print("Heading accuracy not good enough! Estimated heading:", self.trueHeading, "Heading accuracy:", self.headingAccuracy, "Last updated", int(time.time()-self.lastHeadingTime), "s ago. GPS accuracy", self.gpsAccuracy, "mm")

			self.navStatus = [3]
			if self.motionType[0] == "waiting": # robot has been waiting
				
				# go through the debugging possibilities
				

				if self.gpsModule.debugOptions["heading board connected"][0] == False:
					print("The heading board was not connected for", int(time.time()) - self.gpsModule.debugOptions["heading board connected"][0], "s")


			
				if self.gpsModule.debugOptions["RTK signal available"][0]:
					if self.gpsModule.debugOptions["RTK signal used"][0]:
						print(connectionTypesLabels[self.connectionType], "There are RTK signals and it has been used as of", int(time.time()) - self.gpsModule.debugOptions["RTK signal used"][1], "s ago")
					else:
						print(connectionTypesLabels[self.connectionType], "There are RTK signals, but they weren't used for", int(time.time()) - self.gpsModule.debugOptions["RTK signal used"][1], "s")

				else:
					print(connectionTypesLabels[self.connectionType], ", and no RTK signals found")
				print("")


				return 1 # keep on waiting


			else: # robot was navigating normally when the heading stopped working
				print("heading stopped working")
				self.motionType = ["waiting", time.time()]
				self.targetSpeed = [0,0] # stop and try to calibrate by staying put
				return 0.1
		else:
			return 0



	def navigateToPoint(self, target):
		""" 
		Manages any errors and sets the target speed and heading

		parameters: target - dictionary with coordinates and heading
		returns: time to wait before calling this function again
		"""

		# if time.time()-self.lastTargetPathTime > 2:
		# 	self.targetPath = nav_functions.makePath(self.coords, self.trueHeading, target["coord"])
		# 	self.targetPathID = random.randint(0,1000)
		# 	self.lastTargetPathTime = time.time()

#		self.coords = [40.4221268, -86.9161606]

		if self.camConnected:
			if self.cam.aboutToHit:
				print("about to hit")
				self.targetSpeed=[0,0]
				self.navStatus = [18,19]
				return 0.3
		

		gpsQual = self.checkGPS()

		if gpsQual != 0: # The fix of the GPS is not good enough to navigate
			self.targetSpeed = [0, 0]
			return gpsQual

		else:
			# gps and heading accuracy is acceptable. Start navigating!


			# get first destination
			targetCoords = target["coord"]

			# find how far away the robot is from the target (ft)
			distToTarget = nav_functions.findDistBetween(self.coords, targetCoords)


			finalHeading = False
			if "finalHeading" in target:
				
				finalHeading = target["finalHeading"]
				# print("final heading present", finalHeading, "\n\n\n")


			# find target heading
			self.targetHeading = math.degrees(nav_functions.findAngleBetween(self.coords, targetCoords))

			
			if self.reachedDestination:
				if finalHeading == False: # reached destination with no heading requirement
					self.targetSpeed = [0,0]
					print("reached destination, no heading requirement")
					self.navStatus = [8]
					time.sleep(2)
					return -1
				if abs(nav_functions.findShortestAngle(finalHeading, self.trueHeading)) < 5: # reached destination with the right heading
					print("reached destination with right heading")
					self.navStatus = [8]
					self.targetSpeed = [0, 0]
					time.sleep(1)
					if abs(nav_functions.findShortestAngle(finalHeading, self.trueHeading)) < 5:
						print("actually at the right heading")
						time.sleep(2)
						return -1
					else:
						print("stopped being at right heading")
						self.navStats = [9]
						return 0.3
				else:
					print("at destination with incorrect heading")
					self.navStatus = [9]
					self.targetHeading = finalHeading
					self.atDestinationTolerance = 10000

			elif distToTarget[0]**2 + distToTarget[1]**2 < self.atDestinationTolerance ** 2: # reached the destination
				self.reachedDestination = True
				print(" -_-_-_-_-_-_-_- reached destination -_-_-_-_-_-_-_-_-_")
				self.targetSpeed = [0,0]
				time.sleep(1)
				
			else:
				self.reachedDestination = False



			# set ideal speed given the heading and the distance from target
			targetSpeedPercent = nav_functions.findDiffSpeeds(self.coords, targetCoords, self.trueHeading, self.targetHeading, 
				finalHeading = finalHeading, turnConstant = 2, destTolerance = self.atDestinationTolerance, obstacles = self.obstacles)
			
			zeroPt = False
			if abs(targetSpeedPercent[1]-targetSpeedPercent[0]) > 100:
				zeroPt = True
			elif abs(targetSpeedPercent[0]) + abs(targetSpeedPercent[1]) > 80:
				if abs(targetSpeedPercent[1]-targetSpeedPercent[0]) > abs(targetSpeedPercent[0]) + abs(targetSpeedPercent[1]) / 5:
					zeroPt = True



			if zeroPt and self.motionType[0] != "0-pt-turn": # started doing a 0-point turn
				self.motionType = ["0-pt-turn", time.time()]
				print("started a 0-pt-turn")
				self.navStatus = [6]

			elif not zeroPt and self.motionType[0] != "straight-nav": # started moving normally
				self.motionType = ["straight-nav", time.time()]
				print("doing regular navigation")
				self.navStatus = [4]

			elif  self.motionType[0] == "0-pt-turn" and self.motionType[1]+1 < time.time(): # doing a 0-pt turn for 4 seconds. Try waiting for a second to get its heading
				self.motionType = ["waiting", time.time()]
				self.targetSpeed = [0,0]
				print("pausing after partial 0-pt-turn")
				self.navStatus = [7]
				return 0.5
			else:
				self.navStatus = [4]
			print("navStatus", self.navStatus)


			# conver the target speed from the above function to a real speed. target speed percent is -100 to 100. the robot should usually move -2 to 2 mph
			self.targetSpeed = [targetSpeedPercent[0]*self.topSpeed/100, targetSpeedPercent[1]*self.topSpeed/100]

			if zeroPt and self.alerts != "0-PT TURN!!!":
				self.alerts = "0-PT TURN!!!"
				self.alertsChanged = True
			elif not zeroPt and self.alerts != "no zeropt":
				self.alerts = "no zero pt turn"
				self.alertsChanged = True


			# Print status
			self.printStatus(targetCoords, distToTarget)
			


			# give appropriate time to wait
			return 0.3


	def printStatus(self, targetCoords, distToTarget):
		# prints relevant variables for the robot. This should be the only place where stuff is printed
		print("heading:", self.trueHeading, "target heading:", self.targetHeading, "heading accuracy:", self.headingAccuracy)

		# print("current coords:", self.coords, "target coords:", targetCoords, "fix type:", self.connectionType, "(accuracy:", self.gpsAccuracy, ")")
		print("target speeds:", self.targetSpeed,"real speeds:", self.realSpeed, "distance from target:", distToTarget, "\n")
		# print("motion type:", self.motionType)
		print("runtime:", int(time.time()-self.startTime))





	def interRowNavigate(self, destination = False):
		"""
		Call the navigation time

		Parameters:
			cam - object from video_navigation.py
			destination - location that the robot will travel to. Coordinates (list)
		
		Returns:
			time to delay before calling this function again

		"""

		if self.cam.stop or not self.notCtrlC:
			# the camera stopped. Something happened in the video navigation program
			# stop streaming and go to the next destination
			self.cam.stopStream()
			print("stopped camera stream")
			return -1

		# copy the variable to this scope because there is a slight chance it will change as it is being analyzed (threading thing)
		camHeading = self.cam.heading 

		if type(camHeading) == list: # there are multiple rows that the robot can enter
			if self.rowDirection == -1: # no specific row direction, go into the center row
				camHeading = camHeading[int(len(camHeading)/2)]
			else:
				bestHeading = camHeading[0]
				bestHeadingDif = abs(nav_functions.findShortestAngle(self.rowDirection, camHeading[0]+self.trueHeading))
				
				ind = 0
				bestInd = 0
				print("true heading", self.trueHeading)
				for i in camHeading:

					headingDif = abs(nav_functions.findShortestAngle(self.rowDirection, i+self.trueHeading))
					print("heading dif", headingDif, "angle", i)
					if bestHeadingDif > headingDif:
						bestHeadingDif = headingDif
						bestHeading = i
						bestInd = ind
					ind+=1


				print("best index", bestInd, "\n")
				camHeading = bestHeading


		if abs(camHeading)<100:
			travelDirection = -camHeading-1 # find direction it should be facing (degrees)
		else:	
			travelDirection = 0

		if self.cam.aboutToHit:
			if self.motionType[0] != "obstacleStop":
				print("about to hit")
				self.targetSpeed = [0,0]
				print("speed", self.targetSpeed)
				self.motionType = ["obstacleStop", time.time()]
				return 0.3
			else:
				print("backing up - obstacle in sight")
				self.targetSpeed = [-self.topSpeed*0.1, -self.topSpeed*0.1]
				self.motionType = ["reversing", time.time()]
				return 0.3

		elif self.motionType[0] == "reversing" and time.time()-self.motionType[1] < 5: # continue to back up for a bit even when there is no longer an obstacle
			print("backing up - just to be safe")
			self.targetSpeed = [-self.topSpeed*0.1, -self.topSpeed*0.1]
			return 0.3
		else:
			self.motionType = ["normal", 0]


		print("heading", camHeading)
		if abs(camHeading) < 10: # facing approximately correct way
			maxSpeed = self.topSpeed*0.5
			self.navStatus = [10, 14]

		elif camHeading == 1000 or abs(camHeading) > 44: # does not know where it is facing
			print("dont know what way its facing")
			maxSpeed = self.topSpeed*0.1
			h = 0
			self.navStatus = [10, 13]

		else: # facing wrong way but knows where it is facing
			maxSpeed = self.topSpeed*0.2
			self.navStatus = [10, 15]


		# speed range is how different the faster wheel and slower wheel can go. 
		# when speed range is target speed, the slower wheel goes 0 mph
		# when speed range is 2x target speed, the slower wheel goes backwards at the same speed as the faster wheel
		speedRange = abs(maxSpeed)


		# note the /60 part of the equation is a hard-coded value. Change as seen fit
		if self.cam.distFromCorn > 1200:
			print("outside row")

			rowDirAngle = nav_functions.findShortestAngle(self.rowDirection, self.trueHeading)
			print("row angle", rowDirAngle)
			if abs(rowDirAngle) > 10 and self.rowDirection !=-1:
				if abs(rowDirAngle) > 20:
					rowDirAngle = 20*abs(rowDirAngle)/rowDirAngle

				travelDirection = rowDirAngle
				self.navStatus += [17]

			slowerWheelSpeed = maxSpeed - (abs(travelDirection)/40 *  speedRange)

			self.navStatus += [11]

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



def beginRobot():
	# this function begins the navigation function. 
	# For some reason you need to call a function outside of the Robot when threading for it to work properly.
	myRobot.navigate(navDestinations)




def signal_handler(sig, frame):

	#   stopStream()
	print('You pressed Ctrl+C!')

	myRobot.notCtrlC = False
	time.sleep(0.5)
	exit()



if __name__ == "__main__":
	signal.signal(signal.SIGINT, signal_handler)

	myRobot = Robot()


	robotThread = Thread(target=beginRobot)

	print("made robot")

	robotThread.start()

	print("starting website")
	robot_website.myRobot = myRobot
	robot_website.app.run(debug=False, port=8000, host='0.0.0.0')

	print("done starting website")




