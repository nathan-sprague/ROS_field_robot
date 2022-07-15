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

# change to False to prevent the robot from moving automatically. Useful for collecting log data without real movement.
moveRobot = True

navDestinations = exampleDests.acreBayCorn #acreBayCorn # destinations the robot will go to



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


		# vestigial variables used for the website. Remove eventually
		self.destinations = []
		self.subPoints = []

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
		self.coords2 = [-1,-1] # lat, long
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

		# to get the target speed to automatically be sent to the ESP controller whenever the target speed is changed, properties are used.
		self.targetSpeed = [0, 0] # mph

		self.targetHeading = 0 # deg (north = 0)
		self.targetDestination = [0, 0]

		self.targetPath = []
		self.targetPathID = random.randint(0, 1000)
		self.lastTargetPathTime = time.time()

		self.destID = random.randint(0, 1000)


		##############################
		# Optimization variables
		##############################
		self.estimatedEnergyConsumed = 0 # joules
		self.lastCalcPowerTime = time.time()


		self.obstacles = []
		self.obstaclesID = random.randint(0,1000)

		###############################################
		# logging variables and set up recording thread
		###############################################
		self.loggingData = False
		if not testing:
			self.loggingData = True
			self.filename = "log_" + str(self.startTime) + ".txt"
			self.recordThread = Thread(target=self.logData, args = [self.filename])
			self.recordThread.start()

		################################
		# set up the esp's
		###############################
		self.espList = [] # list of ESP objects
		i=0
		self.gpsConnected = False

		if not testing:

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
						break

		elif testing:
			esp = Esp(self, 0)
			esp.begin()
			self.espList +=[esp]

			self.gpsModule = Gps()



	def closeRobot(self):
		"""
		Closes all threads in robot.
		Shuts down everything gracefully

		parameters: none
		returns: none
		"""

		print("shutting down robot")
		self.targetSpeed = [0, 0]

		for esp in self.espList:
			esp.endEsp()
		
		if self.gpsConnected:
			self.gpsModule.endGPS()

		robotThread.join()


		# close recording thread
		if self.loggingData:
			self.recordThread.join()
			if time.time() - self.startTime < 5:
				print("Ran for < 5 seconds. Not worth keeping the log file")
				os.remove(self.filename)
			else:
				print("saved log file as", self.filename)


	def logData(self, filename):
		"""
		Logs the variables related to the robot's status into a text file. 
		This is helpful for viewing what the robot thought was going on during a past run.

		Parameters: filename (text file to log to)
		Returns: None
		"""

		print("recording to:", filename, "\n")


		while abs(self.coords[0])<2 and self.notCtrlC:
			time.sleep(0.3)

		while self.targetSpeed[0] == 0 and self.targetSpeed[1] == 0 and self.notCtrlC:
			time.sleep(0.3)

		if not self.notCtrlC:
			print("never logged anything")
			return

		print("coords are real and the robot is moving, it is worth logging now")

	

		
		recordDestID = -2
		lastMsg = [0]*20

		while self.notCtrlC:
			time.sleep(0.3)
		

			""" 
			log the important variables:
					1. time since start
					2. heading
					3. target heading
					4. real left speed
					5. real right speed
					6. target left speed
					7. target right speed
					8. current coordinates[0]
					9. current coordinates[1]
					10. target coordinates[0]
					11. target coordinates[1]
					12. heading accuracy
					13. connection type
			"""

			if recordDestID != self.destID and len(self.destinations) > 0:
				msg = "d,"
				for d in self.destinations:
					msg += str(d["coord"][0]) + "," + str(d["coord"][1]) + ","

				with open(filename, 'a+') as fileHandle:
					fileHandle.write(str(msg) + "\n")
					fileHandle.close()
				recordDestID = self.destID



			importantVars = [int(time.time()) - self.startTime,
							self.trueHeading,
							int(self.targetHeading),
							self.realSpeed[0],
							self.realSpeed[1],
							self.targetSpeed[0],
							self.targetSpeed[1],
							self.coords[0],
							self.coords[1],
							self.targetDestination[0],
							self.targetDestination[1],
							self.headingAccuracy,
							self.gpsAccuracy,
							self.connectionType
							]


			# converts the variables to a string and logs them
			msg = ""
			j=0
			for i in importantVars:
				dif = i-lastMsg[j]
				if abs(dif)>0.000001:
					msg += str(i) + ","
					lastMsg[j] = i
				else:
					msg += ","
				j+=1

			with open(filename, 'a+') as fileHandle:
				fileHandle.write(str(msg) + "\n")
				fileHandle.close()


	def calculateTrueHeading(self):
		"""
		Used to calculated the heading given all the senors. Currently just using the GPS so ignore this function
		"""
		return


	def navigate(self, destinations):
		"""
		Calls functions to navigate based on the destination and navigation type.

		Parameters:
		destinations: list of dictonaries, with the dictionary having destination attributes
															i.e location & navigation type
		Returns:
		none
		""" 

		

		if "obstacles" in destinations[0]:
			print("there are obstacles")
			self.obstacles = destinations[0]["obstacles"]
			self.obstaclesID = random.randint(0,1000)
			destinations = destinations[1::]


		else:
			print("no obstacles")
			exit()

		# add destinations to a list for the website. Possibly remove later, idk
		self.destinations = destinations


		if interRowNavigation:
			cam = video_navigation.RSCamera(useCamera=True, saveVideo=True, filename= "object_detections/rs_" + str(self.startTime) + ".bag")
		#	cam = video_navigation.RSCamera(useCamera=False, saveVideo=False, filename= "object_detections/rs_1652890000.bag")
			videoThread = Thread(target=cam.videoNavigate, args = [False])
			videoThread.start()
			while self.notCtrlC:
				delayTime = self.interRowNavigate(cam)
        
				if delayTime > 0:
					time.sleep(delayTime)
				else:
					break
			self.notCtrlC = True
			cam.stopStream()
			cam.stop = True
			print("cam stopped")
			videoThread.join()
			return

		while self.notCtrlC and len(self.destinations) > 0:
			dest = destinations[0]
			self.targetDestination = dest["coord"]
			self.atDestinationTolerance = self.defaultAtDestinationTolerance
			if "destTolerance" in dest:
				self.atDestinationTolerance = dest["destTolerance"]
			navType = dest["destType"]
			self.calculateTrueHeading()

			if navType == "point":
				# gps-based point-to-point navigation
				delayTime = self.navigateToPoint(dest)

			elif navType == "row":
				# find the best direction to move wheels
				delayTime = self.interRowNavigate(cam, dest)

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

				if len(destinations) > 0:
					destinations = destinations[1::] # remove the old destination from the destination list
					self.destinations = destinations[:]
					self.destID = random.randint(0, 1000)
					
					# if the new destination type is through a row, start the video navigation thread
					dest = destinations[0]
					if dest["destType"] == "row":
						if testing:
							cam = video_navigation.RSCamera(useCamera=False, saveVideo=False,
							filename= "/home/nathan/bag_files/rs_1655483324.bag", robot=self)
						else:
							cam = video_navigation.RSCamera(useCamera=True, saveVideo=True, 
							filename= "object_detections/rs_" + str(self.startTime) + ".bag", robot=self)

						videoThread = Thread(target=cam.videoNavigate, args=([testing]))
						videoThread.start()

				else:
					print("reached all destinations")
					self.closeRobot()
					exit()



	def calcPowerConsumption(self):

		t = time.time() - self.lastCalcPowerTime

		# 0-pt turn is 30 amp
		# straight is 1 amp

		if abs(self.realSpeed[0]-self.realSpeed[1]) > 2.5: # 0-pt turn
			self.estimatedEnergyConsumed += 1*30 / t
			print("0-pt turn")
		elif self.realSpeed == [0,0]:
			self.estimatedEnergyConsumed += 0.5 / t
		else:
			self.estimatedEnergyConsumed += 1*24 / t
			print("normal movement")


		self.lastCalcPowerTime = time.time()



	def manageErrors(self):
		# run through any errors and correct them.

		for i in self.errorList:
 
			if i[0]==0: # motor powered but not moving
				print("stuck")

			if i[0]==1: # ESP cannot connect to GPS
				print("gps not connected")

			if i[0]==2: 
				print("gyro not connected. Stil running but heading will be less accurate")

		self.errorList = []
		return


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

			return 2

		elif self.headingAccuracy > 90 or self.lastHeadingTime+2 < time.time(): # you can't trust the heading.
			print("Heading accuracy not good enough! Estimated heading:", self.trueHeading, "Heading accuracy:", self.headingAccuracy, "Last updated", int(time.time()-self.lastHeadingTime), "s ago. GPS accuracy", self.gpsAccuracy, "mm")

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


		# manage any errors the sensors give
		self.manageErrors()
#		self.coords = [40.4221268, -86.9161606]
		
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
					time.sleep(3)
					return -1
				if abs(nav_functions.findShortestAngle(finalHeading, self.trueHeading)) < 5: # reached destination with the right heading
					print("reached destination with right heading")
					self.targetSpeed = [0, 0]
					time.sleep(3)
					return -1
				else:
					self.targetHeading = finalHeading

			elif distToTarget[0]**2 + distToTarget[1]**2 < self.atDestinationTolerance ** 2: # reached the destination
				self.reachedDestination = True
				print(" -_-_-_-_-_-_-_- reached destination -_-_-_-_-_-_-_-_-_")
				self.targetSpeed = [0,0]
				
			else:
				self.reachedDestination = False



			# set ideal speed given the heading and the distance from target
			targetSpeedPercent = nav_functions.findDiffSpeeds(self.coords, targetCoords, self.trueHeading, self.targetHeading, finalHeading = finalHeading, turnConstant = 2, destTolerance = self.atDestinationTolerance, obstacles = self.obstacles)
			
			zeroPt = False
			if abs(targetSpeedPercent[1]-targetSpeedPercent[0]) > 100:
				zeroPt = True
			elif abs(targetSpeedPercent[0]) + abs(targetSpeedPercent[1]) > 80:
				if abs(targetSpeedPercent[1]-targetSpeedPercent[0]) > abs(targetSpeedPercent[0]) + abs(targetSpeedPercent[1]) / 5:
					zeroPt = True



			if zeroPt and self.motionType[0] != "0-pt-turn": # started doing a 0-point turn
				self.motionType = ["0-pt-turn", time.time()]
				print("started a 0-pt-turn")

			elif not zeroPt and self.motionType[0] != "straight-nav": # started moving normally
				self.motionType = ["straight-nav", time.time()]
				print("doing regular navigation")

			elif False and self.motionType[0] == "0-pt-turn" and self.motionType[1]+1 < time.time(): # doing a 0-pt turn for 4 seconds. Try waiting for a second to get its heading
				self.motionType = ["waiting", time.time()]
				self.targetSpeed = [0,0]
				print("pausing after partial 0-pt-turn")
				return 1


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
		self.calcPowerConsumption()
		print("runtime:", int(time.time()-self.startTime))





	def interRowNavigate(self, cam, destination = False):
		"""
		Call the navigation time

		Parameters:
			cam - object from video_navigation.py
			destination - location that the robot will travel to. Coordinates (list)
		
		Returns:
			time to delay before calling this function again

		"""

		if cam.stop or not self.notCtrlC:
			# the camera stopped. Something happened in the video navigation program
			# stop streaming and go to the next destination
			cam.stopStream()
			print("stopped camera stream")
			return -1

		# manage any error codes
		self.manageErrors()

		if abs(cam.heading)<100:
			travelDirection = -cam.heading-1 # find direction it should be facing (degrees)
		else:	
			travelDirection = 0

		print("heading", cam.heading)
		if abs(cam.heading) < 10: # facing approximately correct way
			maxSpeed = 1

		elif cam.heading == 1000: # does not know where it is facing
			print("dont know what way its facing")
			maxSpeed = 0.1
			cam.heading = 0

		else: # facing wrong way but knows where it is facing
			maxSpeed = 0.5



		# speed range is how different the faster wheel and slower wheel can go. 
		# when speed range is target speed, the slower wheel goes 0 mph
		# when speed range is 2x target speed, the slower wheel goes backwards at the same speed as the faster wheel
		speedRange = abs(maxSpeed)


		# note the /60 part of the equation is a hard-coded value. Change as seen fit
		slowerWheelSpeed = maxSpeed - (abs(travelDirection)/60 *  speedRange)


		if slowerWheelSpeed > maxSpeed:
			print("something is wrong! the slower wheel speed is faster. Fix this code!!!!")
			self.closeRobot()
			exit()

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
	myRobot.closeRobot()
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




