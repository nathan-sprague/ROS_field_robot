import time

import math

from threading import Thread

import signal # for graceful exits


# For determining wheter it is running on a laptop or not
import platform
import os

# import python files in folder
from esp_controller import Esp
import nav_functions
import destinations as exampleDests


# change to True to navigate through a row endlessly. Useful for testing
interRowNavigation = False

navDestinations = exampleDests.abeNorth # destinations the robot will go to

notCtrlC = True # tag used to have graceful shutdowns. When this is set to false, would-be inifnite loops stop


microProcessor = False # tag for whether it is a microprocessor

def checkDeviceType():
	"""
	Checks if the device is linux (Raspberry Pi or Jetson Nano)
	This can affect what external devices are used. 
	If it is run on a laptop, it wont give any errors if things like the GPS don't connect
	
	Parameters: none
	Returns: bool (true = linux, false = other)
	"""

	piRunning = False
	if platform.system() == "Linux":
		print("run on microprocessor")
		return True
	else:
		print("run on laptop")
		return False


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

		###############################
		# robot attributes
		###############################
		self.atDestinationTolerance = 3 # at destination when x feet from target
		self.topSpeed = 4 # mph

		#################################
		# true position-related variables
		#################################
		self.coords = [-1, -1] # lat, long
		self.gpsAccuracy = 10000 # feet
		self.heading = 0 # degrees
		self.wheelSpeed = [0, 0] # mph
		self.distanceTraveled = 0 # feet

		###############################
		# position-target variables
		###############################
		self.targetWheelSpeed = [0, 0] # mph
		self.targetHeading = 0 # deg (north = 0)

		###############################################
		# logging variables and set up recording thread
		###############################################
		filename = "logs/log_" + str(self.startTime) + ".txt"
		self.recordThread = Thread(target=self.logData, args = [filename])
		self.recordThread.start()

		################################
		# set up the esp's
		###############################
		self.espList = [] # list of objects
		i=0
		while i<4:

			esp = Esp(self, i)
			if esp.begin():
				self.espList += [esp]
				i+=1
			else:
				break 

		print("set up", i, "ESPs")


	def closeRobot(self):
		"""
		Closes all threads in robot.
		Shuts down everything gracefully

		parameters: none
		returns: none
		"""

		print("shutting down robot")
		self.targetWheelSpeed = [0, 0]

		for esp in self.espList:
			esp.endEsp()

		navigationThread.join()

		# close recording thread
		self.recordThread.join()



	def endSensors(self):
		"""
		Stops all sensor threads

		parameters: None
		returns: None
		"""

		# stop the recording thread
		self.recordThread.join()

		# stop all the esp threads
		for esp in self.espList:
			esp.endEsp()

		navigationThread.join()



	def logData(self, filename):
		"""
		Logs the variables related to the robot's status into a text file. 
		This is helpful for viewing what the robot thought was going on during a past run.

		Parameters: filename (text file to log to)
		Returns: None
		"""

		print("recording to:", filename)

		while notCtrlC:
			time.sleep(0.5)
		

			""" 
			log the important variables:
					1. time since start
					2. heading
					3. target heading
					4. left wheel speed
					5. right wheel speed
					6. target left wheel speed
					7. target right wheel speed
					8. current coordinates[0]
					9. current coordinates[1]
					10. target coordinates[0]
					11. target coordinates[1]
			"""

			importantVars = [int(time.time()) - self.startTime,
							self.heading,
							int(self.targetHeading),
							self.wheelSpeed[0],
							self.wheelSpeed[1],
							self.targetWheelSpeed[0],
							self.targetWheelSpeed[1],
							self.coords[-1][0],
							self.coords[-1][1],
							self.destinations[0]["coord"][0],
							self.destinations[0]["coord"][1],
							]

			# converts the variables to a string and logs them
		msg = ""
		for i in importantVars:
				msg += str(i) + ","
				with open(self.filename, 'a+') as fileHandle:
					fileHandle.write(str(msg) + "\n")
					fileHandle.close()

	def navigate(self, destinations):
		"""
		Calls functions to navigate based on the destination and navigation type.

		Parameters:
		destinations: list of dictonaries, with the dictionary having destination attributes
																	i.e location & navigation type
		Returns:
		none
		""" 

		while notCtrlC:
			dest = destinations[0]
			navType = dest["destType"]

			if navType == "point":
				# gps-based point-to-point navigation
				delayTime = self.navigateToPoint(dest)

			elif navType == "row":
				# find the best direction to move wheels
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

				if len(destinations) > 0:
					destinations = destinations[1::] # remove the old destination from the destination list
					
					# if the new destination type is through a row, start the video navigation thread
					dest = destinations[0]
					if dest["destType"] == "row":
						cam = video_navigation.RSCamera(useCamera=True, saveVideo=True, 
							filename= "object_detections/rs_" + str(self.startTime) + ".bag")

						videoThread = Thread(target=cam.videoNavigate, args=(["flag", "row"], False))
						videoThread.start()

				else:
					print("reached all destinations")
					self.closeRobot()
					exit()



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



	def navigateToPoint(self, target):
		""" 
		Manages any errors and sets the target speed and heading

		parameters: target - dictionary with coordinates and heading
		returns: time to wait before calling this function again
		"""


		# manage any errors the sensors give
		self.manageErrors()

		if self.gpsAccuracy == 1000:
			# gps never connected
			print("waiting for GPS to connect. If this continues, try restarting the GPS ESP")

			return 2

		elif self.gpsAccuracy > 10:
		# gps accuracy not good enough to navigate

			print("waiting for gps accuracy to improve (accuracy: " + str(self.gpsAccuracy) + " m)")
			return 0.7

		else:
			# gps accuracy is acceptable. Start navigating!

			# get first destination
			targetCoords = target["coord"]

			# find how far away the robot is from the target (ft)
			distToTarget = nav_functions.findDistBetween(self.coords, targetCoords)


			# reached the destination
			if distToTarget < atDestinationTolerance:
				print("reached destination")
				return -1


			# find target heading
			self.targetHeading = math.degrees(nav_functions.findAngleBetween(self.coords, targetCoords))

			# check if there is a heading it should end at
			target["finalHeading"] = False
			if "finalHeading" in target:
				finalHeading = target["finalHeading"]


			# set ideal wheel speed given the heading and the distance from target
			targetSpeedPercent = nav_functions.findDiffWheelSpeeds(distToTarget, self.heading, finalHeading, 10, atDestinationTolerance)

			self.targetWheelSpeed = [targetSpeedPercent[0]*self.topSpeed/100, targetSpeedPercent[1]*self.topSpeed/100]



			# Print status
			print("heading:", self.heading, "target heading:", self.targetHeadingAngle)
			print("current coords:", self.coords, "target coords:", targetCoords, "(accuracy:", self.gpsAccuracy,")")
			print("target wheel speeds:", self.targetWheelSpeed)
			print("real wheel speeds:", self.wheelSpeed) 
			print("distance from target:", distToTarget)


			# give appropriate time to wait
			return 0.3





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
			return -1

		# manage any error codes
		self.manageErrors()

		if abs(cam.heading)<100:
			travelDirection = -cam.heading-1 # find direction it should be facing (degrees)
		else:	
			travelDirection = 0


		if abs(cam.heading) < 10: # facing approximately right way
			targetSpeed = 1

		elif cam.heading == 1000: # does not know where it is facing
			targetSpeed = 0.1

		else: # facing wrong way but knows where it is facing
			targetSpeed = 1

		# speed range is how different the faster wheel and slower wheel can go. 
		# when speed range is target speed, the slower wheel goes 0 mph
		# when speed range is 2x target speed, the slower wheel goes backwards at the same speed as the faster wheel
		speedRange = abs(targetSpeed)


		# note the /60 part of the equation is a hard-coded value. Change as seen fit
		slowerWheelSpeed = targetSpeed - (abs(travelDirection)/60 *  speedRange)

		if slowerWheelSpeed > targetSpeed:
			print("something is wrong! the slower wheel speed is faster. Fix this code!!!!")
			self.closeRobot()
			exit()

		if travelDirection > 0: # should turn right
			self.wheelSpeed = [targetSpeed, slowerWheelSpeed]

		elif travelDirection < 0: # should turn left
			self.wheelSpeed = [slowerWheelSpeed, targetSpeed]


		# return a 0.3 second delay time
		return 0.3





def signal_handler(sig, frame):

	#   stopStream()
	print('You pressed Ctrl+C!')

	myRobot.notCtrlC = False
	myRobot.closeRobot()
	time.sleep(0.5)
	exit()


signal.signal(signal.SIGINT, signal_handler)






if __name__ == "__main__":
	microProcessor = checkDeviceType()

	myRobot = Robot()
	robotThread = Thread(target=myRobot.navigate(navDestinations))
	robotThread.start()









