import time

import math

from threading import Thread

import signal # for graceful exits


# For determining wheter it is running on a laptop or not
import platform
import os


testing = False

# import python files in folder
if testing:
	from esp_tester import Esp
else:
	from esp_controller import Esp
	from gps_controller import Gps
import nav_functions
import destinations as exampleDests
import robot_website
import video_navigation



# change to True to navigate through a row endlessly. Useful for testing
interRowNavigation = True

navDestinations = exampleDests.abeNorth # destinations the robot will go to



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

		self.notCtrlC = True # tag used to have graceful shutdowns. When this is set to false, would-be inifnite loops stop

		self.errorList = []
		self.alerts = "none"


		# vestigial variables used for the website. Remove eventually
		self.coordListVersion = 0 
		self.destinations = []
		self.subPoints = []

		###############################
		# robot attributes
		###############################
		self.atDestinationTolerance = 5 # at destination when x feet from target
		self.topSpeed = 2 # mph

		#################################
		# true position-related variables
		#################################
		self.coords = [-1,-1] # lat, long
		self.coords2 = [-1,-1] # lat, long
		self.gpsAccuracy = 12345 # meters

		self.gpsHeadingAvailable = False
		self.gpsHeading = 0 # degrees	
		self.gyroHeading = 0 # degrees
		self.lastConfirmedHeading = 0 # degrees
		self.gyroAtConfirmedHeading = 0 # degrees
		self.trueHeading = 0 # degrees

		self.useGPSHeading = True
		self.realSpeed = [0, 0] # mph
		self.connectionType = 0

		###############################
		# position-target variables
		###############################
		self.targetSpeed = [0, 0] # mph
		self.targetHeading = 0 # deg (north = 0)
		self.targetDestination = [0, 0]

		###############################################
		# logging variables and set up recording thread
		###############################################
		filename = "log_" + str(self.startTime) + ".txt"
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
		if not testing:
			self.gpsModule = Gps(self)	

			print("set up", i, "ESPs")

			if  self.gpsModule.begin():
				print("began GPS")
			else:
				print("GPS failed")
				while True:
					time.sleep(1)
					print("set up GPS before continuing")


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
		
		self.gpsModule.endGPS()

		robotThread.join()

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

		robotThread.join()



	def logData(self, filename):
		"""
		Logs the variables related to the robot's status into a text file. 
		This is helpful for viewing what the robot thought was going on during a past run.

		Parameters: filename (text file to log to)
		Returns: None
		"""

		print("recording to:", filename)

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
			"""

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
							self.gpsHeading,
							self.gyroHeading
							]

			# converts the variables to a string and logs them
			msg = ""
			for i in importantVars:
					msg += str(i) + ","
			with open(filename, 'a+') as fileHandle:
				fileHandle.write(str(msg) + "\n")
				fileHandle.close()


	def calculateTrueHeading(self):
		"""
		Calculates the real heading based on GPS heading, gyro heading, and current speed
		"""
		# if you are moving and not making a sharp turn, just use the GPS heading
		self.trueHeading = self.gyroHeading
		return
		if self.gpsHeadingAvailable:
			if (self.realSpeed[0] > 1.5 and self.realSpeed[1] > 1.5 and abs(self.realSpeed[0]-self.realSpeed[1]) < 0.2): # you are moving straight forward
					print("heading confirmed")
					self.trueHeading = self.gpsHeading
					self.lastConfirmedHeading = self.trueHeading
					self.gyroAtConfirmedHeading = self.gyroHeading
					self.alerts = "heading confirmed"
					return
			
			else:
				pass
			#	print("not going forward enough")
			
		else:
			pass
			#print("heading wasnt available")
		self.alerts = "heading not confirmed"
	
		# at least one of the above conditions were not met. Use the gyro with the last known heading
		headingChange = self.gyroHeading - self.gyroAtConfirmedHeading
		self.trueHeading = self.lastConfirmedHeading + headingChange
	#	print("unconfimed heading:")
	#	print("gyro was:", self.gyroAtConfirmedHeading, "gyro is now:", self.gyroHeading)
	#	print("correction:", headingChange, "last confirmed:", self.lastConfirmedHeading, "estimated heading", self.trueHeading)


	def navigate(self, destinations):
		"""
		Calls functions to navigate based on the destination and navigation type.

		Parameters:
		destinations: list of dictonaries, with the dictionary having destination attributes
															i.e location & navigation type
		Returns:
		none
		""" 

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
			videoThread.join()
			return

		while self.notCtrlC:
			dest = destinations[0]
			self.targetDestination = dest["coord"]
			navType = dest["destType"]
			self.calculateTrueHeading()

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
					self.destinations = destinations[:]
					
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
#		self.coords = [40.4221268, -86.9161606]
		if self.gpsAccuracy > 90 or self.connectionType == 0:
			# gps never connected
			print("waiting for GPS to lock.")

			return 2

		else:
			# gps accuracy is acceptable. Start navigating!

			# get first destination
			targetCoords = target["coord"]

			# find how far away the robot is from the target (ft)
			distToTarget = nav_functions.findDistBetween(self.coords, targetCoords)


			# reached the destination
			if distToTarget[0]**2 + distToTarget[1]**2 < self.atDestinationTolerance:
				print("reached destination")

				return -1


			# find target heading
			self.targetHeading = math.degrees(nav_functions.findAngleBetween(self.coords, targetCoords))

			# check if there is a heading it should end at
			target["finalHeading"] = False
			if "finalHeading" in target:
				finalHeading = target["finalHeading"]


			# set ideal speed given the heading and the distance from target

			targetSpeedPercent = nav_functions.findDiffSpeeds(distToTarget, self.trueHeading, self.targetHeading, finalHeading = finalHeading, turnConstant = 2, destTolerance = self.atDestinationTolerance)

			self.targetSpeed = [targetSpeedPercent[0]*self.topSpeed/100, targetSpeedPercent[1]*self.topSpeed/100]



			# Print status
			print("\nheading:", self.trueHeading, "target heading:", self.targetHeading)
			print("current coords:", self.coords, "target coords:", targetCoords, "(accuracy:", self.gpsAccuracy, ")")
			print("target speeds:", self.targetSpeed)
			print("real speeds:", self.realSpeed) 
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

	microProcessor = checkDeviceType()

	myRobot = Robot()


	robotThread = Thread(target=beginRobot)

	print("made robot")

	robotThread.start()

	print("starting website")
	robot_website.myRobot = myRobot
	robot_website.app.run(debug=False, port=8000, host='0.0.0.0')

	print("done starting website")














