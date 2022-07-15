import time

import math
import random

from threading import Thread
import signal
import nav_functions
import pyproj
import destinations as exampleDests



testing = True


import robot_website


class FakeRobot:
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

		self.updateSpeed = 0.3

		self.startTime =  int(time.time()) # seconds

		self.notCtrlC = True # flag used to have graceful shutdowns. When this is set to false, would-be inifnite loops stop

		self.testCoords = False
		self.lastWordList = [0]*20
		self.legacyReader = True


		self.errorList = []
		self.alerts = "None"
		self.alertsChanged = True


		# vestigial variables used for the website. Remove eventually
		self.coordListVersion = 0 
		self.destinations = [] # exampleDests.acreBayCornNorth
		self.subPoints = []

		###############################
		# robot attributes
		###############################
		self.atDestinationTolerance = 3 # at destination when x feet from target
		self.topSpeed = 4 # mph

		#################################
		# true position-related variables
		#################################
		self.coords = [-1, -1] # lat, long
		self.coords2 = [-1, -1]
		self.lastCoords = [-1, -1]
		self.gpsAccuracy = 10000 # feet
		self.trueHeading = -1000 # degrees
		self.gyroHeading = 0
		self.gpsHeading = 0
		self.gyroAtConfirmedHeading = 0
		self.lastConfirmedHeading = 0
		self.realSpeed = [0, 0] # mph
		self.pastCoords = []
		self.destID = random.randint(0,1000)
		self.connectionType = 0

		self.obstacles = [[[40.46977123481278,-86.99552120541317], [40.46993606641275,-86.99534717740178], [40.4701423474349,-86.99534608852393], [40.47013772209357,-86.99572404600923], [40.469767727757464,-86.99572404600923] ]]



		self.obstaclesID = 0

		self.targetPath = []
		self.targetPathID = 0

		###############################
		# position-target variables
		###############################
		self.targetSpeed = [0, 0] # mph
		self.targetHeading = 0 # deg (north = 0)
		self.targetDestination = [0, 0]

		###############################################
		# logging variables and set up recording thread
		###############################################

	def begin(self):
		self.filename = "/home/nathan/logs/logs_july14/log_1656706153.txt" #"/home/nathan/log_success/log_1655999853.txt" #"log_1654787004.txt"
		self.running = True
		self.recordThread = Thread(target=self.readLogs, args = [self.filename])
		self.recordThread.start()
		print("thread started")


	def calculateTrueHeading(self):
		"""
		Calculates the real heading based on GPS heading, gyro heading, and current speed
		"""
		#return # dont usually use during playback
		# if you are moving and not making a sharp turn, just use the GPS heading
		# self.trueHeading = 180+self.trueHeading #self.gyroHeading
		return

		
		self.targetHeading = self.gyroHeading
		
		if True:
			if (abs(self.realSpeed[0]) > 1.5 and abs(self.realSpeed[1]) > 1.5 and abs(self.realSpeed[0]-self.realSpeed[1]) < 0.2): # you are moving straight forward
				#	print("heading confirmed")
					self.alerts = "heading confirmed"
					# time.sleep(2)
					self.trueHeading = self.gpsHeading
					self.lastConfirmedHeading = self.trueHeading
					self.gyroAtConfirmedHeading = self.gyroHeading
					return
			
			
		
		# at least one of the above conditions were not met. Use the gyro with the last known heading
		headingChange = self.gyroHeading - self.gyroAtConfirmedHeading
		self.trueHeading = self.lastConfirmedHeading + headingChange
		self.alerts = "heading not confirmed" + str(headingChange)
		print("unconfimed heading:")
		print("gyro was:", self.gyroAtConfirmedHeading, "gyro is now:", self.gyroHeading)
		print("correction:", headingChange, "last confirmed:", self.lastConfirmedHeading, "estimated heading", self.trueHeading)



	def closeRobot(self):
		"""
		Closes all threads in robot.
		Shuts down everything gracefully

		parameters: none
		returns: none
		"""

		print("shutting down robot")
		self.targetSpeed = [0, 0]

		# close recording thread
		self.recordThread.join()


	def estimateCoords(self):
		p = pyproj.Proj('epsg:2793')

		turningSpeedConst = 3.7
		movementSpeedConst = 0.35 / self.updateSpeed * 0.1

		if self.targetSpeed[0] != 0 :
			self.realSpeed[0] = abs(self.realSpeed[0]) * self.targetSpeed[0] / abs(self.targetSpeed[0])

		if self.targetSpeed[1] != 0:
			self.realSpeed[1] = abs(self.realSpeed[1]) * self.targetSpeed[1] / abs(self.targetSpeed[1])
		
		realHeadingChange = (self.realSpeed[0]-self.realSpeed[1])*turningSpeedConst

		#self.trueHeading += realHeadingChange

		distMoved = (self.realSpeed[0] + self.realSpeed[1]) * 5280/3600/3.28 * movementSpeedConst
		# print("dmove", distMoved)

		# print("og coords", self.trueCoords)
		x, y = p(self.coords[1], self.coords[0])
		dy = distMoved * math.cos(self.trueHeading*math.pi/180) * 0.29
		dx = distMoved * math.sin(self.trueHeading*math.pi/180) * 0.29
		x += dx
		y += dy

		# print("og cart coords", dx,dy)

		y2,x2 = p(x, y, inverse=True)


		self.coords = [x2, y2]


	def readLogs(self, filename):
		"""
		Logs the variables related to the robot's status into a text file. 
		This is helpful for viewing what the robot thought was going on during a past run.

		Parameters: filename (text file to log to)
		Returns: None
		"""
		time.sleep(2)
		coordList = []

		print("reading from:", filename)
		lastCoords = [0,0]
		lineNum = 0
		with open(filename) as fileHandle:
			line = fileHandle.readline()
			while line and self.notCtrlC:
				wordList = []
				word = ""
				for i in line:
					if i ==",":
						wordList+=[word]
						word = ""
					else:
						word+=i
				wordList += word



				if wordList[0] == "d":
					i = 1
					self.destinations = []
					print(wordList)
					while i < len(wordList)-1:
						self.destinations += [{"coord": [wordList[i], wordList[i+1]], "destType": 'point'}]
						i+=2
					self.destID = random.randint(0,1000)

				elif wordList[0] == "o":
					i = 1
					self.obstacles = []
					print(wordList)
					while i < len(wordList)-1:
						self.destinations += [{"coord": [wordList[i], wordList[i+1]], "destType": 'point'}]
						i+=2
					self.destID = random.randint(0,1000)

				else:
					i=0
					while i<len(wordList):
						if wordList[i]!='' and wordList[i]!="\n":

							val = float(wordList[i])

							if i==0:
								self.timer = val
								print(self.timer)
							elif i==1:
								self.trueHeading = val
							elif i==2:
								self.targetHeading = val
							elif i==3:
								self.realSpeed[0] = val
							elif i==4:
								self.realSpeed[1] = val
							elif i==5:
								self.targetSpeed[0] = val
							elif i==6:
								self.targetSpeed[1] = val
							elif i==7:
								if not self.testCoords or abs(self.coords[0]-val) > 1:
									self.coords[0] = val
							elif i==8:
								if not self.testCoords or abs(self.coords[0]-val) > 1:
									self.coords[1] = val
							elif i==9:
								self.targetDestination[0] = val
							elif i==10:
								self.targetDestination[1] = val
							elif i==11:
								self.headingAccuracy = val
							elif i==12:
								self.gpsAccuracy = val
							elif i==13:
								self.connectionType = val
								print("connection type:", self.connectionType)

						i+=1

						# self.calculateTrueHeading()

					if self.testCoords:
						self.estimateCoords()

					# print(wordList)

					if lastCoords!=self.coords:
						time.sleep(self.updateSpeed)
					lastCoords = self.coords[:]

				lineNum += 1
				line = fileHandle.readline()
					
			print("All done")
			self.running = False
			exit()
			# robot_website.shutdownServer()



def signal_handler(sig, frame):

	#   stopStream()
	print('You pressed Ctrl+C!')

	myRobot.notCtrlC = False
	myRobot.closeRobot()
	# robot_website.shutdownServer()
	time.sleep(0.5)
	exit()





if __name__ == "__main__":
	signal.signal(signal.SIGINT, signal_handler)
	myRobot = FakeRobot()

	robotThread = Thread(target=myRobot.begin)

	print("made robot")

	robotThread.start()


	print("starting website")
	robot_website.myRobot = myRobot
	robot_website.app.run(debug=False, port=8000, host='0.0.0.0')

	print("done")
	myRobot.closeRobot()
	print("done starting website")














