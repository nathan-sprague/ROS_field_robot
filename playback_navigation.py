import time

import math

from threading import Thread
import signal
import nav_functions


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

		self.startTime =  int(time.time()) # seconds

		self.notCtrlC = True # flag used to have graceful shutdowns. When this is set to false, would-be inifnite loops stop

		self.errorList = []
		self.alerts = "none"


		# vestigial variables used for the website. Remove eventually
		self.coordListVersion = 0 
		self.destinations = [{"coord": [40.470283, -86.995008], "destType": "point"},
                    {"coord": [40.470155, -86.995092], "destType": "point"},
                    {"coord": [40.470048, -86.995006], "destType": "point"},
                    {"coord": [40.469964, -86.995044], "destType": "point"},
                    {"coord": [40.470160, -86.995063], "destType": "point"}]
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
		self.trueHeading = 0 # degrees
		self.gyroHeading = 0
		self.gpsHeading = 0
		self.gyroAtConfirmedHeading = 0
		self.lastConfirmedHeading = 0
		self.realSpeed = [0, 0] # mph
		self.pastCoords = []

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
		self.filename = "/home/nathan/log_success/log_1655999853.txt" #"log_1654787004.txt"
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
		if self.coords != self.lastCoords and abs(self.lastCoords[0] > 10):
			newHeading = nav_functions.findAngleBetween(self.lastCoords, self.coords) * 180 / 3.1416
			if newHeading < 0:
				newHeading = 360 + newHeading
			newDist = nav_functions.findDistBetween(self.coords, self.lastCoords)
			newDist = (newDist[0]**2 + newDist[1]**2)**0.2
			print("new h", newHeading)
			self.pastCoords += [[newHeading, newDist]]
		
		self.lastCoords = self.coords[:]
		if False: #len(self.pastCoords) > 1:
			avgHeading = 0
			totDist = 0
			for i in self.pastCoords:
				# print(i)
				avgHeading += i[0]*i[1]
				totDist += i[1]
			avgHeading /= len(self.pastCoords)
			avgHeading /= (totDist/len(self.pastCoords))
			self.pastCoords = self.pastCoords[1::]
			print("tot dist", totDist)
			if totDist > 0.75:
				self.trueHeading = avgHeading
				self.alerts = "tot dist used"
				print("set real heading", avgHeading)
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




	def readLogs(self, filename):
		"""
		Logs the variables related to the robot's status into a text file. 
		This is helpful for viewing what the robot thought was going on during a past run.

		Parameters: filename (text file to log to)
		Returns: None
		"""
		time.sleep(2)

		print("reading from:", filename)
		lastCoords = [0,0]
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
				# print(wordList)

				self.timer = float(wordList[0])
				self.trueHeading = float(wordList[1])
				self.targetHeading = float(wordList[2])
				self.realSpeed[0] = float(wordList[3])
				self.realSpeed[1] = float(wordList[4])
				self.targetSpeed[0] = float(wordList[5])
				self.targetSpeed[1] = float(wordList[6])
				# print("real", self.realSpeed)
				self.coords[0] = float(wordList[7])
				self.coords[1] = float(wordList[8])
				self.targetDestination[0] = float(wordList[9])
				self.targetDestination[1] = float(wordList[10])
				if len(wordList) > 12: # remove this later, this is for backwards compatability
					# print("wl", wordList[12])
					self.headingAccuracy = float(wordList[11])
					self.gpsAccuracy = float(wordList[12])

					self.calculateTrueHeading()

				# print(wordList)

				if lastCoords!=self.coords:
					time.sleep(0.1)
				lastCoords = self.coords[:]
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














