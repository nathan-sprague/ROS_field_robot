import time

import math

from threading import Thread
import signal


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

		self.notCtrlC = True # tag used to have graceful shutdowns. When this is set to false, would-be inifnite loops stop

		self.errorList = []


		# vestigial variables used for the website. Remove eventually
		self.coordListVersion = 0 
		self.destinations = [{"coord": [40.422266, -86.916176], "destType": "point"},
                    {"coord": [40.422334, -86.916240], "destType": "point"},
                    {"coord": [40.422240, -86.916287], "destType": "point"},
                    {"coord": [40.422194, -86.916221], "destType": "point"},
                    {"coord": [40.422311, -86.916329], "destType": "point"}]
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
		self.gpsAccuracy = 10000 # feet
		self.heading = 0 # degrees
		self.realSpeed = [0, 0] # mph

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
		self.filename = "logs2/NEW.txt"
		self.running = True
		self.recordThread = Thread(target=self.readLogs, args = [self.filename])
		self.recordThread.start()
		print("thread started")





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

		print("reading from:", filename)
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
				print(wordList)

				self.timer = float(wordList[0])
				self.heading = float(wordList[1])
				self.targetHeading = float(wordList[2])
				self.realSpeed[0] = float(wordList[3])
				self.realSpeed[1] = float(wordList[4])
				self.targetSpeed[0] = float(wordList[5])
				self.targetSpeed[1] = float(wordList[6])
				self.coords[0] = float(wordList[7])
				self.coords[1] = float(wordList[8])
				self.targetDestination[0] = float(wordList[9])
				self.targetDestination[1] = float(wordList[10])





				time.sleep(0.5)
				line = fileHandle.readline()
				
			print("All done")
			self.running = False
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
	robot_website.app.run(debug=False, port=8003, host='0.0.0.0')

	print("done")
	myRobot.closeRobot()
	print("done starting website")














