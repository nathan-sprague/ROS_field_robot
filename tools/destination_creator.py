from gps_controller import Gps

from threading import Thread
import time
import serial
import signal


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

		self.notCtrlC = True # tag used to have graceful shutdowns. When this is set to false, would-be inifnite loops stop

		self.destinations = []
		self.subPoints = []

		self.defaultAtDestinationTolerance = 1.5
		self.atDestinationTolerance = self.defaultAtDestinationTolerance # at destination when x feet from target
	

		#################################
		self.coords = [-1,-1] # lat, long
		self.gpsAccuracy = 12345 # mm

		self.headingAccuracy = 360 # degrees
		self.headingAccuracyTimer = time.time() # used when doing point-to-point navigation. If the robot doesn't know where it's going, there's no point moving
		self.lastHeadingTime = time.time() # time since a heading was obtained
		self.trueHeading = 0 # degrees

		
		self.connectionType = 0 # gps connection type (0=none, 1=dead rekoning, 2=2D fix, 3=3D fix, 4=GNSS+dead rekoning, 5=time fix)

		self.obstacles = []

		i=0
		self.gpsConnected = False


		mainGpsPort = "none"

		# get a list of all of the ports and port information
		portsConnected = [tuple(p) for p in list(serial.tools.list_ports.comports())]

		
		for i in portsConnected: # connect to all of the relevant ports
			if i[1] == "u-blox GNSS receiver":
				print("gps main on port", i[0])
				mainGpsPort = i[0]
				self.gpsConnected = True

			else:
				print("unknown device port", i)

		if False:
				# initialize the GPS with the port found
			self.gpsModule = Gps(self, mainGpsPort, verbose=False)	

			if  self.gpsModule.begin():
				print("began GPS")
			else:
				print("GPS failed")
				while True:
					time.sleep(1)
					print("set up GPS before continuing")
			
		self.filename = "dest_" + str(int(time.time())) + ".py"
		print("saving file as", self.filename)
		# self.createDestination(self.filename)
		self.recordThread = Thread(target=self.createDestination, args = [self.filename])
		self.recordThread.start()


	def createDestination(self, filename):

		msg = "autoDests=["
		while True:
			if msg == "":
				msg += "},\n"
			msg += "{'coord':" + str(self.coords) + ", "
			i = input("->")
			print("what was given:", i)
			if i == "":
				pass
			elif i=="h":
				msg += "'heading': " + str(self.trueHeading) + ","



			if i=="r":
				msg += "'destType': 'row',"
			else:
				msg += "'destType': 'point',"

				
			print(msg)
			with open(filename, 'a+') as fileHandle:
				fileHandle.write(str(msg[0:-1]))
				fileHandle.close()
				print("wrote")
				msg = ""
""""


acreBayCornNorth = [{"obstacles": [ [[40.470660, -86.995236], [40.470560, -86.995229], [40.470554, -86.995360], [40.469927, -86.995355], [40.469834, -86.995500], [40.469767, -86.995515], [40.469751, -86.996771], [40.470864, -86.996777] ] ]},
                 #   {"coord": [40.4705652, -86.9952982], "destType": "point"}, # test point inside of obstacle
                    {"coord": [40.4705552, -86.9952982], "finalHeading": 1, "destType": "point"},
                    {"coord": [40.4705552, -86.9952982], "destType": "row"},
                    {"coord": [40.470558, -86.995371], "destType": "point"},
                    {"coord": [40.470558, -86.995259], "destType": "point"}
                    ]
       """
       


def signal_handler(sig, frame):
	print('You pressed Ctrl+C!')

	with open(myRobot.filename, 'a+') as fileHandle:
		fileHandle.write("}]")
		fileHandle.close()
		# print("wrote")
		msg = ""
	# myRobot.recordThread.join()

	exit()

signal.signal(signal.SIGINT, signal_handler)




if __name__ == "__main__":
	
	myRobot = Robot()
