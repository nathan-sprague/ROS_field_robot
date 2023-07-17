# log with: python3 deleteme2.py >> deleteme.txt 2>&1

import time

import math

from threading import Thread

import os

import random

import signal # for graceful exits

# For finding which ports to connect to (GPS, ESP controller, etc)
import serial.tools.list_ports


_run_mode = "sim" # real, sim, playback

if len(list(serial.tools.list_ports.comports())) == 0:
	print("no devices connected, assuming this is a simulation or playback") 
else:
	print("devices connected. This is not a test")
	_run_mode = "real"

if _run_mode != "real":
	from esp_tester import Esp, Gps

else:
	from esp_controller import Esp
	from gps_controller import Gps

import robot_navigation
import destinations as exampleDests

import video_navigation



_nav_destination = [
			{"field": [[[40.4711748, -86.994099], [40.471862, -86.994105], [40.471862, -86.995176], [40.471174, -86.995170]]]}, # google maps Quinns field?
			{"coord": [40.4713166, -86.9952095], "destType": "point"}, # map
			{"coord": [40.471191753333336, -86.995175], "destType": "point", "finalHeading":90}, # google maps
			{"coord": [40.471191753333336, -86.994056], "destType": "row", "heading": 359},
			{"coord": [40.471191753333336, -86.994056], "destType": "point"},
			{"coord": [40.471165, -86.994079], "destType": "point"},
			{"coord": [40.471166, -86.995200], "destType": "point"}
                ]
# Qual: 4, Lat: 40.471191753333336, long: -86.99412397166667, alt: 215.2308


class Robot:
	"""
	It stores the robot's status variables, along with its commands
	It sends the commands to the external components through other python files
	This allows it to navigate autonomously through both corn rows and gps-based points
	"""

	def __init__(self, run_mode):
		"""
		Sets up the variables related to the robot's conditions and targets
	
		Parameteres: none
		Returns: none
		"""


		print("creating robot")


		self.start_time =  int(time.time()) # seconds

		self.notCtrlC = True # tag used to have graceful shutdowns. When this is set to false, would-be inifnite loops stop	


		self.run_mode = run_mode

		self.navStatus = {0} # number representing a message. See html file for the status types
		self.lastNavStatus = {0} 


		###############################
		# robot attributes
		###############################
		self.destination_tolerance = 0.3 # at destination when x feet from target
		self.top_speed = 2 # mph
		self.at_destination = False

		#################################
		# true position-related variables
		#################################
		self.coords = {"coords": [-1,-1], # lat, long
						"accuracy": 12345,
						"time": time.time(),
						"fix": -1, # gps connection type (0=none, 1=dead rekoning, 2=2D fix, 3=3D fix, 4=GNSS+dead rekoning, 5=time fix)
						"siv": 0
						}


		self.heading = {"heading": -1,
						"accuracy": 360,
						"time": time.time()}

		self.real_speed = [0, 0] # mph (left, right)


		###############################
		# position-target variables
		###############################

		self.destinations = []


		

		# target speed read by esp_controller.py to set the esp speed
		self.target_speed = [0, 0] # mph

		self.target_heading = 0 # deg (north = 0)
		self.target_destination = [0, 0]

		self.destID = -10

		self.obstacles = []
		self.obstaclesID = -10


		################################
		# set up the sensors (ESP, GPS and camera)
		###############################

		self.save_dir = "log_" + str(self.start_time)
		if self.run_mode == "real":
			

			os.mkdir(self.save_dir)

		self.navStatus = set()

		self.in_row_time = 0

		# motion types are: {waiting, straight-nav, 0-pt-turn, calibration-forward}
		self.motionType = ["waiting", time.time()]


		
		if self.run_mode == "real": # not testing
			portsConnected = [tuple(p) for p in list(serial.tools.list_ports.comports())]

			for i in portsConnected: # connect to all of the relevant ports
				if i[1] == "CP2102 USB to UART Bridge Controller - CP2102 USB to UART Bridge Controller":
					
					self.espModule = Esp(self, i[0]) # make a new ESP object and give it this class and the port name
					if self.espModule.begin(True):
						print("Connected to ESP on port", i[0])
					else:
						print("Connection for ESP on port", i[0], "not successful!")
						exit()

				elif i[1] in ["u-blox GNSS receiver", "Septentrio USB Device - CDC Abstract Control Model (ACM)"]: # initialize the GPS with the port found
					self.gpsModule = Gps(self, verbose=False)	
					if self.gpsModule.begin():
						print("Connected to GPS on port", i[0])
					else:
						print("Connection for GPS on port", i[0], "not successful!")
						exit()
				else:
					print("unknown device port", i[0])



			self.cam = video_navigation.Camera(self, save_dir=self.save_dir,
						process_images=True, playback_type=run_mode)
			self.cam.begin()
			self.video_thread = Thread(target=self.cam.video_navigate)
			self.video_thread.start()


			self.log_thread = Thread(target=self.log_status)
			self.log_thread.start()

		else: # simulation or playback
			self.espModule = Esp(self, 0)
			self.gpsModule = Gps()

			self.cam = video_navigation.Camera(self, save_dir="",
						process_images=True, playback_type=run_mode)
			print("made cam")

			if self.run_mode == "sim": # simulation
				self.espModule.begin()
				print("began esp")

			self.cam.begin()

			self.video_thread = Thread(target=self.cam.video_navigate, args=[True])
			self.video_thread.start()


	def begin(self, destinations):
		self.destinations = destinations

		navigator = robot_navigation.Navigator(self)
		while self.notCtrlC:
			navigator.navigate()
			break


	def printStatus(self, targetCoords, distToTarget):
		# prints relevant variables for the robot. This should be the only place where stuff is printed (in theory)
		print("heading:", self.heading["heading"], "target heading:", self.target_heading, "heading accuracy:", self.heading["accuracy"])

		# print("current coords:", self.robotcoords, "target coords:", targetCoords, "fix type:", self.robot.connectionType, "(accuracy:", self.robot.gpsAccuracy, ")")
		print("target speeds:", self.target_speed, "real speeds:", self.real_speed, "distance from target:", distToTarget, "\n")
		# print("motion type:", self.robot.motionType)
		print("runtime:", int(time.time()-self.start_time))


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

		
		self.espModule.endEsp()
		
		
		self.gpsModule.endGPS()

		robotThread.join()


	def log_status(self):
		while self.notCtrlC:
			log_vars = [
						self.cam.frame_num,
						int(time.time()),
						self.coords["coords"][0],
						self.coords["coords"][1],
						int(self.coords["time"]),
						self.coords["fix"],
						self.coords["siv"],
						self.heading["heading"],
						self.heading["accuracy"],
						int(self.heading["time"]),
						self.real_speed[0],
						self.real_speed[1],
						self.target_speed[0],
						self.target_speed[1],
						self.target_heading
						# self.move_status
						]
			save_str = str(log_vars)[1:-1] + "\n"
			file = open(os.path.join(self.save_dir, "status.txt"), "a")  # append mode
			file.write(save_str)
			file.close()


			time.sleep(1)



def beginRobot(nav_destinations):
	# this function begins the navigation function. 
	# For some reason you need to call a function outside of the robot when threading for it to work properly.
	myRobot.begin(nav_destinations) 




def signal_handler(sig, frame):

	#   stopStream()
	print('You pressed Ctrl+C!')

	myRobot.notCtrlC = False
	time.sleep(0.5)
	exit()



if __name__ == "__main__":
	if True: #runMode != 0:
		signal.signal(signal.SIGINT, signal_handler)

	myRobot = Robot(run_mode=_run_mode)

	robotThread = Thread(target=beginRobot, args=[_nav_destination])

	print("made robot")

	robotThread.start()

	if True:
		import robot_website

		print("starting website")
		robot_website.myRobot = myRobot
		robot_website.app.run(debug=False, port=8000, host='0.0.0.0')

		print("done starting website")
	else:
		print("not using website")
