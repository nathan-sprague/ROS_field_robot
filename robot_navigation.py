import math
import random
import time
import nav_functions


class Navigator:
	"""
	This class controls the functions of the robot.
	It stores the robot's status variables, along with its commands
	It sends the commands to the external components through other python files
	This allows it to navigate autonomously through both corn rows and gps-based points
	"""

	def __init__(self, robot):
		"""
		Sets up the variables related to the robot's conditions and targets
	
		Parameteres: none
		Returns: none
		"""
		self.robot = robot


	def navigate(self):
		"""
		Calls functions to navigate based on the destination and navigation type.

		Parameters:
		destinations: list of dictonaries, with the dictionary having destination attributes
															i.e. location & navigation type
		Returns:
		none
		"""

		if self.robot.run_mode=="playback": # full playback doesn't use this
			return
			
		print("destinations")

		if "field" in self.robot.destinations[0]:
			self.robot.obstacles = self.robot.destinations[0]["field"]
		if not "coord" in self.robot.destinations[0]: # there was the obstacles and/or rows info in the first element of the destinations list
			self.robot.destinations = self.robot.destinations[1::]


		while self.robot.notCtrlC and len(self.robot.destinations) > 0:


			self.robot.navStatus = set() # nav status is a set of numbers, each corresponding to a message that can be viewed on the website. See the index.html to see what they are
			
			 # get the details from the current destination
			dest = self.robot.destinations[0]
			self.robot.target_destination = dest["coord"]

			if "destTolerance" in dest:
				self.robot.destination_tolerance = dest["destTolerance"]


			# navigate according to the destination type
			navType = dest["destType"]

			if navType == "point":
				# gps-based point-to-point navigation
				self.robot.cam.navMethod = 0
				self.robot.cam.process_images = True
				delayTime = self.navigate_to_point(dest)

			elif navType == "row":
				# find the best direction to move wheels
				self.robot.cam.navMethod = 1
				self.robot.cam.process_images = True
				self.robot.cam.idle = False
				delayTime = self.inter_row_navigate(dest)

			elif navType == "sample":
				# just a placeholder for sampling for now. It will eventually do something.
				print("sampling")
				time.sleep(5)
				delayTime = 0

			else:
				print("unknown navigation type:", navType)
				self.robot.closeRobot()
				exit()


			if delayTime > 0:
				# call the function again
				time.sleep(delayTime)

			elif delayTime < 0:
				# reached the destination
				self.robot.at_destination = False
				print("removed destination")
				if len(self.robot.destinations) > 1:
					destinations = self.robot.destinations[1::] # remove the old destination from the destination list
					self.robot.destinations = destinations[:]
					self.robot.destID = random.randint(0, 1000)

				else:
					print("reached all destinations")
					self.robot.closeRobot()
					time.sleep(0.5)
					os.kill(os.getpid(), signal.SIGUSR1)




	def check_GPS(self):
		"""
		Checks the GPS accuracy and heading to determine whether it is good enough to use for point navigation
		It also prints information on why the GPS is bad if there is an issue

		parameters: 
		none

		returns:
		0 (if GPS accuracy and heading is good)
		waitTime (number) (if GPS is bad)
		"""

		connectionTypesLabels = ["Position not known", "Position known without RTK", "DGPS", "UNKNOWN FIX LEVEL 3", "RTK float", "RTK fixed"]


		if self.robot.coords["accuracy"] > 1000 or self.robot.coords["fix"] <= 0:

			# gps connected but no lock
			print("No GPS fix.", self.robot.coords["siv"], "SIV as of", int(time.time() - self.robot.coords["time"]), "s ago")

			return 2

		elif self.robot.heading["accuracy"] > 40 or time.time() - self.robot.heading["time"] > 2: # you can't trust the heading.
			print("Heading accuracy not good enough! Estimated heading:", self.robot.heading["heading"], "Heading accuracy:", self.robot.heading["accuracy"], 
				"Last updated", int(time.time()-self.robot.heading["time"]), "s ago. GPS accuracy", self.robot.heading["accuracy"], "mm")

			return 2

		else:
			return 0



	def navigate_to_point(self, target):

		""" 
		Manages any errors and sets the target speed and heading

		parameters: target - dictionary with coordinates and heading
		returns: time to wait before calling this function again
		"""
		print("point nav")
		if self.robot.cam.about_to_hit:
			print("about to hit")
			self.robot.target_speed=[0,0]		

		gps_quality = self.check_GPS()

		if gps_quality != 0: # The fix of the GPS is not good enough to navigate
			self.robot.target_speed = [0, 0]
			return gps_quality

		else:
			# gps and heading accuracy is acceptable. Start navigating!

			# get first destination
			target_coords = target["coord"]

			# find how far away the robot is from the target (ft)
			distToTarget = nav_functions.findDistBetween(self.robot.coords["coords"], target_coords)


			final_heading = False
			if "finalHeading" in target:
				final_heading = target["finalHeading"]


			# find target heading
			self.robot.target_heading = math.degrees(nav_functions.findAngleBetween(self.robot.coords["coords"], target_coords))

			
			if self.robot.at_destination:
				if final_heading == False: # reached destination with no heading requirement
					self.robot.target_speed = [0,0]
					print("reached destination, no heading requirement")
					self.robot.navStatus.add(8)
					time.sleep(2)
					return -1

				if abs(nav_functions.findShortestAngle(final_heading, self.robot.heading["heading"])) < 5: # reached destination with the right heading
					print("reached destination with right heading")
					self.robot.navStatus.add(8)
					self.robot.target_speed = [0, 0]
					time.sleep(1)
					return -1
					
				else:
					print("at destination with incorrect heading")
					self.robot.navStatus.add(9)
					self.robot.target_heading = final_heading
					# self.robot.destination_tolerance = 10000

			elif distToTarget[0]**2 + distToTarget[1]**2 < self.robot.destination_tolerance ** 2: # reached the destination
				self.robot.at_destination = True
				print(" -_-_-_-_-_-_-_- reached destination -_-_-_-_-_-_-_-_-_")
				self.robot.target_speed = [0,0]
				time.sleep(1)
				
			else:
				self.robot.at_destination = False


			# set ideal speed given the heading and the distance from target

			targetSpeedPercent = nav_functions.findDiffSpeeds(self.robot.coords["coords"], target_coords, self.robot.heading["heading"], self.robot.target_heading, 
				finalHeading = final_heading, turnConstant = 2, destTolerance = self.robot.destination_tolerance, obstacles = [])
			
			zeroPt = False
			if abs(targetSpeedPercent[1]-targetSpeedPercent[0]) > 100:
				zeroPt = True
			elif abs(targetSpeedPercent[0]) + abs(targetSpeedPercent[1]) > 80:
				if abs(targetSpeedPercent[1]-targetSpeedPercent[0]) > abs(targetSpeedPercent[0]) + abs(targetSpeedPercent[1]) / 5:
					zeroPt = True


			if zeroPt and self.robot.motionType[0] != "0-pt-turn": # started doing a 0-point turn
				self.robot.motionType = ["0-pt-turn", time.time()]
				print("started a 0-pt-turn")
				self.robot.navStatus.add(6)

			elif not zeroPt and self.robot.motionType[0] != "straight-nav": # started moving normally
				self.robot.motionType = ["straight-nav", time.time()]
				print("doing regular navigation")
				print(self.robot.navStatus)
				self.robot.navStatus.add(4)

			elif self.robot.motionType[0] == "0-pt-turn" and self.robot.motionType[1]+1 < time.time(): # doing a 0-pt turn for 4 seconds. Try waiting for a second to get its heading
				self.robot.motionType = ["waiting", time.time()]
				self.robot.target_speed = [0,0]
				print("pausing after partial 0-pt-turn")
				self.robot.navStatus.add(7)
				return 0.5
			else:
				self.robot.navStatus.add(4)

			# conver the target speed from the above function to a real speed. target speed percent is -100 to 100. the robot should usually move -2 to 2 mph
			self.robot.target_speed = [targetSpeedPercent[0]*self.robot.top_speed/100, targetSpeedPercent[1]*self.robot.top_speed/100]

			# Print status
			self.robot.printStatus(self.robot.target_destination, distToTarget)
			


			# give appropriate time to wait
			return 0.3


	

		


	def inter_row_navigate(self, rowInfo = False):
		"""
		use the video_navigation's processed video information (heading, etc.) to set the best target speed.

		Parameters:
			cam - object from video_navigation.py
			destination - location that the robot will travel to. Coordinates (list)
		
		Returns:
			time to delay before calling this function again

		"""

		maxSpeed = self.robot.top_speed

		if self.robot.cam.stop or not self.robot.notCtrlC:
			# the camera stopped. Something happened in the video navigation program
			# stop streaming and go to the next destination
			self.robot.cam.stopStream()
			print("stopped camera stream")
			return -1


		if len(self.robot.obstacles) > 0:
			if nav_functions.is_point_in_polygon(self.robot.coords["coords"], self.robot.obstacles[0]) and self.robot.in_row_time == 0:
				self.robot.in_row_time = time.time()
			elif not nav_functions.is_point_in_polygon(self.robot.coords["coords"], self.robot.obstacles[0]) and self.robot.in_row_time > 0 and time.time()-self.robot.in_row_time > 15:
				return -1


		print(self.robot.cam.robot_pose)

		if self.robot.cam.inside_row:
			maxSpeed = 0.8
		else:
			maxSpeed = 0.2
		# maxSpeed = 0.

		
		slowerWheelSpeed = maxSpeed - (abs(self.robot.cam.robot_pose)/60) # note the /60 part of the equation is a hard-coded value. Change as seen fit


		if slowerWheelSpeed > maxSpeed:
			print("something is wrong! the slower wheel speed is faster. Fix this code!!!!")
			self.robot.closeRobot()
			exit()

		self.robot.target_heading = self.robot.heading["heading"]

		if self.robot.cam.robot_pose > 0: # should turn right
			print("slight right")
			self.robot.target_speed = [maxSpeed, slowerWheelSpeed]

		elif self.robot.cam.robot_pose < 0: # should turn left
			print("slight left")
			self.robot.target_speed = [slowerWheelSpeed, maxSpeed]
		else:
			self.robot.target_speed = [maxSpeed, maxSpeed]

		print("speed:", self.robot.target_speed)

		# return a 0.3 second delay time
		return 0.3



	def process_row_obstacles(self, camHeading):
		if self.robot.cam.about_to_hit:
			if not self.robot.cam.outside_row:
				print("about to hit but inside row. Just going slowly")
				self.robot.navStatus.add(18)
				self.robot.navStatus.add(23)
				maxSpeed*=0.5


			elif self.robot.motionType[0] != "obstacleStop" and self.robot.motionType[0] != "reversing":
				print("about to hit")
				self.robot.target_speed = [0,0]
				print("speed", self.robot.target_speed)
				self.robot.motionType = ["obstacleStop", time.time()]
			
				self.robot.navStatus.add(18)
				self.robot.navStatus.add(19)
				return 0.3
			else:
				print("backing up - obstacle in sight")
				self.robot.target_speed = [-self.robot.top_speed*0.3, -self.robot.top_speed*0.3]
				self.robot.motionType = ["reversing", time.time()]
				self.robot.navStatus.add(18)
				self.robot.navStatus.add(20)
				
				return 0.3

		elif self.robot.motionType[0] == "reversing" and time.time()-self.robot.motionType[1] < 1: # normal reverse for a bit more
			print("backing up - just to be safe")
			self.robot.target_speed = [-self.robot.top_speed*0.2, -self.robot.top_speed*0.2]
			self.robot.navStatus.add(21)
			if len(self.robot.motionType)==2 and camHeading>0:
				self.robot.bounces += 1
				self.robot.motionType += [1]
			elif len(self.robot.motionType) == 2: # did not add the direction yet (first time this if statement is called)
				self.robot.motionType += [-1]
				self.robot.bounces += 1
			return 0.3

		elif self.robot.motionType[0] == "reversing" and self.robot.bounces > 2 and time.time()-self.robot.motionType[1] < 10 + self.robot.bounces*2: # it went back and forth too many times
			print("backing up a lot more- break the bouncing cycle")
			self.robot.target_speed = [-self.robot.top_speed*0.2, -self.robot.top_speed*0.2]
			self.robot.navStatus.add(21)

			if time.time()-self.robot.motionType[1] > 9+ self.robot.bounces*2:
				self.robot.motionType = ["normal", 0]
			return 0.3

		elif self.robot.motionType[0] == "reversing" and time.time()-self.robot.motionType[1] < 10: # back up and turn a little bit to line up with target heading
			if self.robot.motionType[2] < 0:
				self.robot.target_speed = [-self.robot.top_speed*0.2, -self.robot.top_speed*0.1]
			else:
				self.robot.target_speed = [-self.robot.top_speed*0.1, -self.robot.top_speed*0.2]
			print("turning while reversing")
			return 0.3

		elif self.robot.motionType[0] == "reversing" and time.time()-self.robot.motionType[1] < 15: # just turn to face heading
			if self.robot.motionType[2] > 0:
				self.robot.target_speed = [-self.robot.topSpeed*0.1, self.robot.top_speed*0.1]
			else:
				self.robot.target_speed = [self.robot.top_speed*0.1, -self.robot.top_speed*0.1]
			print("turning while going forward - after obstacle")
			return 0.3

		else:
			self.robot.motionType = ["normal", 0]

		return -1 # No longer about to hit. Resume normal navigation




if __name__ == "__main__":
	print("dont run from here")
