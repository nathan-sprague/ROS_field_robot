# ROS_field_robot
Files to run the robot with a Jetson Nano. The robot will autonomously travel to specified areas and then travel through rows given GPS destinations. It is capable carrying extra sensors to collect data in a field.

# What It Can Do Right Now
## Inter-row navigation
So far, navigation through small corn via webcam works well if the camera is positioned correctly.
Navigation through taller corn works as well, when the camera is a little bit under the canopy (~18" from the ground)

## Point-to-point navigation
Point-to-point navigation works when provided with good RTK correction signals


## Simulations
- If provided with a .bag file, the robot can display the center of the row estimations.
- Point-to-point navigation can be simulated. The robot's movement parameters, such as turning radius and simulated gps accuracy, can be modified in the esp32_tester.py file.

- To run the simulations, run the same robot_navigation.py file on your computer. It is smart enough to know that you are running it from a personal computer, not the robot.

<br><br>

# Control

## Connecting to Jetson Nano

Make sure the access point ESP is connected. It will create a network called "robot" and has no passsword. The Jetson nano will connect to it automatically but you must connect to it on your PC.<br>

Next open up any UNIX-based terminal on your PC. In the terminal, enter: <br>
```console
$ ssh john@jetson.local
```
It may ask you if you are willing to connect to an unsecured device. Say yes.<br>
Enter the password, which is ```raspberry```

## Running the program
### Starting the program
Enter the command below:
```console
$ /usr/bin/python3 Desktop/ROS_field_robot/robot_navigation2.py
```
The previous terminal commands can be found by pressing the up arrow key so you may not need to type it.

### Monitoring the program
If you are doing point-based navigation, you can monitor the robot's status on the website. When connected to the same network, (which you should be anyway to run the program) go to <http://jetson.local:8000/>. <br>
You can see the map with the points at the bottom. Overriding and adding new destinations using the website currently does not work.

### Exiting the program
If the robot is set navigate from point to point, it will stop when it reaches the last destination. If it is set to navigate between rows or you wish to stop sooner, enter ```ctrl+c``` and the program will end and the robot will stop. <br>If it keeps moving even after the program stops, something is wrong and you should use the E-stop. This is extremely unlikely.

## Editing the program
There may be reasons to edit the file when ssh'd in. I suggest using nano, for example:

```console 
$ nano /home/john/Desktop/ROS_field_robot/robot_navigation.py
```
press ```ctrl+x``` to exit and press y to save

## Other useful terminal commands
```ls / cd + [filename]```: Change directory and see what is in the directory you are in. <br>
```rm + [filename]```: remove the file. Use to get rid of .bag files taking up too much space. <br>
```mkdir + [filename]``` make a folder. <br>
Making a file: nano or vi a file that does not exist and it will make it.

# About the python files
## robot_navigation.py
This is the main python file used to navigate point-to-point or between rows.

### Switch between between-row navigation and point-to-point navigation
The default navigation method is to go from point-to-point. The robot will switch to inter-row navigation automatically when specified in the destinations dictionary. To make the robot always run in inter-row navigation mode, change the variable "interRowNavigation" to True in robot_navigation2.py

## destinations.py
There are some example destinations that the robot can use. They include examples of the different kinds of parameters the robot can understand, such as final heading and obstacles.

## video_navigation.py
Run this program to just view the stream or camera feed. It just calculates the center of the row. There are different video-based navigation methods that this file can call. The default is video_nav_types/standard.py

## gps_controller.py
Read the GPS and updates relevant information to the robot object.

## robot_esp_control.py
Sends commands and recieves statuses from the ESP32s. Sends commands based on the robot's target speed.

## esp_tester.py
Simulates robot_esp_control.py when there are no ESP32s connected. Modify this to change the simualtion parameters .

## robot_website.py
Hosts a website for the user to view the robot's status. The website html and js files are in the templates and static folders.

## nav_functions.py
Various functions that are used in other programs, such as converting coordinates.

# Hardware
## Controllers
### ESP32
You need at least 1 ESP32. Check the movement.ino file for specific pinouts. Most of the pins are used.

### Jetson Nano
Any Jetson Nano or even a Raspberry Pi would do, but we are using the 4GB model of the Jetson Nano.

### Access point
To ssh into the Jetson nano, you need an access point. An ESP32 can be used to make an access point easily. Using the motor controlling ESP32 is not reccomended because using wifi may slow down the speed and reliability.

## Sensors
### Ardusimple simpleRTK2B+heading board
  For the robot to know its heading, it needs to have RTK corrections. Therefore you will need a base station nearby. The robot can run with a different heading sensor, but the code will need to be modified
  
### intel Realsense camera
<br>

# Software
  Upload the movement.ino Sketch to the ESP32.
  Run the file "robot_navigation.py" on the Jetson Nano.

<br>


# Initial software setup process
1. Flash the Jetson Nano .iso image to an SD card and set up the Jetson Nano
2. Clone this repository
3. Install the pyrealsense2 library. It must be compiled from source, which may take a while.
4. Download the remaining required libraries using pip. 
5. Change the permissions to enable the program to access the serial ports. To give access one time, enter in the terminal:
```console
 sudo chmod -777 /dev/ttyACM0
 ```
 6. To give access forever, enter:
 ```console
sudo adduser dialout [your username]
```
7. Run robot_navigation.py and the robot will go!


<br>

# Things Left To Do
- The robot struggles to enter the row coming from point-based navigation
- The robot does not move through all types of corn height.
- Destinations and movement must be given to the robot manually. While it can do basic obstacle avoidance, it cannot optimize a path around the obstacles.