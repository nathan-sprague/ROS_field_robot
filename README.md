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
### Change the destinations 
At the top there are some example dictionaries with points. Make your own dictionary in the same format as the others and comment the others out. <br>
Each point a has few other things about them, such as destination type and heading. Destination type should usually be just "point" and heading is where the robot will face when it reaches the point. Leave the heading blank if you don't care where it ends up at.

### Switch between between-row navigation and point-to-point navigation
The default navigation method is to go from point-to-point. The robot will switch to inter-row navigation automatically when specified in the destinations dictionary. To make the robot always run in inter-row navigation mode, change the variable "interRowNavigation" to True in robot_navigation2.py

## video_navigation.py
Run this program to just view the stream or camera feed. It just calculates the center of the row. There are different video-based navigation methods that this file can call. The default is video_nav_types/standard.py

## GPS controller.py
Read the GPS and updates relevant information to the robot object. No need to look at this.

## robot_esp_control.py
Controls the ESP32. No need to look at this.

## esp_tester.py
Simulates robot_esp_control.py when there are no ESP32s connected. Modify this to change the simualtion parameters 

## robot_website.py
Controls the website. No need to look at this.

## nav_functions.py
Various functions that are used in other programs. No need to look at this.

# Hardware
## Controllers
### ESP32
You need at least 1 ESP32. Check the movement.ino file for specific pinouts. Most of the pins are used.

### Jetson Nano
Any Jetson Nano or even a Raspberry Pi would do, but we are using the 4GB model of the Jetson Nano.

## Sensors
### Ardusimple simpleRTK2B+heading board
  For the robot to know its heading, it needs to have RTK corrections. Therefore you will need a base station nearby. The robot can run with a different heading sensor, but the code will need to be modified
  
# Software
  Upload the movement.ino Sketch to the ESP832.
  Run the file "robot_navigation.py" on the Jetson Nano.
 
<br>

# Things Left To Do
- The robot struggles to enter the row coming from point-based navigation
- The robot does not move through any type of corn height.
- Destinations and movement must be given to the robot manually. While it can do basic obstacle avoidance, it cannot optimize a path around the obstacles.