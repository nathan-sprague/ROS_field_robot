# ROS_field_robot
Files to run the robot with a Jetson Nano. The robot can autonomously travel to specified areas and then travel through rows given GPS destinations. It is capable carrying extra sensors to collect data in a field.


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
$ /usr/bin/python3 /home/john/Desktop/ROS_field_robot/robot_navigation.py
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
#### Change the destinations 
At the top there are some example dictionaries with points. Make your own dictionary in the same format as the others and comment the others out. <br>
Each point a has few other things about them, such as destination type and heading. Destination type should usually be just "point" and heading is where the robot will face when it reaches the point. Leave the heading blank if you don't care where it ends up at.

#### Switch between between-row navigation and point-to-point navigation
Near the bottom of the file in the function called beginRobot(), uncomment/comment the lines pointNavigate()/interRowNavigate()

## video_navigation.py
Run this program to just view the stream or camera feed. It just calculates the center of the row. You can modify the variables beginning with an underscore at the top of the file. Edit the list "stepsShown" to include the step numbers you want to see.

## robot_esp_control.py
Controls the ESP32. No need to look at this.

## robot_website.py
Controls the website. No need to look at this.

## nav_functions.py
Various functions that are used in other programs. No need to look at this.

# Hardware
## Frame
The CAD files are in the zipped folder

## Controllers
### ESP32
You need at least 1 ESP32. Check the movement.ino file for specific pinouts. Most of the pins are used.

### Jetson Nano
Any Jetson Nano or even a Raspberry Pi would do, but we are using the 4GB model of the Jetson Nano.

## Sensors
### Sparkfun RTK2 GPS Module
  You can also use a second RTK2 and XBee radio to have better accuracy using RTK.
  
### Magnetic Encoder
  Specifications and CAD file will be released soon.

# Software
  Upload the movement.ino Sketch to the ESP832.
  Run the file "robot_navigation.py" on the Jetson Nano.
 
