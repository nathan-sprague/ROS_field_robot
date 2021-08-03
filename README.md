# ROS_field_robot
Files to run the robot with a Jetson Nano (originally Raspberry Pi) and soon ROS. The robot can autonomously travel to a specific area and then travel through rows given GPS destinations. It is capable carrying extra sensors to collect data in a field.


# Control

## Connecting to Jetson Nano

Make sure the access point ESP is connected. It will create a network called "robot" and has no passsword. The Jetson nano will connect to it automatically but you must connect to it on your PC.<br>

Next open up any UNIX-based terminal. In the terminal, enter: <br>
```console
user@pc:~$ ssh john@jetson.local
```
It may ask you if you are willing to connect to an unsecured device. Say yes.<br>
Enter the password, which is ```raspberry```

## Running the program
The previous terminal commands can be found by pressing the up arrow key. 
Enter the command below:
```console
john@jetson:~$ /usr/bin/python3 /home/john/Desktop/ROS_field_robot/robot_navigation.py
```

## Editing the program
There may be reasons to edit the file when ssh'd in. I generally advise using nano, for example:
```console
john@jetson:~$ nano /home/john/Desktop/ROS_field_robot/robot_navigation.py
```
press ```ctrl+x``` to exit and press y to save

## Other useful terminal commands
```ls / cd + filename```: Change directory and see what is in the directory you are in. <br>
```rm + filename```: remove the file. Use to get rid of .bag files taking up too much space. <br>
```mkdir + filename``` make a folder. <br>
Making a file: nano or vi a file that does not exist and it will make it.

# About the python files
## robot_navigation.py
This is the main python file used to navigate point-to-point or between rows.
#### Change the points 

#### Switch between between-row navigation and point-to-point navigation
Near the bottom of the file in the function called beginRobot(), uncomment/comment the lines pointNavigate()/interRowNavigate()

## video_navigation.py
Run this program to view the video

## robot_esp_control.py

## robot_website.py

## nav_functions.py

# Hardware
## Frame
The CAD files are in the zipped folder

## Controllers
### ESP8266
You need at least 2 ESP8266 microcontrollers, one to control the steering and one to control the speed. You can also use another as an access point. The ESP8266s should be connected to the Jetson Nano.

### Jetson Nano
Any Jetson Nano or even a Raspberry Pi would do, but we are using the 4GB model of the Jetson Nano.

## Sensors
### Sparkfun RTK2 GPS Module
  You can also use a second RTK2 and XBee radio to have better accuracy using RTK.
  
### Magnetic Encoder
  Specifications and CAD file will be released soon.

# Software
  Upload the 2 Sketches of steering and speed to the ESP8266s.
  Run the file "robot_navigation.py" on the Jetson Nano.
 
