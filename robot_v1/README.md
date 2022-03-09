# Robot_v1
Files to run the original robot with a Jetson Nano. The robot is built with 8020 and Ackerman steering. It still works, but has a few shortcomings compared to the new robot, namely it lacks power and the steering system prevents sharp turns away from corn stalks.


# Control
So far the robot works the same as the new robot. I will update this readme if there ever is a significant change.


# About the python files
## robot_navigation.py
This is the main python file used to navigate point-to-point or between rows. It is same as the new robot file

## video_navigation.py
Functions identically to the new robot file. Only difference is some algorithms are a bit older.

## robot_esp_control.py
Controls the ESP8266 (not ESP32s). Functions in a similar way.

## robot_website.py
Controls the website, basically the same as the new one

## nav_functions.py
Same as new one

# Hardware
# Frame
Built with 8020. Contact me for the CAD model and pictures if you want to see.

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
 
