# ROS_field_robot
Files to run the robot with a Jetson Nano (originally Raspberry Pi) and soon ROS. The robot can autonomously travel to a specific area and then travel through rows given GPS destinations. It is capable carrying extra sensors to collect data in a field.

# Hardware
## Frame
The CAD files are still being finalized, but as long as it can hold the rest of the sensors, it should be fine.

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
 
