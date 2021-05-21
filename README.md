# ROS_field_robot
Files to run the robot with a raspberry pi and ROS. The robot can autonomously travel through rows given various GPS destinations. It is capable of extra sensors to collect data in a field.

# Setup

## Software
We are using the 32-bit Raspberry Pi OS

As of now, we are not using ROS. However, we plan to use ROS Noetic on a Raspberry Pi 4. Use this tutorial to install it<br />
https://varhowto.com/install-ros-noetic-raspberry-pi-4/ <br />

## Components Necessary
### Compass
$ sudo apt-get install git i2c-tools python-smbus python3 python-pip python-virtualenv python3-setuptools
VCC -> 3.3V (Pin 1), GND -> GND (Pin 6), SCL -> GPIO3 (Pin 5), SDA -> GPIO2 (Pin 3) 
enable I2C


### GPS
$ pip3 install gps
$ pip3 install geographiclib

### Serial
(Already there)
