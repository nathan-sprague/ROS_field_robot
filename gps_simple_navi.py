from gps import *
import math
import time
import numpy as np
from numpy.linalg import norm
from geographiclib.geodesic import Geodesic
import smbus  
from time import sleep  
import math

#some MPU6050 Registers and their Address
Register_A     = 0              #Address of Configuration register A
Register_B     = 0x01           #Address of configuration register B
Register_mode  = 0x02           #Address of mode register

X_axis_H    = 0x03              #Address of X-axis MSB data register
Z_axis_H    = 0x05              #Address of Z-axis MSB data register
Y_axis_H    = 0x07              #Address of Y-axis MSB data register
declination = -0.084352         #define declination angle of location where measurement going to be done
pi          = 3.14159265359     #define pi value

bus = smbus.SMBus(1)  # or bus = smbus.SMBus(0) for older version boards
Device_Address = 0x1e   # HMC5883L magnetometer device address

#geo coordinates of destination
dest_lat = 40.4214
dest_lon = -86.9202


#initialize magcompass
def Magnetometer_Init():
        
        #write to Configuration Register A
        bus.write_byte_data(Device_Address, Register_A, 0x70)

        #Write to Configuration Register B for gain
        bus.write_byte_data(Device_Address, Register_B, 0xa0)

        #Write to mode Register for selecting mode
        bus.write_byte_data(Device_Address, Register_mode, 0)

#format raw mag data
def read_raw_data(addr):
    
        #Read raw 16-bit value
        high = bus.read_byte_data(Device_Address, addr)
        low = bus.read_byte_data(Device_Address, addr+1)

        #concatenate higher and lower value
        value = ((high << 8) | low)

        #to get signed value from module
        if(value > 32768):
            value = value - 65536
        return value

#code to get data from gps
def NaviData(dest_lat,dest_lon):
    
    gpsd = gps(mode=WATCH_ENABLE|WATCH_NEWSTYLE)

    while True:
        
        global lat,lon,brng
        
        report = gpsd.next() #
        if report['class'] == 'TPV':  
            lat = getattr(report,'lat',0.0)
            lon = getattr(report,'lon',0.0)
            
            #Vector Calculation for computing Azimutj
            dest_vec = np.array([np.cos(dest_lat)*np.cos(dest_lon),np.cos(dest_lat)*np.sin(dest_lon),np.sin(dest_lat)])
            curr_vec = np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
            normal = np.array([0,0,1])
            normal_curr_n = np.cross(curr_vec,normal)
            normal_curr_dest = np.cross(curr_vec,dest_vec)
            angle = np.arccos(np.dot(norm(normal_curr_n),norm(normal_curr_dest))) * 180 / math.pi
            manual_azimuth = math.atan2(np.sin(dest_lon-lon)*np.cos(dest_lat),np.cos(lat)*np.sin(dest_lat)-np.sin(lat)*np.cos(dest_lat)*np.cos(dest_lon-lon)) * 180/math.pi
            
            #Get Azimuth from geographiclibary
            brng = Geodesic.WGS84.Inverse(lat, lon, dest_lat, dest_lon)['azi1']
            data = "lat: " + str(lat) + " " + "lon: " + str(lon) + " azimuth: " + str(brng)
            print(data)
            return brng
            
            
#code for magnectic heading
def MagneticHeading():
    
    global HeadingAngle

    #Read Accelerometer raw value
    x = read_raw_data(X_axis_H)
    z = read_raw_data(Z_axis_H)
    y = read_raw_data(Y_axis_H)

    heading = math.atan2(y, x) + declination
        
    #Due to declination check for >360 degree
    if(heading > 2*pi):
            heading = heading - 2*pi

    #check for sign
    if(heading < 0):
            heading = heading + 2*pi

    #convert into angle
    HeadingAngle = int(heading * 180/pi)
    
    print ("Heading Angle = %d°" %HeadingAngle)

#send some driving parameter
def DriveSignal(brng,HeadingAngle):
    
    global AngVec,LinVec
    
    #angular and linear velocity
    AngleDiff = HeadingAngle - brng
    if AngleDiff > 0:
        
        #turn left
        AngVec = 10
        print("turn left")
    elif AngleDiff < 0:
        
        #turn right:
        AngVec = -10
        print("turn right")
    else:
        
        #go straight
        LinVec = 5

#main code
if __name__ == "__main__":
    
    Magnetometer_Init() # initialize HMC5883L magnetometer
    print ("MagCompass Ready")

    while True:
        MagneticHeading()
        NaviData(dest_lat,dest_lon)
        DriveSignal(brng,HeadingAngle)
