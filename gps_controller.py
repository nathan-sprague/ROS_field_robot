
from re import L
import serial
import time
from threading import Thread
import pynmea2
import nav_functions


class Gps():
    def __init__(self, robot, verbose = False):
        """
        This class for the GPS board. It continually looks for position and accuracy of the robot
        """
        self.robot = robot
        self.running = True
        self.verbose = verbose
        self.headingDeviceConnected = False
        self.headingCoords = [0,0]
        self.headingFixLevel = 0

        self.t = 0
        self.ht = 0


        """
        Connection numbers:
        -1: not connected to GPS board
        0: position not known
        1: position known without rtk
        2: DGPS
        3: N/A
        4: RTK float
        5: RTK fixed
        """
        self.connectionType = -1


    def begin(self):
        """
        Sets up the GPS serial port and begins the read and write threads

        returns
            boolean - whether it successfully set up
        """
        print("setting up GPS")

        # self.device = serial.Serial(port='/dev/ttyUSB0', baudrate=38400, timeout=1)
        self.device = serial.Serial(port='/dev/ttyACM1', baudrate=115200, timeout=1)
        time.sleep(0.5)
        self.rxThread = Thread(target=self.readSerial, args=[self.device, "main"])
        self.rxThread.start()

        try:
            self.headingDevice = serial.Serial(port='/dev/ttyUSB0', baudrate=460800, timeout=1)
            self.rxThread = Thread(target=self.readSerial, args=[self.headingDevice, "heading"])
            self.rxThread.start()
            self.headingDeviceConnected = True
            print("connected to heading gps")

        except:
            print("unable to connect to heading GPS")
    
        return True
    


    def endGPS(self):
        """
            joins the read thread
        """
        self.running = False
        self.rxThread.join()

        return True


    def readSerial(self, deviceName, deviceType):
        """
            Read the incoming serial messages from the GPS module.
            Process any incoming messages.
            Repeat this indefinitely (should be a thread)
        """

#        self.robot.coords = [40.4221268, -86.9161606]
        lastTime = 0
        while self.running:
            lineDecoded = False
            try:
                line = deviceName.readline()
                line = line.decode("utf-8")
                lineDecoded = True
            except:
                l = str(line)
                if "$GNRMC" in str(line):
                    i = 6
                    while i < len(l):
                        if l[i-6:i] == "$GNRMC":
                            line = l[i-6:-5]
                            #print(line)
                            lineDecoded = True
                            break
                        i+=1
            
            if lineDecoded:
                # print(line)
                self.parseGps(line, deviceType)
            # elif deviceType == "heading":
                # print(line)
 
            time.sleep(0.01)

    def parseGps(self, nmeaData, deviceType):
       # print(nmeaData)
#        self.robot.coords = [40.4214268, -86.9161606]
 #       return
        # if deviceType == "heading":
            # print(nmeaData)
        try:
           msg = pynmea2.parse(nmeaData)
        except:
          # print("couldn't parse data")
          
          return
        #if self.verbose:
        # print(nmeaData)
        if deviceType == "main":
            try:
                self.robot.coords[0] = float(msg.latitude)
                self.robot.coords[1] = float(msg.longitude)
                if self.verbose and self.t+1<time.time() and self.robot.connectionType > 1:
                    self.t = time.time()

                    # print("got pos", self.robot.coords)
            except:
                pass
                
            try:
                if int(msg.gps_qual) != self.robot.connectionType:
                   print("new main fix level", int(msg.gps_qual))
                self.robot.connectionType = int(msg.gps_qual)
                # print("got qual")
            except:
                pass
                
            try:
                self.robot.gpsAccuracy = float(msg.horizontal_dil)
                # print("got dil", self.robot.gpsAccuracy)
            except:
                pass

            if "$POLYA" in nmeaData:
                print("yo")
            
            if "$GNVTG" in nmeaData and not self.headingDeviceConnected:
                if self.verbose:
                    pass
                    # print("msg is", msg)
                    # print("heading obtainable", nmeaData)
           
                try:
                    self.robot.gpsHeading = float(msg.true_track)
                    self.robot.gpsHeadingAvailable = True
                    if self.verbose:
                        print("got single gps heading", self.robot.GPSheading)
                except:
                    self.robot.gpsHeadingAvailable = False
                  #  print("GPS heading not found")
        elif deviceType == "heading":
            try:
                self.headingCoords[0] = float(msg.latitude)
                self.headingCoords[1] = float(msg.longitude)
            
                try:
                    if int(msg.gps_qual) != self.headingFixLevel:
                       print("new heading fix level", int(msg.gps_qual))
                    self.headingFixLevel = int(msg.gps_qual)
                    # print("got qual")
                except:
                    pass

                if self.headingCoords[0] != 0 and self.robot.coords[0] != 0:
                    self.robot.heading = nav_functions.findAngleBetween(self.headingCoords, self.robot.coords) * 180 / 3.14159

                if self.verbose and self.ht+1<time.time() and self.headingFixLevel > 1 and self.robot.connectionType>1:
                    self.ht = time.time()
                    d = nav_functions.findDistBetween(self.headingCoords, self.robot.coords)

                    print("got heading pos:", self.robot.coords, self.headingCoords)
                    print("heading estimate:", self.robot.heading)
                    print("distance between antenna:", d)
                    print("")

            except:
                pass


  

if __name__ == "__main__":
    class Blah:
        def __init__(self):
            self.coords = [-1,-1]
            self.heading = 0
            self.gpsAccuracy = 0
            self.connectionType = 0
    blah = Blah()
    gps = Gps(blah, True)
    gps.begin()
