
from re import L
import serial
import serial.tools.list_ports
import time
import pynmea2
import nav_functions
from threading import Thread, Lock
from io import BufferedReader
from pyubx2 import (
    UBXMessage,
    UBXReader,
    POLL,
    UBX_MSGIDS,
)


class Gps():
    def __init__(self, robot, mainGpsPort, verbose = False):
        """
        This class for the GPS board. It continually looks for position and accuracy of the robot
        """
        self.robot = robot
        self.running = True
        self.verbose = verbose
        self.headingDeviceConnected = False
        self.headingCoords = [0,0]
        self.headingFixLevel = 0
        self.mainGpsPort = mainGpsPort

        self.debugOptions = {"heading board connected": [False, 0], 
                            "SIV": [0, 0], 
                            "RTK signal available": [False, 0],
                            "RTK signal used": [False, 0]}

        self.t = 0
        self.ht = time.time()


        """
        Connection numbers:
        0: no fix
        1: dead reckoning only
        2: 2D-fix
        3: 3D-fix
        4: GNSS + dead reckoning combined
        5: time only fix
        """
        self.connectionType = -1


    def begin(self):
        """
        Sets up the GPS serial port and begins the read and write threads

        returns
            boolean - whether it successfully set up
        """
        print("setting up GPS")

        if self.mainGpsPort == "none":
            print("no GPS Connected to computer")
            return False
        # self.device = serial.Serial(port='/dev/ttyUSB0', baudrate=38400, timeout=1)
        try:
            self.device = serial.Serial(port=self.mainGpsPort, baudrate=115200, timeout=1)
        except:
            print("unable to connect to GPS, maybe check permissions?", self.mainGpsPort)
            return False
        time.sleep(0.5)
        print("connected to main gps on port", self.mainGpsPort)

           # create UBXReader instance, reading only UBX messages
        ubr = UBXReader(BufferedReader(self.device))
        serial_lock = Lock()


        self.rxThread = Thread(target=self.readUBX, args=[self.device, serial_lock, ubr])
        self.rxThread.start()


        return True
    


    def endGPS(self):
        """
            joins the read thread
        """
        self.running = False
        self.rxThread.join()

        return True


    def readUBX(self, stream, lock, ubxreader):
        """
        reads, parses and prints out incoming UBX and NMEA messages
        """
    # pylint: disable=unused-variable, broad-except

        while self.robot.notCtrlC:
            if stream.in_waiting:
                try:
                    lock.acquire()
                    (raw_data, parsed_data) = ubxreader.read()
                    lock.release()

                    if parsed_data:
                        # print(parsed_data.identity)
                        # print(parsed_data)
                        # print("")

                        # print(parsed_data.identity)


                        if parsed_data.identity == "NAV-RELPOSNED": #hasattr(parsed_data, "relPosHeading"):
                            # print("identity")#, #parsed_data.identity)
                            

                            headingAccuracy = parsed_data.accHeading
                            heading = parsed_data.relPosHeading
                            self.robot.trueHeading = parsed_data.relPosHeading # add a constant if the antennas aren't in line with the robot
                            self.robot.trueHeading %= 360


                            if headingAccuracy != 0.0 or heading != 0: # good heading
                                self.robot.headingAccuracy = headingAccuracy
                            else:
                                self.robot.headingAccuracy = 360
                                self.robot.trueHeading = 0

                            if self.verbose:
                                print("heading", self.robot.trueHeading, "accuracy:", self.robot.headingAccuracy)
                                print("")

                            self.robot.lastHeadingTime = time.time()

                        if parsed_data.identity == "NAV-PVT": #hasattr(parsed_data, "lon"):
                            # if parsed_data.identity == "NAV-PVT":
                            #     print("nav pvt data type")
                            if self.verbose:
                                print("coords:", parsed_data.lon, parsed_data.lat, "fix type:", parsed_data.fixType, "accuracy:", parsed_data.hAcc, "mm")
                            self.robot.coords = [parsed_data.lat, parsed_data.lon]
                            # self.robot.connectionType = parsed_data.fixType
                            self.robot.gpsAccuracy = parsed_data.hAcc

    
                        if parsed_data.identity == "RXM-RTCM":
                            # RTK2B heading msg: <UBX(RXM-RTCM, version=2, crcFailed=0, msgUsed=2, subType=0, refStation=0, msgType=1230)>
                            # RTK base msg:      <UBX(RXM-RTCM, version=2, crcFailed=0, msgUsed=1, subType=1, refStation=0, msgType=4072)>

                            if parsed_data.msgType == 1230: # heading board id is 1230. I think it should be parsed_data.refStation so pyubx updates may break this in the future.
                                self.debugOptions["heading board connected"] = [True, int(time.time())]
                            else: # rtk base message
                                self.debugOptions["RTK signal available"] = [True, int(time.time())]

                                if parsed_data.msgUsed > 1:
                                    self.debugOptions["RTK signal used"] = [True, int(time.time())]
                                elif time.time() - self.debugOptions["RTK signal used"][1] > 2:
                                    self.debugOptions["RTK signal used"] = [False, int(time.time())]

                                                        
                        if parsed_data.identity == "GNGGA":
                            if self.verbose:
                                print("quality:", parsed_data.quality)
                            self.robot.connectionType = parsed_data.quality
                            self.debugOptions["SIV"] = [parsed_data.numSV, int(time.time())]
                            
                except:
                    pass


if __name__ == "__main__":
    class Blah:
        def __init__(self):
            self.coords = [-1,-1]
            self.heading = 0
            self.gpsAccuracy = 0
            self.connectionType = 0
            self.notCtrlC = True


    portsConnected = [tuple(p) for p in list(serial.tools.list_ports.comports())]

    mainGpsPort = "none"
    for i in portsConnected: # connect to all of the relevant ports

        if i[1] == "u-blox GNSS receiver":
            mainGpsPort = i[0]
        
    blah = Blah()

    gps = Gps(blah, mainGpsPort, verbose = True)    

    gps.begin()
