
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

        # self.device = serial.Serial(port='/dev/ttyUSB0', baudrate=38400, timeout=1)
        try:
            self.device = serial.Serial(port=self.mainGpsPort, baudrate=115200, timeout=1)
        except:
            print("unable to connect to GPS")
            return False
        time.sleep(0.5)
        print("connected to main gps on port", self.mainGpsPort)

           # create UBXReader instance, reading only UBX messages
        ubr = UBXReader(BufferedReader(self.device), protfilter=2)
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
        reads, parses and prints out incoming UBX messages
        """
    # pylint: disable=unused-variable, broad-except

        while self.robot.notCtrlC:
            if stream.in_waiting:
                try:
                    lock.acquire()
                    (raw_data, parsed_data) = ubxreader.read()
                    lock.release()
                    if parsed_data:
                        # print(parsed_data)

                        if hasattr(parsed_data, "relPosHeading"):
                            if self.verbose:
                                print("heading", parsed_data.relPosHeading, "accuracy:", parsed_data.accHeading)
                            headingAccuracy = parsed_data.accHeading
                            if headingAccuracy != 0.0: # good heading
                                self.robot.trueHeading = parsed_data.relPosHeading
                                self.robot.headingAccuracy = headingAccuracy
                            else:
                                self.robot.headingAccuracy = 360
                            self.robot.lastHeadingTime = time.time()

                            print("")
                        if hasattr(parsed_data, "lon"):
                            if self.verbose:
                                print("coords:", parsed_data.lon, parsed_data.lat, "fix type:", parsed_data.fixType, "accuracy:", parsed_data.hAcc, "mm")
                            self.robot.coords = [parsed_data.lat, parsed_data.lon]
                            self.robot.connectionType = parsed_data.fixType
                            self.robot.gpsAccuracy = parsed_data.hAcc
                            
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

    headingPort = False
    mainGpsPort = "none"
    for i in portsConnected: # connect to all of the relevant ports

        if i[1] == "u-blox GNSS receiver":
            mainGpsPort = i[0]
        
    blah = Blah()

    gps = Gps(blah, mainGpsPort, verbose = True)    

    gps.begin()
