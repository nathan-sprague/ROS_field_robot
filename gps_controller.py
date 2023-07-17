
from re import L

import time
import pynmea2
import nav_functions
from threading import Thread, Lock
import datetime
import socket


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

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('192.168.3.1', 28784)
#        server_address = ('localhost', 28784)
        self.sock.connect(server_address)
       
        self.rxThread = Thread(target=self.read_gps)
        self.rxThread.start()
        print("done setting up")

        return True
    


    def endGPS(self):
        """
            joins the read thread
        """
        self.running = False
        self.rxThread.join()

        return True



    def read_gps(self):
        
        
        try:
            data = ""
            while True:
                chunk = self.sock.recv(1).decode('UTF-8')
                data += chunk

                if data and data[-1] == '\n':
                    try:
                        sentence = data.strip('\n\r')
                        
                        if sentence:
                            msg = pynmea2.parse(sentence)
                            try:
                                if isinstance(msg, pynmea2.types.talker.GGA):
                                    self.robot.coords["coords"] = [float(msg.latitude), float(msg.longitude)]
                                    self.robot.coords["time"] = time.time()
                                    self.robot.coords["fix"] = int(msg.gps_qual)
                                    if msg.gps_qual > 1:
                                        self.robot.coords["accuracy"] = 0.1
                                    elif msg.gps_qual == 1:
                                        self.robot.coords["accuracy"] = 10
                                    else:
                                        self.robot.coords["accuracy"] = 123456
                                    if self.verbose:
                                        print(f"Qual: {msg.gps_qual}, Lat: {msg.latitude}, long: {msg.longitude}, alt: {msg.altitude}")

                                if isinstance(msg, pynmea2.types.talker.HDT):
                                    self.robot.heading["heading"] = float(msg.heading)
                                    self.robot.heading["time"] = time.time()
                                    self.robot.heading["accuracy"] = 0.1
                                    if self.verbose:
                                        print(f"heading: {msg.heading}")
                            except:
                                print("bad")

                    except pynmea2.ParseError:
                        print("gps bad")
                    data = ""
        finally:
            self.sock.close()


if __name__ == "__main__":
    class Blah:
        def __init__(self):
            self.coords = {"coords":[-1,-1], "time":0, "fix":0, "accuracy":0}
            self.heading = {"heading":0, "time":0, "accuracy":0}



    blah = Blah()

    gps = Gps(blah, verbose = True)    

    gps.begin()
