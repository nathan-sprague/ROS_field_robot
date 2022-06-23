
import serial
import serial.tools.list_ports
import time
from threading import Thread

class Esp():
    def __init__(self, robot, portName):
        """
        This class is associated with a single ESP. It uses parameters given by the robot's targets and sends the message to the esp32s.
    
        Parameters:
            robot - robot object with all the parameters
            portName - The port number that the ESP32 is on. It is also can be used as a unique identifier. 

        Returns:
        nothing
        """
        self.robot = robot

        self.portName = portName

        self.espType = "unknown"

        # These are the messages to send (dictionary)
        # the format is:
        #   {"Character prefix": [value to send, whether it was refreshed (needs to be sent)]}
        # The prefix is tells the ESP32 what kind of message it is, and the boolean tag specifying xwhether it should be sent prevents too much serial traffic
        self.messagesToSend = {"l": ["", True], "r": ["", True], "g":["", True]}

        # this is the serial device that you use to read and write things. It is intialized in the begin function. For now it is just a boolean
        self.device = False

        # Status of the ESP32
        self.stopped = False


    def begin(self):
        """
        Sets up the ESP32 serial port and begins the read and write threads

        returns
            boolean - whether it successfully set up
        """
        print("setting up esp")
        try:
            self.device = serial.Serial(port=self.portName, baudrate=115200, timeout=.1)
        except:
            return False
        time.sleep(0.5)
        self.rxThread = Thread(target=self.readSerial)
        self.txThread = Thread(target=self.sendSerial)
        self.rxThread.start()
        self.txThread.start()
        return True
    


    def endEsp(self):
        """
            joins the read and write threads and tells the ESP to stop
        """

        self.rxThread.join()

        self.txThread.join()

        self.device.write(bytes("s", 'utf-8'))

        return True


    def readSerial(self):
        """
            Read the incoming serial messages from the ESP32.
            Process any incoming messages.
            Repeat this indefinitely (should be a thread)
        """

        while self.robot.notCtrlC:
            line = self.device.readline()
            self.processSerial(line)
            time.sleep(0.1)


    def processSerial(self, message_bin):
        """
        process the incoming serial message and do the appropriate actions

        It still has some old ids and commands that are no longer used by the robot, like the role of the ESP.
            I may add it back in someday so I won't remove it yet.

        Parameters:
            message_bin - a message with some characters at the beginning and end. It is technically a string, not binary

        Returns:
            nothing
        """

        message = str(message_bin)
        if len(message_bin) > 0:

            try:
                message = message[2:-5] # the binary message has some extra things at the beginning and end
                msgType = message[0] # first character shows message type

            except: # strange message format
                return

            # print("got message:", message)

            if msgType == ".": # suppressed print statement
                
                return

            elif msgType == "-":

                keyName = message[1] # since the 1st character was a minus, the 2nd character is the ID character and the following characters are the value
              #  print(self.messagesToSend[keyName])
                try:
                    if self.messagesToSend[keyName][0] == "" or int(float(self.messagesToSend[keyName][0])) == int(float(message[2::])): # ESP correctly got the message we are trying to send
                        self.messagesToSend[keyName][1] = False
                        # print("got it right")

                    else: # esp incorrectly got the message we are trying to send
                        # print("wrong", self.messagesToSend[keyName][0], message[2::])
                        pass
                except:
                    pass
                    # print("wrong type", self.messagesToSend[keyName][0], message[2::])
                return


            if msgType == "s": # The ESP32 is stopped. No more analysis necessary
               self.stopped = True
               return

            # convert the message to a float
            try:
                res = float(message[1::])
            except:
                if len(message)>3:
                    firstChars = message[0:3]
                else:
                    firstChars = ""

                # when the ESP32 restarts, it sometimes writes these phrases through serial. Ignore them.
                restartChars = ["chk", "csu", "v00", "~ld", "loa", "tai"] 

                if firstChars == " et": # the first characters the ESP32 prints when it restarts
                    print(self.espType + " restarting")
                elif firstChars not in restartChars:
                    print("invalid message from " + self.espType + ": " + message)
                return


            # analyze the message based on the prefix

            if msgType == "x":  # compass latitude
                
                self.robot.coords2[0] = res
                # self.robot.lastUpdatedDist = self.robot.distanceTraveled

            elif msgType == "y":  # compass longitude
            
                self.robot.coords2[1] = res
                # self.robot.lastUpdatedDist = self.robot.distanceTraveled

            elif msgType == "l":  # left wheel speed
                self.robot.realSpeed[0] = res

            elif msgType == "r":  # right wheel speed
                self.robot.realSpeed[1] = res
#                print("real speeed: ", self.robot.realSpeed[1])

            elif msgType == "h": # heading from BNO
                self.robot.gyroHeading = res
                

            elif msgType == "a": # accuracy of GPS
                pass
                # self.robot.gpsAccuracy = res
                # headingCorrection = res
            
            elif msgType == "s": # stopped
                self.stopped = res
            
            elif msgType == "o": # An error occured represented by a number. Deal with it elsewhere (eventually)
                pass

            elif msgType == "e": # role of ESP
                espTypes = ["unknown", "movement", "access point"]
                if res < len(espTypes):
                    if espTypes[int(res)] != self.espType:
                        self.setESPType(espTypes[int(res)])
    
        


    def setESPType(self, espType):
        """
        Sets up the messages associated to send the ESP32 depending on the type of ESP it is.
        """
#        if espType == "speed":
            # send:
                # left target speed
                # right target speed
                # stop
                # go
        self.messagesToSend = {"l": ["0", False], "r": ["0", False], "s": ["", False], "g": ["", False], "r": ["", False], "l": ["", False]}

            # nothing to send the ap esp32
        if espType == "access point":
            self.messagesToSend = {}
        
        print("set up esp " + espType)
        self.espType = espType
 

    def sendSerial(self):
        """
        Repeatedly sends the messages in the messages to send dictionary
        This is effectively an infinite loop like the readSerial function, so put this in a thread.
    
        """
        # return
        while self.robot.notCtrlC:


            if self.device != False: # check if the device is real

                self.updateMessages() # refresh the dictionary of messages to send

                messagesSent = 0 # how many messages from the dictionary are sent (counter used for debugging)

                for prefix in self.messagesToSend:

                    # check to see if the message was sent in the past. Dont bother to send the same command twice (avoids serial traffic)
                    if self.messagesToSend[prefix][1]: 

                         # send the prefix and value to the ESP32
                        msg = prefix + self.messagesToSend[prefix][0]
                        self.device.write(bytes(msg, 'utf-8'))
                 #       print("sending message:", msg)
                        messagesSent+=1
                        time.sleep(0.2)

                if messagesSent == 0:
                
                    # send an empty message to prevent the ESP from assuming an error in communication
                    self.device.write(bytes(".", 'utf-8'))
                    time.sleep(0.2)
   

    def restart(self):
        """
        simply send the command to restart the ESP32.
        This is useful if the ESP32 is acting up or something is not properly connecting
        """

        self.device.write(bytes("r", 'utf-8'))


    def updateMessages(self):
        """
        Updates the messages to send based on parameters from the robot object given when this object was initialized.
        Checks to see if the messages are different from what was sent before

        """

        # Set the various message names to what the robot is targeting, namely the wheel speed
        fullMessages = {"l": str(int(self.robot.targetSpeed[0]*100)/100), "r": str(int(self.robot.targetSpeed[1]*100)/100), "s": "", "g": ""}
        # fullMessages = {"l": 1, "r": 1}

        # manually set he message if the robot is told to stop or go.
        # if self.robot.stopNow:
        #     self.messagesToSend["s"][1] = True
            
        # elif self.stopped:
        #     self.stopped = False 
        #     self.messagesToSend["g"][1] = True


        # check to see if the messages the ESP is sending is different from what was sent earlier. 
        for i in self.messagesToSend:
            if self.messagesToSend[i][0] != fullMessages[i]:
                self.messagesToSend[i][0] = str(fullMessages[i])
                self.messagesToSend[i][1] = True



if __name__ == "__main__":

    class blah:
        def __init__(self):
            self.notCtrlC = True
            self.targetSpeed = [0,0]
            self.wheelSpeed = [0,0]
            self.heading = 0
            self.gyroHeading = 0
    b = blah()
    espList = [] # list of objects
    i=0
    portsConnected = [tuple(p) for p in list(serial.tools.list_ports.comports())]

    for i in portsConnected: # connect to all of the relevant ports
        if i[1] == "CP2102 USB to UART Bridge Controller - CP2102 USB to UART Bridge Controller":
            print("esp on port", i[0])
            esp = Esp(b, i[0]) # make a new ESP object and give it this class and the port name
            if esp.begin():
                espList += [esp] 
      

    print("set up", len(espList), "ESPs")
