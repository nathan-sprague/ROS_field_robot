
import serial
import serial.tools.list_ports
import time
from threading import Thread
import math

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

        self.lastSent = {}
        self.lastSentTime = 0
        self.messagesToSend = {}

        # this is the serial device that you use to read and write things. It is intialized in the begin function. For now it is just a boolean
        self.device = False

        # Status of the ESP32
        self.stopped = False



    def begin(self, doTx=True):
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
        self.rxThread.start()
        if doTx:
            self.txThread = Thread(target=self.sendSerial)
            self.txThread.start()
        else:
            self.txThread = False
        
        
        return True
    


    def endEsp(self):
        """
            joins the read and write threads and tells the ESP to stop
        """

        self.rxThread.join()

        if self.txThread != False:
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
            if self.processSerial(line):
                pass
            else:
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

    def processSerial(self, message_bin):
        """
        process the incoming serial message and do the appropriate actions

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
                if keyName in self.lastSent:
                    if int(float(self.lastSent[keyName][0])) == int(float(message[2::])): # ESP correctly got the message we are trying to send
                        self.lastSent[keyName][1] = True
                        self.lastSent[keyName][3] = time.time()

                   #     print("ESP got correct message", message[1::])
                        # print(self.lastSent)

                    else: # esp incorrectly got the message we are trying to send
                 #       print("esp got the wrong message. expected", keyName, self.lastSent[keyName][0], "got", keyName, message[2::])
                        self.lastSent.pop(keyName) # remove from list of sent messages to send it again

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
            if msgType == "l":  # left wheel speed
                self.robot.realSpeed[0] = res

            elif msgType == "r":  # right wheel speed
                self.robot.realSpeed[1] = res
#                print("real speeed: ", self.robot.realSpeed[1])

            
            elif msgType == "o": # An error occured represented by a number. Deal with it elsewhere (todo)
                pass
            return True
        else:
            return False




    def sendSerial(self):
        """
        sends relevant messages to the ESP32
        """
        while self.robot.notCtrlC:
            messagesToSend = self.updateMessages()
            
            for prefix in messagesToSend:

                # check to see if the message was sent in the past. Dont bother to send the same command twice (avoids serial traffic)
            
                 # send the prefix and value to the ESP32
                msg = prefix + str(messagesToSend[prefix])
                self.device.write(bytes(msg, 'utf-8'))
                self.lastSent[prefix] = [messagesToSend[prefix], False, time.time(), 0] # [value, confirmed, sent time, response time]
                self.lastSentTime = time.time()
         #       print("sending message:", msg)
                time.sleep(0.05)

            if messagesToSend =={} and time.time()-self.lastSentTime > 0.5:
                # send an empty message to keep the ESP informed that there is a connection
                self.device.write(bytes(".", 'utf-8'))
                self.lastSentTime = time.time()
                # print("sending empty")
                time.sleep(0.05)
            elif messagesToSend == {}:
                time.sleep(0.05)
            # time.sleep(0.1)
         #   print("done sending messages")


    def updateMessages(self):
        """
        Updates the messages to send based on parameters from the robot object given when this object was initialized.
        Checks to see if the messages are different from what was sent before

        """

        # Set the various message names to what the robot is targeting, namely the wheel speed
        fullMessages = {"l": int(self.robot.targetSpeed[0]*100)/100, "r": int(self.robot.targetSpeed[1]*100)/100}
        messagesToSend = {}


        # check to see if the messages the ESP is sending is different from what was sent earlier. 
        for i in fullMessages:
            if i in self.lastSent:
                if self.lastSent[i][0] != fullMessages[i]:
                    # print("different", i, self.lastSent[i], fullMessages[i])
                    messagesToSend[i] = fullMessages[i]

                elif self.lastSent[i][1] == False and time.time()-self.lastSent[i][2] > 1:
                    # print("timeout")
                    messagesToSend[i] = fullMessages[i]

            else:
                # print("not there", i, fullMessages[i])
                messagesToSend[i] = fullMessages[i]

        return messagesToSend

     



if __name__ == "__main__":

    class FakeRobot:
        def __init__(self):
            self.notCtrlC = True
            self.targetSpeed = [0,0]
            self.realSpeed = [0,0]
            self.heading = 0
            self.gyroHeading = 0
            self.esp = ""
            self.observers = []
            self.testType = 2

        

        def run(self):
            while True:
                if self.testType == 1:
                    i=0
                    while i<2:
                        # self.targetSpeed = 5
                        self.targetSpeed[i] = (self.targetSpeed[i]+1)%4

                        i+=1
                    
                    timer = time.time()
                    self.targetSpeed = [self.targetSpeed[0], self.targetSpeed[1]]
                    print("changed")

                    speedupTime = 0
                    
                    while True:
                        try:
                            if self.esp.lastSent['l'][1] and self.esp.lastSent['r'][1]:
                                if self.esp.lastSent['l'][0] == self.targetSpeed[0] and self.esp.lastSent['r'][0] == self.targetSpeed[1]:
                                    if speedupTime == 0:
                                        speedupTime = time.time()
                                    if abs(self.targetSpeed[0]-self.realSpeed[0]) < 0.2 and abs(self.targetSpeed[0]-self.realSpeed[0]) < 0.2:
                                        break
                        except:
                            pass
                        time.sleep(0.01)
                    # print("target", self.targetSpeed, "real", [self.esp.lastSent['l'][0], self.esp.lastSent['r'][0]])
                    speedupTime = time.time()-speedupTime
                    print("test took", time.time()-timer, "s overall, ", self.esp.lastSent['l'][2]-timer, "s to send the message", self.esp.lastSent['l'][3]-self.esp.lastSent['l'][2], "s for esp to read message,", speedupTime, "to speed up")
                    print("")
                    time.sleep(2)

                elif self.testType == 2:
                    self.targetSpeed = [math.cos(time.time()/5)*2, math.cos(time.time()/5)*2]
                    time.sleep(0.2)
                    print(self.targetSpeed[0], self.targetSpeed[1], self.realSpeed[0], self.realSpeed[1])
                
                # print(self.targetSpeed, self.realSpeed)


    b = FakeRobot()
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
    if len(espList) > 0:
        b.esp = espList[0]
        b.run()

    
    
