
import serial
import time
from threading import Thread

class Esp():
    def __init__(self, robot, espNum):
        self.robot = robot
        self.espNum = espNum
        self.espType = "unknown"
        self.messagesToSend = {"l": ["", True], "g":["",False]}
        self.device = False
        self.stopped = False


    def begin(self):
        try:
            self.device = serial.Serial(port='/dev/ttyUSB' + str(self.espNum), baudrate=115200, timeout=.1)
        except:
            return False
        time.sleep(0.5)
        self.rxThread = Thread(target=self.readSerial)
        self.txThread = Thread(target=self.sendSerial)
        self.rxThread.start()
        self.txThread.start()
        return True
    
    def endEsp(self):

        self.rxThread.join()

        self.txThread.join()

        self.device.write(bytes("s", 'utf-8'))

        return True


    def readSerial(self):
        while self.robot.notCtrlC:
            line = self.device.readline()
            self.processSerial(line)
            time.sleep(0.1)


    def processSerial(self, message_bin):
        message = str(message_bin)
        if len(message_bin) > 0:

            try:
                message = message[2:-5] # the binary message has some extra things at the beginning and end
                msgType = message[0] # first character shows message type

            except: # strange message format
                return

           

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
            if msgType == "s":
             #  print(self.espType, "stopped")
               self.stopped = True
               return

            try:
                res = float(message[1::])
            except:
                if len(message)>3:
                    firstChars = message[0:3]
                else:
                    firstChars = ""
            #    print(firstChars)
                restartChars = ["chk", "csu", "v00", "~ld", "loa", "tai"]
                if firstChars == " et":
                    print(self.espType + " restarting")
                elif firstChars not in restartChars:
                    print("invalid message from " + self.espType + ": " + message)
                return

            if msgType == "x":  # compass latitude
                self.robot.coords[0] = res
                self.robot.lastUpdatedDist = self.robot.distanceTraveled

            elif msgType == "y":  # compass longitude
                self.robot.coords[1] = res
                self.robot.lastUpdatedDist = self.robot.distanceTraveled

            elif msgType == "w":  # wheel speed
                self.robot.wheelSpeed = res

            elif msgType == "d":  # distance
                self.robot.distanceTraveled = res
                self.robot.updateCoords()

            elif msgType == "a":  # steering angle
                self.robot.steeringAngle = res
                self.robot.steeringDirection = self.robot.heading + res

            elif msgType == "h": # heading from GPS
                self.robot.heading = res

            elif msgType == "c": # heading from compass
                pass
#                self.robot.heading = res

            elif msgType == "t": # accuracy of GPS
                self.robot.gpsAccuracy = res
            
            elif msgType == "s": # stopped
                self.stopped = res
            
            elif msgType == "o": # An error occured represented by a number. Deal with it elsewhere
                if [res, self] not in self.robot.errorList:
                    self.robot.errorList += [[res, self]]

            elif msgType == "e": # role of ESP
                espTypes = ["unknown", "steer", "speed", "access point"]
                if res < len(espTypes):
                    if espTypes[int(res)] != self.espType:
                        self.setESPType(espTypes[int(res)])
    
        


    def setESPType(self, espType):
        if espType == "steer":
            self.messagesToSend = {"p": ["0", False], "s": ["0", False], "g": ["", False], "r": ["", False], "l": ["", False]}
        
        if espType == "speed":
            self.messagesToSend = {"f": ["0", False], "m": ["0", False], "s": ["", False], "g": ["", False], "r": ["", False], "l": ["", False]}

        if espType == "access point":
            self.messagesToSend = {"l": ["", False], "s": ["", False], "g": ["", False], "r": ["", False]}
        
        print("set up esp " + espType)
        self.espType = espType
 

    def sendSerial(self):
        print("send serial")
        while self.robot.notCtrlC:
            if self.device != False:
                self.updateMessages()
                messagesSent = 0 
                for i in self.messagesToSend:
                    if self.messagesToSend[i][1]:
                        msg = i + self.messagesToSend[i][0]
                        self.device.write(bytes(msg, 'utf-8'))
                     
                        # print("sent", msg, "to", self.espType)
                        messagesSent+=1
                        time.sleep(0.2)

                if messagesSent == 0:
                    # send an empty message to prevent the ESP from assuming an error in communication
                    self.device.write(bytes(".", 'utf-8'))
                    time.sleep(0.2)
                    # print("sent empty")
              #      print(self.messagesToSend)

    def restart(self):
        self.device.write(bytes("r", 'utf-8'))


    def updateMessages(self):
        fullMessages = {"p": str(self.robot.targetWheelAngle), "f": str(self.robot.targetSpeed),
         "m": str(self.robot.targetMoveDist), "r": "", "s": "", "g": "", "l": ""}

        if self.robot.stopNow:
            self.messagesToSend["s"][1] = True
            
        elif self.stopped:
            self.stopped = False 
            self.messagesToSend["g"][1] = True


        for i in self.messagesToSend:
            if self.messagesToSend[i][0] != fullMessages[i]:
                self.messagesToSend[i][0] = str(fullMessages[i])
                self.messagesToSend[i][1] = True

