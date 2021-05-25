# importing
import tkinter
import random
import time
import pickle

runGPIO = False
try:
    import RPi.GPIO as GPIO

    runGPIO = True
    print("imported GPIO library")
except:
    print("cannot connect to GPIO")

import time


class CanvasController:
    def __init__(self, importedGPIO):
        self.motor = Motors(18, 33, importedGPIO)
        self.root = tkinter.Tk()
        self.canvas = tkinter.Canvas(self.root, bg=("white"), height=100, width=100)  # 9000,width=1200)
        self.root.grid()
        self.canvas.pack()
        self.get_commands()

    def get_commands(self):
        self.root.bind("<Key>", self.process_key)

    def process_key(self, event):

        if event.keysym == "Up":
            self.motor.y_speed = 1

        elif event.keysym == "Down":
            self.motor.y_speed = -1

        elif event.keysym == "Right":
            self.motor.x_speed = 1


        elif event.keysym == "Left":
            self.motor.x_speed = -1

        elif event.keysym == "space":
            self.motor.x_speed = 0
            self.motor.y_speed = 0

        elif event.keysym == "BackSpace":
            self.motor.destroy()
        else:
            print("unrecognized", event.keysym)
        self.motor.move()


class Motors:
    def __init__(self, motorPin, steerPin, runGPIO=False):
        self.steerPin = steerPin
        self.motorPin = motorPin
        self.runGPIO = runGPIO
        self.x_speed = 0
        self.y_speed = 0

        if self.runGPIO:
            GPIO.setmode(GPIO.BOARD)

            # set up motor
            GPIO.setup(self.motorPin, GPIO.OUT)
            GPIO.output(self.motorPin, GPIO.LOW)
            self.motorPWM = GPIO.PWM(self.motorPin, 100)  # Set Frequency
            self.motorPWM.start(0)  # Set the starting Duty Cycle

            # set up steering
            GPIO.setup(self.steerPin, GPIO.OUT)
            GPIO.output(self.steerPin, GPIO.LOW)
            self.steerPWM = GPIO.PWM(self.steerPin, 100)  # Set Frequency
            self.steerPWM.start(0)  # Set the starting Duty Cycle
            print("setup complete")

    def move(self):
        if self.runGPIO:
            print("moving", self.x_speed, self.y_speed)
            if self.y_speed > 0:
                self.motorPWM.ChangeDutyCycle(20)
            elif self.y_speed < 0:
                self.motorPWM.ChangeDutyCycle(10)
            else:
                self.motorPWM.ChangeDutyCycle(0)

            if self.x_speed > 0:
                self.steerPWM.ChangeDutyCycle(20)
            elif self.x_speed < 0:
                self.steerPWM.ChangeDutyCycle(10)
            else:
                self.steerPWM.ChangeDutyCycle(0)

    def destroy(self):
        if self.runGPIO:
            self.steerPWM.stop()
            self.motorPWM.stop()
            GPIO.output(self.steerPin, GPIO.LOW)
            GPIO.output(self.motorPin, GPIO.LOW)
            GPIO.cleanup()


if __name__ == '__main__':
    myController = CanvasController(runGPIO)
