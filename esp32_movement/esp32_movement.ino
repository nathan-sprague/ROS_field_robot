/*
  This program controls all 4 motors through both serial and radio.
  What it does:
    -Calculates and reports speed through serial
    -Reads input PWM signal from the radio
    -Drives each motor independently to make them go at their specified speed


  This program only works for the ESP32.
  In order to control all 4 motors independently, it needs 10 input pins and 4 output pins.
  Only the ESP32 has a sufficient amount of GPIO pins.
*/



/*
  TO-DOs:
  general movement:
    verify main loop
  Serial control:
    enable proper communication (single letter commands)
   PID control:
    verify control
    enable "D" control
    fix all of the situation controls, like stuck
   radio control:
    verify

*/

#include "MotorController.h"


#define NUM_MOTORS 4

// define all the pins
const byte forwardRadioPin = 12;
const byte turnRadioPin = 13;


const byte rfEncoderPin1 = 2;
const byte rfEncoderPin2 = 4;

const byte rbEncoderPin1 = 5;
const byte rbEncoderPin2 = 18;

const byte lfEncoderPin1 = 19;
const byte lfEncoderPin2 = 21;

const byte lbEncoderPin1 = 22;
const byte lbEncoderPin2 = 23;


const byte rfWheelPin = 13;
const byte rbWheelPin = 12;
const byte lfWheelPin = 14;
const byte lbWheelPin = 27;



byte controlType = 1; //  movement is controlled by radio or serial. 0 = radio; 1 = serial

bool stopNow = false;


// interrupt for encoders
void ICACHE_RAM_ATTR handleInterrupt();

// various timers
unsigned long lastMoveTime = 0;
unsigned long lastPrintTime = 0;

// 4 motors: rf, rb, lf, lb
MotorController motors[4];

// speeds (mph)
float targetSpeeds[] = {0, 0, 0, 0}; // rf, rb, lf, lb



void interrupt1() {
  motors[0].updateEncoder1();
}

void interrupt2() {
  motors[0].updateEncoder2();
}








void setup() {
  Serial.begin(115200);

  Serial.println("setting up radio");

  setupRadioPins();


  Serial.println("adding interrupts");

  // set interrupt to switch to radio control if there is a radio signal
  //  attachInterrupt(digitalPinToInterrupt(forwardRadioPin), setRadioControl, CHANGE);
  //    attachInterrupt(digitalPinToInterrupt(turnRadioPin), setRadioControl, CHANGE);


  Serial.println("setting motor pins");
  // set pins for rf motor
  motors[0].encoderPin1 = rfEncoderPin1;
  motors[0].encoderPin2 = rfEncoderPin2;
  motors[0].pwmPin = rfWheelPin;
  motors[0].motorChannel = 1;
  motors[0].setupPins();
  attachInterrupt(digitalPinToInterrupt(rfEncoderPin1), interrupt1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(rfEncoderPin2), interrupt2, CHANGE);


  // set pins for rb motor
  motors[1].encoderPin1 = rbEncoderPin1;
  motors[1].encoderPin2 = rbEncoderPin2;
  motors[1].pwmPin = rbWheelPin;
  motors[1].motorChannel = 2;
  motors[1].setupPins();

  // set pins for lf motor
  motors[2].encoderPin1 = lfEncoderPin1;
  motors[2].encoderPin2 = lfEncoderPin2;
  motors[2].pwmPin = lfWheelPin;
  motors[2].motorChannel = 3;
  motors[2].setupPins();

  // set pins for lb motor
  motors[3].encoderPin1 = lbEncoderPin1;
  motors[3].encoderPin2 = lbEncoderPin2;
  motors[3].pwmPin = lbWheelPin;
  motors[3].motorChannel = 4;
  motors[3].setupPins();



  Serial.println("done");

  delay(1000);
}



void loop() {
  if (controlType == 0) { // radio control

    radioControl();


    for (int i = 0; i < NUM_MOTORS; i++) {
      motors[i].pwmOut = targetSpeeds[i];
    }



  } else { // serial control

    processSerial(); // check if there are any serial commands


    // find ideal pwm of each motor
    //    for (int i = 0; i < NUM_MOTORS; i++) {
    //      motors[i].pwmOut = motors[i].findIdealPWM(targetSpeeds[i]);
    //    }

  }


  // estimate, print, and set speed of each motor
  for (int i = 0; i < NUM_MOTORS; i++) {

    //    motors[i].calculateSpeed();

    //    motors[i].setMotorSpeed(stopNow);

  }

  if (millis() - 500 > lastPrintTime) {
    lastPrintTime = millis();
    Serial.println("current speeds: " + String(motors[0].currentSpeed, 2) + ", " + String(motors[2].currentSpeed, 2));
    Serial.println("pwm speeds: " + String(motors[0].pwmOut) + ", " + String(motors[2].pwmOut));
    Serial.println("target speeds: " + String(targetSpeeds[0]) + ", " + String(targetSpeeds[2]) + "\n");
  }


}
