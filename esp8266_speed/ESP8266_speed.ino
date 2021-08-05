/*
  Pins labeled on ESP vs pins used by program
  static const uint8_t D0   = 16;
  static const uint8_t D1   = 5;
  static const uint8_t D2   = 4;
  static const uint8_t D3   = 0;
  static const uint8_t D4   = 2;
  static const uint8_t D5   = 14;
  static const uint8_t D6   = 12;
  static const uint8_t D7   = 13;
  static const uint8_t D8   = 15;
  static const uint8_t D9   = 3;
  static const uint8_t D10  = 1;
*/


#define DEBUGMODE false

// at least one of these should be true for the robot to move
#define USE_PROPORTIONAL true
#define USE_INTEGRATED false
#define USE_DERIVATIVE false


const int motorPin = 16;  // motor (D0)
const int hallPin = 5; // speed sensor pin 1 (D1)
const int otherHallPin = 4; // speed sensor pin 2 (D2)


unsigned long lastMsgTime = 0; // timer to prevent over-printing

const float wheelCircum = 13 * 2 * 3.14159;
const int numHoles = 16;

float smoothSpeed = 0; // speed used for all calculations. Smooths and averages speeds to limit outliers
float smoothDist = 0; // distance travelled since start in inches

// movement variables
bool setMovement = false;
long targetDist = 0;
long startDist = 0;
bool goingForward = true;

const float tickTimeTomph = (wheelCircum / 12.0) / numHoles * 1000.0 * 3600 / 5280; // constant for calculating wheel speed


// speed variables for the left wheel
long distTraveled = 0;
int lastRead = 0;
unsigned long lastHitTime = 0;
float wheelSpeed = 0;
unsigned int timeBetweenHoles = 0;


// speed variables for the right wheel
long distTraveledY = 0;
int lastReadY = 0;
unsigned long lastHitTimeY = 0;
float wheelSpeedY = 0;
unsigned int timeBetweenHolesY = 0;

// variables related to serial commands
unsigned long lastCommandTime = 0;
bool stopNow = false; // emergency stop
String stopReason = "Unknown";
bool manual = false; // set pwm manually, does not work right now
float targetSpeed = 0;
float startSpeed = 0;
bool reachedTarget = false;

float PWMSignal = 0; // pwm signal applied to

const int maxPWM = 200 - 0;
const int minPWM = 110 + 0;

unsigned long lastTargetChange = 0;
int pwmChangeTime = 100; // time to reassess the given PWM speed (milliseconds)

unsigned long stuckTime = 0;
bool stuck = false;

unsigned long lastTalkTime = 0;

// variables used for setting pwm
int atTarget = 0;
int lastTarget = 0;
float intError = 0;
int lastPWM = 0;

float acceleration = 0;
unsigned  long lastAccelerationTime = 0;



void processSerial() {
  /*
    Read the serial message and obey the given command.

    The serial is given as a single letter ID and a value. Letter IDs are listed below:

    Serial Message Prefixes:

    Informational:
    x: gps latitude
    y: gps longitude
    u: destination latitude
    v: destination longitude
    w: wheel speed
    d: distance
    a: steering angle
    h: heading

    Command:
    p: steering angle
    f: motor speed
    l: destination latitude
    t: destination longitude
    l: request type of esp

    Other:
    .: console log, the Pi/Nano ignore these messages
    -: acknowledge the message
    +: say the message is irrelevant to this microcontroller


  */
  if (Serial.available()) {


    if (DEBUGMODE) {
      Serial.println(".got message");
    }

    // get keyword
    char commandType = Serial.read();
    String serialMsg = "";


    while (Serial.available()) {
      char b = Serial.read();
      serialMsg += b;
    }

    if (DEBUGMODE) {
      Serial.println(serialMsg);
    }

    if (commandType == 'f') { //// set speed
      targetSpeed = serialMsg.toFloat();
      Serial.println("-f" + String(int(targetSpeed)));
      if (DEBUGMODE) {
        Serial.println(".target speed is set to " + serialMsg + ", was: " + String(targetSpeed));
      }
      //      if (targetSpeed>0){
      //        targetSpeed = 1.5;
      //      }
      setMovement = false;
      manual = false;


    } else if (commandType == 'z') { // manually give PWM
      if (DEBUGMODE) {
        Serial.println(".manual");
      }
      setMovement = false;
      manual = true;
      analogWrite(motorPin, serialMsg.toInt());
      PWMSignal = serialMsg.toInt();

      Serial.println("-z" + String(PWMSignal));

    } else if (commandType == 'l') { // requested esp type. Tell it this is speed
      Serial.println("e2");

    } else if (commandType == 's') { // emergency stop. Stops all PWM until explicitly told to continue
      if (DEBUGMODE) {
        Serial.println(".emergency stop");
        Serial.println("-s");
      }
      stopNow = true;
      stopReason = "Serial stop";

    } else if (commandType == 'g') { // go, used to continue after an emergency stop is performed
      if (DEBUGMODE) {
        Serial.println(".go");
      }
      Serial.println("-g");
      stuck = false;
      stopNow = false;
      stopReason = "Unknown";
    } else if (commandType == 'r') { // restart
      Serial.println(".restarting");
      Serial.println("-r");
      ESP.restart();

    } else if (commandType == 'm') { // move for a given distance
      int moveFor = serialMsg.toInt();
      targetDist = smoothDist + moveFor;
      startDist = smoothDist;
      setMovement = true;
      
    } else if (commandType == '.'){ // empty message, ignore
      
    } else { // irrelevant or unrecognized message, ask pi/nano not to send it again
      Serial.println("+" + commandType);
    }
  }
}

unsigned long ltt = 0;
float lastSpeed = 0;

void getWheelSpeed() {
  /*
    This function calculates the vehicle speed
    Execute this function as frequently as possible to detect if the hall sensor runs over a hole.

  */
  bool speedChange =  false; // something happened, like a hole detected, so the moving average functions should be run
  int  x = digitalRead(hallPin);


  if (lastRead != x && x == 0 && millis() - lastHitTime > 0) { // There was a change in what it sensed and did sense something
    timeBetweenHoles = millis() - lastHitTime;

    //    float rpm = 60.0 / numHoles / (timeBetweenHoles / 1000.0);

    wheelSpeed = tickTimeTomph / timeBetweenHoles; // mph
    if (goingForward) {
      distTraveled += wheelCircum / numHoles;
    } else {
      distTraveled -= wheelCircum / numHoles;
    }
    lastHitTime = millis();
    speedChange = true;

  } else if (timeBetweenHoles < millis() - lastHitTime) { // if it is taking longer than before to reach a hole
    wheelSpeed = tickTimeTomph / (millis() - lastHitTime); // use that value for speed
    wheelSpeed = int(wheelSpeed); // the value is likely not exact so get rid of floating point precision
    speedChange = true;
  }

  lastRead = x; // use reading to detect change next time


  // repeat above for other wheels
  int  y = digitalRead(otherHallPin);

  if (lastReadY != y && y == 0 && millis() - lastHitTimeY > 0) {

    timeBetweenHolesY = millis() - lastHitTimeY;
    wheelSpeedY = tickTimeTomph / timeBetweenHolesY; // mph

    if (goingForward) {
      distTraveledY += wheelCircum / numHoles;
    } else {
      distTraveledY -= wheelCircum / numHoles;
    }
    lastHitTimeY = millis();
    speedChange = true;

  } else if (timeBetweenHolesY < millis() - lastHitTimeY) {

    wheelSpeedY = tickTimeTomph / (millis() - lastHitTimeY);
    wheelSpeedY = int(wheelSpeedY);
    speedChange = true;
  }
  lastReadY = y;

  // if going backwards, negate the wheel speeds
  if (!goingForward) {
    wheelSpeed = abs(wheelSpeed) * -1;
    wheelSpeedY = abs(wheelSpeedY) * -1;
  }

  // sanity check: if the smooth speed is way to high, bring it to the most recent value
  if (abs(smoothSpeed) > 100) {
    Serial.println(".wheel speed way too high");
    smoothSpeed = (wheelSpeed + wheelSpeedY) / 2;
  }

  if (speedChange) { // no need to reasses everything unless there is a change in speed

    // units are miles/(hour * second)

    if (lastAccelerationTime + 50 < millis()) {
      acceleration = (acceleration * 2 + (((wheelSpeed + wheelSpeedY) / 2) - lastSpeed) / ((millis() - lastAccelerationTime) / 1000.0) ) / 3;
      lastSpeed = smoothSpeed;
      lastAccelerationTime = millis();

      //      Serial.println(".acceleration: "  + String(acceleration, 5));
    }

    // get average speed between the wheels and past speeds. #s 3 and 5 were arbitrarily chosen
    smoothSpeed = (smoothSpeed * 3 + (wheelSpeed + wheelSpeedY)) / 5;
    smoothDist = (distTraveled + distTraveledY) / 2;





    // Direction may change if the speed is zero
    if (smoothSpeed > -0.1 && smoothSpeed < 0.1) { // cant compare it directly to zero because a float

      if ((PWMSignal >= 155 || int(PWMSignal) == 0) && targetSpeed > 0) { // signal tells it to go forwards, go forwards
        if (lastMsgTime + 500 < millis()) {
          if (DEBUGMODE) {
            Serial.println(".stopped and going forwards");
          }
        }
        goingForward = true;

      } else { // it is going backwards
        if (lastMsgTime + 500 < millis()) {
          if (DEBUGMODE) {
            Serial.println(".stopped and going backwards");
          }
        }
        goingForward = false;
      }
    }
  }

  // print results every 0.5 seconds
  if (lastMsgTime + 500 < millis()) {
    if (DEBUGMODE) {
      Serial.println(".vehicle speed: " + String(wheelSpeed) + ", " + String(wheelSpeedY) + " (" + String(smoothSpeed) + ")");
    } else {
      Serial.println("d" + String(smoothDist));
      Serial.println("w" + String(smoothSpeed));
    }
    lastMsgTime = millis();
  }
}


int lastGoodPWM = 0;

void setMotorSpeed() {
  /*
    This function sets finds the best PWM value  to move the robot at the desired speed using PID logic.

    With no wheel resistance, these PWM values correspond to the following speeds (mph)
    200 -> 14.5
    190 -> 9.7
    180 -> 7.3
    170 -> 4.1
    160 -> 1.1
    155 -> 0
    150 -> -0.4
    140 -> -2.9
    130 -> -5.8
    120 -> -9.7
    110 -> -14.5
  */

  if (lastTargetChange + pwmChangeTime < millis()) { // don't set the speed too often or else it may surge
    lastTargetChange = millis();

    if (abs(lastTarget - targetSpeed) > 0.1) { // commanded target changed
      
      lastGoodPWM = 155 + targetSpeed / 14.5 * (90.0 / 2); // estimation for correct pwm

      lastTarget = int(targetSpeed);
      startSpeed = smoothSpeed;
      reachedTarget = false;
      if (DEBUGMODE) {
        Serial.println(".target changed");
      }
      // this is to figure out the whether you passed the speed and calculate the integrated error and correct for it.
      intError = 0;
      if (smoothSpeed > targetSpeed)
        atTarget = 1;
      else {
        atTarget = -1;
      }
    }
  } else {
    return;
  }

  //  Serial.println(".wheel target speed: " + String(targetSpeed));

  if (targetSpeed == 0) {
    analogWrite(motorPin, 0);
    PWMSignal = 0;
    return;

  } else if (smoothSpeed != targetSpeed) {



    float pwmChange = 0;



    // proportional part of PID
    if (USE_PROPORTIONAL) {


      if (abs(smoothSpeed) - abs(targetSpeed) < 2) {  // your speed is not much more than the target

        // convert difference in speed to a difference in PWM
        float PWMdif = ((targetSpeed - smoothSpeed) / 14.5) * (90.0 / 2); // map (-14.5)-(14.5) to (110)-(200)

        const int steps = 10; // change this number to modify how quickly and how much the PWM is updated

        // change the PWM signal according to the error
        pwmChange = PWMdif / 20; // steps;
        pwmChangeTime = 500 / 10; // steps;
        //    Serial.println(".pwm change: " + String(pwmChange));


        if (abs(acceleration) < 0.5 && abs(smoothSpeed - targetSpeed) < 0.1) {
          // going the right speed sustainably. Remember this so that you can quickly return to the right speed after being stuck
          lastGoodPWM = (lastGoodPWM * 9 + PWMSignal) / 10;
        }

        if (abs(smoothSpeed) < 0.1 && abs(acceleration) < 0.1 && PWMSignal > 155 + 2 * (targetSpeed / 14.5 * (90.0 / 2) + 5)) { // you are likely stuck
          pwmChangeTime = 50;
          pwmChange = 0;
          PWMSignal = 200;
          if (!stuck) {
            stuckTime = millis();
          }
          stuck = true;


        } else if (stuck && abs(smoothSpeed) < targetSpeed && acceleration < 1) { // probably still stuck
          pwmChange = 0;
          PWMSignal = 200;
          //          Serial.print(".still stuck but maybe moving ");


        } else { // not stuck

          pwmChangeTime = 200 / steps;

          if (stuck) { // you were stuck but now are not
            PWMSignal = lastGoodPWM;

            pwmChange = 0;
            stuck = false;
          }
        }

      } else {
        // you are going faster than target speed by a stable amount, just go to close to default speed

        PWMSignal = lastGoodPWM; 
        pwmChange = 0;
        pwmChangeTime = 1;

      }
      if (stuck && millis() - stuckTime > 1000) {
        // stuck even when at full throttle. Just give up until you are given different instructions
        stopNow = true;
        stuck = false;
        stopReason = "stuck";


      } else if (stuck) {
        pwmChange = 45;
        PWMSignal = 155;
      }
    }

    // derivative part of PID
    if (USE_DERIVATIVE) {
      float desiredAcceleration = (targetSpeed - smoothSpeed); // mph/second

      float accelerationDif = desiredAcceleration - acceleration;

      pwmChange = (accelerationDif / 14.5) * (90.0 / 2); // map to pwm
      pwmChangeTime = 60;
    }



    // integral part of PID
    if (USE_INTEGRATED) { // may not need the integrated error
      // you went from being too slow/fast to the other way around, begin collecting integrated differences
      if (atTarget == -1 && smoothSpeed > targetSpeed)
        atTarget = 0;
      else if (atTarget == 1 && smoothSpeed < targetSpeed) {
        atTarget = 0;
      }

      // add up the integrated error
      if (atTarget == 0) {
        intError = (intError * 5 + (targetSpeed - smoothSpeed)) / 6;
        if (intError < 0.1) {
          pwmChange += 1;
        } else if (intError > 0.1) {
          pwmChange -= 1;
        }
      }
    }


    // pwm values 155 and 0 both mean no movement. Use 155
    if (int(PWMSignal) == 0) {
      PWMSignal = 155;
    }

    // add the change
    PWMSignal = PWMSignal + pwmChange;



    // prevent PWM from going outside of recognized PWM range
    if (PWMSignal > maxPWM) {
      PWMSignal = maxPWM;
    } else if (PWMSignal < minPWM) {
      PWMSignal = minPWM;
    }

    // sanity check: You must be sending pwm signals in a certain range to go the direction you want
    if (targetSpeed > 0 and PWMSignal < 155 and int(PWMSignal) != 0) {
      //      Serial.println(".going wrong way probably. Your pwm is " + String(PWMSignal) + " but you want to go forward");
      PWMSignal = 155;
    } else if (targetSpeed < 0 and PWMSignal > 155) {
      //      Serial.println(".going wrong way probably. Your pwm is " + String(PWMSignal) + " but you want to go backward");
      PWMSignal = 155;
    }

    // when you are going forwards but want to go backwards, you need to stop first
    if (targetSpeed < 0 && smoothSpeed > 0 && goingForward) {
      Serial.println(".stopping backwards first");
      PWMSignal = 155;

    } else if (targetSpeed > 0 && smoothSpeed < 0 && !goingForward) {
      Serial.println(".stopping forwards first");
      PWMSignal = 155;
    }

    if (int(PWMSignal) == 155) {
      analogWrite(motorPin, 0);
    } else if (lastPWM != int(PWMSignal)) { // &&  abs(smoothSpeed)<5){
      //  Serial.println("pwm: " + String(int(PWMSignal)));
      analogWrite(motorPin, int(PWMSignal));
    }
    lastPWM = int(PWMSignal);

  }

}


void setup() {
  Serial.begin(115200);
  pinMode(hallPin, INPUT);
  pinMode(otherHallPin, INPUT);
  analogWriteFreq(100);
  analogWriteRange(1023);
  delay(1000);

}

unsigned long gotSerialTime = 0;
bool gotSerial = false;

void loop() {


  if (Serial.available() && !gotSerial) {
    // serial just showed up. Don't do anything, just wait for the rest of the message to come in
    gotSerialTime = millis();
    gotSerial = true;
  }
  if (gotSerial && gotSerialTime + 15 < millis()) {
    // waited long enough for the message to come in. Process the serial
    processSerial();
    gotSerial = false;
  }
  
  if (gotSerialTime + 1500 < millis()) {
    // it has been a while since serial was sent. Something is wrong so stop
    analogWrite(motorPin, 0);
    PWMSignal = 0;
    targetSpeed = 0;
    return;
  }

  getWheelSpeed();


  if (stopNow) {
    targetSpeed = 0;
    PWMSignal = 0;
    analogWrite(motorPin, 0);
    if (lastTalkTime + 500 < millis()) {
      lastTalkTime = millis();
      if (stopReason == "stuck") { // report this error
        Serial.println("o1");
      }
      Serial.println("s");
      //      Serial.println(".wheel stopped. (" + stopReason + ")");
    }
    return;
  }


  if (lastTalkTime + 500 < millis()) {
    lastTalkTime = millis();
    //    Serial.println(".PWM speed: " + String(PWMSignal));
  }


  // setMovement is for the robot to go a specified distance and stop. It is given this command through serial. It is not usually used.
  if (setMovement) {
    int sign = (targetDist - startDist) / abs(targetDist - startDist);
    if ((sign > 0 && smoothDist >= targetDist) || (sign < 0 && smoothDist <= targetDist)) { // reached destination
      targetSpeed = 0;
      setMovement = false;
      //  Serial.println(".overshot by " + String(targetDist - smoothDist));
    } else {
      targetSpeed = pow(1.1, 1 / (0.04 * (targetDist - startDist)));
    }

  }


  if (!manual) {
    setMotorSpeed();
  }


}
