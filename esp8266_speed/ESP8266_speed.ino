/*
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

/*
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

  Other:
  .: console log
  -: error
*/


//#include <ESP8266WiFi.h>
//#include <ESP8266WiFiMulti.h>
//#include <ESP8266WebServer.h>
//#include <ESP8266mDNS.h>
//#include <ESP8266HTTPClient.h>
//
//#include <WiFiClient.h>
//
//ESP8266WiFiMulti WiFiMulti;
//
//ESP8266WebServer server(80);

#define DEBUGMODE false

const int motorPin = 16;  // motor (D0)
const int hallPin = 5; // speed sensor pin 1 (D1)
const int otherHallPin = 4; // speed sensor pin 2 (D2)


unsigned long lastMsgTime = 0; // timer to prevent over-printing

const float wheelCircum = 13 * 2 * 3.14159;
const int numHoles = 16;

float smoothSpeed = 0; // speed used for all calculations. Smooths and averages speeds to limit outliers
float smoothDist = 0;

bool setMovement = false;
long targetDist = 0;
long startDist = 0;

bool goingForward = true;

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

unsigned long lastCommandTime = 0;


bool stopNow = false;
bool manual = false;
int PWMSignal = 0;

float targetSpeed = 0;

unsigned long lastTargetChange = 0; // timer to reassess the speed every half second

unsigned long lastTalkTime = 0;

String stopReason = "Unknown";


void processSerial() {
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

    if (commandType == 'f')  {
      targetSpeed = serialMsg.toFloat(); // set speed
      Serial.println("-f" + String(int(targetSpeed)));
      if (DEBUGMODE){
        Serial.println(".target speed is set to " + serialMsg + ", was: " + String(targetSpeed));
      }
      setMovement = false; 
      manual = false;


    } else if (commandType == 'z') {
      if (DEBUGMODE) {
        Serial.println(".manual");
      }
      setMovement = false;
      manual = true;
      analogWrite(motorPin, serialMsg.toInt());
      PWMSignal = serialMsg.toInt();
      
      Serial.println("-z" + String(PWMSignal));

    } if (commandType == 'p') {
      Serial.println("+p");
      
    } else if (commandType == 's') {
      if (DEBUGMODE) {
        Serial.println(".emergency stop");
        Serial.println("-s");
      }
      stopNow = true;
      stopReason = "Serial stop";

    } else if (commandType == 'g') {
      if (DEBUGMODE) {
        Serial.println(".go");
        Serial.println("-g");
      }
      stopNow = false;
      stopReason = "Unknown";
    } else if (commandType == 'r') {
      Serial.println(".restarting");
      Serial.println("-r");
      ESP.restart();
      
    } else if (commandType == 'm') {
      int moveFor = serialMsg.toInt();
      targetDist = smoothDist + moveFor;
      startDist = smoothDist;
      setMovement = true;
    }
    




  }
}



const float tickTimeTomph = (wheelCircum / 12.0) / numHoles * 1000.0 * 3600 / 5280; // constant for calculating wheel speed

void getWheelSpeed() {
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

  if (speedChange) { // Don't reasses the situation unless there is a change in speed

    // get average speed between the wheels and past speeds. #s 3 and 5 were arbitrarily chosen
    smoothSpeed = (smoothSpeed * 3 + (wheelSpeed + wheelSpeedY)) / 5;
    smoothDist = (distTraveled + distTraveledY)/2;

    // Direction may change if the speed is zero
    if (smoothSpeed > -0.1 && smoothSpeed < 0.1) { // cant compare it directly to zero because a float

      if ((PWMSignal >= 155 || PWMSignal == 0) && targetSpeed > 0) { // signal tells it to go forwards, go forwards
        if (lastMsgTime + 500 < millis()) {
          if (DEBUGMODE) {
            Serial.println("stopped and going forwards");
          }
        }
        goingForward = true;

      } else { // it is going backwards
        if (lastMsgTime + 500 < millis()) {
          if (DEBUGMODE) {
            Serial.println("stopped and going backwards");
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




unsigned long lastPotPrint = 0;
float averagePotSpeed = 0;
float lastPotVal = 0;
unsigned long lastPotReading = 0;
float lastDistFromCenter = 0;
float lastDistFromFront = 0;

void getFeelerHits() {
  if (lastPotReading +  10 < millis()) {

    // left 45 degrees 845
    // right 45 degrees 525
    // resting 675
    // therefore 845-525 is roughly 300
    
    float potAngle = (analogRead(A0) - 675.0) * 90 / 300.0;
    
    

    if ((abs(potAngle)>5 && lastPotPrint + 100 < millis())|| lastPotPrint + 500 < millis()){
        Serial.println("o" + String(potAngle));
        lastPotPrint = millis();
    }
    
//    unsigned long potReadingTime = millis();
//    float potSpeed = float(potAngle - lastPotVal) * 1000 / (potReadingTime - lastPotReading);
//
//    averagePotSpeed = (averagePotSpeed * 2 + potSpeed) / 3;
//
//    float distFromCenter = 1000;
//    float distFromFront = -1000;
//
//
//    if (averagePotSpeed * potAngle < 0) {
//      // the signs are different, meaning the pot is returning to zero
//      distFromCenter = 2000;
//      distFromFront = -2000;
//
//    } else if (abs(averagePotSpeed) > 0.1 && abs(potAngle) > 0.1) {
//      // d = v * sin(theta) / (dtheta/dt)
//      distFromCenter = smoothSpeed * 5280 * 12 / 3600 / (averagePotSpeed * 3.14159 / 180);
//      distFromFront = distFromCenter / tan(potAngle * 3.1415 / 180);
//    }
//    if (abs(lastDistFromCenter-distFromCenter)>0.1 && abs(lastDistFromFront-distFromFront)>0.1){
//      if (abs(distFromCenter) < 30 && abs(distFromFront) < 30 && lastPotPrint + 100 < millis()) {
//        Serial.println("k" + String(distFromCenter));
//        Serial.println("l" + String(distFromFront));
//        Serial.println("o" + String(potAngle));
//        lastPotPrint = millis();
//        lastDistFromCenter = distFromCenter;
//        lastDistFromFront = distFromFront;
//        //Serial.println("from center: " + String(distFromCenter) + ", from front: " + String(distFromFront));
//      }
      
//    }
//
//    lastPotVal = potAngle;
//    lastPotReading = potReadingTime;
  }
}


unsigned long lastPrintTime2 = 0;

int atTarget = 0;
int lastTarget = 0;
float intError = 0;
int lastPWM = 0;

void setMotorSpeed() {
  // 200 -> 14.5
  // 190 -> 9.7
  // 180 -> 7.3
  // 170 -> 4.1
  // 160 -> 1.1
  // 155 -> 0
  // 150 -> -0.4
  // 140 -> -2.9
  // 130 -> -5.8
  // 120 -> -9.7
  // 110 -> -14.5

  if (lastTargetChange + 500 < millis()) {
    lastTargetChange = millis();

    if (lastTarget != int(targetSpeed)) { // commanded target changed
      if (DEBUGMODE) {
        Serial.println("target changed");
      }
      // you are not at the target speed
      lastTarget = int(targetSpeed);
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

  Serial.println(".wheel target speed: " + String(targetSpeed));

  if (targetSpeed == 0) {
    analogWrite(motorPin, 0);
    PWMSignal = 0;
    return;

  } else if (smoothSpeed != targetSpeed) {

    // convert difference in speed to a difference in PWM
    float PWMdif = ((targetSpeed - smoothSpeed) / 14.5) * (90.0 / 2); // map (-14.5)-(14.5) to (110)-(200)

    // round the float to the nearest whole number (not int)
    if (PWMdif - int(PWMdif) > 0.5) {
      PWMdif = int(PWMdif) + 1;
    }

    if (PWMdif - int(PWMdif) < -0.5) {
      PWMdif = int(PWMdif) - 1;
    }

    if (DEBUGMODE) {
      Serial.print(".old pwm: " + String(PWMSignal));
    }


    if (PWMSignal == 0) {
      PWMSignal = 155;
    }
    // change the PWM signal according to the error
    PWMSignal = PWMSignal + PWMdif;

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
        PWMSignal += 1;
      } else if (intError > 0.1) {
        PWMSignal -= 1;
      }
    }


    if (DEBUGMODE) {
      Serial.print(", pwmdif: " + String(PWMdif));

      Serial.print(", new pwm: " + String(PWMSignal));

      Serial.println(", Integrated error: " + String(intError));
    }


    // prevent overshooting
    if (PWMSignal > 200) {
      PWMSignal = 200;
    } else if (PWMSignal < 110) {
      PWMSignal = 110;
    }

    // when you are going forwards but want to go backwards, you need to stop first
    if (targetSpeed < 0 && smoothSpeed > 0 && goingForward) {
      Serial.println(".stopping backwards first");
      PWMSignal = 155;

    } else if (targetSpeed > 0 && smoothSpeed < 0 && !goingForward) {
      Serial.println(".stopping forwards first");
      PWMSignal = 155;
    }

    if (PWMSignal == 155) {
      analogWrite(motorPin, 0);
    }
    if (lastPWM != PWMSignal) { // &&  abs(smoothSpeed)<5){
      analogWrite(motorPin, PWMSignal);
    }
    lastPWM = PWMSignal;

  }
}


void setup() {
  Serial.begin(115200);
  pinMode(hallPin, INPUT);
  pinMode(otherHallPin, INPUT);
  analogWriteFreq(100);
//  setupAP();
//    connectToWIFI();
  delay(1000);

}

unsigned long gotSerialTime = 0;
bool gotSerial = false;

void loop() {

  
  if (Serial.available() && !gotSerial) {
    gotSerialTime = millis();
    gotSerial = true;
  }
  if (gotSerial && gotSerialTime+15 < millis()) {
    processSerial();
    gotSerial = false;
  } 
  if (gotSerialTime+1500 < millis()) {
    analogWrite(motorPin, 0);
    PWMSignal = 0;
    targetSpeed = 0;
    return;
  }
  
  getWheelSpeed();

  //getFeelerHits();

  if (stopNow) {
    targetSpeed = 0;
    PWMSignal = 0;
    analogWrite(motorPin, 0);
    if (lastTalkTime + 500 < millis()) {
      lastTalkTime = millis();
      Serial.println(".wheel stopped");
      Serial.println(".Stop reason: " + stopReason);
    }
    return;
  }


  if (lastTalkTime + 500 < millis()) {
    lastTalkTime = millis();
    Serial.println(".PWM speed:" + String(PWMSignal));
  }

  if (setMovement){
    int sign = (targetDist - startDist) / abs(targetDist - startDist);
    if ((sign>0 && smoothDist >= targetDist) || (sign<0 && smoothDist <= targetDist)) { // reached destination
      targetSpeed = 0;
      setMovement = false;
      Serial.println(".overshot by " + String(targetDist - smoothDist));
    } else {
      targetSpeed = pow(1.1, 1/(0.04*(targetDist - startDist)));
    }
    
  }


  if (!manual) {
    setMotorSpeed();
  }


}
