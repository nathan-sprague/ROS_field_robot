/*
  static const uint8_t D0   = 16;
  static const uint8_t D1   = 5; // (SCL)
  static const uint8_t D2   = 4; // (SDA)
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
#define WIRELESSDEBUG false
#define SERIALTIMEOUT false

#include <Wire.h>
#include <SparkFun_u-blox_GNSS_Arduino_Library.h> //http://librarymanager/All#SparkFun_u-blox_GNSS
SFE_UBLOX_GNSS myGNSS;

#include <Arduino.h>

const int steerPin = 16;  // motor (D0)

bool useGPS = false;
unsigned long lastPosReadTime = 0;


int pwmSpeed = 0;
bool stopNow = false;
int targetPosition = 0;
unsigned long lastPrint = 0;



bool manualPWM = false;

int passedTarget = 10;

float averagePosition = 0;



void processSerial() {
  if (Serial.available()) {

    char commandType = Serial.read();

    String serialMsg = "";

    while (Serial.available()) {
      char b = Serial.read();
      serialMsg += b;
    }
    int commandVal = serialMsg.toInt();


    if (commandType == '_') {
      return;
    }

    if (commandType == 'z') {
      Serial.println(".manual");

      pwmSpeed = commandVal;
      analogWrite(steerPin, commandVal);
      Serial.println("-z" + String(commandVal));
      manualPWM = true;
    
    } else if (commandType == 'l') { // requested esp type. Tell it this is speed
      Serial.println("e1");

    } else if (commandType == 'p') {
      manualPWM = false;
      if (commandVal != targetPosition) {
        Serial.println("-p" + String(int(commandVal)));
        Serial.println(".serial says move to angle: " + String(commandVal));
        // Serial.println(".");
        targetPosition  = commandVal;
      }
      passedTarget = 0;

    } else if (commandType == 's') {
      manualPWM = false;
      Serial.println("-s");
      Serial.println(".emergency stop");
      stopNow = true;

    } else if (commandType == 'g') {
      manualPWM = false;
      Serial.println("-g");
      stopNow = false;
      
    } else if (commandType == 'r') {
      manualPWM = false;
      Serial.println("-r");
      Serial.println(".restarting");
      ESP.restart();
    } else {
      Serial.println("+" + String(commandType));
    }
  }
}

bool beginGPS() {
  Wire.begin();

  if (myGNSS.begin() == false) { //Connect to the u-blox module using Wire port
    Serial.println(".u-blox GNSS not detected");
    return false;
  }
  Serial.println("gps set up successfully");
  myGNSS.setI2COutput(COM_TYPE_UBX); //Set the I2C port to output UBX only (turn off NMEA noise)
  myGNSS.saveConfigSelective(VAL_CFG_SUBSEC_IOPORT); //Save (only) the communications port settings to flash and BBR
  return true;
}


void printFractional(int32_t fractional, uint8_t places) {
  if (places > 1) {
    for (uint8_t place = places - 1; place > 0; place--) {
      if (fractional < pow(10, place)) {
        Serial.print("0");
      }
    }
  }
  Serial.print(fractional);
}




void getPosition() {
  Serial.println(".getting position");

  float heading = float(myGNSS.getHeading()) / 100000;
  Serial.println("h" + String(heading));

  // First, let's collect the position data
  int32_t latitude = myGNSS.getHighResLatitude();
  int8_t latitudeHp = myGNSS.getHighResLatitudeHp();
  int32_t longitude = myGNSS.getHighResLongitude();
  int8_t longitudeHp = myGNSS.getHighResLongitudeHp();
  uint32_t accuracy = myGNSS.getHorizontalAccuracy();

  // Defines storage for the lat and lon units integer and fractional parts
  int32_t lat_int; // Integer part of the latitude in degrees
  int32_t lat_frac; // Fractional part of the latitude
  int32_t lon_int; // Integer part of the longitude in degrees
  int32_t lon_frac; // Fractional part of the longitude

  // Calculate the latitude and longitude integer and fractional parts
  lat_int = latitude / 10000000; // Convert latitude from degrees * 10^-7 to Degrees
  lat_frac = latitude - (lat_int * 10000000); // Calculate the fractional part of the latitude
  lat_frac = (lat_frac * 100) + latitudeHp; // Now add the high resolution component
  if (lat_frac < 0) // If the fractional part is negative, remove the minus sign
  {
    lat_frac = 0 - lat_frac;
  }
  lon_int = longitude / 10000000; // Convert latitude from degrees * 10^-7 to Degrees
  lon_frac = longitude - (lon_int * 10000000); // Calculate the fractional part of the longitude
  lon_frac = (lon_frac * 100) + longitudeHp; // Now add the high resolution component
  if (lon_frac < 0) // If the fractional part is negative, remove the minus sign
  {
    lon_frac = 0 - lon_frac;
  }

  // Print the lat and lon
  Serial.print("x");
  Serial.print(lat_int); // Print the integer part of the latitude
  Serial.print(".");
  printFractional(lat_frac, 9); // Print the fractional part of the latitude with leading zeros
  Serial.println("");


  Serial.print("y");
  Serial.print(lon_int); // Print the integer part of the latitude
  Serial.print(".");
  printFractional(lon_frac, 9); // Print the fractional part of the latitude with leading zeros
  Serial.println("");

  // Now define float storage for the heights and accuracy
  float f_accuracy;


  // Convert the horizontal accuracy (mm * 10^-1) to a float
  f_accuracy = accuracy;
  // Now convert to m
  f_accuracy = f_accuracy / 10000.0; // Convert from mm * 10^-1 to m


  Serial.print("t");
  Serial.println(f_accuracy, 4); // Print the accuracy with 4 decimal places

  byte RTK = myGNSS.getCarrierSolutionType();
  if (RTK == 0) Serial.print(".no special fix");
  else if (RTK == 1) Serial.println(".floating fix");
  else if (RTK == 2) Serial.print(".High precision fix");


}


void setup() {
  Serial.begin(115200);
  delay(1000);
  analogWriteRange(1023);
  analogWriteFreq(100);
  averagePosition = analogRead(A0);
  useGPS = beginGPS();
}

void targetToSpeed() {
  int distToTarget = targetPosition - averagePosition ;
  // possible difference in angle:
  // on left, go to right -> 50 - (-50) = 100
  // on right, go to left -> -50 - 50 = -100
  // acceptable speeds:
  // 110 - 135 (left)
  // 175 - 200 (right)

  bool canPrint = false;
  if (lastPrint + 200 < millis()) {
    lastPrint = millis();
    canPrint = true;
  }
  if (canPrint && DEBUGMODE) {

    Serial.println(".dist to target: " + String(distToTarget));
  }
  if (distToTarget < -1) { // target to the left
    pwmSpeed = distToTarget * (30.0 / 100.0) + 135;
  } else if (distToTarget > 1) { // target to the right
    pwmSpeed = (distToTarget * (30.0 / 100.0)) + 175;

  } else {
    if (canPrint && DEBUGMODE) {
      Serial.println(".at correct angle!");
    }
    pwmSpeed = 0;
  }
  if (canPrint && DEBUGMODE) {
    Serial.println(".PWM speed: " + String(pwmSpeed));
  }


}

float potToAngle(int pot) {
  // convert potentiometer values to wheel angle in degrees
  // center 240
  // right 330
  // left 190

  float angle = 0;
  if (pot > 240) { // right
    angle = (pot - 240) * 45 / (330.0 - 240);
  } else { // left
    angle = (pot - 240) * 45 / (240 - 170.0);
  }

  return angle + 1;
}

bool limitOvershoot() {
  // if the wheel angle is too large, stop moving
  if ((averagePosition < 50 || pwmSpeed < 150) && (averagePosition > -50 || pwmSpeed == 0 || pwmSpeed > 150)) {
    return true;
  } else  {
    Serial.println(".stopped");
    analogWrite(steerPin, 0);
    pwmSpeed = 0;
    return false;
  }
}

unsigned long lastRequest = 0;
int lastPWMspeed = 0;

float lastReading = 0;

unsigned long lastAngleReading = 0;

unsigned long gotSerialTime = 0;
bool gotSerial = false;

void loop() {


  if (Serial.available() && !gotSerial) {
    gotSerialTime = millis();
    gotSerial = true;
  }
  if (gotSerial && gotSerialTime + 15 < millis()) {
    processSerial();
    gotSerial = false;
  }


  if (useGPS && lastPosReadTime + 1000 < millis()) {
    getPosition();
    lastPosReadTime = millis();
  }


  if (gotSerialTime + 1500 < millis()  && SERIALTIMEOUT) {
    analogWrite(steerPin, 0);
    pwmSpeed = 0;
    return;
  } else {
    pwmSpeed = lastPWMspeed;
  }

  if (lastRequest + 300 < millis()) {
    Serial.println("a" + String(int(averagePosition)));
    lastRequest = millis();
    //    Serial.println(".moving " + String(pwmSpeed));
  }

  //  Serial.println("raw: " + String(analogRead(A0)));
  if (stopNow) {
    analogWrite(steerPin, 0);
    pwmSpeed = 0;
    delay(10);
    //    Serial.println(".stopped");
    return;
  }

  if (millis() - 10 < lastAngleReading) {
    return;
  }
  delay(0);

  lastAngleReading = millis();

  int currentPosition = potToAngle(analogRead(A0));
  averagePosition = (averagePosition * 5 + currentPosition) / 6;


  if (targetPosition < -40) {
    targetPosition = -40;
  } else if (targetPosition > 40) {
    targetPosition = 40;
  }
  targetToSpeed();


  if ((lastReading > targetPosition && averagePosition < targetPosition) || (lastReading < targetPosition && averagePosition > targetPosition)) {
    passedTarget++;
    if (passedTarget == 4) {
      Serial.println(".done wobbling");
    }
  }
  if (passedTarget > 4) {

    pwmSpeed = 0;
  }
  lastReading = averagePosition;


  if (limitOvershoot() && pwmSpeed != lastPWMspeed && !manualPWM) {
    Serial.println(".moving " + String(pwmSpeed));
    analogWrite(steerPin, pwmSpeed);
    lastPWMspeed = pwmSpeed;
  }


}
