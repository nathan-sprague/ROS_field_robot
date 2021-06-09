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

#define DEBUGMODE false
#define WIRELESSDEBUG false

#define USE_GPS true

#include <Wire.h>
#include <SparkFun_u-blox_GNSS_Arduino_Library.h> //http://librarymanager/All#SparkFun_u-blox_GNSS
SFE_UBLOX_GNSS myGNSS;

#include <Arduino.h>

#include <ESP8266WiFi.h>
#include <ESP8266WiFiMulti.h>

#include <ESP8266HTTPClient.h>

#include <WiFiClient.h>

ESP8266WiFiMulti WiFiMulti;


const int steerPin = 16;  // motor (D0)
const int sdaPin = 4;  // (D2)
const int sclPin = 5;  // (D1)


int pwmSpeed = 0;
bool stopNow = false;
int targetPosition = 0;
unsigned long lastPrint = 0;
unsigned long lastPrint2 = 0;

int passedTarget = 10;

int averagePosition = 0;


void processSerial() {
  if (Serial.available()) {

    char commandType = Serial.read();
    String serialMsg = "";

    while (Serial.available()) {
      char b = Serial.read();
      serialMsg += b;
    }
    int commandVal = serialMsg.toInt();


    if (commandType == 'z') {
      Serial.println(".manual");
      pwmSpeed = commandVal;
      analogWrite(steerPin, commandVal);

    } else if (commandType == 'p') {
      if (commandVal!=targetPosition){
        Serial.println(".move to angle: " + String(commandVal));
        Serial.println(".");
        targetPosition  = commandVal;
      
        passedTarget = 0;
      }

    } else if (commandType == 's') {
      Serial.println(".emergency stop");
      stopNow = true;

    } else if (commandType == 'g') {
      Serial.println(".go");
      stopNow = false;
    } else if (commandType == 'r') {
      Serial.println(".restarting");
      ESP.restart();
    }
  }
}

void beginGPS() {
  Wire.begin();

  if (myGNSS.begin() == false) { //Connect to the u-blox module using Wire port
    Serial.println(".u-blox GNSS not detected");
    while (1);
  }

  myGNSS.setI2COutput(COM_TYPE_UBX); //Set the I2C port to output UBX only (turn off NMEA noise)
  myGNSS.saveConfigSelective(VAL_CFG_SUBSEC_IOPORT); //Save (only) the communications port settings to flash and BBR

}

void getPosition() {

  float latitude = myGNSS.getLatitude();
  Serial.println("x" + String(latitude));

  float longitude = myGNSS.getLongitude();
  Serial.println("y" + String(longitude));
}


void talkToESP() {
  if ((WiFiMulti.run() == WL_CONNECTED)) {

    WiFiClient client;

    HTTPClient http;
    if (WIRELESSDEBUG) {
      Serial.print(".[HTTP] begin...\n");
    }

    if (http.begin(client, "http://192.168.4.1/_angle?a=" + String(averagePosition))) {  // HTTP


      if (WIRELESSDEBUG) {
        Serial.print(".[HTTP] GET...\n");
      }
      // start connection and send HTTP header
      int httpCode = http.GET();

      // httpCode will be negative on error
      if (httpCode > 0) {
        // HTTP header has been send and Server response header has been handled
        if (WIRELESSDEBUG) {
          Serial.printf(".[HTTP] GET... code: %d\n", httpCode);
        }

        // file found at server
        if (httpCode == HTTP_CODE_OK || httpCode == HTTP_CODE_MOVED_PERMANENTLY) {
          String payload = http.getString();
          if (payload.length()
          >0){
            if (payload == "s") {
              stopNow = true;
              return;
            } else {
              stopNow = false;
              targetPosition = payload.toInt();
              passedTarget = 0;
              //          Serial.println(payload);
            }
          }
        }
      } else {
        if (WIRELESSDEBUG) {
          Serial.printf(".[HTTP] GET... failed, error: %s\n", http.errorToString(httpCode).c_str());
        }
      }

      http.end();
    } else {
      if (WIRELESSDEBUG) {
        Serial.printf(".[HTTP} Unable to connect\n");
      }
    }
  }
}

void connectToWIFI() {
  for (uint8_t t = 4; t > 0; t--) {
    if (WIRELESSDEBUG) {
      Serial.printf(".[SETUP] WAIT %d...\n", t);
    }
    Serial.flush();
    delay(1000);
  }
  WiFi.mode(WIFI_STA);
  WiFiMulti.addAP("robot");
}

void setup() {
  Serial.begin(115200);

  connectToWIFI();


  analogWriteFreq(100);

  if (USE_GPS) {
    beginGPS();
  }
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
  if (lastPrint2 + 200 < millis()) {
    lastPrint2 = millis();
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
  if ((averagePosition < 50 || pwmSpeed < 150) && (averagePosition > -50 || pwmSpeed == 0 || pwmSpeed > 150)) {
    return true;
  } else  {
    Serial.println(".stopped");
    analogWrite(steerPin, 0);
    pwmSpeed = 0;
    return false;
  }
}

unsigned long lastPosReadTime = 0;

unsigned long lastRequest = 0;
int lastPWMspeed = 0;

int lastReading = 0;

void loop() {
  processSerial();

  if (lastRequest + 300 < millis()) {
    talkToESP();
    Serial.println("a" + String(averagePosition));
    lastRequest = millis();
//    Serial.println(".moving " + String(pwmSpeed));
  }

  //  Serial.println("raw: " + String(analogRead(A0)));
  if (stopNow) {
    analogWrite(steerPin, 0);
    pwmSpeed = 0;
    delay(10);
    //    Serial.println("stopped");
    return;
  }

  int currentPosition = potToAngle(analogRead(A0));

  averagePosition = (averagePosition * 5 + currentPosition) / 6;


  if (targetPosition<-40){
    targetPosition = -40;
  }
  targetToSpeed();

  if (lastPrint + 400 < millis()) {
    lastPrint = millis();
    if (DEBUGMODE) {
      Serial.println(".angle: " + String(averagePosition) + ", Speed: " + String(pwmSpeed));
    }

  }

  if (USE_GPS && lastPosReadTime + 2000 < millis()) {
    getPosition();
    lastPosReadTime = millis();
  }

  if ((lastReading > targetPosition && averagePosition < targetPosition) || (lastReading < targetPosition && averagePosition > targetPosition)) {
    passedTarget++;
    if (passedTarget == 3){
      Serial.println(".done wobbling");
    }
  }
  if (passedTarget > 2) {

    pwmSpeed = 0;
  }
  lastReading = averagePosition;


  if (limitOvershoot() && pwmSpeed != lastPWMspeed) {
    analogWrite(steerPin, pwmSpeed);
    lastPWMspeed = pwmSpeed;
  }

  delay(10);

}
