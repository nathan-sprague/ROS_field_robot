
// set USE_WIFI to 0 for faster compiling. It won't let you make the website though
#define USE_WIFI 0
#define USE_AP 0

#define USE_GPS 0

void ICACHE_RAM_ATTR handleInterrupt();



#if USE_WIFI

#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiAP.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <ESPmDNS.h>

#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

AsyncWebServer server(80);

#else
#if USE_AP
#include <ESPmDNS.h>
#include <WiFi.h>
#include <WiFiAP.h>
#endif

#endif

#if USE_GPS
#include <Wire.h>
#include <SparkFun_u-blox_GNSS_Arduino_Library.h> //http://librarymanager/All#SparkFun_u-blox_GNSS

#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
SFE_UBLOX_GNSS myGNSS;
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
#endif

bool stopNow = false;

unsigned long lastSerialTime = 0;

const int radioPinL = 33;
const int radioPinR = 32;

const int motorPinL = 23;
const int motorPinR = 22;

const int hallPin1 = 27;
const int hallPin2 = 26;
const int hallPin3 = 18;
const int hallPin4 = 19;

bool pwmControl = true;

const int motorChannel = 1;
const int outputFreq = 100;
const int resolution = 10;

int pwmLimitLow = 130;
int pwmLimitHigh = 180;

// unless otherwise specified, {left, right} for all variables
volatile int pwmIn[] = {0, 0};
volatile int prevTime[] = {0, 0};

int leadEncoder[2] = {0, 0};
unsigned long lastHitTime[] = {0, 0};
long encoderTicks[] = {0, 0};


int maximumSpeed = 4;

float wheelSpeed[] = {0, 0};

float proportionalError[] = {0, 0};
float derivativeError[] = {0, 0};
float integratedError[] = {0, 0};
bool altEncoderActivated[] = {false, false};

float targetSpeed[] = {0, 0};
float pwmSpeed[] = {155, 155};

float kp = 1;
float ki = 0;
float kd = 0;

int goingForward[] = {1, 1};



// function declarations
void setMotorSpeed();




void setup() {
  Serial.begin(115200);
  attachInterrupt(digitalPinToInterrupt(hallPin1), updateEncoder1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(hallPin2), updateEncoder2, CHANGE);

  attachInterrupt(digitalPinToInterrupt(hallPin3), updateEncoder3, CHANGE);
  attachInterrupt(digitalPinToInterrupt(hallPin4), updateEncoder4, CHANGE);

  pinMode(radioPinL, INPUT);
  pinMode(radioPinR, INPUT);

  attachInterrupt(digitalPinToInterrupt(radioPinL), leftPwmIntRise, RISING);
  attachInterrupt(digitalPinToInterrupt(radioPinR), rightPwmIntRise, RISING);


  ledcSetup(1, outputFreq, resolution);
  pinMode(motorPinR, OUTPUT);
  ledcAttachPin(motorPinR, 1);

  ledcSetup(2, outputFreq, resolution);
  pinMode(motorPinL, OUTPUT);
  ledcAttachPin(motorPinL, 2);


#if USE_WIFI
  setupAP();
  setupPages();
  server.begin();
#else
#if USE_AP
  setupAP();
#endif
#endif

#if USE_GPS
  beginGPS();
  bno.begin();
#endif



}

void calculateSpeed() {

  int wheelNum = 0;


  while (wheelNum < 2) {

    float timeBetweenHits = (millis() - lastHitTime[wheelNum]);
    float ws = 0;

    if (timeBetweenHits != 0) {
      ws = encoderTicks[wheelNum] / timeBetweenHits / 10;
    }
    //
    encoderTicks[wheelNum] = 0;

    lastHitTime[wheelNum] = millis();


    // this backwards calculation doesn't work (one encoder doesnt work right now for some reason)
    if (leadEncoder[0] == 0 && leadEncoder[1] == 1) { // backwards
      ws *= -1;
    }

    // this backwards calculation should
    if (pwmSpeed[wheelNum] < 155 && pwmSpeed[wheelNum] > 90 && abs(ws) < 0.3) { // going backwards
      goingForward[wheelNum] = -1;


    } else if (pwmSpeed[wheelNum] > 155 && abs(ws) < 0.3) {  // going forwards
      goingForward[wheelNum] = 1;
    }


    wheelSpeed[wheelNum]  = ws * goingForward[wheelNum];


    derivativeError[wheelNum] += (proportionalError[wheelNum] - (targetSpeed[wheelNum] - wheelSpeed[wheelNum])) / (timeBetweenHits / 100.0);

    integratedError[wheelNum] += (targetSpeed[wheelNum] - wheelSpeed[wheelNum]) / (timeBetweenHits / 100.0);

    proportionalError[wheelNum] = (targetSpeed[wheelNum] - wheelSpeed[wheelNum]);


    wheelNum++;
  }

}

void updateEncoder1() {
  if (digitalRead(hallPin1) == HIGH) { // pin is turning high

    // directional control not working for some reason
    if (altEncoderActivated[1]) {
      //      goingForward[0] = false; // going backwards

    } else {
      //      goingForward[0] = true; // going forwards
    }
  }
  encoderTicks[0]++; // add 1 to the number of ticks experienced
}

void updateEncoder2() {
  if (hallPin1 == HIGH) {
    altEncoderActivated[0] = true;
  } else {
    altEncoderActivated[0] = false;
  }
}

void updateEncoder3() {
  if (digitalRead(hallPin3) == HIGH) { // pin is turning high

    // directional control not working for some reason
    if (altEncoderActivated[1]) {
      //      goingForward[1] = false; // going backwards

    } else {
      //      goingForward[1] = true; // going forwards
    }
  }
  encoderTicks[1]++; // add 1 to the number of ticks experienced
}

void updateEncoder4() {
  if (hallPin3 == HIGH) {
    altEncoderActivated[1] = true;
  } else {
    altEncoderActivated[1] = false;
  }
}


unsigned long lastPrintTime2 = 0;

void loop() {

  readSerial();

  calculateSpeed();

  if (millis() - lastSerialTime > 1000 || (abs(pwmIn[0] - 155) < 50 && abs(pwmIn[0] - 155) > 5) ) { // haven't gotten a serial message for a second. Switch over to radio control
    //    readRadioSpeed();
    pwmControl = false;
    ledcWrite(1, pwmIn[0]);
    ledcWrite(2, pwmIn[1]);
    if (millis() - lastPrintTime2 > 300) {
      //      Serial.println("radio " + String(pwmIn[0]) + ", " + String(pwmIn[1]));
      lastPrintTime2 = millis();
    }
  } else {
    pwmControl = true;
  }

#if USE_GPS
  readGPS();
#endif

  if (stopNow) {
    targetSpeed[0] = 0;
    targetSpeed[1] = 0;
    pwmSpeed[0] = 0;
    pwmSpeed[1] = 0;
    ledcWrite(1, 0);
    ledcWrite(2, 0);
  } else if (pwmControl) {
    setMotorSpeed();
  }

  sendSerial();

  delay(5);

}
