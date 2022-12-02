#define USEWIFI 0

#ifdef USEWIFI
String apName = "robot";

#include <DNSServer.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiAP.h>
#include <WebServer.h>
#include <HTTPClient.h>

#include <ESPmDNS.h>  // not needed for ap

#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
//
//
DNSServer dnsServer;
AsyncWebServer server(80);
#endif


void ICACHE_RAM_ATTR handleInterrupt();


bool stopNow = false;

unsigned long lastSerialTime = 0;


const int radioPinL = 33;
const int radioPinR = 32;

const int motorPinRF = 2;
const int motorPinRB = 25;

const int motorPinLF = 23;
const int motorPinLB = 22;

//rf
const int hallPin1 = 27;
const int hallPin2 = 26;

//rb
const int hallPin3 = 16;
const int hallPin4 = 17;

//lf
const int hallPin5 = 19;
const int hallPin6 = 21;

//lb
const int hallPin7 = 34;
const int hallPin8 = 35;

bool pidControl = true;

const int motorChannel = 1;
const int outputFreq = 100;
const int resolution = 10;

int pwmLimitLow = 110;
int pwmLimitHigh = 200;

// unless otherwise specified, {left, right} for all variables
volatile int pwmIn[] = { 0, 0 };
volatile int prevTime[] = { 0, 0 };

int leadEncoder[] = { 0, 0, 0, 0 };
unsigned long lastHitTime[] = { 0, 0, 0, 0 };
long encoderTicks[] = { 0, 0, 0, 0 };


int maximumSpeed = 4;

float wheelSpeed[] = { 0, 0, 0, 0 };

float proportionalError[] = { 0, 0, 0, 0 };
float derivativeError[] = { 0, 0, 0, 0 };
float integratedError[] = { 0, 0, 0, 0 };

float targetSpeed[] = { 0, 0, 0, 0 };
float pwmSpeed[] = { 155, 155, 155, 155 };

float kp = 2;
float ki = 0;
float kd = 0;

int goingForward[] = { 1, 1, 1, 1 };


// function declarations
void setMotorSpeed();


#ifdef USEWIFI
void setupPages() {

  server.on("/", [](AsyncWebServerRequest* request) {
    request->send(200, "text/html", "");
  });

  server.on("/_info", [](AsyncWebServerRequest* request) {
    if (request->hasParam("pwmControl")) {
      AsyncWebParameter* p = request->getParam("pwmControl");
      int pwmControl = (p->value()).toInt();
    }
    if (request->hasParam("maxSpeed")) {
    }

    request->send(200, "text/plain", "good");
  });

  server.begin();
}
#endif


void setup() {
  Serial.begin(115200);

  attachInterrupt(digitalPinToInterrupt(hallPin1), updateEncoder1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(hallPin2), updateEncoder2, CHANGE);

  attachInterrupt(digitalPinToInterrupt(hallPin3), updateEncoder3, CHANGE);
  attachInterrupt(digitalPinToInterrupt(hallPin4), updateEncoder4, CHANGE);

  attachInterrupt(digitalPinToInterrupt(hallPin5), updateEncoder5, CHANGE);
  attachInterrupt(digitalPinToInterrupt(hallPin6), updateEncoder6, CHANGE);

  attachInterrupt(digitalPinToInterrupt(hallPin7), updateEncoder7, CHANGE);
  attachInterrupt(digitalPinToInterrupt(hallPin8), updateEncoder8, CHANGE);

  pinMode(radioPinL, INPUT);
  pinMode(radioPinR, INPUT);

  attachInterrupt(digitalPinToInterrupt(radioPinL), leftPwmIntRise, RISING);
  attachInterrupt(digitalPinToInterrupt(radioPinR), rightPwmIntRise, RISING);


  ledcSetup(1, outputFreq, resolution);  // ledcSetup(uint8_t channel, double freq, uint8_t resolution_bits);
  pinMode(motorPinRF, OUTPUT);
  ledcAttachPin(motorPinRF, 1);

  ledcSetup(2, outputFreq, resolution);
  pinMode(motorPinRB, OUTPUT);
  ledcAttachPin(motorPinRB, 2);

  ledcSetup(3, outputFreq, resolution);  // ledcAttachPin(uint8_t pin, uint8_t chan);
  pinMode(motorPinLF, OUTPUT);
  ledcAttachPin(motorPinLF, 3);

  ledcSetup(4, outputFreq, resolution);
  pinMode(motorPinLB, OUTPUT);
  ledcAttachPin(motorPinLB, 4);

  #ifdef USEWIFI
  WiFi.softAP(apName.c_str());

  IPAddress myIP = WiFi.softAPIP();
  Serial.print(".IP address: " + WiFi.localIP());

  String mdnsName = "ESP32";
  if (MDNS.begin(mdnsName.c_str())) {
    // Serial.println(".MDNS responder started as http://" + mdnsName + ".local/");
  }
  setupPages();
  #endif
}


void calculateSpeed() {
  int wheelNum = 0;


  while (wheelNum < 4) {

    float timeBetweenHits = (millis() - lastHitTime[wheelNum]);
    float ws = encoderTicks[wheelNum] / timeBetweenHits / 10;


    if (timeBetweenHits == 0) {
      return;
    }
    encoderTicks[wheelNum] = 0;

    lastHitTime[wheelNum] = millis();


    wheelSpeed[wheelNum] = ws * goingForward[wheelNum];

    derivativeError[wheelNum] += (proportionalError[wheelNum] - (targetSpeed[wheelNum] - wheelSpeed[wheelNum])) / (timeBetweenHits / 100.0);

    integratedError[wheelNum] += (targetSpeed[wheelNum] - wheelSpeed[wheelNum]) / (timeBetweenHits / 100.0);

    proportionalError[wheelNum] = (targetSpeed[wheelNum] - wheelSpeed[wheelNum]);

    wheelNum++;
  }
}

void updateEncoder1() {
  /*
    Forward:
    hall pin 1 low
    hall pin 2 high
    hall pin 1 high
    hall pin 2 low
    hall pin 1 low
    hall pin 2 high
    hall pin 1 high
    hall pin 2 low

    Backward:
    hall pin 1 low
    hall pin 1 high
    hall pin 2 high
    hall pin 1 low
    hall pin 2 low
    hall pin 1 high

  */
  bool p1 = digitalRead(hallPin1);
  bool p2 = digitalRead(hallPin2);

  if (p1 == p2) {
    goingForward[0] = 1;
  } else {
    goingForward[0] = -1;
  }
  encoderTicks[0]++;  // add 1 to the number of ticks experienced
}

void updateEncoder2() {
  bool p1 = digitalRead(hallPin1);
  bool p2 = digitalRead(hallPin2);

  if (p1 == p2) {
    goingForward[0] = -1;
  } else {
    goingForward[0] = 1;
  }
}

void updateEncoder3() {
  bool p3 = digitalRead(hallPin3);
  bool p4 = digitalRead(hallPin4);

  if (p3 == p4) {
    goingForward[1] = -1;
  } else {
    goingForward[1] = 1;
  }
  encoderTicks[1]++;  // add 1 to the number of ticks experienced
}

void updateEncoder4() {
  bool p3 = digitalRead(hallPin3);
  bool p4 = digitalRead(hallPin4);

  if (p3 == p4) {
    goingForward[1] = 1;
  } else {
    goingForward[1] = -1;
  }
}

void updateEncoder5() {
  bool p3 = digitalRead(hallPin5);
  bool p4 = digitalRead(hallPin6);

  if (p3 == p4) {
    goingForward[2] = 1;
  } else {
    goingForward[2] = -1;
  }
  encoderTicks[2]++;  // add 1 to the number of ticks experienced
}

void updateEncoder6() {
  bool p3 = digitalRead(hallPin5);
  bool p4 = digitalRead(hallPin6);

  if (p3 == p4) {
    goingForward[2] = -1;
  } else {
    goingForward[2] = 1;
  }
}

void updateEncoder7() {
  bool p7 = digitalRead(hallPin7);
  bool p8 = digitalRead(hallPin8);

  if (p7 == p8) {
    goingForward[3] = -1;
  } else {
    goingForward[3] = 1;
  }
  encoderTicks[3]++;  // add 1 to the number of ticks experienced
}

void updateEncoder8() {
  bool p7 = digitalRead(hallPin7);
  bool p8 = digitalRead(hallPin8);

  if (p7 == p8) {
    goingForward[3] = 1;
  } else {
    goingForward[3] = -1;
  }
}


unsigned long lastPrintTime2 = 0;

void loop() {

  calculateSpeed();

  if (pidControl) {  // set speed relative to maximum speed

    for (int i=0; i<4; i++){targetSpeed[i]=0;}
    // Serial.println(String(pwmIn[0]) + ", " + String(pwmIn[1]));
    if (pwmIn[0] > 110){
      targetSpeed[0] = (pwmIn[0] - 155) / 45.0 * maximumSpeed;
      targetSpeed[1] = (pwmIn[0] - 155) / 45.0 * maximumSpeed;
    }
    if (pwmIn[1] > 110){
      targetSpeed[2] = (pwmIn[1] - 155) / 45.0 * maximumSpeed;
      targetSpeed[3] = (pwmIn[1] - 155) / 45.0 * maximumSpeed;
    }
    for (int i=0; i<4; i++){
      if (targetSpeed[i] < -maximumSpeed) {targetSpeed[i] = -maximumSpeed;}
      if (targetSpeed[i] > maximumSpeed) {targetSpeed[i] = maximumSpeed;}
    }

    setMotorSpeed();

  } else {  // radio control with no target speed
    ledcWrite(1, pwmIn[0]);
    ledcWrite(2, pwmIn[0]);

    ledcWrite(3, pwmIn[1]);
    ledcWrite(4, pwmIn[1]);
  }

  sendSerial();

  delay(50);
}