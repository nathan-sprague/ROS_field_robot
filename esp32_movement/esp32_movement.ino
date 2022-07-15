

void ICACHE_RAM_ATTR handleInterrupt();


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

float kp = 2;
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


    wheelSpeed[wheelNum]  = ws * goingForward[wheelNum];


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
  encoderTicks[0]++; // add 1 to the number of ticks experienced
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
    goingForward[1] = 1;
  } else {
    goingForward[1] = -1;
  }
  encoderTicks[1]++; // add 1 to the number of ticks experienced
}

void updateEncoder4() {
  bool p3 = digitalRead(hallPin3);
  bool p4 = digitalRead(hallPin4);

  if (p3 == p4) {
    goingForward[1] = -1;
  } else {
    goingForward[1] = 1;
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

  } else {
    pwmControl = true;
  }



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
  //
  //    if (millis() - lastPrintTime2 > 300) {
  //      Serial.println("forward: " + String(goingForward[0]) + ", " + String(goingForward[1]) );
  //      lastPrintTime2 = millis();
  //    }


  sendSerial();

  delay(5);

}
