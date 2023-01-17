void ICACHE_RAM_ATTR handleInterrupt();


bool stopNow = false;

unsigned long lastSerialTime = 0;


const int radioPinL = 33;
const int radioPinR = 32;


// pins to control the motors
const int motorPinRF = 2;
const int motorPinRB = 25;
const int motorPinLF = 23;
const int motorPinLB = 22;


// pins to control the encoders
// lf
const int encoderPinLF1 = 19; // 5
const int encoderPinLF2 = 21; // 6

//lb
const int encoderPinLB1 = 34;
const int encoderPinLB2 = 35;

//rf
const int encoderPinRF1 = 27;
const int encoderPinRF2 = 26;

//rb
const int encoderPinRB1 = 16;
const int encoderPinRB2 = 17;


bool pidControl = true; // pidControl=true means the motors are being controlled by serial, not the radio

// PWM characteristics
const int outputFreq = 100;
const int resolution = 10;

// saturation PWM output limits (only used for PID speed control)
int pwmLimitLow = 65;
int pwmLimitHigh = 245;

// left/right radio
volatile int pwmIn[] = {0, 0}; // volatile type because it is handled by interrupts
volatile int prevTime[] = {0, 0};


// unless otherwise specified, {left front, left back, right front, right back} for all variables
int leadEncoder[] = {0, 0, 0, 0};
unsigned long lastSpeedCalcTime[] = {0, 0, 0, 0};
long encoderTicks[] = {0, 0, 0, 0};


float wheelSpeed[] = {0, 0, 0, 0};

float proportionalError[] = {0, 0, 0 ,0};
float derivativeError[] = {0, 0, 0, 0};
float integratedError[] = {0, 0, 0, 0};

float targetSpeed[] = {0, 0, 0, 0};
float pwmSpeed[] = {155, 155, 155, 155};

float kp = 1;
float ki = 0;
float kd = 0;

int goingForward[] = {1, 1, 1, 1};

unsigned long lastSetSpeedCallTime = 0;



// function declarations
void setMotorSpeed();



void setup() {
  Serial.begin(115200);


  // have to enable interrupts individually, for some reason
  attachInterrupt(digitalPinToInterrupt(encoderPinLF1), updateEncoderLF1, CHANGE);

  attachInterrupt(digitalPinToInterrupt(encoderPinLB1), updateEncoderLB1, CHANGE);

  attachInterrupt(digitalPinToInterrupt(encoderPinRF1), updateEncoderRF1, CHANGE);

  attachInterrupt(digitalPinToInterrupt(encoderPinRB1), updateEncoderRB1, CHANGE);

  pinMode(encoderPinLF2, INPUT);
  pinMode(encoderPinLB2, INPUT);
  pinMode(encoderPinRF2, INPUT);
  pinMode(encoderPinRB2, INPUT);


  // pins for radio controller
  pinMode(radioPinL, INPUT);
  pinMode(radioPinR, INPUT);
  attachInterrupt(digitalPinToInterrupt(radioPinL), leftPwmIntRise, RISING);
  attachInterrupt(digitalPinToInterrupt(radioPinR), rightPwmIntRise, RISING);


  ledcSetup(1, outputFreq, resolution); // ledcSetup(uint8_t channel, double freq, uint8_t resolution_bits);
  pinMode(motorPinLF, OUTPUT);
  ledcAttachPin(motorPinLF, 1);

  ledcSetup(2, outputFreq, resolution);
  pinMode(motorPinLB, OUTPUT);
  ledcAttachPin(motorPinLB, 2);

  ledcSetup(3, outputFreq, resolution); // ledcAttachPin(uint8_t pin, uint8_t chan);
  pinMode(motorPinRF, OUTPUT);
  ledcAttachPin(motorPinRF, 3);

  ledcSetup(4, outputFreq, resolution);
  pinMode(motorPinRB, OUTPUT);
  ledcAttachPin(motorPinRB, 4);
}


void calculateSpeed() {
  /*
   calculate speed and error for each wheel
   */
  

  for (int wheelNum=0; wheelNum < 4; wheelNum++) {

    float timeBetweenHits = (millis() - lastSpeedCalcTime[wheelNum]);
    if (timeBetweenHits == 0) { return;} // shouldnt be a problem, but avoiding a divide by zero error

    // calculate wheel speed. 16,000 ticks/rev, circumference of tire is 1 meter
    wheelSpeed[wheelNum] = (encoderTicks[wheelNum] * 2.2394) / (16.0 * timeBetweenHits);

    // reset the encoder tick count
    encoderTicks[wheelNum] = 0;

    lastSpeedCalcTime[wheelNum] = millis();

//    // dampen wheel speed calculation
//    if (abs(wheelSpeed[wheelNum]-ws * goingForward[wheelNum]) < 1.5){
//       wheelSpeed[wheelNum] = (wheelSpeed[wheelNum]*0.9 + ws * goingForward[wheelNum] * 0.1);
//
//    } else {
//      wheelSpeed[wheelNum] = (wheelSpeed[wheelNum]*0.5 + ws * goingForward[wheelNum] * 0.5);
//    }

    
    
    // calculate the proportional/integral/derivative errors
    float error = (targetSpeed[wheelNum] - wheelSpeed[wheelNum]);

    derivativeError[wheelNum] += (proportionalError[wheelNum]) - error;

    integratedError[wheelNum] = integratedError[wheelNum] + error;

    proportionalError[wheelNum] = error;
  }

}




  /*
    Have to update the encoders in separate functions for some reason due to the way how interreupts work.
    Format:
    
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

 
// If A channel is freshly high when B channel is high, wheel is going one way.  
// When it is freshly low, the other.
// Only counting changes of A channel when B channel active b/c of encoder issues.
// Another solution could be debouncing, but we don't need that high of prec.

static inline int handleEncoder(int encoderPin, int encoderPinPair) {
  if(digitalRead(encoderPinPair)) {
    return digitalRead(encoderPin) ? 1 : -1;
  }
  return 0;
}

void updateEncoderLF1() {
  encoderTicks[0] += handleEncoder(encoderPinLF1, encoderPinLF2);
}

void updateEncoderLB1() {
  encoderTicks[1] -= handleEncoder(encoderPinLB1, encoderPinLB2);
}

void updateEncoderRF1() {
  encoderTicks[2] += handleEncoder(encoderPinRF1, encoderPinRF2);
}

void updateEncoderRB1() {
  encoderTicks[3] -= handleEncoder(encoderPinRB1, encoderPinRB2);
}

void loop() {

  readSerial(); // look for serial messages

  calculateSpeed(); // calculate the wheel speed


  if (millis() - lastSerialTime > 1000 || (abs(pwmIn[0] - 155) < 50 && abs(pwmIn[0] - 155) > 5) ) { // haven't gotten a serial message for a second. Switch over to radio control
    pidControl = false;
    int limit = 10;
    int j = 0;

//  limit the maximum PWM the radio can give
//    while (j < 2){
//      if (pwmIn[j] > 155 + limit){pwmIn[j] = 155+limit;}
//       if (pwmIn[j] < 155 - limit){pwmIn[j] = 155-limit;}
//       j+=1;
//    }
    
    ledcWrite(1, pwmIn[0]);
    ledcWrite(2, pwmIn[0]);

    ledcWrite(3, pwmIn[1]);
    ledcWrite(4, pwmIn[1]);

  } else {
    pidControl = true;
  }


  if (stopNow) { // a serial message 's' was sent. Tell the robot not to move in every way
    for (int wheelNum = 0; wheelNum < 4; wheelNum++){
      targetSpeed[wheelNum] = 0;
      ledcWrite(wheelNum+1, 0);
      pwmSpeed[wheelNum] = 0;
    }
    
  } else if (pidControl) { // PID serial control
    setMotorSpeed();
  }

  sendSerial();

  delay(1);

}
