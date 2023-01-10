#define RUNFREQ 1 //hz
#define IN_SAMPLE_TIME 1000 //ms
#define BUFFSIZE 10000 // size of storage buffer

volatile int interruptCounter;
int interruptBackups = 0;

bool enabled = false;

hw_timer_t* timer = NULL;

portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

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

// PWM characteristics
const int outputFreq = 100;
const int resolution = 10;

int encoderTicks[4] = {0};

unsigned short wheelSpeeds[4][BUFFSIZE];
int wheelSpeedIndex = 0;

float currTimeMS = 0;
float timeStepMS = 1000/(RUNFREQ);

uint8_t input = {155 155 155 165 165 165};
float endTime = (sizeof(input)/sizeof(*input))*IN_SAMPLE_TIME;


void setup() {
  Serial.begin(115200);

  timer = timerBegin(0, ESP.getCpuFreqMhz(), true); // 1,000,000 ticks/s
  timerAttachInterrupt(timer, &onTimer, true);
  timerAlarmWrite(timer, 1_000_000 / RUNFREQ, true);
  timerAlarmEnable(timer);

  attachInterrupt(digitalPinToInterrupt(encoderPinLF1), updateEncoderLF, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderPinLB1), updateEncoderLB, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderPinRF1), updateEncoderRF, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderPinRB1), updateEncoderRB, RISING);

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

void IRAM_ATTR onTimer() {
  portENTER_CRITICAL_ISR(&timerMux);
  interruptCounter++;
  portEXIT_CRITICAL_ISR(&timerMux);
}

static inline void handleEncoder(int wheelNum) {
  if(enabled) {
    encoderTicks[wheelNum]++;
  }
}

void updateEncoderLF() {
  handleEncoder(0);
}

void updateEncoderLB() {
  handleEncoder(1);
}

void updateEncoderRF() {
  handleEncoder(2);
}

void updateEncoderRB() {
  handleEncoder(3);
}

void calculateSpeed() {
  if(wheelSpeedIndex > BUFFSIZE) {
    Serial.println("Exceeded buffer size");
  }

  for(int wheelNum=0; wheelNum < 4; wheelNum++) {
    unsigned short ticks_per_ms = encoderTicks[wheelNum] * RUNFREQ/1000;
    wheelSpeeds[wheelNum][wheelSpeedIndex] = ticks_per_ms;
  }
  wheelSpeedIndex++;
}

void setSpeed(float currTimeMS) {
  int lowerIndex = currTimeMS / (IN_SAMPLE_TIME);
  int remainder = currTimeMS % (IN_SAMPLE_TIME);
  int slope = (input[lowerIndex+1]-input[lowerIndex])/(IN_SAMPLE_TIME);
  for(int wheelNum = 0; wheelNum < 4; wheelNum++) {
    ledcWrite(wheelNum + 1, slope*remainder);
  }
}

void printResults() {
  Serial.println("Wheel Speeds [ticks/ms]");
  Serial.println("LF, LB, RF, RB");
  for(int wheelNum = 0; wheelNum < 4; wheelNum++) {
    for(int i = 0; i < wheelSpeedIndex; i++) {
      Serial.print(wheelSpeeds[wheelNum][wheelSpeedIndex]);
      Serial.print(", ");
    }
    Serial.println();
  }
}

void loop() {
  if(interruptCounter > 1) {
    interruptBackups++;
  }

  if(!enabled && Serial.available()) {
    if(Serial.read() == 'g') {
      Serial.println("Starting...");
      enabled == true;
    }
  }

  if(interruptCounter > 0) {
    portENTER_CRITICAL(&timerMux);
    interruptCounter--;
    portEXIT_CRITICAL(&timerMux);

    if(enabled) {
      currTimeMS += timeStepMS;

      if(currTime > endTime) {
        enabled = false;
        printResults();
        wheelSpeedIndex = 0;
        memset(wheelSpeeds, 0, sizeof(wheelSpeeds));
        Serial.print("Interrupt Backups: ");
        Serial.println(interruptBackups);
        return;
      }

      calculateSpeed();              
      setSpeed(currTime);
    }
  }
}
