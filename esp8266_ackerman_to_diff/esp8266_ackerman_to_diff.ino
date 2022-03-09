/*
  This program takes the turn and forward radio PWM signals and converts them to left and right wheel commands.
*/



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

int forwardRadioPin = 12; // D6
int turnRadioPin = 13; // D7

int rfWheel = 2; // d4
int rbWheel = 5; // d1
int lfWheel = 14; // d5
int lbWheel = 4; // d2

unsigned long lastMoveTime = 0;

unsigned long lastPrintTime = 0;

void setup() {
  Serial.begin(115200);

  // set the radio pins for the motor controller
  pinMode(forwardRadioPin, INPUT); // D0 - FORWARD RADIO
  pinMode(turnRadioPin, INPUT); // D1 - TURN RADIO

  // set the output pins controlling the motor
  pinMode(rfWheel, OUTPUT);
  pinMode(rbWheel, OUTPUT);
  pinMode(lfWheel, OUTPUT);
  pinMode(lbWheel, OUTPUT);

  // set the write frequency and range (makes it the same as the ESP32)
  analogWriteFreq(100);
  analogWriteRange(1023);
}

void loop() {

  // max pwm is 200
  // min pwm is 110
  // neutral is 155

  // read the PWM values from the two radio pins
  // the "/ 10 + 5" part is to make it have the same 10-bit precision as the output PWM
  int forwardPWM = pulseIn(forwardRadioPin, HIGH) / 10 + 5;
  int turnPWM = pulseIn(turnRadioPin, HIGH) / 10 + 5;


  if (forwardPWM == 0 || turnPWM == 0) {
    // not connected to controller, set all speeds to 0

    analogWrite(rfWheel, 0);
    analogWrite(rbWheel, 0);

    analogWrite(lfWheel, 0);
    analogWrite(lbWheel, 0);

    delay(10);
    return;
  }
//
//  forwardPWM = 155 - forwardPWM + 155;
//  turnPWM = 155 - turnPWM + 155;

  // it is connected to the controller; print the incoming PWM
  Serial.print("input: ");
  Serial.print(forwardPWM);
  Serial.print(", ");
  Serial.println(turnPWM);

  // if the PWM is close to neutral, make it neutral
  if (abs(forwardPWM - 155) < 2) {
    forwardPWM = 155;
  }
  if (abs(turnPWM - 155) < 2) {
    turnPWM = 155;
  }

  // if the PWM is less than expected, make it neutral
  if (forwardPWM < 50) {
    forwardPWM = 155;
  }
  if (turnPWM < 50) {
    turnPWM = 155;
  }


  // if the PWM is slightly more/less than expected, make it the maximum
  if (turnPWM < 110) {
    turnPWM = 110;
  } else if (turnPWM > 200) {
    turnPWM = 200;
  }
  if (forwardPWM < 110) {
    forwardPWM = 110;
  } else if (forwardPWM > 200) {
    forwardPWM = 200;
  }


  // initialize the output PWM
  int rightPWM = 0;
  int leftPWM = 0;

  // map the turn value from -1 to 1
  float turnDifPercent = (turnPWM - 155.0) / 45.0;

  // this is the magnitude of the range the PWM is outputted
  int speedRange = abs(forwardPWM - 155) * 2;

  if (turnDifPercent > 0) { // go left
    leftPWM = forwardPWM;
    rightPWM = forwardPWM - (turnDifPercent * speedRange);

  } else if (turnDifPercent < 0) { // go right
    rightPWM = forwardPWM;
    leftPWM = forwardPWM - ((-turnDifPercent) * speedRange);

  } else { // straight
    rightPWM = forwardPWM;
    leftPWM = forwardPWM;

  }


  if (abs(rightPWM - 155) < 3) {
    rightPWM = 155;
  }
  if (abs(leftPWM - 155) < 3) {
    leftPWM = 155;
  }

  // for some reason the motor clicks if it is left in neutral for too long. Turn off all PWM signals after 20 seconds
  if (leftPWM == 155 && rightPWM == 155) {
    if (lastMoveTime - millis() > 20000) {
      leftPWM = 0;
      rightPWM = 0;
    }
  } else {
    lastMoveTime = millis();
  }


  // print relevant variables 2x every second
  if (lastPrintTime - millis() > 500) {
    Serial.println("percent: " + String(turnDifPercent, 3));
    Serial.println(String(leftPWM) + ", " + String(rightPWM));
    lastPrintTime = millis();
  }


  // set the PWM of the wheels
  analogWrite(rfWheel, rightPWM);
  analogWrite(rbWheel, rightPWM);

  analogWrite(lfWheel, leftPWM);
  analogWrite(lbWheel, leftPWM);


  delay(10);

}
