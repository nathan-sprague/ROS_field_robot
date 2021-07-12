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

const int steerPin = 16;  // motor (D0)


int pwmSpeed = 0;
bool stopNow = false;
int targetPosition = 0;
unsigned long lastPrint = 0;
unsigned long lastPrint2 = 0;

bool manualPWM = false;

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

    if (commandType=='_'){
      return;
    }

    if (commandType == 'z') {
      Serial.println(".manual");
      
      pwmSpeed = commandVal;
      analogWrite(steerPin, commandVal);
      Serial.println("-z" + String(commandVal));
      manualPWM = true;

    } else if (commandType == 'p') {
      manualPWM = false;
      if (commandVal!=targetPosition){
        Serial.println("-p" + String(int(commandVal)));
        Serial.println(".serial says move to angle: " + String(commandVal));
       // Serial.println(".");
        targetPosition  = commandVal;
      
        passedTarget = 0;
      }

    } else if (commandType == 's') {
      manualPWM = false;
      Serial.println("-s");
      Serial.println(".emergency stop");
      stopNow = true;

    } else if (commandType == 'g') {
      manualPWM = false;
      Serial.println("-g");
//      Serial.println(".go");
      stopNow = false;
    } else if (commandType == 'r') {
      manualPWM = false;
      Serial.println("-r");
      Serial.println(".restarting");
      ESP.restart();
    } else if (commandType == 'f') {
      Serial.println("+f");
    } else if (commandType == 'b') {
      Serial.println("+b");
    }
  }
}





void setup() {
  Serial.begin(115200);

  analogWriteFreq(100);
  
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

unsigned long lastRequest = 0;
int lastPWMspeed = 0;

int lastReading = 0;

unsigned long lastAngleReading = 0;

unsigned long gotSerialTime = 0;
bool gotSerial = false;

void loop() {

  if (Serial.available() && !gotSerial) {
    gotSerialTime = millis();
    gotSerial = true;
  }
  if (gotSerial && gotSerialTime+15 < millis()){
    processSerial();
    gotSerial = false;
  }
  if (gotSerialTime+1500 < millis()){
    analogWrite(steerPin, 0);
    pwmSpeed = 0;
    return;
  } else {
    pwmSpeed = lastPWMspeed;
  }

  if (lastRequest + 300 < millis()) {
//    talkToESP();
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

  if (millis()-10<lastAngleReading) {
    return;
  }
  
  lastAngleReading = millis();
  
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


  if (limitOvershoot() && pwmSpeed != lastPWMspeed && !manualPWM) {
    analogWrite(steerPin, pwmSpeed);
    lastPWMspeed = pwmSpeed;
  }


}
