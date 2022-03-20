#define DEBUGMODE true

const int motorPin = 4;  // motor (D2)
const int hallPin = 5; // speed sensor pin (D1)
const int otherHallPin = 0; // speed sensor pin (D3)

bool stopNow = false;
bool manual = false;
int PWMSignal = 0;

float targetSpeed = 0;

unsigned long lastTargetChange = 0; // timer to reassess the speed every half second

void processSerial() {
  if (Serial.available()) {

    if (DEBUGMODE) {
      Serial.println(".got message");
    }
    
    // get keyword
    char commandType = Serial.read();
    String serialMsg = "";

    // convert the rest of the message to an int
    while (Serial.available()) {
      char b = Serial.read();
      serialMsg += b;
    }
    int commandVal = serialMsg.toInt();

    if (DEBUGMODE) {
      Serial.println(serialMsg);
    }
    if (commandType == 'f')  {
      if (DEBUGMODE) {
        Serial.println(".target speed " + String(commandVal));
      }
      targetSpeed = commandVal; // set speed
      manual = false;
      lastTargetChange = millis();

    } else if (commandType == 'z') {
      if (DEBUGMODE) {
        Serial.println(".manual");
      }
      manual = true;
      analogWrite(motorPin, commandVal);
      PWMSignal = commandVal;

    } else if (commandType == 's') {
      if (DEBUGMODE) {
        Serial.println(".emergency stop");
      }
      stopNow = true;

    } else if (commandType == 'g') {
      if (DEBUGMODE) {
        Serial.println(".go");
      }
      stopNow = false;
    }

  }
}


unsigned long lastMsgTime = 0; // timer to prevent over-printing

const float wheelCircum = 13 * 2 * 3.14159;
const int numHoles = 16;

float smoothSpeed = 0; // speed used for all calculations. Smooths and averages speeds to limit outliers

bool goingForward = true;

// speed variables for the left wheel
unsigned long distTraveled = 0;
int lastRead = 0;
unsigned long lastHitTime = 0;
float wheelSpeed = 0;
unsigned int timeBetweenHoles = 0;


// speed variables for the right wheel
unsigned long distTraveledY = 0;
int lastReadY = 0;
unsigned long lastHitTimeY = 0;
float wheelSpeedY = 0;
unsigned int timeBetweenHolesY = 0;

const float tickTimeTomph = (wheelCircum / 12.0) / numHoles * 1000.0 * 3600 / 5280; // constant for calculating wheel speed

void getWheelSpeed() {
  bool speedChange =  false;
  int  x = digitalRead(hallPin);


  if (lastRead != x && x == 0 && millis() - lastHitTime > 0) { // There was a change in what it sensed and did sense something
    timeBetweenHoles = millis() - lastHitTime;

    //    float rpm = 60.0 / numHoles / (timeBetweenHoles / 1000.0);

    wheelSpeed = tickTimeTomph / timeBetweenHoles; // mph
    distTraveled += wheelCircum / numHoles;

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
    distTraveledY += wheelCircum / numHoles;


    lastHitTimeY = millis();

  } else if (timeBetweenHolesY < millis() - lastHitTimeY) {
    
    wheelSpeedY = tickTimeTomph / (millis() - lastHitTimeY);
    wheelSpeedY = int(wheelSpeedY);
    speedChange = true;
  }
  lastReadY = y;
  
  // if going backwards, negate the wheel speeds
  if (!goingForward){
    wheelSpeed = abs(wheelSpeed)*-1;
    wheelSpeedY = abs(wheelSpeedY)*-1;
  }

  // sanity check: if the smooth speed is way to high, bring it to the most recent value
  if (abs(smoothSpeed)>100){
    smoothSpeed = (wheelSpeed+wheelSpeedY)/2;
  }

  
  if (speedChange) { // Don't reasses the situation unless there is a change in speed

    // get average speed between the wheels and past speeds. #s 3 and 5 were arbitrarily chosen
    smoothSpeed = (smoothSpeed * 3 + (wheelSpeed + wheelSpeedY)) / 5;

    // Direction may change if the speed is zero
    if (smoothSpeed > -0.1 && smoothSpeed < 0.1){ // cant compare it directly to zero because a float
      
      if (PWMSignal > 155){ // signal tells it to go forwards, go forwards
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
      Serial.println("d" + String(distTraveled));
      Serial.println("w" + String(smoothSpeed));
    }
    lastMsgTime = millis();
  }


}




int atTarget = 0;
int lastTarget = 0;
float intError = 0;

void setMotorSpeed() {
  //  if (wifiControl && (millis() - lastCommandTime)> 1500){
  //    Serial.println(".error: not enough communication to be safe (>1.5s timeout");
  //    analogWrite(motorPin, 0);
  //    return;
  //  }


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
  bool changePWM = false; // should you change the PWM value

  if (lastTargetChange + 500 < millis()) {
    changePWM = true;
    lastTargetChange = millis();

    if (lastTarget!=int(targetSpeed)){ // commanded target changed
      if (DEBUGMODE) {
        Serial.println("target changed");
      }
      
      // you are not at the target speed
      lastTarget = targetSpeed;
      intError = 0;
      if (smoothSpeed>targetSpeed)
        atTarget = 1;
      else {
        atTarget = -1;
      }
    }

    
  }

  if (targetSpeed == 0) {
    analogWrite(motorPin, 0);
    PWMSignal = 155;

  } else if (smoothSpeed != targetSpeed && changePWM) {



    // convert difference in speed to a difference in PWM
    float PWMdif = ((targetSpeed - smoothSpeed) / 14.5) * (90.0 / 2); // map (-14.5)-(14.5) to (110)-(200)

    // round the float to the nearest whole number (not int)
    if (PWMdif - int(PWMdif)>0.5) {PWMdif = int(PWMdif)+1;}

    if (PWMdif - int(PWMdif)<-0.5) {PWMdif = int(PWMdif)-1;}

    if (DEBUGMODE) {
      Serial.print("old pwm: " + String(PWMSignal));
    }

    // change the PWM signal according to the error
    PWMSignal = PWMSignal + PWMdif;

    // you went from being too slow/fast to the other way around, begin collecting integrated differences
    if (atTarget==-1 && smoothSpeed>targetSpeed)
      atTarget = 0;
    else if (atTarget==1 && smoothSpeed<targetSpeed){
      atTarget = 0;
    }

    // add up the integrated error
    if (atTarget == 0){
      intError = (intError*5 + (targetSpeed - smoothSpeed))/6;
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
    if (targetSpeed<0 && smoothSpeed>0 && goingForward){
      PWMSignal = 155;
    }

    analogWrite(motorPin, PWMSignal);

  }
}


void setup() {
  Serial.begin(115200);
  pinMode(hallPin, INPUT);
  pinMode(otherHallPin, INPUT);
  analogWriteFreq(100);

}



void loop() {
  processSerial();

  if (stopNow) {
    targetSpeed = 0;
    PWMSignal = 0;
    analogWrite(motorPin, 0);
    delay(10);
    return;
  }

  getWheelSpeed();
  
  if (!manual) {
    setMotorSpeed();
  }
  delay(10);

}