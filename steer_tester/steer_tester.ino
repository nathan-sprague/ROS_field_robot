// 330 max to right
// 190 max to left
#define USE_GPS false

#include <Wire.h>
#include <SparkFun_u-blox_GNSS_Arduino_Library.h> //http://librarymanager/All#SparkFun_u-blox_GNSS
SFE_UBLOX_GNSS myGNSS;
const int motorPin = 4;  // motor (D2)
int pwmSpeed = 0;
bool stopNow = false;
int targetPosition = 0;
unsigned long lastPrint = 0;
unsigned long lastPrint2 = 0;

int averagePosition = 0;


void processSerial(){
  if (Serial.available()){
      Serial.println(".got message");
      char commandType = Serial.read();
      String serialMsg = "";
      
    while (Serial.available()){
      char b = Serial.read();
      serialMsg += b;
    }
    int commandVal = serialMsg.toInt();
    
    Serial.println(serialMsg);
  

    if (commandType == 'z'){
      Serial.println("manual");
      pwmSpeed = commandVal;
      analogWrite(motorPin, commandVal);

    } else if (commandType == 'p'){
      Serial.println("to angle");
      targetPosition  = commandVal;      

    } else if (commandType == 's'){
      Serial.println("emergency stop");
      stopNow = true;
      
    } else if (commandType == 'g'){
      Serial.println("go");
      stopNow = false;  
    }
    
  }
}

void beginGPS(){
  Wire.begin();

  if (myGNSS.begin() == false) { //Connect to the u-blox module using Wire port
    Serial.println(F("u-blox GNSS not detected at default I2C address. Please check wiring. Freezing."));
    while (1);
  }

  myGNSS.setI2COutput(COM_TYPE_UBX); //Set the I2C port to output UBX only (turn off NMEA noise)
  myGNSS.saveConfigSelective(VAL_CFG_SUBSEC_IOPORT); //Save (only) the communications port settings to flash and BBR

}

void getPosition(){

  float latitude = myGNSS.getLatitude();
  Serial.println("x" + String(latitude));
    
  float longitude = myGNSS.getLongitude();
  Serial.println("y" + String(longitude));
}



void setup() {
  Serial.begin(115200);
  analogWriteFreq(100);

   if (USE_GPS){
    beginGPS();
  }
}

void targetToSpeed(){
  int distToTarget = targetPosition - averagePosition ;
  // possible difference in angle:
  // on left, go to right -> 50 - (-50) = 100
  // on right, go to left -> -50 - 50 = -100
  // acceptable speeds:
  // 170 - 190 (right)
  // 120 - 140 (left)
  bool canPrint = false;
  if (lastPrint2+200<millis()){
    lastPrint2 = millis();
    canPrint = true;
  }
  if (canPrint) {
    Serial.println("dist to target: " + String(distToTarget));
  }
  if (distToTarget < 0) { // target to the left
    pwmSpeed = distToTarget * (30.0 / 100.0) + 150;
  } else if (distToTarget > 0) { // target to the right
    pwmSpeed = (distToTarget * (30.0 / 100.0)) + 160;

  } else {
    if (canPrint){
      Serial.println("arrived!");
    }
    pwmSpeed = 0;
  }
  if (canPrint){
    Serial.println("PWM speed: " + String(pwmSpeed));
  }
  
 
  analogWrite(motorPin, pwmSpeed);
  
}

float potToAngle(int pot) {
// center 240
// right 330
// left 190
    float angle = 0;
    if (pot>240) { // right
      angle = (pot-240) * 45 / (330.0-240);
    } else { // left
      angle = (pot-240) * 45 / (240-190.0);
    }

    return angle;
}

void limitOvershoot(){
    if (averagePosition>50 && pwmSpeed > 150){
      Serial.println("stopped (too far right)");
      analogWrite(motorPin, 0);
      pwmSpeed = 0;
      return;
  } else if (averagePosition<-50 && pwmSpeed < 150 && pwmSpeed > 5){
      Serial.println("stopped (too far left)");
      analogWrite(motorPin, 0);
      pwmSpeed = 0;
      return;
  }
}


unsigned long lastPosReadTime = 0;


void loop() {
  processSerial();
  
  if (stopNow){
    analogWrite(motorPin, 0);
    pwmSpeed = 0;
    delay(10);
    return;
  }

  int currentPosition = potToAngle(analogRead(A0));
  
  averagePosition = (averagePosition*5 + currentPosition)/6;


  targetToSpeed();
  
  if (lastPrint+200<millis()){
    lastPrint = millis();
    Serial.println("angle: " + String(averagePosition) + ", Speed: " + String(pwmSpeed));

  }

  if (USE_GPS && lastPosReadTime+2000<millis()){
    getPosition();
    lastPosReadTime = millis();
  }

 
  limitOvershoot();
  
  delay(10);

}
