#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

int readRate = 200; // milliseconds

float bnoHeading = 0; // degrees

/*
  The connections are:

  SDA - D21
  SCL - D22


*/



unsigned long lastBNOReadTime = 0;


void updateHeading() {
  sensors_event_t orientationData, linearAccelData;
  bno.getEvent(&orientationData, Adafruit_BNO055::VECTOR_EULER);
  bnoHeading = orientationData.orientation.x;
}



void setup() {

  Serial.begin(115200);
  delay(1000);
  if (bno.begin()){
    Serial.println("connected to BNO");
  } else {
    Serial.println("unable to connect to BNO");
  }

}


void loop() {

  if (lastBNOReadTime + readRate < millis()) { // time to read BNO position
    updateHeading();
    
    Serial.println("h" + String(bnoHeading));
    
    lastBNOReadTime = millis();
  }





}
