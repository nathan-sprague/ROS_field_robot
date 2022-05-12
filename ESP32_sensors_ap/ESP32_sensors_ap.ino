#include <Wire.h>
#include <SparkFun_u-blox_GNSS_Arduino_Library.h> //http://librarymanager/All#SparkFun_u-blox_GNSS



#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>



SFE_UBLOX_GNSS myGNSS;

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);


float bnoCorrection = 0;
float bnoHeading = 0;
float heading = 0;
bool gpsUpdated = false;

float lastHeadingLong = 0;
float lastHeadingLat = 0;

/*
  The connections are:

  SDA - D21
  SCL - D22


*/


//bool useMag = false;
bool useGPS = false;
bool useBNO = false;

unsigned long lastPosReadTime = 0;

unsigned long lastHeadReadTime = 0;

unsigned long lastBNOReadTime = 0;

unsigned long gotSerialTime = 0;
bool gotSerial = false;


void processSerial() {
  if (Serial.available()) {

    char commandType = Serial.read();

    String serialMsg = "";

    while (Serial.available()) {
      char b = Serial.read();
      serialMsg += b;
    }
    int commandVal = serialMsg.toInt();


    if (commandType == 'e') { // requested esp type. Tell it this is ap
      Serial.println("-e");
      //      Serial.println("3");

      //    } else if (commandType == 'r') {
      //      Serial.println(".restarting");
      //      ESP.restart();

    } else if (commandType == '.') { // empty message, ignore

    } else {
      Serial.println("-" + String(commandType));
    }
  }
}


bool beginGPS() {
  Serial.println(".starting gps");
  Wire.begin();

  if (myGNSS.begin() == false) { //Connect to the u-blox module using Wire port
    Serial.println(".u-blox GNSS not detected");
    return false;
  }

  Serial.println(".gps set up successfully");
  myGNSS.setI2COutput(COM_TYPE_UBX); //Set the I2C port to output UBX only (turn off NMEA noise)
  myGNSS.saveConfigSelective(VAL_CFG_SUBSEC_IOPORT); //Save (only) the communications port settings to flash and BBR
  return true;
}

void getPosition() {
  //  Serial.println(".getting position");




  float latitude = float(myGNSS.getLatitude()) / 10000000;
  Serial.println("x" + String(latitude, 7));

  float longitude = float(myGNSS.getLongitude()) / 10000000;
  Serial.println("y" + String(longitude, 7));

  uint32_t accuracy = myGNSS.getHorizontalAccuracy();
  // Now define float storage for the heights and accuracy
  float f_accuracy;


  // Convert the horizontal accuracy (mm * 10^-1) to a float
  f_accuracy = accuracy;
  // Now convert to m
  f_accuracy = f_accuracy / 10000.0; // Convert from mm * 10^-1 to m


  float longDif = longitude - lastHeadingLong;
  float latDif = latitude - lastHeadingLat;

  Serial.println(".compare: change=" + String((longDif*longDif + latDif*latDif)*100000) + ", accuracy=" + String(accuracy));
  
  if ((longDif*longDif + latDif*latDif)*100000 > accuracy){// only count heading change if robot moved much
    float heading = float(myGNSS.getHeading()) / 100000;
    bnoCorrection = heading - bnoHeading;
    lastHeadingLong = longitude;
    lastHeadingLat = latitude;
    Serial.println(".change good enough for heading calculations");

    
  } else {
    Serial.println(".change not good enough for heading calculations");
  }
  

  Serial.print("a");
  Serial.println(f_accuracy, 4); // Print the accuracy with 4 decimal places

}

void updateHeading() {
  sensors_event_t orientationData, linearAccelData;
  bno.getEvent(&orientationData, Adafruit_BNO055::VECTOR_EULER);
  bnoHeading = orientationData.orientation.x;
}



void setup() {

  Serial.begin(115200);

  delay(1000);
  useGPS = beginGPS();
  useBNO = bno.begin();
  if (!useGPS) {
    ESP.restart();
  }
}






void loop() {

  // process serial
  if (Serial.available() && !gotSerial) {
    gotSerialTime = millis();
    gotSerial = true;
  }
  if (gotSerial && gotSerialTime + 15 < millis()) { // wait 15 milliseconds for the whole serial message to come through
    processSerial();
    gotSerial = false;
  }


  if (useGPS && lastPosReadTime + 600 < millis()) { // time to read position
    getPosition();
    lastPosReadTime = millis();

  } else if (lastPosReadTime + 1000 < millis()) {
    //    Serial.println("o2");
    lastPosReadTime = millis();
  }



  if (useBNO && lastBNOReadTime + 300 < millis()) { // time to read BNO position
    updateHeading();
    heading = bnoCorrection + bnoHeading;
    lastBNOReadTime = millis();

  } else if (lastBNOReadTime + 1000 < millis()) {
    //    Serial.println("o3");
    lastBNOReadTime = millis();
  }

  if (lastHeadReadTime + 300 < millis()) {
    Serial.println("h" + String(heading));
    lastHeadReadTime = millis();
  }





}
