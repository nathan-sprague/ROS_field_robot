#if USE_GPS








float bnoCorrection = 0;
float bnoHeading = 0;
float heading = 0;
bool gpsUpdated = false;



/*
The connections are:

SDA - D21
SCL - D22


 */


//bool useMag = false;
bool useGPS = true;
bool useBNO = true;

unsigned long lastPosReadTime = 0;

unsigned long lastHeadReadTime = 0;

unsigned long lastBNOReadTime = 0;



bool beginGPS() {
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

  float heading = float(myGNSS.getHeading()) / 100000;
  bnoCorrection = heading - bnoHeading;
  

  float latitude = float(myGNSS.getLatitude()) / 10000000;
  Serial.println("x" + String(latitude, 7)); // lat

  float longitude = float(myGNSS.getLongitude()) / 10000000;
  Serial.println("y" + String(longitude, 7)); // long

  uint32_t accuracy = myGNSS.getHorizontalAccuracy();
  // Now define float storage for the heights and accuracy
  float f_accuracy;


  // Convert the horizontal accuracy (mm * 10^-1) to a float
  f_accuracy = accuracy;
  // Now convert to m
  f_accuracy = f_accuracy / 10000.0; // Convert from mm * 10^-1 to m


  Serial.print("a");
  Serial.println(f_accuracy, 4); // Print the accuracy with 4 decimal places

}

void updateHeading() {
  sensors_event_t orientationData , linearAccelData;
  bno.getEvent(&orientationData, Adafruit_BNO055::VECTOR_EULER);
  bnoHeading = orientationData.orientation.x;
}





void readGPS(){

  
  if (useGPS && lastPosReadTime + 600 < millis()) {
    getPosition();
    lastPosReadTime = millis();
    
  } else if (lastPosReadTime + 1000 < millis()) {
    Serial.println("o2"); // error
    lastPosReadTime = millis();
  }
  

  
  if (useBNO && lastBNOReadTime + 300 < millis()) {
    updateHeading();
    heading = bnoCorrection + bnoHeading;
    lastBNOReadTime = millis();
    
  } else if (lastBNOReadTime + 1000 < millis()) {
    Serial.println("o3"); // error
    lastBNOReadTime = millis();
  }

  if (lastHeadReadTime + 300 < millis()) {
    Serial.println("h" + String(heading)); // heading
    lastHeadReadTime = millis();
  }

  

  

}

#endif
