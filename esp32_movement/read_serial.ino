

bool haveSerial = false;
unsigned long lastSerialReadTime = 0;


unsigned long lastPrintTime = 0;

void readSerial() {
  /*
    Read the serial message and obey the given command.

    The serial is given as a single letter ID and a value. Letter IDs are listed below:

    Serial Message Prefixes:

    Informational:
    x: gps latitude
    y: gps longitude
    l: left wheel speed
    r: right wheel speed
    h: heading

    Command:
    l: left wheel speed
    r: right wheel speed
    s: stop
    g: go

    Other:
    .: console log, the Pi/Nano ignore these messages
    -: acknowledge the message
    +: say the message is irrelevant to this microcontroller


  */
  if (Serial.available()) {
   
    lastSerialTime = millis(); // timer looking to reset radio. Different from lastSerailReadTime

    if (haveSerial && lastSerialReadTime > 30) { // must wait for >30 milliseconds for the full serial message to arrive
      haveSerial = false; // reset tag for next message
      char commandType = Serial.read();
      String serialMsg = "";

     

      while (Serial.available()) {
        char b = Serial.read();
        serialMsg += b;
      }
      Serial.println("serial message: " + serialMsg);

      if (commandType == 'l') {
        targetSpeed[0] = serialMsg.toFloat();
        Serial.println("-l" + String(int(targetSpeed[0])));

      } else if (commandType == 'r') {
        targetSpeed[1] = serialMsg.toFloat();
        Serial.println("-r" + String(int(targetSpeed[1])));

      } else if (commandType == 's') {
        stopNow = true;
        targetSpeed[0] = 0;
        targetSpeed[1] = 0;
        pwmSpeed[0] = 0;
        pwmSpeed[1] = 0;
        Serial.println("-s");

      } else if (commandType == 'g') {
        stopNow = false;
        Serial.println("-g");
      }



    } else if (!haveSerial) {
      haveSerial = true;
      lastSerialReadTime = millis();
    }



  }

}


void sendSerial() {
  if (millis() - lastPrintTime > 1000) {
//    Serial.println("target speed: " + String(targetSpeed[0]));
//    Serial.println("pwm speed: " + String(pwmSpeed[0]));
    Serial.println("l" + String(wheelSpeed[0], 3));
    Serial.println("r" + String(wheelSpeed[1], 3));
    lastPrintTime = millis();
    
  }
}
