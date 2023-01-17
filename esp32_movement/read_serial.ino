

bool haveSerial = false;
unsigned long lastSerialReadTime = 0;


unsigned long lastPrintTime = 0;

void readSerial() {
  /*
    Read the serial message and obey the given command.

    The serial is given as a single letter ID and a value. Letter IDs are listed below:

    Serial Message Prefixes:

    Informational:
    w: each wheel speed

    Command:
    w: both wheel speed (mph) (separated by a comma, ex: w1.5,1.2)
    l: left wheel speed (mph) (old)
    r: right wheel speed (mph) (old)
    s: stop - sets target speed to 0
    g: go (unused)

    Other:
    .: console log, the Pi/Nano ignore these messages
    -: acknowledge the message
    +: say the message is irrelevant to this microcontroller (unused currently)


  */
  if (Serial.available()) {

    lastSerialTime = millis(); // timer looking to reset radio. Different from lastSerialReadTime

    if (haveSerial && lastSerialReadTime > 30) { // wait for > 30 milliseconds for the full serial message to arrive
      haveSerial = false; // reset tag for next message
      char commandType = Serial.read();
      String serialMsg = "";



      while (Serial.available()) {
        char b = Serial.read();
        serialMsg += b;
      }


      if (commandType == 'w') {
        int i = 0;
        bool readLeft = true;
        String leftSpeed = "";
        String rightSpeed = "";
        while (i < serialMsg.length()) {
          if (serialMsg[i] == ',') {
            readLeft = false;
          } else if (readLeft) {
            leftSpeed += serialMsg[i];
          } else {
            rightSpeed += serialMsg[i];
          }
          i++;
        }
        targetSpeed[0] = leftSpeed.toFloat();
        targetSpeed[1] = leftSpeed.toFloat();
        
        targetSpeed[2] = rightSpeed.toFloat();
        targetSpeed[3] = rightSpeed.toFloat();


         for (int j=0; j<4; j++){
          integratedError[j] = 0;
          derivativeError[j] = 0;
         }
        
      //  Serial.println("target speed" + String(targetSpeed[0]) + ", "+ String(targetSpeed[2]));
        
        
        Serial.println("-w" + String(int(targetSpeed[0]) * 10 + int(targetSpeed[2])));

      } else if (commandType == 'l') {
        targetSpeed[0] = serialMsg.toFloat();
        targetSpeed[1] = serialMsg.toFloat();
        Serial.println("-l" + String(int(targetSpeed[0])));

      } else if (commandType == 'r') {
        targetSpeed[2] = serialMsg.toFloat();
        targetSpeed[3] = serialMsg.toFloat();
        Serial.println("-r" + String(int(targetSpeed[1])));

      } else if (commandType == 's') {
        //        stopNow = true;
        targetSpeed[0] = 0;
        targetSpeed[1] = 0;
        pwmSpeed[0] = 0;
        pwmSpeed[1] = 0;
        Serial.println("-s");

      } else if (commandType == 'g') {
        //        stopNow = false;
        Serial.println("-g");
      }

    } else if (!haveSerial) {
      haveSerial = true;
      lastSerialReadTime = millis();
    }

  }

}


void sendSerial() {
  /*
   Print out whatever relevant information here
   */
  
  if (millis() - lastPrintTime > 100) {
//        Serial.println("target speed: " + String(targetSpeed[0]));
//        Serial.println("pwm speed: " + String(pwmSpeed[0]));
//        Serial.println("l" + String(wheelSpeed[0], 3));
    //
//        Serial.println("r" + String(wheelSpeed[1], 3));
    Serial.println(String(wheelSpeed[0], 3) + "," + String(wheelSpeed[1], 3) + "," + String(wheelSpeed[2], 3) + "," + String(wheelSpeed[3], 3));
 //   Serial.println("," + String(pwmSpeed[0], 3) + "," + String(pwmSpeed[1], 3) + "," + String(pwmSpeed[2], 3) + "," + String(pwmSpeed[3], 3));
  //  Serial.println("," + String(proportionalError[0], 3) + "," + String(proportionalError[1], 3) + "," + String(proportionalError[2], 3) + "," + String(proportionalError[3], 3));

    lastPrintTime = millis();

  }
}
