

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
    d: both wheel distances to travel (ft) and top speed when doing it (mph) (separated by a comma and letter 's', ex: d1.5,1.2s3.1)
    w: both wheel speed (mph) (separated by a comma, ex: w1.5,1.2)
    l: left wheel speed (mph) (old)
    r: right wheel speed (mph) (old)
    s: stop - sets target speed to 0, clears distance goals
    g: go (unused)

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
        targetSpeed[1] = rightSpeed.toFloat();
        useTargetDist[0] = false;
        useTargetDist[1] = false;
        
        Serial.println("-w" + String(int(targetSpeed[0]) * 10 + int(targetSpeed[1])));

      } else if (commandType == 'd') {
        int i = 0;
        bool readLeft = true;
        bool readDist = true;
        String leftDist = "";
        String rightDist = "";
        String topSpeed = "";
        while (i < serialMsg.length()) {
          if (serialMsg[i] == ',') {
            readLeft = false;
          } else if (serialMsg[i] == 's') {
            readDist = false;
          } else if (readLeft) {
            leftDist += serialMsg[i];
          } else if (readDist) {
            rightDist += serialMsg[i];
          } else {
            topSpeed += serialMsg[i];
          }
          i++;
        }

        targetDistance[0] = leftDist.toFloat();
        targetDistance[1] = rightDist.toFloat();
        distTravelled[0] = 0;
        distTravelled[1] = 0;
        if (topSpeed != "") {
          topDistSpeed = topSpeed.toFloat();
        }
        useTargetDist[0] = true;
        useTargetDist[1] = true;
        Serial.println("-d" + String(int(targetDistance[0]) * 10 + int(targetDistance[1])));

      } else if (commandType == 'l') {
        useTargetDist[0] = false;
        targetSpeed[0] = serialMsg.toFloat();
        Serial.println("-l" + String(int(targetSpeed[0])));

      } else if (commandType == 'r') {
        useTargetDist[1] = false;
        targetSpeed[1] = serialMsg.toFloat();
        Serial.println("-r" + String(int(targetSpeed[1])));

      } else if (commandType == 's') {
        //        stopNow = true;
        targetSpeed[0] = 0;
        targetSpeed[1] = 0;
        pwmSpeed[0] = 0;
        pwmSpeed[1] = 0;
        useTargetDist[0] = false;
        useTargetDist[1] = false;
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
  if (millis() - lastPrintTime > 200) {
    //    Serial.println("target speed: " + String(targetSpeed[0]));
      //  Serial.println("pwm speed: " + String(pwmSpeed[1]));
    //    Serial.println("l" + String(wheelSpeed[0], 3));
    //
    //    Serial.println("r" + String(wheelSpeed[1], 3));
    Serial.println("w" + String(wheelSpeed[0], 3) + "," + String(wheelSpeed[1], 3));

    if (useTargetDist[0] || useTargetDist[1]) {
      Serial.println("d" + String(distTravelled[0], 3) + "," + String(distTravelled[1], 3));
    }

        //  Serial.println(".pwm control: " + String(pwmIn[0]) + ", " + String(pwmIn[1]));

    lastPrintTime = millis();

  }
}
