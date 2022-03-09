

void processSerial() {
  /*
    Read the serial message and obey the given command.

    The serial is given as a single letter ID and a value. Letter IDs are listed below:

    Serial Message Prefixes:

    Informational:
    a: left wheel speed
    b: right wheel speed
    c: esp type


    Command:
    d: left wheel speed
    e: right wheel speed
    f: request type of esp

    s: emergency stop
    g: go (when stoped from emergency)

    Other:
    .: console log, the Pi/Nano ignores whatever comes after
    +: acknowledge the message
    -: say the message is irrelevant to this microcontroller


  */


  if (Serial.available()) {

    delay(10); // wait a bit for the whole serial message to come in


    // get keyword
    char commandType = Serial.read();

    // get the rest of the message
    String serialMsg = "";
    while (Serial.available()) {
      char b = Serial.read();
      serialMsg += b;
    }

    Serial.println(serialMsg);


    if (commandType == 'd') { // set left wheel speed
      targetSpeeds[0] = serialMsg.toFloat();
      targetSpeeds[1] = serialMsg.toFloat();
      Serial.println("+d" + String(int(targetSpeeds[0])) ); // confirm message given


    } else if (commandType == 'e') { // set left wheel speed
      targetSpeeds[2] = serialMsg.toFloat();
      targetSpeeds[3] = serialMsg.toFloat();
      Serial.println("+e" + String(int(targetSpeeds[1])) ); // confirm message given

    } else if (commandType == 'c') { // asking for esp type
      Serial.println("c1"); // say it is the movement esp

    } else if (commandType == 's') { // emergency stop
      stopNow = true;
      Serial.println("+s");

    } else if (commandType == 'g') { // emergency stop
      stopNow = false;
      Serial.println("+g");

    } else if (commandType == '.') { // empty message, ignore

    } else { // irrelevant or unrecognized message, ask pi/nano not to send it again to reduce serial traffic
      Serial.println("-" + commandType);
    }

  }
}
