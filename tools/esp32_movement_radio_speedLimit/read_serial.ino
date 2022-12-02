

bool haveSerial = false;
unsigned long lastSerialReadTime = 0;


unsigned long lastPrintTime = 0;



void sendSerial() {
  if (millis() - lastPrintTime > 500) {
//        Serial.println("target speed: " + String(targetSpeed[0]));
//        Serial.println("pwm speed: " + String(pwmSpeed[0]));
//        Serial.println("l" + String(wheelSpeed[0], 3));
    //
//        Serial.println("r" + String(wheelSpeed[1], 3));
    Serial.println("w" + String(wheelSpeed[0], 3) + "," + String(wheelSpeed[1], 3) + "," + String(wheelSpeed[2], 3) + "," + String(wheelSpeed[3], 3));


//          /Serial.println(".pwm control: " + String(pwmIn[0]) + ", " + String(pwmIn[1]));

    lastPrintTime = millis();

  }
}
