/*
  Deals with the radio controls:
  Wakes up and stops serial control when there is a signal from the radio
  Reads the input PWM from the radio controller
  Converts it to differential speeds

*/



void setupRadioPins() {
  /*
    Sets up the pins used for radio input
  */
  pinMode(forwardRadioPin, INPUT); // D0 - FORWARD RADIO
  pinMode(turnRadioPin, INPUT); // D1 - TURN RADIO

}


void setRadioControl() {
  /*
    This function is called as an interrupt when there is some input from the radio.
    This means radio has priority over serial control.
    If the robot is out of control, turn on the radio and it will stop
  */
  controlType = 0;
  Serial.println("set to radio control");



  for (int i = 0; i < NUM_MOTORS; i++) {
    if (abs(targetSpeeds[i]) < 2) {
      targetSpeeds[i] = 0;
    }
  }


  // now that it is controlled by radio, detached interrupts triggering this type of control
  //  detachInterrupt(digitalPinToInterrupt(forwardRadioPin));
  //  detachInterrupt(digitalPinToInterrupt(turnRadioPin));
  return;
}



void radioControl() {
  /*
    Gets the PWM input from the radio and translates it to differential steering output values.
  */

  // input from 1100 to 1950 (neutral is 1525, range is 850)
  int forwardPWM = pulseIn(forwardRadioPin, HIGH);
  int turnPWM = pulseIn(turnRadioPin, HIGH);

  if (forwardPWM == 0 || turnPWM == 0) { // no signal
//    Serial.println("no radio signal. now serial control");


    // no signal suggests it should be controlled by serial. Switch control over
//    controlType = 1;
    for (int i = 0; i < NUM_MOTORS; i++) {
      if (abs(targetSpeeds[i]) < 2) {
        targetSpeeds[i] = 0;
      }
    }

    // attach interrupts to be able to switch control back
    attachInterrupt(digitalPinToInterrupt(forwardRadioPin), setRadioControl, CHANGE);
    attachInterrupt(digitalPinToInterrupt(turnRadioPin), setRadioControl, CHANGE);

    return;
  }

  // map pwm to value from -100 to 100
  int forwardSpeed = (forwardPWM - 1525.0) / 4.25;
  int turnSpeed = (turnPWM - 1525.0) / 4.25;


  // the speed is basically neutral, so set it to be neutral (avoid burning out motor)
  if (abs(forwardSpeed) < 2) {
    forwardSpeed = 0;
  }
  if (abs(turnSpeed) < 2) {
    turnSpeed = 0;
  }

  // prevent speed from going over the limits
  if (abs(forwardSpeed) > 100) {
    forwardSpeed = (forwardSpeed / abs(forwardSpeed)) * 100;
  } if (abs(turnSpeed) > 100) {
    turnSpeed = (turnSpeed / abs(turnSpeed)) * 100;
  }

  // get direction of motor (-1 or 1 or 0)
  int dir = 0;
  if (forwardPWM !=  155) {
    dir = abs(forwardPWM - 155) / (forwardPWM - 155);
  }

  // map the turn value from -1 to 1
  float turnDifPercent = (turnPWM - 155.0) / 45.0;

  int speedRange = abs(forwardPWM) * 2;


  int rightPWM = 0;
  int leftPWM = 0;

  if (turnDifPercent > 0) { // go left
    leftPWM = forwardPWM;
    rightPWM = forwardPWM - dir * (turnDifPercent * speedRange);
    Serial.println("left");

  } else if (turnDifPercent < 0) { // go right
    Serial.println("right");
    rightPWM = forwardPWM;
    leftPWM = forwardPWM + dir * ((turnDifPercent) * speedRange);

  } else { // straight
    rightPWM = forwardPWM;
    leftPWM = forwardPWM;

  }

  targetSpeeds[0] = rightPWM;
  targetSpeeds[1] = rightPWM;
  targetSpeeds[2] = leftPWM;
  targetSpeeds[3] = leftPWM;


  // the speed is close enough to 0, just make it 0 to avoid burning out motors
  for (int i = 0; i < NUM_MOTORS; i++) {
    if (abs(targetSpeeds[i]) < 2) {
      targetSpeeds[i] = 0;
    }
  }

  return;
}
