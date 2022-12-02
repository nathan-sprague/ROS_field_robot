
void setMotorSpeed() {
   const float timeConstant = 25; // for how fast the motor controller's values can change

  unsigned long timeSinceCalc = millis() - lastSetSpeedCallTime;
  lastSetSpeedCallTime = millis();

  
  for (int wheelNum = 0; wheelNum < 4; wheelNum++) {

    float totalChange = (-proportionalError[wheelNum] * kp) + (-integratedError[wheelNum] * ki) + (derivativeError[wheelNum] * kd);

    if (abs(totalChange * timeSinceCalc / timeConstant) > 40) { // limit the maximum amount of change
      totalChange = 40 * timeConstant / timeSinceCalc;
    }

    pwmSpeed[wheelNum] += totalChange * timeSinceCalc / timeConstant;


    // cap the top speeds
    if (pwmSpeed[wheelNum] < 20) {
      pwmSpeed[wheelNum] = 0;
    } else if (pwmSpeed[wheelNum] > pwmLimitHigh) {
      pwmSpeed[wheelNum] = pwmLimitHigh;
    } else if (pwmSpeed[wheelNum] < pwmLimitLow) {
      pwmSpeed[wheelNum] = pwmLimitLow;
    }


    // sanity check: The motors should go in the direction of the target speed, regardless of what the PID value says. If not, go neutral.
    if (pwmSpeed[wheelNum] > 155 && targetSpeed[wheelNum] > 0 && pwmSpeed[wheelNum] > 20) { // telling it to go backward when it should be going forward
      pwmSpeed[wheelNum] = 155;
   //   Serial.println("wrong way");
    } else if (pwmSpeed[wheelNum] < 155 && targetSpeed[wheelNum] < 0) { // telling it to go forward when it should be going backward
      pwmSpeed[wheelNum] = 155;
  //    Serial.println("wrong way");
    } else if (targetSpeed[wheelNum] == 0){ // shouldn't move
      pwmSpeed[wheelNum] = 155;
    }

    int out = int(pwmSpeed[wheelNum]);


    // if the PWM speed is close to neutral, don't output any PWM at all
    if (abs(pwmSpeed[wheelNum] - 155) < 5) {
      out = 0;
    }


    if (abs(targetSpeed[wheelNum]) < 0.1) { // if you are not told to move, dont move.
      out = 0;
      pwmSpeed[wheelNum] = 155;
    }


    ledcWrite(wheelNum + 1, out);
  }
  

}
