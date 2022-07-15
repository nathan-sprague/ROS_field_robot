

unsigned long lastSetSpeedCallTime = 0;
const float timeConstant = 25;


int ii = 0;


void setMotorSpeed() {

  //  if ( millis() - lastSetSpeedCallTime < 280) {
  //    return;
  //  }


  unsigned long timeSinceCalc = millis() - lastSetSpeedCallTime;
  lastSetSpeedCallTime = millis();

  int wheelNum = 0;




  while (wheelNum < 2) {


    float totalChange = (-proportionalError[wheelNum] * kp); // + integratedError[wheelNum] * ki + derivativeError[wheelNum] * kd);

    if (abs(totalChange * timeSinceCalc / timeConstant) > 40) { // limit the maximum amount of change
      totalChange = 40 * timeConstant / timeSinceCalc;
    }

    pwmSpeed[wheelNum] += totalChange * timeSinceCalc / timeConstant;


    if (pwmSpeed[wheelNum] < 20) {
      pwmSpeed[wheelNum] = 0;
    } else if (pwmSpeed[wheelNum] > pwmLimitHigh) {
      pwmSpeed[wheelNum] = pwmLimitHigh;
    } else if (pwmSpeed[wheelNum] < pwmLimitLow) {
      pwmSpeed[wheelNum] = pwmLimitLow;
    }

    if (pwmSpeed[wheelNum] > 155 && targetSpeed[wheelNum] > 0 && pwmSpeed[wheelNum] > 20) { // telling it to go backward when it should be going forward
      pwmSpeed[wheelNum] = 155;
      //  Serial.println("forwards/backwards" + String(pwmSpeed[wheelNum]));
    } else if (pwmSpeed[wheelNum] < 155 && targetSpeed[wheelNum] < 0) { // telling it to go forward when it should be going backward
      pwmSpeed[wheelNum] = 155;
   //   Serial.println("backwards/forwards");
    }

    int output = int(pwmSpeed[wheelNum]);


    if (abs(pwmSpeed[wheelNum] - 155) < 5) {
      output = 0;
    }


    if (abs(targetSpeed[wheelNum]) < 0.1) { // if you are not told to move, dont move.
      output = 0;
      pwmSpeed[wheelNum] = 155;
    }


    ledcWrite(wheelNum + 1, output);
    wheelNum++;
  }

}
