
void rightPwmIntRise() {
  attachInterrupt(radioPinR, rightPwmIntFall, FALLING);
  prevTime[0] = micros();
}

void rightPwmIntFall() {
  attachInterrupt(radioPinR, rightPwmIntRise, RISING);
  pwmIn[0] = micros() - prevTime[0];
}

void leftPwmIntRise() {
  attachInterrupt(radioPinL, leftPwmIntFall, FALLING);
  prevTime[1] = micros();
}

void leftPwmIntFall() {
  attachInterrupt(radioPinL, leftPwmIntRise, RISING);
  pwmIn[1] = micros() - prevTime[1];
}


void readRadioSpeed() {

  int rightInputPwm = (pwmIn[0] - 1525.0) / 4.25 + 5;

  int leftInputPwm = (pwmIn[1] - 1525.0) / 4.25 + 5;




  if (abs(rightInputPwm) > 200) {
    rightInputPwm = 0;
  } if (abs(leftInputPwm) > 200) {
    leftInputPwm = 0;
  }

  if (rightInputPwm < -100) {
    rightInputPwm = -100;
  } if (leftInputPwm < -100) {
    leftInputPwm = -100;
  }


  if (rightInputPwm > 100) {
    rightInputPwm = 100;
  } if (leftInputPwm > 100) {
    leftInputPwm = 100;
  }

  // some problem with the input pwm. added a deadband. Maybe look into later.
  if (abs(rightInputPwm) < 6) {
    rightInputPwm = 0;
  }
  if (abs(leftInputPwm) < 6) {
    leftInputPwm = 0;
  }


  

  targetSpeed[0] = int(maximumSpeed * rightInputPwm / 10) / 10.0;
  targetSpeed[1] = int(maximumSpeed * leftInputPwm / 10) / 10.0;

  if (!pwmControl){
    ledcWrite(1, int(rightInputPwm*0.45 + 155));
    ledcWrite(2, int(leftInputPwm*0.45 + 155));
  }

}
