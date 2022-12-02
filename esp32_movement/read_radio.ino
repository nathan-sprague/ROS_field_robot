/*
Interrupts to read the radio's PWM
 */


void rightPwmIntRise() {
  attachInterrupt(radioPinR, rightPwmIntFall, FALLING);
  prevTime[0] = micros();
}

void rightPwmIntFall() {
  attachInterrupt(radioPinR, rightPwmIntRise, RISING);
  pwmIn[0] = (micros() - prevTime[0])/10 + 5;
  if (abs(pwmIn[0] - 155) < 3){
    pwmIn[0] = 155;
  }
}

void leftPwmIntRise() {
  attachInterrupt(radioPinL, leftPwmIntFall, FALLING);
  prevTime[1] = micros();
}

void leftPwmIntFall() {
  attachInterrupt(radioPinL, leftPwmIntRise, RISING);
  pwmIn[1] = (micros() - prevTime[1])/10 + 5;
  if (abs(pwmIn[1] - 155) < 3){
    pwmIn[1] = 155;
  }
}
