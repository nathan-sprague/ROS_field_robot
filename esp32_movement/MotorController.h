

#ifndef Test_h
#define Test_h

class MotorController {
  public:

    const byte maxSpeed = 5; // maximum speed of the robot. Used to get approximate map of speed to PWM. Test to verify.

    float currentSpeed = 0;
    float acceleration = 0;

    int pwmOut = 0;
    int encoderPin1;
    int encoderPin2;
    int pwmPin;
    int motorChannel = 1;



    bool altEncoderActivated = false; // encoder 2

    unsigned long encoderTicks = 0; // encoder 1

    bool goingForward = true;

    unsigned long lastHitTime = 0;




    void calculateSpeed() {
      /*
        Calculates speed depending on the number of ticks the encoders sense.

        Sets the variables wheelspeed and acceleration to the calculated speed/acceleration

        No inputs, no returns. Just sets the public variables
      */

      float timeBetweenHits = (millis() - lastHitTime); // time since the speed was last calculated
      float ws = 0;

      if (timeBetweenHits != 0) {
        ws = encoderTicks / timeBetweenHits / 10;
      }

      lastHitTime = millis(); // reset timer

      // reset encoder tick count to 0
      encoderTicks = 0;

      if (!goingForward) { // wheel going backwards
        ws *= -1;
      }

      // calculate acceleration
      float wa = (ws - currentSpeed) * 100.0 / timeBetweenHits;
      acceleration = (acceleration * 4 + wa) / 5;

      // calculate integrated speed (sort of)
      integratedSpeed = (integratedSpeed * 19 + ws) / 20;


      // se the global speed variable to the calculated wheel speed
      currentSpeed = ws;
    }


    void updateEncoder1() {
      /*
        updates the encoder
      */
      Serial.println("hit!");

      if (digitalRead(encoderPin1) == HIGH) { // pin is turning high

        if (altEncoderActivated) {
          goingForward = false; // going backwards

        } else {
          goingForward = true; // going forwards
        }
      }

      encoderTicks++; // add 1 to the number of ticks experienced
    }

    void updateEncoder2() {
      if (encoderPin2 == HIGH) {
        altEncoderActivated = true;
      } else {
        altEncoderActivated = false;
      }
    }

    void setupPins() {

      ledcSetup(motorChannel, 100, 10);

      pinMode(pwmPin, OUTPUT);

//      attachInterrupt(rfEncoderPin1, motors[0].updateEncoder1, CHANGE);
//      attachInterrupt(rfEncoderPin2, motors[0].updateEncoder2, CHANGE);

    }


    int findIdealPWM(int targetSpeed) {
      /*
        This function finds the best PWM value to move the motor at the desired speed
        This motor control aims to:
          - limit PWM changes to prevent surging
          - quickly bring the robot up to speed using typical mapping to PWM values
          - change the PWM depending where the speed stabilizes
          - recognize the robot is stuck
            - apply maximum power to get robot unstuck
            - if the robot is still stuck, turn off to avoid blowing a fuse
            - if the robot gets unstuck, quickly reduce power to avoid going fast
          - clean the PWM values

        inputs: target speed
        output: best PWM
      */

      float pwmChange = 0;

      if ((targetSpeed > 0 && currentSpeed < 0) || (targetSpeed > 0 && currentSpeed < 0)) { // going opposite direction of target speed. Stop first.
        return 0;
      }

      if (targetSpeed == 0) { // you don't want to move
        return 0;
      }

      if (lastPwmCalcTime == millis()) { // this function was just called. (should never happen)
        return pwmOut;
      }
      // the target speed changed
      if (currentTargetSpeed != int(targetSpeed)) {
        // set to default estimated PWM
        lastGoodPWM = (targetSpeed * 100 / maxSpeed);
        currentTargetSpeed = int(targetSpeed);
        return lastGoodPWM;
      }


      byte moveDir = (abs(targetSpeed) / targetSpeed); // -1 or 1



      //        if ((currentSpeed - targetSpeed) * moveDir < 2) { // your speed is slightly faster than the target
      //
      //
      //        } else {
      //          // you are going faster than target speed by a stable amount, just go to default speed
      //
      //        }
      //
      //
      //        if (false) { // stuck even when at full throttle for a second.
      //          // don't move until you are given different instructions
      //
      //        } else if (false) { // temporarily stuck
      //          // keep on going full throttle
      //        }

      if (abs(currentSpeed < 0.1) && abs(acceleration < 0.1)) { // you are likely stuck


      } else { // not stuck



        if (false) { // you were stuck but now are not
          // go to the last good pwm

        }
      }






      // set the pwm
      int pwmSignal = pwmOut + pwmChange / (millis() - lastPwmCalcTime);


      // prevent PWM from going outside of recognized PWM range



      // sanity check: You must be sending pwm signals in a certain range to go the direction you want
      if (pwmSignal > 100) {
        pwmSignal = 100;
      } else if (pwmSignal < -100) {
        pwmSignal = -100;
      }



      return pwmSignal;
    }






    void setMotorSpeed(bool stopNow) {
      /*
        Output PWM Frequency is 1024 (default)
        PWM range is:
          110 -> backwards full power
          155 -> neutral
          200 -> forwards full power

      */

      //      Serial.println("setting speed of " + String(pwmOut));


      //convert the easier to use -100 - +100 signal to true PWM
      int truePWM = (pwmOut / 100.0) * 90 + 155;
      //      Serial.println("true pwm: " + String(truePWM));

      if (stopNow) {

        Serial.println(".stopped");
        delay(200);
      }

      ledcWrite(motorChannel, truePWM);


      delay(10);

    }



  private:


    unsigned long timeBetweenHits = 0;
    int currentTargetSpeed = 0;
    int lastGoodPWM = 0;
    float integratedSpeed = 0;
    unsigned long stuckTimer = 0;
    unsigned long lastPwmCalcTime = 0;



    int findPWM_stuck(float targetSpeed) {
      /*
        stuck and trying to get out. Apply maximum power for 1/2 second

        parameters: target speed
        returns: best PWM change
      */

      if (stuckTimer > 0) {
        if (millis() - stuckTimer > 500) { // trying to get out for 0.5 seconds, just stop
          return 0;
        }
      } else {
        stuckTimer = millis();
      }

      return (targetSpeed / abs(targetSpeed)) * 100; // send the maximum speed in the direction you are going
    }

    int findPWM_goodSpeed(float targetSpeed) {
      /*
        At a good speed. Make small adjustments to stay at this speed

        parameters: target speed
        returns: best PWM change
      */

      if (acceleration < 0.5 && abs(currentSpeed - targetSpeed) < 0.1) { // not accelerating and going at the right speed
        // Remember this pwm value so that you can quickly return to the right speed after being stuck
        lastGoodPWM = lastGoodPWM * 9 + pwmOut;
      }

      // change the PWM signal according to the error (PID control)
      float pwmProportional = (targetSpeed - currentSpeed);

      float pwmIntegral = (targetSpeed - integratedSpeed);

      float pwmDerivative = 0; // still trying to find a way to get the derivative term to work


      // sum all of the PID adjustments
      float pwmChange = pwmProportional * 1 + pwmIntegral * 0.1 + pwmDerivative * 0.1;




      return pwmChange;
    }





};

#endif
