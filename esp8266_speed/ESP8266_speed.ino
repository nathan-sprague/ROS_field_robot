/*
static const uint8_t D0   = 16;
static const uint8_t D1   = 5;
static const uint8_t D2   = 4;
static const uint8_t D3   = 0;
static const uint8_t D4   = 2;
static const uint8_t D5   = 14;
static const uint8_t D6   = 12;
static const uint8_t D7   = 13;
static const uint8_t D8   = 15;
static const uint8_t D9   = 3;
static const uint8_t D10  = 1;
 */

/*
Serial Message Prefixes:

Informational:
x: gps latitude
y: gps longitude
u: destination latitude
v: destination longitude
w: wheel speed
d: distance
a: steering angle
h: heading

Command:
p: steering angle
f: motor speed
l: destination latitude
t: destination longitude

Other:
.: console log
-: error
*/


#include <ESP8266WiFi.h>
#include <ESPAsyncTCP.h>
#include <ESPAsyncWebServer.h>
//#include <ESP8266WebServer.h>
#include <ESP8266mDNS.h>
AsyncWebServer server(80);

#define DEBUGMODE false

const int motorPin = 4;  // motor (D2)
const int hallPin = 5; // speed sensor pin (D1)
const int otherHallPin = 0; // speed sensor pin (D3)

// General Robot information for website and control
// steering variables 
int targetAngle = 0;
int realAngle = 0;

// Navigation variables
long latitude = 0;
long longitude = 0;
long targetX = 0;
long targetY = 0;
int heading = 0;


unsigned long lastMsgTime = 0; // timer to prevent over-printing

const float wheelCircum = 13 * 2 * 3.14159;
const int numHoles = 16;

float smoothSpeed = 0; // speed used for all calculations. Smooths and averages speeds to limit outliers

bool goingForward = true;

// speed variables for the left wheel
unsigned long distTraveled = 0;
int lastRead = 0;
unsigned long lastHitTime = 0;
float wheelSpeed = 0;
unsigned int timeBetweenHoles = 0;


// speed variables for the right wheel
unsigned long distTraveledY = 0;
int lastReadY = 0;
unsigned long lastHitTimeY = 0;
float wheelSpeedY = 0;
unsigned int timeBetweenHolesY = 0;


bool wifiControl = false;
unsigned long lastCommandTime = 0;


bool stopNow = false;
bool manual = false;
int PWMSignal = 0;

float targetSpeed = 0;

unsigned long lastTargetChange = 0; // timer to reassess the speed every half second


void setupAP() {
    WiFi.softAP("robot");

    
    Serial.println("robot network made");
    Serial.print("IP address: " + WiFi.localIP());
    
    if (MDNS.begin("robot")) {
        Serial.println("MDNS responder started as http://robot.local/");
    }

    setupPages();
    server.begin();
} 


void setupPages(){
    server.on("/", [](AsyncWebServerRequest * request) {
    Serial.println(".main requested");
    request->send(200, "text/html",
"<!DOCTYPE html>\n<script>\n\nvar x=0;\nvar y=0;\n\nvar realX = 0;\nvar realY = 0;\n\nvar stop = 0;\n\nvar targPosApplied = false;\n\nready = true;\nfunction sendInfo(){  \n if (ready){\n   // ready = false;\n     var xhttp = new XMLHttpRequest();\n     xhttp.onreadystatechange = function() {\n       if (this.readyState == 4 && this.status == 200) {\n           desLong = 0;\n           desLat = 0;\n           console.log(this.responseText);\n           var i = 0;\n            text = this.responseText;\n           var val = \"\";\n           while (i<text.length) {\n             if (text[i]=='x'){\n                realX = parseInt(val);\n                val = \"\";\n             } else if (text[i]=='y'){\n               realY = parseInt(val);\n                val = \"\";\n             } else if (text[i]=='h'){\n               heading = parseInt(val);\n                val = \"\";\n             } else if (text[i]=='l'){\n               desLong = parseInt(val);\n                val = \"\";\n             } else if (text[i]=='t'){\n               desLat = parseInt(val);\n                val = \"\";\n             } else {\n                val += text[i];\n             }\n\n             i+=1;\n           }\n\n           document.getElementById(\"speed\").innerHTML = realY;\n           document.getElementById(\"angle\").innerHTML = realX;\n           document.getElementById(\"desLong\").innerHTML = desLong/10000000.0;\n           document.getElementById(\"desLat\").innerHTML = desLat/10000000.0;\n\n           ready = true;\n\n       } else if (this.status == 0){\n         // ready = false;\n         // console.log(\"rs\", this.status);\n        }\n     };\n\n      //displaySpeed();\n\n\n     shouldOverride = 1;\n     if (document.getElementById(\"observe\").checked){\n        shouldOverride = 0;\n        argList = \"?override=0\";\n     } else {\n      argList = \"?override=1\";\n       if (!stop){\n      \n        argList += \"&p=\" + x + \"&f=\" + y;\n      }\n     }\n     if (stop){\n        argList += \"&s=1\"\n     }\n     if (targPosApplied & document.getElementById(\"desLatInput\").value !== \"\" & document.getElementById(\"desLongInput\").value !== \"\"){\n        argList += \"&l=\" + document.getElementById(\"desLatInput\").value;\n        argList += \"&t=\" + document.getElementById(\"desLongInput\").value;\n        targPosApplied = false;\n      }\n      \n\n    console.log(argList);\n     xhttp.open('GET', '/_info' + argList, true);\n     xhttp.send();\n } else {\n    console.log(\"not ready\");\n }\n}\n\nvar requestInt = setInterval(sendInfo, 500);\n\n\n\nfunction changeSpeed(val){\n    if ((val == 87 || val == 38) && y<15) { // forward\n     y+=1;\n    } else if ((val == 83 || val == 40) && y>-15) {  // backward\n     y-=1;\n\n    } else if ((val == 65 || val == 37) && x>-40) { // left\n     x-=5;\n    } else if ((val == 68 || val == 39) && x<40) { //right\n     x+=5;\n    }\n    console.log(x,y);\n    \n    if ((val == 32)){ // stop\n      x=0;\n      y=0;\n      sendInfo();\n    }\n    displaySpeed();\n}\n\ndocument.onkeydown = function(evt) {\n    evt = evt || window.event;\n    // console.log(evt.keyCode);\n\n    changeSpeed(evt.keyCode)\n    // sendInfo();\n    //w 87 38\n    //a 65 37\n    //s 83 40\n    //d 68 39\n    //space 32\n    \n};\n\nfunction displaySpeed(){\n\n     document.getElementById(\"desSpeed\").innerHTML = y;\n      document.getElementById(\"desAngle\").innerHTML = x;\n    }\n\nfunction stopNow(){\n  if (document.getElementById(\"stop\").innerHTML == \"STOP\"){\n    stop = 1;\n    changeSpeed(32);\n    document.getElementById(\"stop\").innerHTML = \"GO\";\n  } else {\n    stop = 0;\n    document.getElementById(\"stop\").innerHTML = \"STOP\";\n  }\n}\n\nfunction setCoords(){\n\n\n}\n\nfunction toggleOverride(){\n  if (document.getElementById(\"observe\").checked){\n    // document.getElementById(\"targetPositionOverride\").style.display = \"none\";\n\n  } else {\n    // document.getElementById(\"targetPositionOverride\").style.display = \"inline-block\";\n  }\n\n}\n\n\n</script>\n\n<html>\n<title>Robot Controller</title>\n\n<!-- <input type=\"checkbox\" id=\"override\" checked = \"true\" onclick=\"displaySpeed();\"> Override -->\n\n<input type=\"radio\" id=\"observe\" name=\"action\" value=\"observe\" onclick=\"toggleOverride();\" checked>\n<label for=\"observe\">observe</label><br>\n<input type=\"radio\" id=\"override\" name=\"action\" value=\"override\" onclick=\"toggleOverride();\">\n<label for=\"override\">override</label><br>\n\n\n\n<br>\nReal Speed: <span id=\"speed\">__</span> mph<br>\nDesired Speed: <span id=\"desSpeed\">__</span> mph<br>\n\n<br>\nReal steering angle: <span id=\"angle\">__</span> degrees<br>\nDesired angle: <span id=\"desAngle\">__</span> degrees<br>\n<br>\n\n\nCurrent Position: <span id=\"lat\">__</span>, <span id=\"long\">__</span><br>\n\n\nTarget Position: \n<span id=\"desLat\">__</span>,  <span id=\"desLong\">__</span>  <br>\n\n<p id= \"targetPositionOverride\" style=\"\">\nOverride Target Position <input type=\"text\" id=\"desLatInput\">, <input type=\"text\" id=\"desLongInput\">\n\n<button type=\"button\" id=\"setTarget\", onclick=\"targPosApplied=true;\">apply</button>\n</p>\n\n\n<br>Heading: <span id=\"heading\">__</span> degrees North<br> \n\n<br>\n<button type=\"button\" id=\"stop\", onclick=\"stopNow();\" >STOP</button><br>\n<center>\n<button type=\"button\" id=\"forward\", onclick=\"changeSpeed(87);\" >forward</button><br>\n<button type=\"button\" id=\"left\", onclick=\"changeSpeed(65);\" >left</button>\n<button type=\"button\" id=\"right\", onclick=\"changeSpeed(68);\" >right</button><br>\n<button type=\"button\" id=\"back\", onclick=\"changeSpeed(83);\" >back</button>\n</center>\n\n<script>\ndisplaySpeed();\n</script>\n<!-- <body onload=\"setup();\"> -->\n\n\n\n</html>"

);
  });

  server.on("/_info", [](AsyncWebServerRequest * request) {
    if (request->hasParam("s")) {
      Serial.println(".emergency stop");
          stopNow = true;
          targetSpeed = 0;
    } 
    
    if (request->hasParam("l") && request->hasParam("t")) {
      Serial.println(".new destination");
      Serial.println("l" + request->getParam("l")->value());
      Serial.println("t" + request->getParam("t")->value());
    }

    
    if (request->hasParam("override")) {
      if (request->getParam("override")->value() == "0"){
          Serial.println(".not requesting override");  
          wifiControl = false;
      } else if (request->getParam("override")->value() == "1"){
        wifiControl = true;
        lastCommandTime = millis();
        if (request->hasParam("p")) {
          String x = request->getParam("p")->value();
          
          targetAngle = x.toInt();
          stopNow = false;
        } 
        if (request->hasParam("f")) {
          String y = request->getParam("f")->value();
          targetSpeed = y.toInt();
          stopNow = false;
        }

       
      }
        
    } else {   
        wifiControl = false;   
        Serial.println(".not requesting override");   
    }

    String response = String(realAngle) + "x" + String(smoothSpeed) + "y" + String(heading) + "h" + String(targetX) + "l" + String(targetY) + "t";
    
    request->send(200, "text/plain", response);
    
  } );
  
  server.on("/_angle", [](AsyncWebServerRequest * request) {
      if (request->hasParam("a")){
        realAngle = (request->getParam("a")->value()).toInt();
      }
      String response = String(targetAngle);
      request->send(200, "text/plain", response);
    
  } );
  
}



void processSerial() {
  if (Serial.available()) {

    if (DEBUGMODE) {
      Serial.println(".got message");
    }
    
    // get keyword
    char commandType = Serial.read();
    String serialMsg = "";

    // convert the rest of the message to an int
    while (Serial.available()) {
      char b = Serial.read();
      serialMsg += b;
    }
    int commandVal = serialMsg.toInt();

    if (DEBUGMODE) {
      Serial.println(serialMsg);
    }
    if (commandType == 'f')  {
      if (DEBUGMODE) {
        Serial.println(".target speed " + String(commandVal));
      }
      targetSpeed = commandVal; // set speed
      manual = false;
      lastTargetChange = millis();

    } else if (commandType == 'z') {
      if (DEBUGMODE) {
        Serial.println(".manual");
      }
      manual = true;
      analogWrite(motorPin, commandVal);
      PWMSignal = commandVal;

    } else if (commandType == 'h') {
      heading = commandVal;
      
    } else if (commandType == 'x') {
      latitude = float(commandVal);
      
    } else if (commandType == 'y') {
      longitude = float(commandVal);

    } else if (commandType == 'u') {
      targetX = float(commandVal);
      
    } else if (commandType == 'v') {
      targetY = float(commandVal);
      
    } else if (commandType == 's') {
      if (DEBUGMODE) {
        Serial.println(".emergency stop");
      }
      stopNow = true;

    } else if (commandType == 'g') {
      if (DEBUGMODE) {
        Serial.println(".go");
      }
      stopNow = false;
    }

  }
}



const float tickTimeTomph = (wheelCircum / 12.0) / numHoles * 1000.0 * 3600 / 5280; // constant for calculating wheel speed

void getWheelSpeed() {
  bool speedChange =  false;
  int  x = digitalRead(hallPin);


  if (lastRead != x && x == 0 && millis() - lastHitTime > 0) { // There was a change in what it sensed and did sense something
    timeBetweenHoles = millis() - lastHitTime;

    //    float rpm = 60.0 / numHoles / (timeBetweenHoles / 1000.0);

    wheelSpeed = tickTimeTomph / timeBetweenHoles; // mph
    distTraveled += wheelCircum / numHoles;

    lastHitTime = millis();
    speedChange = true;

  } else if (timeBetweenHoles < millis() - lastHitTime) { // if it is taking longer than before to reach a hole
    wheelSpeed = tickTimeTomph / (millis() - lastHitTime); // use that value for speed
    wheelSpeed = int(wheelSpeed); // the value is likely not exact so get rid of floating point precision
    speedChange = true;
  }

  lastRead = x; // use reading to detect change next time


  // repeat above for other wheels
  int  y = digitalRead(otherHallPin);
  
  if (lastReadY != y && y == 0 && millis() - lastHitTimeY > 0) {
    
    timeBetweenHolesY = millis() - lastHitTimeY;
    wheelSpeedY = tickTimeTomph / timeBetweenHolesY; // mph
    distTraveledY += wheelCircum / numHoles;


    lastHitTimeY = millis();

  } else if (timeBetweenHolesY < millis() - lastHitTimeY) {
    
    wheelSpeedY = tickTimeTomph / (millis() - lastHitTimeY);
    wheelSpeedY = int(wheelSpeedY);
    speedChange = true;
  }
  lastReadY = y;
  
  // if going backwards, negate the wheel speeds
  if (!goingForward){
    wheelSpeed = abs(wheelSpeed)*-1;
    wheelSpeedY = abs(wheelSpeedY)*-1;
  }

  // sanity check: if the smooth speed is way to high, bring it to the most recent value
  if (abs(smoothSpeed)>100){
    smoothSpeed = (wheelSpeed+wheelSpeedY)/2;
  }

  
  if (speedChange) { // Don't reasses the situation unless there is a change in speed

    // get average speed between the wheels and past speeds. #s 3 and 5 were arbitrarily chosen
    smoothSpeed = (smoothSpeed * 3 + (wheelSpeed + wheelSpeedY)) / 5;

    // Direction may change if the speed is zero
    if (smoothSpeed > -0.1 && smoothSpeed < 0.1){ // cant compare it directly to zero because a float
      
      if (PWMSignal > 155){ // signal tells it to go forwards, go forwards
        if (lastMsgTime + 500 < millis()) {
          if (DEBUGMODE) {
            Serial.println("stopped and going forwards");
          }
        }
        goingForward = true;
        
      } else { // it is going backwards
        if (lastMsgTime + 500 < millis()) {
          if (DEBUGMODE) {
            Serial.println("stopped and going backwards");
          }
        }
        goingForward = false;
      }
    }
  }



  // print results every 0.5 seconds
  if (lastMsgTime + 500 < millis()) {
    if (DEBUGMODE) {
      Serial.println(".vehicle speed: " + String(wheelSpeed) + ", " + String(wheelSpeedY) + " (" + String(smoothSpeed) + ")");
    } else {
      Serial.println("d" + String(distTraveled));
      Serial.println("w" + String(smoothSpeed));
    }
    lastMsgTime = millis();
  }


}




int atTarget = 0;
int lastTarget = 0;
float intError = 0;

void setMotorSpeed() {
    if (wifiControl && (millis() - lastCommandTime)> 1500){
      Serial.println(".error: not enough communication to be safe (>1.5s timeout)");
      analogWrite(motorPin, 0);
      delay(500);
      return;
    }


  // 200 -> 14.5
  // 190 -> 9.7
  // 180 -> 7.3
  // 170 -> 4.1
  // 160 -> 1.1
  // 155 -> 0
  // 150 -> -0.4
  // 140 -> -2.9
  // 130 -> -5.8
  // 120 -> -9.7
  // 110 -> -14.5
  bool changePWM = false; // should you change the PWM value

  if (lastTargetChange + 500 < millis()) {
    changePWM = true;
    lastTargetChange = millis();

    if (lastTarget!=int(targetSpeed)){ // commanded target changed
      if (DEBUGMODE) {
        Serial.println("target changed");
      }
      
      // you are not at the target speed
      lastTarget = targetSpeed;
      intError = 0;
      if (smoothSpeed>targetSpeed)
        atTarget = 1;
      else {
        atTarget = -1;
      }
    }

    
  }

  if (targetSpeed == 0) {
    analogWrite(motorPin, 0);
    PWMSignal = 155;

  } else if (smoothSpeed != targetSpeed && changePWM) {



    // convert difference in speed to a difference in PWM
    float PWMdif = ((targetSpeed - smoothSpeed) / 14.5) * (90.0 / 2); // map (-14.5)-(14.5) to (110)-(200)

    // round the float to the nearest whole number (not int)
    if (PWMdif - int(PWMdif)>0.5) {PWMdif = int(PWMdif)+1;}

    if (PWMdif - int(PWMdif)<-0.5) {PWMdif = int(PWMdif)-1;}

    if (DEBUGMODE) {
      Serial.print("old pwm: " + String(PWMSignal));
    }

    // change the PWM signal according to the error
    PWMSignal = PWMSignal + PWMdif;

    // you went from being too slow/fast to the other way around, begin collecting integrated differences
    if (atTarget==-1 && smoothSpeed>targetSpeed)
      atTarget = 0;
    else if (atTarget==1 && smoothSpeed<targetSpeed){
      atTarget = 0;
    }

    // add up the integrated error
    if (atTarget == 0){
      intError = (intError*5 + (targetSpeed - smoothSpeed))/6;
      if (intError<0.1){
        PWMSignal+=1;
      } else if (intError>0.1){
        PWMSignal-=1;
      }
    }
    

    if (DEBUGMODE) {
      Serial.print(", pwmdif: " + String(PWMdif));
  
      Serial.print(", new pwm: " + String(PWMSignal));

      Serial.println(", Integrated error: " + String(intError));
    }


    // prevent overshooting
    if (PWMSignal > 200) {
      PWMSignal = 200;
    } else if (PWMSignal < 110) {
      PWMSignal = 110;
    }

    // when you are going forwards but want to go backwards, you need to stop first
    if (targetSpeed<0 && smoothSpeed>0 && goingForward){
      PWMSignal = 155;
    }

    analogWrite(motorPin, PWMSignal);

  }
}


void setup() {
  Serial.begin(115200);
  pinMode(hallPin, INPUT);
  pinMode(otherHallPin, INPUT);
  analogWriteFreq(100);
  Serial.println("about to begin ap");
  setupAP();
  Serial.println("done with ap");
  delay(1000);

  

}



void loop() {
//  return;
  if (!wifiControl){
    processSerial();
  }
  
  getWheelSpeed();
  
  if (stopNow) {
    targetSpeed = 0;
    PWMSignal = 0;
    analogWrite(motorPin, 0);
    delay(10);
    return;
  }

  
  
  if (!manual) {
    setMotorSpeed();
  }
  delay(10);

}
