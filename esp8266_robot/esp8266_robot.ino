// wifi libraries
#include <ESP8266WiFi.h>
#include <ESPAsyncTCP.h>
#include <ESPAsyncWebServer.h>
//#include <ESP8266WebServer.h>
#include <ESP8266mDNS.h>

// pin 4 is D2
// pin 5 is D1
AsyncWebServer server(80);
const int steerPin = 5;  // steering (D1)
const int motorPin = 4;  // motor (D2)

const int potPin = A0;

// setting PWM properties
const int freq = 100;
//const int resolution = 8; // how many bits I think


const int steerForwardSpeed = 45;
const int steerBackwardSpeed = 25;


const int driveForwardSpeed = 50;
const int driveBackwardSpeed = 20;

bool serialControl = true;
bool stopNow = false;
bool pass = false;
int destAngle = 0;
int motorSpeed = 0;



int realAngle = 0;
float realSpeed = 0;
const int wheelDiameter = 10;
int realDist = 0;

long passes = 0;
bool lastPass = 0;
unsigned long lastPassTime = 0;

int readings = 0;
int counter = 0;

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

void connectToWIFI() {
    WiFi.mode(WIFI_STA);
    WiFi.begin("CW-104", "deny bunch cycle");
    while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
    }
    Serial.println("\nConnected to CW-104");

    Serial.println(WiFi.localIP());

//    String chipId = String((uint32_t)(ESP.getEfuseMac() >> 24), HEX);
    String mdnsName =  "robot";
    if (MDNS.begin("robot")) {
        Serial.println("MDNS responder started as http://robot.local/");
    }
    
    setupPages();

    
    server.begin();
}


void setupPages(){
    server.on("/", [](AsyncWebServerRequest * request) {
    Serial.println("main requested");
    request->send(200, "text/html",
"<!DOCTYPE html>\n<script>\n\nvar x=0;\nvar y=0;\n\nvar realX = 0;\nvar realY = 0;\n\n\nready = true;\nfunction sendInfo(){  \n if (ready){\n   // ready = false;\n     var xhttp = new XMLHttpRequest();\n     xhttp.onreadystatechange = function() {\n       if (this.readyState == 4 && this.status == 200) {\n\n           console.log(this.responseText);\n           var i = 0;\n            text = this.responseText;\n           var val = \"\";\n           while (i<text.length) {\n             if (text[i]=='x'){\n                realX = parseInt(val);\n                val = \"\";\n             } else if (text[i]=='y'){\n               realY = parseInt(val);\n                val = \"\";\n             } else {\n                val += text[i];\n             }\n\n             i+=1;\n           }\n\n           document.getElementById(\"speed\").innerHTML = realY;\n           document.getElementById(\"angle\").innerHTML = realX;\n\n           ready = true;\n\n       } else if (this.status == 0){\n         // ready = false;\n         // console.log(\"rs\", this.status);\n        }\n     };\n\n      //displaySpeed();\n     var override = 1;\n     if (document.getElementById(\"override\").checked ==false){\n       override = 0;     \n      } if (x==0 && y==0){\n        override = 3;\n     }\n     argList = \"?override=\" + override + \"&x=\" + x + \"&y=\" + y;\n      \n      console.log(argList);\n     xhttp.open('GET', '/_info' + argList, true);\n      xhttp.send();\n } else {\n    console.log(\"not ready\");\n }\n}\n\nvar requestInt = setInterval(sendInfo, 500);\n\nfunction changeSpeed(val){\n      if ((val == 87 || val == 38) && y<100) { // forward\n     y+=10;\n    } else if ((val == 83 || val == 40) && y>-100) {  // backward\n     y-=10;\n\n    } else if ((val == 65 || val == 37) && x>-40) { // left\n     x-=5;\n    } else if ((val == 68 || val == 39) && x<40) { //right\n     x+=5;\n    }\n    console.log(x,y);\n    \n    if ((val == 32)){ // stop\n      x=0;\n      y=0;\n      sendInfo();\n    }\n    displaySpeed();\n}\n\ndocument.onkeydown = function(evt) {\n    evt = evt || window.event;\n    console.log(evt.keyCode);\n\n    changeSpeed(evt.keyCode)\n    // sendInfo();\n    //w 87 38\n    //a 65 37\n    //s 83 40\n    //d 68 39\n    //space 32\n    \n};\n\nfunction displaySpeed(){\n if (document.getElementById(\"override\").checked == false){\n    document.getElementById(\"desSpeed\").innerHTML = \"__\";\n     document.getElementById(\"desAngle\").innerHTML = \"__\";\n    } else {\n     document.getElementById(\"desSpeed\").innerHTML = y;\n      document.getElementById(\"desAngle\").innerHTML = x;\n    }\n}\n\n</script>\n\n<html>\n<title>Robot Controller</title>\n\n<input type=\"checkbox\" id=\"override\" checked = \"true\" onclick=\"displaySpeed();\"> Override\n\n\n\n<br>\nReal Speed: <span id=\"speed\">__</span><br>\nDesired Speed: <span id=\"desSpeed\">__</span><br>\n\n<br>\nReal steering angle: <span id=\"angle\">__</span><br>\nDesired angle: <span id=\"desAngle\">__</span><br>\n<button type=\"button\" id=\"stop\", onclick=\"changeSpeed(32);\" >STOP</button><br>\n<center>\n<button type=\"button\" id=\"forward\", onclick=\"changeSpeed(87);\" >forward</button><br>\n<button type=\"button\" id=\"left\", onclick=\"changeSpeed(65);\" >left</button>\n<button type=\"button\" id=\"right\", onclick=\"changeSpeed(68);\" >right</button><br>\n<button type=\"button\" id=\"back\", onclick=\"changeSpeed(83);\" >back</button>\n</center>\n\n<script>document.getElementById(\"override\").checked=true; \ndisplaySpeed();\n</script>\n<!-- <body onload=\"setup();\"> -->\n\n\n\n</html>"
                 );   
  });

  server.on("/_info", [](AsyncWebServerRequest * request) {
    if (request->hasParam("override") && request->hasParam("x") && request->hasParam("y")) {
        if (request->getParam("override")->value() == "0"){
          serialControl = true;
          Serial.println("not requesting override");  
          
        } else if (request->getParam("override")->value() == "1"){
          String x = request->getParam("x")->value();
          String y = request->getParam("y")->value();
          Serial.println("setting to " + x + ", " + y);
          destAngle = x.toInt();
          motorSpeed = y.toInt();
          serialControl= false;
          stopNow = false;
          
          
        } else {
          Serial.println("emergency stop");
          stopNow = true;
          serialControl = false;
          destAngle = 0;
          motorSpeed = 0;
          serialControl= false;
        }
        
    } else {      
        serialControl = true;
        Serial.println("not requesting override");   
    }

    String response = String(realAngle) + "x" + String(motorSpeed) + "y";
    
    request->send(200, "text/plain", response);
    
  } );
}

long lastPrintTime = 0;
void setup() {
  Serial.begin(115200);
  setupAP();
//  connectToWIFI();

  // attach the channel to the GPIO to be controlled
//  analogWrite(steerPin, 0);
//  analogWrite(motorPin, 0);
  analogWriteFreq(freq);
  Serial.println("done setup");
}


void loop() {
//    return;

//  get message

  if (Serial.available()){
    Serial.println("got message");
    char motorChange = Serial.read();
    String serialMsg = "";
    
    while (Serial.available()){
      char b = Serial.read();
      serialMsg += b;
    }
    
    int command = serialMsg.toInt();
    
    Serial.println(serialMsg);
  
    if (serialControl) {
      if (motorChange == 'x'){
         destAngle = command;

      } else if (motorChange == 'y'){
         motorSpeed = command;
      } else if (motorChange == 'f'){
        analogWriteFreq(command);
      } else if (motorChange == 'm') {
        analogWrite(steerPin, command);
        pass = true;
      } else if (motorChange == 's'){
        Serial.println("emergency stop");
        stopNow = true;
      } else if (motorChange == 'g'){
        Serial.println("go");
        stopNow = false;
      }
    }
  }
  if (stopNow){
    
    analogWrite(motorPin, 0);
    analogWrite(steerPin, 0);
    delay(10);
    return;
  } else if (pass){
    Serial.println(analogRead(potPin));
    delay(250);
    return;
  }

  // take care wheel angle
//  Serial.println("got " + String(destAngle));
  realAngle = potToAngle(analogRead(potPin));
  
  
//  Serial.print ("Real angle: " + String(realAngle) + ", dest: " + String(destAngle) + "   ");
  
  if (realAngle<destAngle-4){
//    Serial.println("forward");
    analogWrite(steerPin, steerForwardSpeed*1024/256-1);
    
  } else if (realAngle>destAngle+4){
//    Serial.println("backward");
    analogWrite(steerPin, steerBackwardSpeed*1024/256-1);
  } else {
//    Serial.println("reached destination");
    analogWrite(steerPin, 0);
  }



  delay(100);   

// y 60 forward top speed
// y 38 - middle
// y 20 backward top speed

  // take care of speed
  if (motorSpeed == 0){
    analogWrite(motorPin, 0);
  } else {
//    Serial.println("motor speed: " + String(motorSpeed));
    int mSpeed = (motorSpeed+100)*18/100+20;
//    Serial.print("mspeed: ");
//    Serial.println(mSpeed);
    analogWrite(motorPin, mSpeed*1024/256-1);
  }
}


float potToAngle(int pot) {
// center 240
// right 320
// left 200
    float angle = 0;
    if (pot>240) { // right
      angle = (pot-240) * 45 / (320.0-240);
    } else { // left
      angle = (pot-240) * 45 / (240-200.0);
    }

    return angle;
}
