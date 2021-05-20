// x50 - Center
// x10 - left
// x80 - right


// y 60 forward top speed
// y 38 - middle
// y 20 backward top speed


// python serial command:
//import serial
//try:
//   arduino = serial.Serial(port='/dev/cu.SLAB_USBtoUART', baudrate=115200, timeout=.1)
//except:
//   print("Cannot set up serial port")
//   return False
//arduino.write(bytes(str("x50"),'utf-8'))


// wifi libraries
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiAP.h>
#include <WebServer.h>
#include <HTTPClient.h>

#include <ESPmDNS.h> // not needed for ap

#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

AsyncWebServer server(80);
const int steerPin = 27;  // steering
const int motorPin = 26;  // motor

const int potPin = 34;

// setting PWM properties
const int freq = 100;
const int steerChannel = 1;
const int motorChannel = 2;
const int resolution = 8; // how many bits I think


const int steerForwardSpeed = 50;
const int steerBackwardSpeed = 20;


const int driveForwardSpeed = 50;
const int driveBackwardSpeed = 20;

bool serialControl = true;
int destAngle = 50;
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
    IPAddress myIP = WiFi.softAPIP();
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

    String chipId = String((uint32_t)(ESP.getEfuseMac() >> 24), HEX);
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
"<!DOCTYPE html>\n<script>\n\nvar x=1;\nvar y=0;\n\noverride = false;\n\n\nvar radius = 50;\nvar midX = 500;\nvar midY = 400;\nvar range = 300;\n\nvar clickedIn = false;\nvar clickY = midY;\nvar clickX = midX;\nvar mouseX = midX;\nvar mouseY = midY;\n\nfunction drawJoystick(){\n var c = document.getElementById(\"mainCanvas\");\n  var ctx = c.getContext(\"2d\");\n\n    ctx.clearRect(0, 0, 1000, 750);\n    ctx.beginPath();\n  ctx.ellipse(midX-clickX+mouseX, midY-clickY+mouseY, radius, radius, 0, 0, 2*Math.PI);\n ctx.stroke();\n ctx.fill();\n\n\n ctx.beginPath();\n  ctx.lineWidth = 10;\n ctx.strokeStyle = \"red\";\n  ctx.ellipse(midX, midY, radius+5, radius+5, 0, 0, 2*Math.PI);\n ctx.stroke();\n ctx.strokeStyle = \"black\";\n\n}\n\nfunction setup(){\n  var c = document.getElementById(\"mainCanvas\");\n    var ctx = c.getContext(\"2d\");\n    ctx.clearRect(0, 0, 1000, 750);\n\n    drawJoystick();\n}\n\n\n\n\n    \nfunction clickEvent(event){\n console.log(\"click\");\n ready = true;\n    x = (event.clientX);\n    y = (event.clientY);\n\n    var midXdif = x-midX;\n    var midYdif = y-midY;\n\n    console.log((midXdif),(midYdif));\n    if ((midXdif)*(midXdif) + (midYdif)*(midYdif) < radius*radius){\n     clickedIn = true;\n     clickX = x;\n     clickY = y;\n    }\n}\n\n\nfunction mouseUp(){\n  \n   if (clickedIn){\n    clickedIn = false;\n    clickY = midY;\n    clickX = midX;\n    mouseY = midY;\n    mouseX = midX;\n    drawJoystick();\n }\n\n}\n\nfunction mouseMove(event){\n    mouseX = event.clientX;\n mouseY = event.clientY;\n if (mouseX < (midX - range)){\n   mouseX = midX - range;\n  } else if (mouseX > (midX + range)){\n    mouseX = midX + range;\n  }\n\n if (mouseY < (midY - range)){\n   mouseY = midY - range;\n  } else if (mouseY > (midY + range)){\n    mouseY = midY + range;\n  }\n\n\n    if (clickedIn){\n      drawJoystick();\n\n    }\n\n}\n\n\nready = true;\nfunction sendInfo(){  \n  if (ready){\n   // ready = false;\n     var xhttp = new XMLHttpRequest();\n     xhttp.onreadystatechange = function() {\n     if (this.readyState == 4 && this.status == 200) {\n\n         console.log(this.responseText);\n         ready = true;\n     } else if (this.status == 0){\n       // ready = false;\n       console.log(\"rs\", this.status);\n     }\n     };\n\n      if (clickedIn){\n       x = 100*(mouseX - midX)/range;\n        y = 100*(mouseY - midY)/range;\n\n        argList = \"?override=1&x=\" + x + \"&y=\" + y;\n     } else {\n        argList = \"\";\n     }\n     console.log(argList);\n     xhttp.open('GET', '/_info' + argList, true);\n      xhttp.send();\n }\n else {\n    console.log(\"not ready\");\n }\n}\n\nvar requestInt = setInterval(sendInfo, 500);\n\n\n\n</script>\n\n<html>\n<canvas id=\"mainCanvas\" width=\"1000px\" height=\"750px\" style=\"position: absolute; background: rgb(200,200,200); \" onmouseup=\"mouseUp(event);\" onmousedown=\"clickEvent(event);\" onmousemove=\"mouseMove(event);\">\n\n\n\n\n<body onload=\"setup();\">\n\n\n\n</html>"
                 );   
  });

  server.on("/_info", [](AsyncWebServerRequest * request) {
    if (request->hasParam("override") && request->hasParam("x") && request->hasParam("y")) {
        String x = request->getParam("x")->value();
        String y = request->getParam("y")->value();
        
        Serial.println("setting to " + x + ", " + y);
        destAngle = (x.toInt() + 100)/2.5+50;
        motorSpeed = y.toInt();
        
      
    } else {
        
        Serial.println("not requesting override");
        
    }


    String info = String(realAngle) + "," + String(realSpeed);
    
    request->send(200, "text/plain", info);
    
  } );
}

long lastPrintTime = 0;
void setup() {
  Serial.begin(115200);
  setupAP();
//  connectToWIFI();
  
  ledcSetup(steerChannel, freq, resolution);
  ledcSetup(motorChannel, freq, resolution);
  // attach the channel to the GPIO to be controlled
  ledcAttachPin(steerPin, steerChannel);
  ledcAttachPin(motorPin, motorChannel);
  Serial.println("done setup");
}


void loop() {
  
//  get message
  if (Serial.available()){
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
         destAngle = angleToXParameter(command);

      } else if (motorChange == 'y'){
         motorSpeed = command;
      }
    }
  }



  // take care wheel angle
//  Serial.println("got " + String(destAngle));
  realAngle = analogRead(potPin)*255/4095;
  Serial.println(realAngle);
     
  if (realAngle<destAngle-3){
    Serial.println("forward");
    ledcWrite(steerChannel, steerForwardSpeed);
    
  } else if (realAngle>destAngle+3){
    Serial.println("backward");
    ledcWrite(steerChannel, steerBackwardSpeed);
  } else {
    Serial.println("reached destination");
    ledcWrite(steerChannel, 0);
  }

  delay(200);   

  // take care of speed
  ledcWrite(motorChannel, motorSpeed);
}

float angleToXParameter(int angle) {
// x50 - Center
// x10 - left
// x80 - right
//-45 = 20
//0 = 50
//45 = 80
    int maxX = 80;
    int minX = 20;
    int centerX = 50;

    int maxAngle = -45
    int minAngle = 45
    int centerAngle = 0;

    float x = (angle-centerAngle) * (maxX - minX) / 90 + centerX;


    if (x<minX){
        x = minX;
    } else if (x> maxX){
        x = maxX;
    }

    return x;

}