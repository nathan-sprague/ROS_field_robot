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


//#include <ESP8266WiFi.h>
//#include <ESP8266WiFiMulti.h>
//#include <ESP8266WebServer.h>
//#include <ESP8266mDNS.h>
//#include <ESP8266HTTPClient.h>
//
//#include <WiFiClient.h>
//
//ESP8266WiFiMulti WiFiMulti;
//
//ESP8266WebServer server(80);

#define DEBUGMODE false

const int motorPin = 4;  // motor (D2)
const int hallPin = 5; // speed sensor pin (D1)
const int otherHallPin = 0; // speed sensor pin (D3)

// General Robot information for website and control
// steering variables
int targetAngle = -1000;
int realAngle = 0;

// values that get passed right on to the website
String websiteInfo = "0h-1x-1y[0]c";

unsigned long lastMsgTime = 0; // timer to prevent over-printing

const float wheelCircum = 13 * 2 * 3.14159;
const int numHoles = 16;

float smoothSpeed = 0; // speed used for all calculations. Smooths and averages speeds to limit outliers

bool goingForward = true;

// speed variables for the left wheel
long distTraveled = 0;
int lastRead = 0;
unsigned long lastHitTime = 0;
float wheelSpeed = 0;
unsigned int timeBetweenHoles = 0;


// speed variables for the right wheel
long distTraveledY = 0;
int lastReadY = 0;
unsigned long lastHitTimeY = 0;
float wheelSpeedY = 0;
unsigned int timeBetweenHolesY = 0;


bool wifiControl = false;
unsigned long lastCommandTime = 0;


bool stopNow = false;
bool manual = false;
int PWMSignal = 0;

int targetSpeed = 0;

unsigned long lastTargetChange = 0; // timer to reassess the speed every half second

unsigned long lastTalkTime = 0;

String stopReason = "Unknown";



//void setupAP() {
//  delay(1000);
//  WiFi.softAP("robot");
//
//  IPAddress myIP = WiFi.softAPIP();
//  Serial.print("AP IP address: ");
//  Serial.println(myIP);
//
//  server.on("/", handleMain);
//
//  server.on("/main", handleJS);
//
//  server.on("/main.js", handleJS);
//
//  server.begin();
//}

//void connectToWIFI() {
//    WiFi.mode(WIFI_STA);
//  WiFi.begin("robot");
//  Serial.println("");
//
//  // Wait for connection
//  while (WiFi.status() != WL_CONNECTED) {
//    delay(500);
//    Serial.print(".");
//  }
//  Serial.println("");
//  Serial.print("Connected to ");
//
//  Serial.print("IP address: ");
//  Serial.println(WiFi.localIP());
//
//  // Set up mDNS responder:
//  // - first argument is the domain name, in this example
//  //   the fully-qualified domain name is "esp8266.local"
//  // - second argument is the IP address to advertise
//  //   we send our IP address on the WiFi network
////  if (!MDNS.begin("esp8266")) {
////    Serial.println("Error setting up MDNS responder!");
////    while (1) {
////      delay(1000);
////    }
////  }
//  Serial.println("mDNS responder started");
//  
//
//  server.on("/", handleMain);
//
//  server.on("/main", handleJS);
//
//  server.on("/main.js", handleJS);
//
//  server.begin();
//}


void handleMain() {
  server.send(200, "text/html", "<!DOCTYPE html>\n<script src=\"main.js\"></script>\n\n\n\n<html>\n<title>Robot Controller</title>\n\n\n\n\n<!-- <input type=\"checkbox\" id=\"override\" checked = \"true\" onclick=\"displaySpeed();\"> Override -->\n\n<input type=\"radio\" id=\"observe\" name=\"action\" value=\"observe\" onclick=\"toggleOverride();\" checked>\n<label for=\"observe\">observe</label><br>\n<input type=\"radio\" id=\"override\" name=\"action\" value=\"override\" onclick=\"toggleOverride();\">\n<label for=\"override\">override</label><br>\n\n\n\n\n<br>\nReal Speed: <span id=\"speed\">__</span> mph<br>\nDesired Speed: <span id=\"desSpeed\">__</span> mph<br>\n\n<br>\nReal steering angle: <span id=\"angle\">__</span> degrees<br>\nDesired angle: <span id=\"desAngle\">__</span> degrees<br>\n<br>\n\n\nCurrent Position: <span id=\"lat\">__</span>, <span id=\"long\">__</span><br>\n\n\n\n\n<select id=\"targets\" size=0 style=\"width:500px;\"></select>\n<button type=\"button\" id=\"upPriority\", onclick=\"updatePriority(-1);\" >priority &#8593;</button>\n<button type=\"button\" id=\"downPriority\", onclick=\"updatePriority(1);\" >priority &#8595;</button>\n<button type=\"button\" id=\"remove\", onclick=\"updatePriority(0);\" >remove</button><br>\n\n\n\n<p id= \"targetPositionOverride\" style=\"\">\nAdd Target Position <input type=\"number\" id=\"desLatInput\">, <input type=\"number\" id=\"desLongInput\">\n\n<button type=\"button\" id=\"setTarget\", onclick=\"addDest(document.getElementById('desLatInput').value, document.getElementById('desLongInput').value);\">apply</button>\n</p>\n\n\n<br>Heading: <span id=\"heading\">__</span> degrees north; Target: <span id=\"targetHeading\">__</span> degrees north<br> \n\n\n<br>\n<button type=\"button\" id=\"stop\", onclick=\"stopNow();\" >STOP</button><br>\n<center>\n<button type=\"button\" id=\"forward\", onclick=\"changeSpeed(87);\" >forward</button><br>\n<button type=\"button\" id=\"left\", onclick=\"changeSpeed(65);\" >left</button>\n<button type=\"button\" id=\"right\", onclick=\"changeSpeed(68);\" >right</button><br>\n<button type=\"button\" id=\"back\", onclick=\"changeSpeed(83);\" >back</button>\n</center>\n\n\n\nscale:<br>\n<input type=\"number\" id=\"scale\" ></input>\n<button type=\"button\" id=\"scaleApply\", onclick=\"canvasScale = document.getElementById('scale').value; drawBoard();\" >apply</button><br>\n\n\n<canvas id=\"mainCanvas\" width=\"600px\" height=\"300px\" onclick=\"clickEvent(event);\" style=\"background:rgb(200,200,200);\">\n\n\n\n<body onload=\"begin();\">\n\n<!-- <body onload=\"setup();\"> -->\n\n\n\n</html>"
             );
}

void handleJS() {
  server.send(200, "text/html", "var desSteer=0;\nvar desSpeed=0;\n\nvar realSteer = 0;\nvar realSpeed = 0;\n\nvar latitude = 0;//40.421779;\nvar longitude = 0;//-86.919310;\n\nvar targetPositions = [];\n// var targetPositions = [[40.421779, -86.919310], [40.421806, -86.919074], [40.421824, -86.918487], [40.421653, -86.918739], [40.421674, -86.919232]];\nvar editedTP = false;\n\nvar stop = 0;\n\nvar targPosApplied = false;\n\nvar heading = 0;\nvar targetHeading = 0;\n\nelements = [1,2,3,4];\n\n\n\nready = true;\nfunction sendInfo(){  \n  console.log(targetPositions);\n if (ready){\n   // ready = false;\n     var xhttp = new XMLHttpRequest();\n     xhttp.onreadystatechange = function() {\n       if (this.readyState == 4 && this.status == 200) {\n          processInfo(this.responseText);\n\n       } else if (this.status == 0){\n         // ready = false;\n        }\n     };\n\n      //displaySpeed();\n\n\n     shouldOverride = 1;\n     if (document.getElementById(\"observe\").checked){\n        shouldOverride = 0;\n        argList = \"?override=0\";\n     } else {\n      argList = \"?override=1\";\n       if (!stop){\n      \n        argList += \"&p=\" + desSteer + \"&f=\" + desSpeed;\n      }\n     }\n     if (stop){\n        argList += \"&s=1\"\n     }\n     if (editedTP){\n        console.log(\"edited the TP\");\n\n        argList+= \"&c=\" +  targetPositions.join();\n\n        editedTP = false;\n     }\n      \n\n    console.log(\"arg list\", argList);\n     xhttp.open('GET', '/_info' + argList, true);\n     xhttp.send();\n } else {\n    console.log(\"not ready\");\n }\n}\n\n\n\n//-40.00a0.00w93h0.0x0.0y[40.421779, -86.91931],[40.421806, -86.919074],[40.421824, -86.918487],[40.421653, -86.918739],[40.421674, -86.919232]c-65.05924190668195t\n\nfunction processInfo(text){\n    var i = 0;\n    var val = \"\";\n    while (i<text.length) {\n     if (text[i]=='a'){\n        realSteer = parseInt(val);\n\n        val = \"\";\n     } else if (text[i]=='w'){\n       realSpeed = parseFloat(val);\n        val = \"\";\n     } else if (text[i]=='x'){\n       latitude = parseFloat(val);\n        val = \"\";\n     } else if (text[i]=='y'){\n       longitude = parseFloat(val);\n        val = \"\";\n     } else if (text[i]=='h'){\n       heading = parseInt(val);\n        val = \"\";\n     } else if (text[i]=='c'){\n        if (editedTP == false) {\n          targetPositions = [];\n          console.log(val);\n          targetPositions = JSON.parse(\"[\" + val + \"]\");\n          console.log(\"tp\", targetPositions);\n        }\n        val = \"\";\n     } else if (text[i]=='t'){\n       targetHeading = parseFloat(val);\n        val = \"\";\n\n     } else {\n        val += text[i];\n     }\n\n     i+=1;\n    }i\n\n    document.getElementById(\"speed\").innerHTML = realSpeed;\n    document.getElementById(\"angle\").innerHTML = realSteer;\n\n\n    document.getElementById(\"lat\").innerHTML = latitude;\n    document.getElementById(\"long\").innerHTML = longitude;\n\n    document.getElementById(\"targetHeading\").innerHTML = targetHeading;\n    document.getElementById(\"heading\").innerHTML = heading;\n    \n    updateDestinations();\n    setScale();\n    drawBoard();\n\n    ready = true;\n}\n\nfunction changeSpeed(val){\n    if ((val == 87 || val == 38) && desSpeed<15) { // forward\n     desSpeed+=1;\n    } else if ((val == 83 || val == 40) && desSpeed>-15) {  // backward\n     desSpeed-=1;\n\n    } else if ((val == 65 || val == 37) && desSteer>-40) { // left\n     desSteer-=1;\n    } else if ((val == 68 || val == 39) && desSteer<40) { //right\n     desSteer+=1;\n    }\n    console.log(desSteer,desSpeed);\n    \n    if ((val == 32)){ // stop\n      desSteer=0;\n      desSpeed=0;\n      sendInfo();\n    }\n    displaySpeed();\n}\n\ndocument.onkeydown = function(evt) {\n    evt = evt || window.event;\n    // console.log(evt.keyCode);\n\n    changeSpeed(evt.keyCode)\n    // sendInfo();\n    //w 87 38\n    //a 65 37\n    //s 83 40\n    //d 68 39\n    //space 32\n    \n};\n\nfunction displaySpeed(){\n\n     document.getElementById(\"desSpeed\").innerHTML = desSpeed;\n      document.getElementById(\"desAngle\").innerHTML = desSteer;\n    }\n\nfunction stopNow(){\n  if (document.getElementById(\"stop\").innerHTML == \"STOP\"){\n    stop = 1;\n    changeSpeed(32);\n    document.getElementById(\"stop\").innerHTML = \"GO\";\n  } else {\n    stop = 0;\n    document.getElementById(\"stop\").innerHTML = \"STOP\";\n  }\n}\n\n\n\n\n\n\n// var targetPositions = [[47.607533, -122.217883], [26.024640, -81.102921], [45.782204, -66.304858]]; // washington, Florida, Maine\n\n// var targetPositions = [[40.424080, -86.925006], [40.423919, -86.913810], [40.418749, -86.913530], [40.419322, -86.924389]];\nfunction drawBoard(){\n      var c = document.getElementById(\"mainCanvas\");\n      var ctx = c.getContext(\"2d\");\n      ctx.clearRect(0, 0, 1000, 500);\n      if (canvasScale==0){\n        setScale();\n      }\n      drawDestinations(ctx);\n      drawRobot(ctx);\n\n\n      // ctx.beginPath();\n      // ctx.moveTo(x, y);\n      // ctx.lineTo(x, y);\n      // ctx.stroke();\n}\n\nvar canvasScale = 0;\n\nfunction makeNewPoint(){\n\n}\n\nfunction setScale(){\n  console.log(\"set scale\");\n    longitude0=0\n  latitude0 =0\n  var longCorrection = Math.cos(latitude*Math.PI/180);\n\n  if (targetPositions.length==0){\n    canvasScale = 1;\n    document.getElementById(\"scale\").value = Math.floor(canvasScale);\n    return;\n  }\n  // longCorrection = 1;\n // console.log(longCorrection);\n\n  var i = 0;\n  var minX = (targetPositions[0][1]-longitude)*longCorrection;\n  var maxX = (targetPositions[0][1]- longitude)*longCorrection;\n  var minY = targetPositions[0][0]-latitude;\n  var maxY = targetPositions[0][0]-latitude;\n  while (i<targetPositions.length){\n    var x = (targetPositions[i][1]-longitude)*longCorrection; // longitude\n    var y = targetPositions[i][0]-latitude; // latitude\n    console.log(\"dist to target:\", Math.sqrt(x*x+y*y)*69/5280, \"miles\"); // 0.3387 miles\n\n    if (x>maxX){\n      maxX = x;\n    } else if (x<minX){\n      minX = x;\n    }\n    if (y>maxY){\n      maxY = y;\n    } else if (y<minY){\n      minY = y\n    }\n    i+=1;\n  }\n\n  i=0;\n  var scale = maxY-minY;\n  if (maxX-minX>scale){\n    scale = maxX-minX;\n  }\n  scale = 300/scale;\n\n\n  canvasScale = scale;\n\n  document.getElementById(\"scale\").value = Math.floor(canvasScale);\n}\nfunction coordToCtx(lat, long){\n  var longCorrection = Math.cos(latitude*Math.PI/180);\n  // longCorrection=1;\n\n  var x = canvasScale*(long - longitude)*longCorrection + 300;\n  var y = 300-(canvasScale*(lat - latitude)+150);\n\n  return [x,y];\n}\n\nfunction ctxToCoord(x, y){\n  var longCorrection = Math.cos(latitude*Math.PI/180);\n\n  var long = (canvasScale*longitude*longCorrection+x-300)/(canvasScale*longCorrection);\n\n  var lat = (canvasScale*latitude-y+150)/canvasScale;\n\n  return [lat,long];\n}\n// 1 degree of Longitude = cosine (latitude in decimal degrees) * length of degree (miles) at equator\n\n\n\nfunction clickEvent(event){\n    var clickedCoords = ctxToCoord(event.offsetX, event.offsetY);\n  console.log(clickedCoords);\n  document.getElementById(\"desLatInput\").value = clickedCoords[0];\n  document.getElementById(\"desLongInput\").value = clickedCoords[1];\n\n}\n\n\nfunction drawDestinations(ctx){\n  var i = 0;\n    // console.log(targetPositions);\n\n\n  while (i<targetPositions.length){\n\n    var coords = coordToCtx(targetPositions[i][0], targetPositions[i][1]);\n    var x = coords[0];\n    var y = coords[1];\n    // console.log(coords);\n    // console.log(targetPositions[i], coords);\n    // console.log(targetPositions[i][1]+1);\n\n    ctx.beginPath();\n    ctx.arc(x, y, 5, 0, 2 * Math.PI);\n    ctx.fill();\n    ctx.stroke();\n    ctx.font = '25px serif';\n    ctx.fillText(i+1, x-5, y-10);\n    ctx.stroke();\n    i+=1;\n  }\n\n\n}\n\nfunction drawRobot(ctx){\n    ctx.beginPath();\n    ctx.fillStyle = \"red\";\n    var coords = coordToCtx(latitude, longitude);\n    var x = coords[0]\n    var y = coords[1]\n    ctx.arc(x,y, 5, 0, 2 * Math.PI);\n   // console.log(x,y);\n\n    ctx.fill();\n    ctx.stroke();\n\n    ctx.beginPath();\n    ctx.moveTo(x,y);\n\n    var otherX = x+Math.cos(heading*Math.PI/180-Math.PI/2)*30;\n    var otherY = y+Math.sin(heading*Math.PI/180-Math.PI/2)*30\n    ctx.lineTo(otherX, otherY);\n  //  console.log(otherX, otherY);\n    ctx.stroke();\n\n\n    ctx.beginPath();\n    ctx.moveTo(x,y);\n    ctx.strokeStyle = \"green\";\n\n    otherX = x+Math.cos((heading+realSteer)*Math.PI/180-Math.PI/2)*30;\n    otherY = y+Math.sin((heading+realSteer)*Math.PI/180-Math.PI/2)*30\n    ctx.lineTo(otherX, otherY);\n    // console.log(otherX, otherY);\n    ctx.stroke();\n\n    ctx.strokeStyle = \"black\";    \n    ctx.fillStyle = \"black\";\n\n}\n\nfunction toggleOverride(){\n  if (document.getElementById(\"observe\").checked){\n    // document.getElementById(\"targetPositionOverride\").style.display = \"none\";\n\n  } else {\n    // document.getElementById(\"targetPositionOverride\").style.display = \"inline-block\";\n  }\n\n}\n\n\nfunction updatePriority(amount){\n\n  var select = document.getElementById(\"targets\");\n  var s = select.selectedIndex;\n  // console.log(s);\n  if (s+amount<0 || s+amount>targetPositions.length || s ==-1){\n    console.log(\"At end, index would be \", s+amount);\n    return;\n  }\n  value = targetPositions[s];\n\n\n  targetPositions.splice(s, 1);\n  if (amount!=0){\n    s+=amount;\n    targetPositions.splice(s, 0, value);\n  }\n  editedTP = true;\n  updateDestinations();\n}\n\nfunction addDest(x,y){\n\n\n  targetPositions.splice(0, 0, [parseFloat(x),parseFloat(y)]);\n  editedTP = true;\n  updateDestinations();\n\n}\n\nfunction updateDestinations(){\n  var select = document.getElementById(\"targets\");\n\n  var length = select.options.length;\n  for (i = length-1; i >= 0; i--) {\n    select.options[i] = null;\n  }\n\n  var i = 0;\n  while (i<targetPositions.length){\n    var option = document.createElement(\"option\");\n    option.text = targetPositions[i][0] + \", \" + targetPositions[i][1];\n    \n    select.add(option);\n    i+=1;\n  }\n  select.size = i;\n  drawBoard();\n}\n\n\n\nfunction begin(){\n\n  displaySpeed();\n  drawBoard();\n  updateDestinations();\n  // processInfo(\"-40.00a0.00w93h40.421669x-86.91911y[40.421779, -86.91931],[40.421806, -86.919074],[40.421824, -86.918487],[40.421653, -86.918739],[40.421674, -86.919232]c-65.05924190668195t\")\n  var requestInt = setInterval(sendInfo, 1000);\n\n}\n\n\n"
             );
}

void handleInfo() {

  for (uint8_t i = 0; i < server.args(); i++) {
    if (server.argName(i) == "s") {
      stopNow = true;
      stopReason = "website emergency stop";
      targetSpeed = 0;

    } else if (server.argName(i) == "c") {
      Serial.println("c" + server.arg(i));

    } else if (server.argName(i) == "override") {
      if (server.arg(i) == "0") {
        wifiControl = false;
      } else {
        wifiControl = true;
        if (server.argName(i) == "p") {
          targetAngle = server.arg(i).toFloat();

        } else if (server.argName(i) == "f") {
          targetSpeed = server.arg(i).toFloat();
          stopNow = false;
          stopReason = "Unknown";
        }

      }
    }
  }

  String response = String(realAngle) + "a" + String(smoothSpeed) + "w" + websiteInfo;

  server.send(200, "text/plain", response);

}



void processSerial() {
  if (Serial.available()) {


    if (DEBUGMODE) {
      Serial.println(".got message");
    }

    // get keyword
    char commandType = Serial.read();
    String serialMsg = "";


    while (Serial.available()) {
      char b = Serial.read();
      serialMsg += b;
    }

    if (DEBUGMODE) {
      Serial.println(serialMsg);
    }

    if (commandType == 'f')  {
      targetSpeed = serialMsg.toInt(); // set speed
      Serial.println("-f" + String(targetSpeed));
      if (DEBUGMODE){
        Serial.println(".target speed is set to " + serialMsg + ", was: " + String(targetSpeed));
      }
      
      manual = false;


    } else if (commandType == 'z') {
      if (DEBUGMODE) {
        Serial.println(".manual");
      }
      manual = true;
      analogWrite(motorPin, serialMsg.toInt());
      PWMSignal = serialMsg.toInt();
      
      Serial.println("-z" + String(PWMSignal));

    } if (commandType == 'p') {
      targetAngle = serialMsg.toInt();
      Serial.println("-p" + String(targetAngle));
        
    } else if (commandType == 'b') {
      Serial.println(".got: " + serialMsg);
      websiteInfo = serialMsg;
      Serial.println("-b" + serialMsg);

    } else if (commandType == 's') {
      if (DEBUGMODE) {
        Serial.println(".emergency stop");
        Serial.println("-s");
      }
      stopNow = true;
      stopReason = "Serial stop";

    } else if (commandType == 'g') {
      if (DEBUGMODE) {
        Serial.println(".go");
        Serial.println("-g");
      }
      stopNow = false;
      stopReason = "Unknown";
    } else if (commandType == 'r') {
      Serial.println(".restarting");
      Serial.println("-r");
      ESP.restart();
    }
    




  }
}



const float tickTimeTomph = (wheelCircum / 12.0) / numHoles * 1000.0 * 3600 / 5280; // constant for calculating wheel speed

void getWheelSpeed() {
  bool speedChange =  false; // something happened, like a hole detected, so the moving average functions should be run
  int  x = digitalRead(hallPin);


  if (lastRead != x && x == 0 && millis() - lastHitTime > 0) { // There was a change in what it sensed and did sense something
    timeBetweenHoles = millis() - lastHitTime;

    //    float rpm = 60.0 / numHoles / (timeBetweenHoles / 1000.0);

    wheelSpeed = tickTimeTomph / timeBetweenHoles; // mph
    if (goingForward) {
      distTraveled += wheelCircum / numHoles;
    } else {
      distTraveled -= wheelCircum / numHoles;
    }
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

    if (goingForward) {
      distTraveledY += wheelCircum / numHoles;
    } else {
      distTraveledY -= wheelCircum / numHoles;
    }
    lastHitTimeY = millis();
    speedChange = true;

  } else if (timeBetweenHolesY < millis() - lastHitTimeY) {

    wheelSpeedY = tickTimeTomph / (millis() - lastHitTimeY);
    wheelSpeedY = int(wheelSpeedY);
    speedChange = true;
  }
  lastReadY = y;

  // if going backwards, negate the wheel speeds
  if (!goingForward) {
    wheelSpeed = abs(wheelSpeed) * -1;
    wheelSpeedY = abs(wheelSpeedY) * -1;
  }

  // sanity check: if the smooth speed is way to high, bring it to the most recent value
  if (abs(smoothSpeed) > 100) {
    Serial.println(".wheel speed way too high");
    smoothSpeed = (wheelSpeed + wheelSpeedY) / 2;
  }

  if (speedChange) { // Don't reasses the situation unless there is a change in speed

    // get average speed between the wheels and past speeds. #s 3 and 5 were arbitrarily chosen
    smoothSpeed = (smoothSpeed * 3 + (wheelSpeed + wheelSpeedY)) / 5;

    // Direction may change if the speed is zero
    if (smoothSpeed > -0.1 && smoothSpeed < 0.1) { // cant compare it directly to zero because a float

      if ((PWMSignal >= 155 || PWMSignal == 0) && targetSpeed > 0) { // signal tells it to go forwards, go forwards
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


unsigned long lastPrintTime2 = 0;

int atTarget = 0;
int lastTarget = 0;
float intError = 0;
int lastPWM = 0;

void setMotorSpeed() {
  if (wifiControl && (millis() - lastCommandTime) > 1500) {
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

  if (lastTargetChange + 500 < millis()) {
    lastTargetChange = millis();

    if (lastTarget != int(targetSpeed)) { // commanded target changed
      if (DEBUGMODE) {
        Serial.println("target changed");
      }
      // you are not at the target speed
      lastTarget = int(targetSpeed);
      intError = 0;
      if (smoothSpeed > targetSpeed)
        atTarget = 1;
      else {
        atTarget = -1;
      }
    }
  } else {
    return;
  }

  Serial.println(".wheel target speed: " + String(targetSpeed));

  if (targetSpeed == 0) {
    analogWrite(motorPin, 0);
    PWMSignal = 0;
    return;

  } else if (smoothSpeed != targetSpeed) {

    // convert difference in speed to a difference in PWM
    float PWMdif = ((targetSpeed - smoothSpeed) / 14.5) * (90.0 / 2); // map (-14.5)-(14.5) to (110)-(200)

    // round the float to the nearest whole number (not int)
    if (PWMdif - int(PWMdif) > 0.5) {
      PWMdif = int(PWMdif) + 1;
    }

    if (PWMdif - int(PWMdif) < -0.5) {
      PWMdif = int(PWMdif) - 1;
    }

    if (DEBUGMODE) {
      Serial.print("old pwm: " + String(PWMSignal));
    }


    if (PWMSignal == 0) {
      PWMSignal = 155;
    }
    // change the PWM signal according to the error
    PWMSignal = PWMSignal + PWMdif;

    // you went from being too slow/fast to the other way around, begin collecting integrated differences
    if (atTarget == -1 && smoothSpeed > targetSpeed)
      atTarget = 0;
    else if (atTarget == 1 && smoothSpeed < targetSpeed) {
      atTarget = 0;
    }

    // add up the integrated error
    if (atTarget == 0) {
      intError = (intError * 5 + (targetSpeed - smoothSpeed)) / 6;
      if (intError < 0.1) {
        PWMSignal += 1;
      } else if (intError > 0.1) {
        PWMSignal -= 1;
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
    if (targetSpeed < 0 && smoothSpeed > 0 && goingForward) {
      Serial.println(".stopping backwards first");
      PWMSignal = 155;

    } else if (targetSpeed > 0 && smoothSpeed < 0 && !goingForward) {
      Serial.println(".stopping forwards first");
      PWMSignal = 155;
    }

    if (PWMSignal == 155) {
      analogWrite(motorPin, 0);
    }
    if (lastPWM != PWMSignal) { // &&  abs(smoothSpeed)<5){
      analogWrite(motorPin, PWMSignal);
    }
    lastPWM = PWMSignal;

  }
}


void setup() {
  Serial.begin(115200);
  pinMode(hallPin, INPUT);
  pinMode(otherHallPin, INPUT);
  analogWriteFreq(100);
  Serial.println(".about to begin ap");
//  setupAP();
//    connectToWIFI();
  Serial.println(".done with ap");
  delay(1000);

}

unsigned long gotSerialTime = 0;
bool gotSerial = false;
void loop() {
//   MDNS.update();
//  server.handleClient();

  if (Serial.available() && !gotSerial) {
    gotSerialTime = millis();
    gotSerial = true;
  }
  if (gotSerial && gotSerialTime+15 < millis()){
    processSerial();
    gotSerial = false;
  }


  getWheelSpeed();

  if (stopNow) {
    targetSpeed = 0;
    PWMSignal = 0;
    analogWrite(motorPin, 0);
    if (lastTalkTime + 500 < millis()) {
      lastTalkTime = millis();
      Serial.println(".wheel stopped");
      Serial.println(".Stop reason: " + stopReason);
    }
    return;
  }


  if (lastTalkTime + 500 < millis()) {
    lastTalkTime = millis();
    Serial.println(".PWM speed:" + String(PWMSignal));
  }


  if (!manual) {
    setMotorSpeed();
  }


}
