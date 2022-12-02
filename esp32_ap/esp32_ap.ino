#include "SPIFFS.h"
#include "FS.h"

// setting variables
String apName = "robot";


#include <DNSServer.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiAP.h>
#include <WebServer.h>
#include <HTTPClient.h>

#include <ESPmDNS.h>  // not needed for ap

#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>


DNSServer dnsServer;
AsyncWebServer server(80);

String serverMsg = "{\"status\": [\"-1\"]}";

unsigned long lastSerialReadTime = 0;
unsigned long lastMsgRequestTime = 0;
bool haveSerial = false;
bool gracefulShutdown = false;
unsigned long lastLogTime = 0;

unsigned int crashTimeout = 10000;

void readSerial() {

  if (Serial.available()) {
    gracefulShutdown = false;

    if (haveSerial && millis() - lastSerialReadTime > 30) {  // must wait for >30 milliseconds for the full serial message to arrive
      haveSerial = false;                                    // reset flag for next message
      char commandType = Serial.read();

      if (commandType == 'm') {

        String serialMsg = "";
        char lastChar = '-';
        while (Serial.available()) {
          char b = Serial.read();
          if (b == '{' && lastChar == 'm') { // sometimes the python script sends too often
            serialMsg = "";
          }
          serialMsg += b;
          lastChar = b;
        }
        serverMsg = serialMsg;
        Serial.println(".msg gotten: " + serverMsg);
      } else if (commandType == 'k') {  // graceful exiting from robot script
        serverMsg = "{\"status\": [\"-2\", \"-1\"]}";
        recordStatus('g');
        gracefulShutdown = true;


      } else if (commandType == 'p') {  // print out crash logs
        File file = SPIFFS.open("/crashLogs.txt");
        if (!file || file.isDirectory()) {
          Serial.println(".No past logs found");
          return;
        }
        while (file.available()) { Serial.print(file.read()); }
        Serial.println("");
        file.close();

      } else {  // unused message
        Serial.print("-");
        Serial.println(commandType);
      }
      while (Serial.available()) { Serial.read(); }  // flush serial

    } else if (!haveSerial) {
      haveSerial = true;
      lastSerialReadTime = millis();
    }
  }
}


void setupPages() {

  server.on("/", [](AsyncWebServerRequest* request) {
    request->send(200, "text/html", "<!DOCTYPE html>\n<html>\n  <head>\n   <title>Robot Monitor</title>\n  </head>\n\n	<body style=\"background-color: rgb(0,120,0);\">\n \n  </body>\n</html>\n\n<script>\n\n\nvar canvasScale = 0\nvar canvasCenterCoords = [40.4699462,-86.9953345];\nvar robotCentered = true;\nvar showPath = true;\nvar pathPoints = [];\n\nvar handMadePts = []\n\nvar robotCoords = [40.4699462,-86.9953345];\nvar destinationPoints = [];\nvar obstaclePoints = [];\nvar rows = [];\nvar gpsAccuracy = -1;\nvar trueHeading = 0;\nvar headingAccuracy = 0;\nvar headingAccuracy = 0;\nvar targetHeading = 0;\nvar desiredSpeed = [0,0];\nvar realSpeed = [0,0];\nvar runID = -1;\nvar runMode = 0\nvar runTime = 0;\n\nvar mapFeatures = [\n[\"green\", [40.4718357, -86.9967665], [40.4697490, -86.9967665], [40.4697678, -86.9955130], [40.4698261, -86.9955234], [40.4699289, -86.9953532], [40.4705472, -86.9953564], [40.4705693, -86.9952035], [40.4718479, -86.9952443], [40.4718402,-86.9959584], [40.4716069,-86.9960634], [40.4716518,-86.9961884], [40.4718329,-86.9961076]], [\"gray\", [40.470539, -86.995253], [40.469761, -86.995264], [40.469761, -86.996155], [40.469697, -86.996155], [40.469697, -86.995155], [40.470540, -86.995163]], [\"blue\", [4], [40.4700035, -86.9954488], [40.4700035, -86.9953695], [40.4705425, -86.9953695], [40.4705425, -86.99526316666666],  [40.4716435, -86.99526316666666], [40.4716435, -86.9954385]]];\n\n// west_base = [40.46995666666667, -86.99515333333333]\n// east_base =[40.4700035, -86.9953855]\n// mid_west_base = [40.470537, -86.9953695]\n// mid_east_base = [40.47053966666667, -86.99527016666667]\n// north_east_base = [40.4716485, -86.99526316666666]\n// north_west_base = [40.4716435, -86.9954385]\n\n\nvar destID = -1;\nvar obstaclesID = -1;\nvar settingsVersion = -1; // old useless variable. Feeling cute, might remove later, IDk\nvar rowsID = -1;\nvar requestInt = setInterval(sendInfo, 500);\nvar logData = \"\";\nvar lastLogVals = [0,0,0,0,0,0,0,0,0,0,0,0,0,0];\n\n\nvar controlVars = {\n	\"saveLogs\": true,\n	\"topSpeed\": 2,\n	\"defaultDestTolerance\": 1.5,\n	\"enableMovement\": true,\n	\"updateSpeed\": 100,\n	\"kill\":0\n}\n\nvar updatedControlVars = [];\n\n\n\nvar mouseDown = false;\nvar dragged = false;\nvar lastMouseSpot = [0,0]\n\nvar lastUpdateTime = 0;\nvar gettingMessages = true;\n\nvar obstructions = [];\n\n\n\nfunction sendInfo(){\n	\n	var xhttp = new XMLHttpRequest();\n	xhttp.onreadystatechange = function() {\n		if (this.readyState == 4 && this.status == 200) {\n			if (gettingMessages == false){\n				clearInterval(requestInt)\n				requestInt = setInterval(sendInfo, 500);\n				document.getElementById(\"connectionStatusBoxBody\").innerHTML = \"Connected\";\n\n				gettingMessages = true;\n			}\n			console.log(\"got info\", this.responseText);\n			info = JSON.parse(this.responseText);\n\n			if (\"coords\" in info)	{\n				robotCoords = [parseFloat(info[\"coords\"][0].toFixed(7)), parseFloat(info[\"coords\"][1].toFixed(7))];\n				if (robotCentered){\n					canvasCenterCoords = [...robotCoords];\n				}\n			}\n\n			if (\"gpsAccuracy\" in info) {\n				gpsAccuracy = info[\"gpsAccuracy\"];\n				connectionType = info[\"connectionType\"];\n				let connectionTypesLabels = [\"Position not known\", \"Position known without RTK\", \"DGPS\", \"UNKNOWN FIX LEVEL 3\", \"RTK float\", \"RTK fixed\"];\n\n				document.getElementById(\"gpsBoxBody\").innerHTML = \"Status: \" + connectionTypesLabels[connectionType] + \" <br><br>Coords: \" +robotCoords + \"<br><br>Accuracy: \" + gpsAccuracy + \" mm\";\n			}\n\n\n			if (\"heading\" in info) {\n				trueHeading = info[\"heading\"];\n				document.getElementById(\"headingBoxBody\").innerHTML = trueHeading.toFixed(2) + \"&#176; N\";\n				headingAccuracy = info[\"headingAccuracy\"];\n			}\n\n			if (\"targetHeading\" in info) {\n				targetHeading = info[\"targetHeading\"];\n				document.getElementById(\"targetHeadingBoxBody\").innerHTML = targetHeading.toFixed(2) + \"&#176; N\";\n\n				if (pathPoints.length>1 && Math.abs(robotCoords[0]) > 2) {\n\n					if (pathPoints[pathPoints.length-1][0] != robotCoords[0] || pathPoints[pathPoints.length-1][1] != robotCoords[1]) {\n						pathPoints.push([...robotCoords]);\n					}\n				} else if (Math.abs(robotCoords[0]) > 2) {\n					pathPoints.push([...robotCoords]);\n				}\n			}\n\n			if (\"realSpeed\" in info) {\n				realSpeed = [info[\"realSpeed\"][0].toFixed(2), info[\"realSpeed\"][1].toFixed(2)];\n				document.getElementById(\"speedBoxBody\").innerHTML = realSpeed + \" mph\";\n			}\n\n			if (\"targetSpeed\" in info) {\n				targetSpeed = [info[\"targetSpeed\"][0].toFixed(2), info[\"targetSpeed\"][1].toFixed(2)];\n				let speedError = [(targetSpeed[0]-realSpeed[0]).toFixed(2), (targetSpeed[1]-realSpeed[1]).toFixed(2)];\n				document.getElementById(\"targetSpeedBoxBody\").innerHTML = targetSpeed + \" mph<br>(\" + speedError + \" off)\";\n			}\n\n			if (runID in info){\n				if (runID != info[\"runID\"]){\n					runID = info[\"runID\"];\n					console.log(\"new run\");\n					console.log(\"run mode\", runID)\n					pathPoints = [];\n					logData = \"\";\n					if (runID[0] == 's'){\n						console.log(\"simulation!!!\")\n						document.title = \"Simulation Robot Monitor\";\n						runMode = 1;\n						document.getElementById(\"saveLogsBox\").style.display=\"none\"\n						document.getElementById(\"maxSpeedBox\").style.display=\"block\";\n						document.getElementById(\"updateSpeedBox\").style.display=\"block\";\n\n						console.log(\"checked record logs\");\n\n					} else if (runID[0] == 'p'){\n						document.title = \"Playback Robot Monitor\";\n						runMode = 2;\n						document.getElementById(\"saveLogsBox\").style.display=\"none\";\n						document.getElementById(\"maxSpeedBox\").style.display=\"none\";\n						document.getElementById(\"updateSpeedBox\").style.display=\"block\";\n\n					} else {\n						document.title = \"Robot Monitor\";\n						runMode = 0;\n						document.getElementById(\"recordLogs\").checked = true;\n						document.getElementById(\"updateSpeedBox\").style.display=\"none\";\n						document.getElementById(\"saveLogsBox\").style.display=\"block\";\n						document.getElementById(\"maxSpeedBox\").style.display=\"block\";\n					}\n					if (runMode != 0){\n						updatedControlVars.push(\"updateSpeed\");\n					}\n				\n				}\n\n\n				document.getElementById(\"runTimeBoxHeader\").innerHTML = \"Run ID: \" + runID;\n			}\n\n			if (\"runTime\" in info) {\n				runTime = parseInt(info[\"runTime\"]);\n				hrs = (runTime/3600).toFixed(0)\n				mins = String((runTime%3600/60).toFixed(0)).padStart(2, '0')\n				secs = String((runTime%60).toFixed(0)).padStart(2, '0')\n				document.getElementById(\"runTimeBoxBody\").innerHTML = \"Runtime: \" + hrs + \":\" + mins + \":\" + secs\n			}\n\n			let newDestinations = false;\n			if (\"destinations\" in info) { // sometimes doesnt send destinations since it is a big array of a lot of data\n        destinationPoints = info[\"destinations\"];\n        console.log(\"destinations:\", destinationPoints);      \n	     	if (\"destID\" in info) { \n	        destID = info[\"destID\"];\n	      }\n	      newDestinations = true;\n	    }\n\n	    let newObstacles = false;\n	    if (\"obstacles\" in info) { // sometimes doesnt send obstacle data since it is a big array of a lot of data\n	      obstaclePoints = info[\"obstacles\"];\n	      console.log(\"obstacles\", obstaclePoints, \"id\", obstaclesID)\n	     	if (\"obstaclesID\" in info) { \n	      	obstaclesID = info[\"obstaclesID\"];\n	      }\n	      newObstacles = true;\n	    }\n\n		if (\"rows\" in info) { // sometimes doesnt send row data since it is a big array of a lot of data\n			rows = info[\"rows\"];\n			console.log(\"rows\", rows, \"id\", rowsID)\n			if (\"rowsID\" in info) { \n				rowsID = info[\"rowsID\"];\n			}\n	    }\n\n	    if (\"obstructions\" in info) { // sometimes doesnt send target path since it is a big array of a lot of data\n	      obstructions = info[\"obstructions\"];\n	    }\n\n	    if (\"status\" in info){\n		    let statusCodes = {'-4': \"unexpected power cycle - ESP shutoff\", '-3': \"Unexpected crash - ESP remained on\", '-2': \"robot ended gracefully\", '-1': \"robot not running\", 0: \"Waiting to start\", 1: \"Waiting for GPS fix\", 2: \"Waiting for better GPS accuracy\", 3: \"Waiting for GPS heading\", 4: \"Moving forward normally\", 5: \"Close to destination, slowing down\", 6: \"Moving in zero point turn\", 7: \"Pausing during a zero point turn\", 8: \"At destination, pausing\", 9: \"Moving to correct heading\", 10: \"Video-based Navigation\", 11: \"Outside row\", 12: \"Don't know which way the robot is facing, slowing down\", 13: \"Know where the robot is facing\", 14: \"Slight correction necessary\", 15: \"major correction necessary\", 16: \"Shutting down robot\", 17: \"Heading too different from the row angle\", 18: \"Obstacle in view\", 19: \"Stopping to avoid obstacle\", 20: \"backing up to avoid obstacle\", 21: \"Backing up more to be safe\", 22: \"Turning to avoid obstacle\", 23: \"Inside row so ignoring obstacle\", 24: \"just left the row\"};\n		    let msg = \"\";\n		    i = 0;\n		    while (i<info[\"status\"].length){\n		    	msg += statusCodes[info[\"status\"][i]] + \"<br>\";\n		    	i+=1;\n		    }\n		    document.getElementById(\"messagesBoxBody\").innerHTML = msg;\n		  }\n\n\n	    \n	    logtoMsg(newDestinations, newObstacles);\n\n\n			drawBoard();\n			drawCamBoard();\n\n		} else if (this.readyState == 4 && this.status == 0) { // did not get correct message\n			// console.log(\"failed to get message\")\n			lastUpdateTime += 2;\n			// console.log(lastUpdateTime)\n\n			if (gettingMessages == true){\n				lastUpdateTime = 0;\n				console.log(\"slowing message request rate\")\n				clearInterval(requestInt)\n				requestInt = setInterval(sendInfo, 2000);\n				gettingMessages = false;\n			}\n			document.getElementById(\"connectionStatusBoxBody\").innerHTML = \"No Connection (\" + lastUpdateTime + \"s)\"\n\n		}\n	};\n	\n	argList = \"?destID=\" + destID + \"&obstaclesID=\" + obstaclesID + \"&settingsID=\" + settingsVersion + \"&rowsID=\" + rowsID + \"&\";\n\n	let i = 0;\n	while (i<updatedControlVars.length){\n		argList += updatedControlVars[i] + \"=\" + controlVars[updatedControlVars[i]] + \"&\";\n		i+=1;\n	}\n	updatedControlVars = [];\n\n\n	\n	argList = argList.slice(0,-1);\n	console.log(\"arg list\", argList);\n	xhttp.open('GET', '/_info' + argList, true);\n	xhttp.send();\n	 \n}\n\n\nfunction logtoMsg(newDestinations, newObstacles){\n	if (document.getElementById(\"recordLogs\").checked){\n    	// console.log(\"recording log\");\n    	msg = \"\";\n    	if (newDestinations){\n    		msg += \"d,\" + destinationPoints + \",\\n\";\n    	}\n\n    	dataToLog = [runTime, trueHeading, targetHeading, realSpeed[0], realSpeed[1], targetSpeed[0], targetSpeed[1], robotCoords[0], robotCoords[1], 0,0, headingAccuracy, gpsAccuracy, connectionType];\n    	let i=0;\n    	while (i<dataToLog.length){\n    		if (dataToLog[i] != lastLogVals[i]){\n    			msg += dataToLog[i] + \",\";\n    		} else{\n    			msg += \",\";\n    		}\n    		i+=1;\n    	}\n    	msg += \"\\n\";\n    	lastLogVals = [...dataToLog] \n    	logData += msg;\n	}\n}\n\n\n\nfunction coordToCtx(lat, long){\n  var longCorrection = Math.cos(canvasCenterCoords[0]*Math.PI/180);\n \n  var x = canvasScale*(long - canvasCenterCoords[1])*longCorrection;\n  var y = -(canvasScale*(lat - canvasCenterCoords[0]));\n\n  return [x,y];\n}\n\nfunction ctxToCoord(x, y){\n  var longCorrection = Math.cos(canvasCenterCoords[0]*Math.PI/180);\n  var long = (x)/(canvasScale*longCorrection);\n  var lat = y/canvasScale;\n  return [lat,long];\n}\n\n\nfunction adjustScale(){\n	if (destinationPoints.length>2){\n		let i=0\n		latRange = [robotCoords[0], robotCoords[0]];\n		longRange = [robotCoords[1], robotCoords[1]]\n		while (i<destinationPoints.length){\n			if (destinationPoints[i][0]>latRange[1]) {latRange[1]=destinationPoints[i][0]}\n			if (destinationPoints[i][0]<latRange[0]){latRange[0]=destinationPoints[i][0]}\n			if (destinationPoints[i][1]>longRange[1]){longRange[1]=destinationPoints[i][1]}\n			if (destinationPoints[i][1]<longRange[0]){longRange[0]=destinationPoints[i][1]}\n			i+=1;\n		}\n		let c = document.getElementById(\"pointMap\");\n		scaleLat = c.height/(latRange[1]-latRange[0]);\n		scaleLong = c.width/(longRange[1]-longRange[0]);\n		// console.log(\"range\", latRange, longRange);\n		// console.log(\"scale\", scaleLat, scaleLong)\n		if (Math.abs(scaleLat) < Math.abs(scaleLong)){\n			canvasScale = -Math.abs(scaleLat) * 0.3;\n		} else {\n			canvasScale = -Math.abs(scaleLong) * 0.3;\n		}\n	}\n}\n\n\n\nfunction drawBoard(){\n	let c = document.getElementById(\"pointMap\");\n	let ctx = c.getContext(\"2d\");\n	let canvasWidth = c.width;\n	let canvasHeight = c.height;\n	  //adjust this!\n\n	let canvasSize = c.getBoundingClientRect();\n\n\n	let centerCtxCoords = [canvasWidth/2, canvasHeight/2]\n\n	ctx.clearRect(0, 0, canvasWidth, canvasHeight);\n\n	if (robotCentered){\n		canvasCenterCoords = [...robotCoords]\n		adjustScale();\n	}\n\n\n\n\n	//draw map features\n	let i=0;\n	if (mapFeatures.length > 0){\n\n		while (i<mapFeatures.length) {\n\n			ctx.beginPath();\n\n			ctx.fillStyle = mapFeatures[i][0]\n\n			let firstCoords = coordToCtx(mapFeatures[i][1][0], mapFeatures[i][1][1]);\n			firstCoords = [centerCtxCoords[0]-firstCoords[0], (centerCtxCoords[1]-firstCoords[1])];\n			moveTo(firstCoords[0], firstCoords[1]);\n			let j = 2;\n			while (j<mapFeatures[i].length){\n\n				let coords = coordToCtx(mapFeatures[i][j][0], mapFeatures[i][j][1]);\n				coords = [centerCtxCoords[0]-coords[0], (centerCtxCoords[1]-coords[1])];\n				ctx.lineTo(coords[0], coords[1])\n		    j+=1;\n		  }\n\n		  // ctx.fillStyle = \"red\";\n		  ctx.lineTo(firstCoords[0], firstCoords[1])\n		  ctx.closePath();\n		  ctx.fill();\n		  ctx.stroke();\n			i+=1\n		}\n	}\n\n\n	//draw obstacles\n	i=0;\n\n	while (i<obstaclePoints.length) {\n		if (obstaclePoints[i].length>3) {\n			ctx.lineWidth = 1\n		} else{\n			ctx.lineWidth = 10;\n			ctx.lineColor = \"red\"\n		}\n		ctx.beginPath();\n		let firstCoords = coordToCtx(obstaclePoints[i][0][0], obstaclePoints[i][0][1]);\n		firstCoords = [centerCtxCoords[0]-firstCoords[0], (centerCtxCoords[1]-firstCoords[1])];\n		moveTo(firstCoords[0], firstCoords[1]);\n		let j = 1;\n		while (j<obstaclePoints[i].length){\n\n			let coords = coordToCtx(obstaclePoints[i][j][0], obstaclePoints[i][j][1]);\n			coords = [centerCtxCoords[0]-coords[0], (centerCtxCoords[1]-coords[1])];\n			ctx.lineTo(coords[0], coords[1])\n	    j+=1;\n	  }\n\n	  ctx.fillStyle = \"red\";\n	  ctx.lineTo(firstCoords[0], firstCoords[1])\n	  ctx.closePath();\n	  ctx.fill();\n	  ctx.stroke();\n		i+=1\n	}\n	\n	ctx.lineWidth = 1\n	ctx.lineColor = \"black\"\n\n	i = 0;\n	while (i<rows.length){\n		ctx.beginPath();\n		let coords = coordToCtx(rows[i][0][0], rows[i][0][1]);\n		coords = [centerCtxCoords[0]-coords[0], (centerCtxCoords[1]-coords[1])];\n		ctx.moveTo(coords[0], coords[1])\n\n		coords = coordToCtx(rows[i][1][0], rows[i][1][1]);\n		coords = [centerCtxCoords[0]-coords[0], (centerCtxCoords[1]-coords[1])];\n		ctx.lineTo(coords[0], coords[1])\n		ctx.stroke()\n		i+=1\n	}\n\n\n	// draw destinations\n	i=0;\n	while (i<destinationPoints.length){\n		let coords = coordToCtx(destinationPoints[i][0], destinationPoints[i][1]);\n		coords = [centerCtxCoords[0]-coords[0], (centerCtxCoords[1]-coords[1])];\n\n		ctx.fillStyle = \"black\"\n		ctx.beginPath();\n		ctx.arc(coords[0], coords[1], 8, 0, 2 * Math.PI);\n		ctx.fill();\n		// ctx.stroke();\n		ctx.font = '25px serif';\n    ctx.fillText(i+1, coords[0]-5, coords[1]-12);\n    ctx.stroke();\n		i+=1\n	}\n\n		// draw hand made points\n		i=0;\n		while (i<handMadePts.length){\n			let coords = coordToCtx(handMadePts[i][0], handMadePts[i][1]);\n			coords = [centerCtxCoords[0]-coords[0], (centerCtxCoords[1]-coords[1])];\n\n			ctx.fillStyle = \"blue\"\n			ctx.beginPath();\n			ctx.arc(coords[0], coords[1], 4, 0, 2 * Math.PI);\n			ctx.fill();\n	    ctx.stroke();\n			i+=1\n		}\n\n\n	// draw robot\n\n	let robotCtxCoords = coordToCtx(robotCoords[0], robotCoords[1]);\n	robotCtxCoords = [centerCtxCoords[0]-robotCtxCoords[0], centerCtxCoords[1] - robotCtxCoords[1]];\n	ctx.fillStyle = \"red\"\n	ctx.beginPath();\n	ctx.arc(robotCtxCoords[0], robotCtxCoords[1], 8, 0, 2 * Math.PI);\n	ctx.fill()\n		ctx.shadowBlur = 100;\n	ctx.shadowColor = \"black\";\n	ctx.stroke();\n	ctx.shadowBlur = 0;\n\n	// draw heading\n	// True heading\n	ctx.beginPath();\n	ctx.moveTo(robotCtxCoords[0],robotCtxCoords[1]);\n	ctx.strokeStyle = \"blue\";\n	otherX = robotCtxCoords[0]+Math.cos((trueHeading)*Math.PI/180-Math.PI/2)*50;\n	otherY = robotCtxCoords[1]+Math.sin((trueHeading)*Math.PI/180-Math.PI/2)*50\n	ctx.lineTo(otherX, otherY);\n	// console.log(otherX, otherY);\n	ctx.stroke();\n	ctx.strokeStyle = \"black\";\n\n	// draw path\n	if (showPath && pathPoints.length>2){\n			\n			let coords = coordToCtx(pathPoints[0][0], pathPoints[0][1]);\n			coords = [centerCtxCoords[0]-coords[0], centerCtxCoords[1]-coords[1]];\n			ctx.beginPath();\n			ctx.strokeStyle = \"rgb(0,0,0)\";\n			ctx.moveTo(coords[0], coords[1])\n			let i=1;\n			while (i<pathPoints.length){\n\n				let coords = coordToCtx(pathPoints[i][0], pathPoints[i][1]);\n				coords = [centerCtxCoords[0]-coords[0], centerCtxCoords[1]-coords[1]];\n				ctx.fillStyle = \"black\"\n				\n				ctx.lineTo(coords[0], coords[1])\n								\n				i+=1\n			}\n			ctx.stroke();\n	}\n}\n\n\n\nfunction drawCamBoard(){\n	let c = document.getElementById(\"camMap\");\n	let ctx = c.getContext(\"2d\");\n	let canvasWidth = c.width;\n	let canvasHeight = c.height;\n	ctx.clearRect(0, 0, canvasWidth, canvasHeight);\n\n	ctx.beginPath()\n	ctx.fillStyle = \"cyan\"\n	ctx.rect(0, 0, canvasWidth, canvasHeight/1.8);\n	ctx.fill();\n	ctx.stroke();\n\n	ctx.beginPath();\n	ctx.fillStyle = \"rgb(100,255,100)\";\n	ctx.rect(0, canvasHeight/1.8, canvasWidth, canvasHeight);\n	ctx.fill();\n	ctx.stroke();\n\n\n\n	let i=0;\n	while (i<obstructions.length) {\n		ctx.beginPath();\n		//int((i[1]-240)*100/480)\n		// (i[1]-wt/2) * 100 / wt\n		// i\n		c = obstructions[i]\n		let ht = canvasHeight;\n		let wt = canvasWidth;\n		let x = (c[0]/100 * wt)+wt/2;\n		let y = (c[1]/100 * ht)+ht/2;\n		let w = (c[2]/100 * wt)+wt/2 - x\n		let h = (c[3]/100 * ht)+ht/2 - y\n		let d1 = c[4]/100;\n		let d2 = c[5]/100;\n		let color = (d1+d2)/2;\n		if (color>255){color=255;}\n		// console.log(color)\n		let coords = [x,y,w,h]\n		ctx.rect(coords[0], coords[1], coords[2], coords[3])\n	  ctx.fillStyle = \"rgb(\"+ (255-color) +\",0,\" + color + \")\";\n	  ctx.fill();\n	  ctx.stroke();\n		i+=1\n	}\n\n	ctx.beginPath();\n	ctx.moveTo(canvasWidth/2, 0);\n	ctx.lineTo(canvasWidth/2, canvasHeight)\n	ctx.stroke();\n\n	let th = targetHeading - trueHeading\n	ctx.beginPath();\n	ctx.moveTo(th/90*canvasWidth/2 + canvasWidth/2, 0);\n	ctx.lineTo(th/90*canvasWidth/2 + canvasWidth/2, canvasHeight)\n	ctx.stroke();\n	\n}\n\n\n\n\n\nfunction makeStatusBoxes()	{\n\n	makeBox([2,18], 15, \"gpsBox\");\n	document.getElementById(\"gpsBoxHeader\").innerHTML = \"GPS\";\n	document.getElementById(\"gpsBoxBody\").innerHTML = \"Status: ___<br><br>Coords: (___, ___)<br><br>Accuracy: __ mm\";\n\n	makeBox([2,65], 15, \"messagesBox\");\n	document.getElementById(\"messagesBoxHeader\").innerHTML = \"Messages\";\n	document.getElementById(\"messagesBoxBody\").innerHTML = \"<br><br><br><br><br><br>\";\n\n\n	makeBox([50-7.5, 0], 15, \"headingBox\");\n	document.getElementById(\"headingBoxBody\").style.padding = \"0px\";\n	document.getElementById(\"headingBoxHeader\").innerHTML = \"Heading\";\n	document.getElementById(\"headingBoxBody\").innerHTML = \"0&#176 N\";\n	document.getElementById(\"headingBoxBody\").style.fontSize = \"30px\";\n	\n	makeBox([25, 3], 15, \"targetHeadingBox\");\n	document.getElementById(\"targetHeadingBoxBody\").style.padding = \"0px\";\n	document.getElementById(\"targetHeadingBoxHeader\").innerHTML = \"Target Heading\";\n	document.getElementById(\"targetHeadingBoxBody\").innerHTML = \"0&#176; N\";\n\n	makeBox([2, 3], 20, \"runTimeBox\");\n	document.getElementById(\"runTimeBoxBody\").style.padding = \"0px\";\n	document.getElementById(\"runTimeBoxHeader\").innerHTML = \"Run ID: ______\";\n	document.getElementById(\"runTimeBoxBody\").innerHTML = \"Runtime: 0:00:00\";\n\n	makeBox([60, 3], 20, \"connectionStatusBox\");\n	document.getElementById(\"connectionStatusBoxBody\").style.padding = \"0px\";\n	document.getElementById(\"connectionStatusBoxHeader\").innerHTML = \"Robot-Computer Connection\";\n	document.getElementById(\"connectionStatusBoxBody\").innerHTML = \"Connected\";\n\n	makeBox([50-15/2,88], 15, \"speedBox\");\n	document.getElementById(\"speedBoxHeader\").innerHTML = \"Wheel Speed\";\n	document.getElementById(\"speedBoxBody\").innerHTML = \"__, __\";\n\n	makeBox([25,88], 15, \"targetSpeedBox\");\n	document.getElementById(\"targetSpeedBoxHeader\").innerHTML = \"Target Speed\";\n	document.getElementById(\"targetSpeedBoxBody\").innerHTML = \"__, __ (__, __ off)\";\n\n	makeBox([63,88], 15, \"selectedCoordBox\");\n	document.getElementById(\"selectedCoordBoxHeader\").innerHTML = \"Selected Coords\";\n	document.getElementById(\"selectedCoordBoxBody\").innerHTML = \"__, __\";\n	document.getElementById(\"selectedCoordBox\").style.display = \"none\";\n\n\n	makeBox([83, 15], 15, \"maxSpeedBox\");\n	document.getElementById(\"maxSpeedBoxHeader\").innerHTML = \"Maximum Speed\";\n	speedInput = document.createElement(\"input\")\n	speedInput.type = \"number\";\n	speedInput.id=\"maxSpeedInput\";\n	speedInput.style.width = \"30px\";\n	document.getElementById(\"maxSpeedBoxBody\").appendChild(speedInput)\n	document.getElementById(\"maxSpeedBoxBody\").innerHTML += \" mph<br> Enable Movement \";\n	document.getElementById(\"maxSpeedInput\").value=controlVars[\"topSpeed\"];\n\n	document.getElementById(\"maxSpeedInput\").addEventListener(\"blur\", function () {\n				console.log(\"changed\");\n				if (this.value != controlVars[\"maxSpeed\"]){\n					controlVars[\"maxSpeed\"] = this.value;\n				if (!updatedControlVars.includes(\"maxSpeed\")){\n						console.log(\"adding value\")\n						updatedControlVars.push(\"maxSpeed\");\n					}\n				}\n			});\n\n\n	movementAllowed = document.createElement(\"input\")\n	movementAllowed.type = \"checkbox\";\n	movementAllowed.checked=true;\n	movementAllowed.id=\"movementAllowed\";\n	document.getElementById(\"maxSpeedBoxBody\").appendChild(movementAllowed);\n\n\n	makeBox([83, 3], 15, \"killBox\");\n	document.getElementById(\"killBoxHeader\").innerHTML = \"Kill Program\";\n	killButton = document.createElement(\"input\")\n	killButton.type=\"button\";\n	killButton.value = \"KILL\";\n	killButton.onclick=function(){\n		updatedControlVars.push(\"kill\");\n	}\n	document.getElementById(\"killBoxBody\").appendChild(killButton);\n\n\n\n\n	makeBox([83, 33], 15, \"saveLogsBox\");\n	document.getElementById(\"saveLogsBoxHeader\").innerHTML = \"Record Logs\";\n	document.getElementById(\"saveLogsBoxBody\").innerHTML = \"Record Logs:\";\n	recordLogBox = document.createElement(\"input\")\n	recordLogBox.type = \"checkbox\";\n	recordLogBox.checked=false;\n	recordLogBox.id=\"recordLogs\";\n	document.getElementById(\"saveLogsBoxBody\").appendChild(recordLogBox);\n\n	downloadLog = document.createElement(\"button\")\n	downloadLog.id=\"saveLog\";\n	downloadLog.innerHTML = \"Save Log Locally\"\n	downloadLog.onclick=function(){\n		console.log(\"save log locally\");\n		downloadLogFile(\"log_\" + runID + \".txt\", logData);\n	}\n	document.getElementById(\"saveLogsBoxBody\").appendChild(downloadLog)\n\n	\n	makeBox([83, 48], 15, \"destToleranceBox\");\n	document.getElementById(\"destToleranceBoxHeader\").innerHTML = \"Destination Tolerance\";\n	let destTolerance= document.createElement(\"input\")\n	destTolerance.type = \"number\";\n	destTolerance.id=\"destToleranceInput\";\n	document.getElementById(\"destToleranceBoxBody\").appendChild(speedInput)\n	document.getElementById(\"destToleranceBoxBody\").innerHTML += \" m\";\n\n\n	makeBox([83, 63], 15, \"updateSpeedBox\");\n	document.getElementById(\"updateSpeedBoxHeader\").innerHTML = \"Update Speed\";\n	let destToleranceSlider = document.createElement(\"input\")\n	destToleranceSlider.type = \"range\";\n	destToleranceSlider.id=\"updateSpeedSlider\";\n	destToleranceSlider.style.width = \"90%\"\n	destToleranceSlider.value = 100;\n	destToleranceSlider.min=1;\n	destToleranceSlider.max=300;\n	destToleranceSlider.oninput=function(){\n		document.getElementById(\"sliderValue\").innerHTML= this.value + \"%\";\n		controlVars[\"updateSpeed\"] = this.value;\n		if (!updatedControlVars.includes(\"updateSpeed\")){\n			console.log(\"adding value\")\n			updatedControlVars.push(\"updateSpeed\");\n		}\n	}\n	document.getElementById(\"updateSpeedBoxBody\").appendChild(destToleranceSlider)\n	\n  let sliderValue = document.createElement(\"span\");\n 	sliderValue.id = \"sliderValue\";\n 	sliderValue.innerHTML = \"100%\"\n 	document.getElementById(\"updateSpeedBoxBody\").appendChild(sliderValue)\n\n\n\n\n\n\n\n	makeBox([2,42], 15, \"MapToolsBox\");\n	document.getElementById(\"MapToolsBoxHeader\").innerHTML = \"Map Tools\";\n	document.getElementById(\"MapToolsBoxBody\").innerHTML = \"Show Path \";\n\n\n	// this is a mass of unorganized stuff to allow the creation of things in the right order. Touch it if you dare\n	let showPathBox = document.createElement(\"input\")\n	showPathBox.type = \"checkbox\";\n	showPathBox.id=\"showPathBox\";\n	document.getElementById(\"MapToolsBoxBody\").appendChild(showPathBox)\n	document.getElementById(\"MapToolsBoxBody\").innerHTML += \"<br>\"\n	let erasePathBox = document.createElement(\"button\");\n	erasePathBox.id = \"erasePathButton\";\n	document.getElementById(\"MapToolsBoxBody\").appendChild(erasePathBox);\n	document.getElementById(\"MapToolsBoxBody\").innerHTML += \"<br><br>Show Grids\"\n	document.getElementById(\"showPathBox\").checked=true;\n	document.getElementById(\"showPathBox\").onclick=function(){console.log(\"show path\"); showPath = !showPath; drawBoard();};\n\n	document.getElementById(\"erasePathButton\").innerHTML = \"Erase Path\";\n	document.getElementById(\"erasePathButton\").onclick=function(){console.log(\"erase path\"); pathPoints=[]; drawBoard();}\n	showGridsBox = document.createElement(\"input\")\n	showGridsBox.type = \"checkbox\";\n	showGridsBox.checked=true;\n	showGridsBox.id=\"showGrids\";\n	showGridsBox.onclick=function(){console.log(\"show grids\");};\n	document.getElementById(\"MapToolsBoxBody\").appendChild(showGridsBox)\n\n	// show path, erase path, show gridlines, show scale,\n}\n\n\nfunction downloadLogFile(filename, text) {\n    var pom = document.createElement('a');\n    pom.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));\n    pom.setAttribute('download', filename);\n\n    if (document.createEvent) {\n        var event = document.createEvent('MouseEvents');\n        event.initEvent('click', true, true);\n        pom.dispatchEvent(event);\n    }\n    else {\n        pom.click();\n    }\n}\n\n\n\n\n\nfunction makeCanvasDiv(){\n	a = document.createElement(\"div\");\n	a.style.position = \"absolute\";\n	a.style.width = \"60%\";\n	a.style.height = \"70%\"\n	a.style.left = \"50%\";\n	a.style.transform = \"translate(-50%, -50%)\";\n	a.style.top = \"50%\";\n\n	document.body.appendChild(a);\n	b = document.createElement(\"canvas\");\n	originalCanvasShape = [b.width, b.height]\n\n	b.id=\"pointMap\"\n\n	b.onmousemove = function(event){\n		if (mouseDown) {\n			let bound = document.getElementById(\"pointMap\").getBoundingClientRect();\n			let currentMouseSpot = [event.clientX - bound.x, event.clientY - bound.y];\n			if (lastMouseSpot[0] != 0) {\n\n				let coordChange = ctxToCoord(lastMouseSpot[0]-currentMouseSpot[0], lastMouseSpot[1]-currentMouseSpot[1])\n				canvasCenterCoords[0] += coordChange[0];\n				canvasCenterCoords[1] -= coordChange[1];\n			}\n			lastMouseSpot = currentMouseSpot;\n\n			drawBoard();\n			if (robotCentered){\n				robotCentered = false;\n				document.getElementById(\"reCenter\").style.display=\"block\";\n			}\n			dragged = true;\n		}\n	};\n	b.onmousedown = function(event) {\n		mouseDown = true;\n		let bound = document.getElementById(\"pointMap\").getBoundingClientRect();\n		lastMouseSpot = [event.clientX - bound.x, event.clientY - bound.y];\n		dragged = false;\n	}\n	b.onmouseup = function(event){\n			mouseDown = false;\n			let bound = document.getElementById(\"pointMap\").getBoundingClientRect();\n			let currentMouseSpot = [event.clientX - bound.x, event.clientY - bound.y];\n			if (dragged == false){\n					let c = document.getElementById(\"pointMap\");\n					let ctx = c.getContext(\"2d\");\n					let canvasWidth = c.width;\n					let canvasHeight = c.height;\n					let canvasSize = c.getBoundingClientRect();\n					let ctxCenterCoords = [canvasWidth/2, canvasHeight/2]\n					let crds = ctxToCoord(ctxCenterCoords[0]-currentMouseSpot[0], currentMouseSpot[1]-ctxCenterCoords[1])\n					crds[0] += canvasCenterCoords[0]\n					crds[1] += canvasCenterCoords[1]\n\n					handMadePts.push(crds)\n					\n				\n					document.getElementById(\"selectedCoordBox\").style.display = \"block\";\n					document.getElementById(\"selectedCoordBoxBody\").innerHTML = \"(\"+ crds[0].toFixed(7) + \", \" + crds[1].toFixed(7) + \")\";\n					drawBoard();\n\n			}\n\n	}\n\n	b.style.width=\"100%\";\n	b.style.height=\"100%\";\n\n	\n	canvasScale=1000000\n	b.addEventListener('wheel',function(event){\n		if ((Math.abs(canvasScale)>1 || event.deltaY>0) && (Math.abs(canvasScale)<10000000 || event.deltaY<0)) {\n		    canvasScale *= (1+event.deltaY/500);\n		    \n		    event.preventDefault();\n		    // console.log(\"scale\", canvasScale)\n		    if (robotCentered){\n					robotCentered = false;\n					document.getElementById(\"reCenter\").style.display=\"block\";\n				}\n				drawBoard();\n	\n		}\n	}, false);\n\n	b.style.background = \"rgb(200,255,200)\";\n	a.appendChild(b);\n	a.style.border = \"2px solid #73AD21\";\n\n	//get DPI\n	let dpi = window.devicePixelRatio;\n	//get canvas\n	let canvas = document.getElementById('pointMap');\n	//get context\n	let ctx = canvas.getContext('2d');\n	function fix_dpi() {\n		//get CSS height\n		//the + prefix casts it to an integer\n		//the slice method gets rid of \"px\"\n		let style_height = +getComputedStyle(canvas).getPropertyValue(\"height\").slice(0, -2);\n		//get CSS width\n		let style_width = +getComputedStyle(canvas).getPropertyValue(\"width\").slice(0, -2);\n		//scale the canvas\n		canvas.setAttribute('height', style_height * dpi);\n		canvas.setAttribute('width', style_width * dpi);\n	}\n	fix_dpi()\n\n	\n		d = document.createElement(\"div\");\n		\n		d.innerHTML = \"Re-Center\"\n		d.id = \"reCenter\";\n		d.style.position = \"absolute\";\n		d.style.textAlign = \"center\";\n		d.style.backgroundColor = \"rgb(100,100,100)\";\n		d.style.left = \"80%\";\n		d.style.transform = \"translate(-50%, -50%)\";\n		d.style.top = \"90%\";\n		d.style.width = \"15%\";\n		d.style.height = \"5%\";\n		d.style.textAlign = \"center\";\n		d.style.padding = \"10px\";\n		d.style.borderRadius = \"10px\";\n		d.onclick = function(){\n			// console.log(\"hello\");\n			robotCentered = true;\n			canvasCenterCoords = [...robotCoords];\n			adjustScale();\n			drawBoard();\n			this.style.display = \"none\"\n		};\n		a.appendChild(d);\n		d.style.display=\"none\";\n	\n\n\n\n\n\n	a2 = document.createElement(\"div\");\n	a2.style.position = \"absolute\";\n	a2.style.width = \"30%\";\n	a2.style.paddingBottom = \"20%\"\n	a2.style.left = \"0%\";\n	a2.style.top = \"0%\";\n\n	a.appendChild(a2);\n	b2 = document.createElement(\"canvas\");\n	originalCanvasShape = [b2.width, b2.height]\n	b2.style.position = \"absolute\";\n\n	b2.id=\"camMap\"\n\n	b2.style.width=\"100%\";\n	b2.style.height=\"100%\";\n\n	\n	b2.style.background = \"rgb(255,255,255)\";\n	a2.appendChild(b2);\n	a2.style.border = \"2px solid #73AD21\";\n\n	a2.style.display = \"none\";\n	\n}\n\n\n// makeBox()\n\nfunction makeBox(location, width, id){\n	d = document.createElement(\"div\");\n	d.id = id\n	a.style.border = \"2px solid rgb(0,0,0)\";\n	d.style.position = \"absolute\";\n	d.style.textAlign = \"center\";\n	d.style.backgroundColor = \"rgb(200,200,200)\";\n	d.style.left = location[0]+\"%\";\n	d.style.transform = \"translate(-50%, -50)\";\n	d.style.top = location[1] + \"%\";\n	d.style.borderRadius = \"10px\";\n	d.style.width = width + \"%\";\n	// d.style.border = \"2px solid #73AD21\";\n	document.body.appendChild(d);\n\n	e = document.createElement(\"div\");\n	e.id = id + \"Header\";\n	// e.innerHTML = \"clcik here\";\n	// e.style.cursor = \"move\";\n	e.style.padding = \"10px\";\n	e.style.borderRadius = \"10px 10px 0px 0px\";\n	e.style.backgroundColor = \"#2196F3\";\n	d.appendChild(e);\n\n	e = document.createElement(\"p\")\n	// e.innerHTML = \"blah\"\n	e.id = id + \"Body\"\n	d.appendChild(e);\n}\n\n\n\nmakeCanvasDiv();\n\ndrawBoard();\nmakeStatusBoxes();\ndrawCamBoard();\n\n</script>\n\n\n\n</body>\n</html>");
  });

  server.on("/_info", [](AsyncWebServerRequest* request) {
    if (request->hasParam("kill")) {
      Serial.println("k");
      serverMsg = "{\"status\": [\"-2\", \"-1\"]}";
      gracefulShutdown = true;
    }

    if (millis() - lastSerialReadTime > crashTimeout && lastSerialReadTime != 0 && !gracefulShutdown) {
      // Serial.println("didnt get a message for a while");
      serverMsg = "{\"status\": [\"-3\", \"-1\"]}";  // didn't get a message in a while and there was no graceful shutdown
    }
    // Serial.print(".server msg");
    // Serial.println(serverMsg);

    request->send(200, "text/plain", serverMsg);
  });

  server.begin();
}

void setupAP() {
  Serial.println("\n.Configuring access point...");

  WiFi.softAP(apName.c_str());

  IPAddress myIP = WiFi.softAPIP();
  // Serial.print(".IP address: " + WiFi.localIP());

  String mdnsName = "ESP32";
  if (MDNS.begin(mdnsName.c_str())) {
    // Serial.println(".MDNS responder started as http://" + mdnsName + ".local/");
  }
  setupPages();
}


void readLastExit() {
  // return;

  if (!SPIFFS.begin(true)) {
    Serial.println(".SPIFFS Mount Failed");
    return;
  } else {
    // Serial.println(".SPIFF Mount Successful");
  }
  File file = SPIFFS.open("/crashLogs.txt");
  if (!file || file.isDirectory()) {
    // Serial.println(".No past logs found");
    return;
  }

  int crashReason = 0;  // 0=graceful, 1=unexpected power cycle, 2=robot python script crash
  unsigned long crashTime = 0;
  while (file.available()) {
    crashReason = 1;
    crashTime = 0;
    char c = file.read();
    if (c == '\n') {
      crashReason = 1;
      crashTime = 0;
    } else if (c == 'g') {
      crashReason = 0;
    } else if (c == 'b') {
      crashReason = 2;
    } else if (c == '-') {
      crashTime += 10;
    }
  }
  if (crashReason == 1) {
    Serial.println(".last crash from unexpected power cycle");
    serverMsg = "{\"status\": [\"-4\", \"-1\"]}";
  } else if (crashReason == 0) {
    serverMsg = "{\"status\": [\"-2\", \"-1\"]}";
    Serial.println("last crash graceful");
  } else if (crashReason == 2) {
    serverMsg = "{\"status\": [\"-3\", \"-1\"]}";
    Serial.println(".last crash from unexpected python error");
  } else {
    Serial.println(".unknown crash type: " + String(crashReason));
  }
  Serial.print(".script lasted for ");
  Serial.println(crashTime);

  if (file.size() > 5000) {
    Serial.println(".crash file too big. Deleting");
    file.close();
    SPIFFS.remove("/crashLogs.txt");
  }
}


void recordStatus(char msg) {

  File file = SPIFFS.open("/crashLogs.txt", FILE_APPEND);
  if (!file) {
    Serial.println(".failed to open file for writing. One more attempt");
    file = SPIFFS.open("/crashLogs.txt", FILE_WRITE);
    if (!file) {
      Serial.println(".yep, writing really doesn't work. Qutting");
      return;
    }
  }
  file.print(msg);
}

void setup() {
  Serial.begin(115200);
  Serial.setRxBufferSize(1024); // increase the serial buffer size for json
  setupAP();
  readLastExit();
}

void loop() {
  if (lastLogTime == 0 && lastSerialReadTime != 0) {
    // Serial.println(".started getting serial. Recording");
    recordStatus('\n');
    lastLogTime = millis();
  } else if (millis() - lastLogTime > 10000 && millis() - lastSerialReadTime < 3000) {  // haven't logged for 10 seconds and is getting serial
    // Serial.println(".have been getting serial normally. Recording");
    recordStatus('-');
    lastLogTime = millis();
  } else if (millis() - lastSerialReadTime > crashTimeout && lastSerialReadTime != 0 && !gracefulShutdown) {  // havent gotten any messages for 3 seconds. Record this.
    Serial.println(".didnt get a message in 10 seconds. assuming this was a crash. logging.");
    recordStatus('b');
    lastSerialReadTime = 0;
    lastLogTime = 0;
    serverMsg = "{\"status\": [\"-3\", \"-1\"]}";
  }

  readSerial();

  if (millis() - lastMsgRequestTime > 500) {
    Serial.println("m");
    lastMsgRequestTime  = millis();
  }
}