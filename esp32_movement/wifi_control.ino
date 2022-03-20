


#if USE_AP

void setupAP() {
  Serial.println(".Configuring access point...");

  WiFi.softAP("robot");
  IPAddress myIP = WiFi.softAPIP();
  Serial.print(".IP address: " + WiFi.localIP());

  String mdnsName =  "ESP32";
  if (MDNS.begin(mdnsName.c_str())) {
    Serial.println(".MDNS responder started as http://" + mdnsName + ".local/");
  }
}
#endif

#if USE_WIFI

void setupPages() {
  server.on("/", [](AsyncWebServerRequest * request) {
    request->send(200, "text/html",
                  "<!DOCTYPE html>\n<script>\n\n\nvar desVars = {\"maxSpeed\": 0, \"pwmLimitHigh\": 190, \"pwmLimitLow\": 120, \"kp\": 1, \"ki\": 0, \"kd\": 0};\n\n\nvar realVars = {};\n\n\nvar statuses = [\"&#10060;\", \"&#128472;\", \"&#9989;\"];\n\n\nconsole.log(\"Starting\");\n\nvar getInfoAttempts = 0;\n\nready = true;\nfunction requestInfo(){  \n  console.log(\"requesting info\");\n\n if (true){\n   // ready = false;\n     var xhttp = new XMLHttpRequest();\n     xhttp.onreadystatechange = function() {\n       if (this.readyState == 4 && this.status == 200) {\n\n          info = JSON.parse(this.responseText);\n          console.log(info);\n            for (const [key, value] of Object.entries(info)) {\n              document.getElementById(key).innerHTML = value;\n              document.getElementById(key+\"Status\").innerHTML = statuses[2];\n            }\n            getInfoAttempts = 0;\n\n\n       } else if (this.status == 0){\n         // ready = false;\n        }\n     };\n\n      //displaySpeed();\n\n\n    console.log(\"arg list: /request\");\n     xhttp.open('GET', '/_request', true);\n     xhttp.send();\n     getInfoAttempts += 1;\n    \n     if (getInfoAttempts > 10){\n        document.getElementById(\"realRightSpeedStatus\").innerHTML = statuses[0]\n        document.getElementById(\"realLeftSpeedStatus\").innerHTML = statuses[0]\n      } else if (getInfoAttempts > 4){\n        document.getElementById(\"realRightSpeedStatus\").innerHTML = statuses[1]\n        document.getElementById(\"realLeftSpeedStatus\").innerHTML = statuses[1]\n      } \n } else {\n    console.log(\"not ready\");\n }\n}\n\nfunction sendInfo(){\n   // ready = false;\n   console.log(\"sending info\");\n   var xhttp = new XMLHttpRequest();\n   xhttp.onreadystatechange = function() {\n\n     if (this.readyState == 4 && this.status == 200) {\n        for (const [key, value] of Object.entries(desVars)) {\n          document.getElementById(key+\"Status\").innerHTML = statuses[2];\n        }\n\n\n     } else if (this.status == 0){\n       // ready = false;\n      }\n   };\n\n    //displaySpeed();\n  args = \"\";\n  desVars = {\"maxSpeed\": document.getElementById(\"maxSpeed\").value, \"pwmLimitHigh\": document.getElementById(\"pwmLimitHigh\").value, \"pwmLimitLow\": document.getElementById(\"pwmLimitLow\").value, \"kp\": document.getElementById(\"kp\").value, \"ki\": document.getElementById(\"ki\").value, \"kd\": document.getElementById(\"kd\").value};\n  if (document.getElementById(\"pwmControl\").checked) {\n    desVars[\"pwmControl\"] = \"1\";\n  } else {\n    desVars[\"pwmControl\"] = \"0\";\n  }\n\n  for (const [key, value] of Object.entries(desVars)) {\n    if (realVars[key] != value){\n      args += \"&\" + key + \"=\" + value;\n    }\n\n  }\n\n  args = \"?\" + args.substring(1);\n\n\n\n  console.log(\"sending: /_info\" + args);\n  xhttp.open('GET', '/_info' + args, true);\n  xhttp.send();\n  console.log(desVars)\n  for (const [key, value] of Object.entries(desVars)) {\n    console.log(key+\"Status\");\n    document.getElementById(key + \"Status\").innerHTML = statuses[1];\n  }\n}\n\n\nfunction displayInfo(){\n  for (const [key, value] of Object.entries(desVars)) {\n    console.log(key);\n    document.getElementById(key).value = value;\n  }\n}\n\n\nvar myInterval = setInterval(requestInfo, 500);\n\n</script>\n\n\n\n<html>\n\n\n\n\n...............................................................Updated:<br><br>\n\nPID Control: <input type=\"checkbox\" id=\"pwmControl\" checked> ..........................................<span id=\"pwmControlStatus\">&#10060;</span><br><br>\n\n\nTrue Right Speed: <span id=\"realRightSpeed\">__</span> mph ............................<span id=\"realRightSpeedStatus\">&#128472;</span><br>\nTrue Left Speed: <span id=\"realLeftSpeed\">__</span> mph ..............................<span id=\"realLeftSpeedStatus\">&#128472;</span><br><br>\n\nMax Speed: <input type=\"number\" id=\"maxSpeed\"> ............<span id=\"maxSpeedStatus\">&#10060;</span><br><br>\n\nPWM Upper Limit: <input type=\"number\" id=\"pwmLimitHigh\"> ...<span id=\"pwmLimitHighStatus\">&#10060;</span><br><br>\n\nPWM Lower Limit: <input type=\"number\" id=\"pwmLimitLow\"> ...<span id=\"pwmLimitLowStatus\">&#10060;</span><br><br>\n\nkp: <input type=\"number\" id=\"kp\"> ..........................<span id=\"kpStatus\">&#10060;</span><br><br>\n\nki: <input type=\"number\" id=\"ki\"> ..........................<span id=\"kiStatus\">&#10060;</span><br><br>\n\nkd: <input type=\"number\" id=\"kd\"> .........................<span id=\"kdStatus\">&#10060;</span><br><br>\n\n\n\n\n\n<button type=\"button\" id=\"setTarget\", onclick=\"sendInfo();\">Update</button>\n</p>\n\n</html>\n\n<script>\ndisplayInfo();\n</script>"
                 );
  });

  server.on("/_info", [](AsyncWebServerRequest * request) {

    String input = String(request->getParam("maxSpeed")->value());
    maximumSpeed = input.toFloat();

    input = String(request->getParam("kp")->value());
    kp = input.toFloat();

    input = String(request->getParam("ki")->value());
    ki = input.toFloat();

    input = String(request->getParam("kd")->value());
    kd = input.toFloat();

    input = String(request->getParam("pwmLimitLow")->value());
    pwmLimitLow = input.toInt();

    input = String(request->getParam("pwmLimitHigh")->value());
    pwmLimitHigh = input.toInt();

    if (request->hasParam("pwmControl")) {
      input = String(request->getParam("pwmControl")->value());
      if (input == "1") {
        pwmControl = true;
      } else {
        pwmControl = false;
      }
    }


    request->send(200, "text/plain", "good");

  } );


  server.on("/_request", [](AsyncWebServerRequest * request) {
    String msg = "{\"realRightSpeed\":" + String(wheelSpeed[0]) + ", \"realLeftSpeed\":" + String(wheelSpeed[1]) + "}";
    request->send(200, "text/plain", msg);

  } );

}

#endif
