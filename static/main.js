
var targetSpeed = [0,0];

var realSpeed = [0,0];

var latitude = 0;
var longitude = 0;

var destinations = [];
var haveDestinations = 0;

var stop = 0;

var targPosApplied = false;

var heading = 0;
var targetHeading = 0;


console.log("Starting");


ready = true;
function sendInfo(){  

 if (ready){
   // ready = false;
  var xhttp = new XMLHttpRequest();
  xhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) { 
      console.log(this.responseText);

      info = JSON.parse(this.responseText);
      console.log(info);

      latitude = info["coords"][0];

      longitude = info["coords"][1];

      realSpeed = info["realSpeed"];

      targetSpeed = info["targetSpeed"]



      heading = info["heading"];
      targetHeading = info["targetHeading"];

      if ("destinations" in info) { // sometimes doesnt send destinations since it is a big array of a lot of data
        destinations = info["destinations"];
        console.log("destinations:", destinations);
        haveDestinations = 0;
      }

      setScale();
      drawBoard();


      document.getElementById("leftSpeed").innerHTML = realSpeed[0];
      document.getElementById("rightSpeed").innerHTML = realSpeed[1];

      document.getElementById("lat").innerHTML = latitude;
      document.getElementById("long").innerHTML = longitude;

      document.getElementById("targetHeading").innerHTML = targetHeading;
      document.getElementById("heading").innerHTML = heading;

      document.getElementById("desLeftSpeed").innerHTML = targetSpeed[0];
      document.getElementById("desRightSpeed").innerHTML = targetSpeed[1];

          
       } else if (this.status == 0){ // did not get correct message
         // ready = false;
        }
     };

    shouldOverride = 1;
    
    argList = "?haveDestinations=" + haveDestinations;
     


    console.log("arg list", argList);
     xhttp.open('GET', '/_info' + argList, true);
     xhttp.send();
 } else {
    console.log("not ready");
 }
}



/*
From here down, I have no idea what happens exactly. 
It takes the coordinates of the robot and the destinations, and draws it.
There are some equations I used to automatically set the scale for the map and translate coordinates to an XY system.
It works so don't touch it unless you plan to re-write everything.
*/

function drawBoard(){
      var c = document.getElementById("mainCanvas");
      var ctx = c.getContext("2d");
      ctx.clearRect(0, 0, 1000, 1000);
      if (canvasScale==0){
        setScale();
      }
      drawDestinations(ctx);
      drawRobot(ctx);
}

var canvasScale = 0;
// drawBoard();

function setScale(){
  console.log("set scale");
  longitude0=0
  latitude0 =0
  var longCorrection = Math.cos(latitude*Math.PI/180);


  if (destinations.length==0){
    canvasScale = 1;
    document.getElementById("scale").value = Math.floor(canvasScale);
    return;
  }

  var tp =  [...destinations];
  tp.push([latitude, longitude]);
  // longCorrection = 1;
 // console.log(longCorrection);

  var i = 0;
  var minX = (tp[0][1]-longitude)*longCorrection;
  var maxX = (tp[0][1]- longitude)*longCorrection;
  var minY = tp[0][0]-latitude;
  var maxY = tp[0][0]-latitude;

  while (i<tp.length){
   
    var x = (tp[i][1]-longitude)*longCorrection; // longitude
    var y = tp[i][0]-latitude; // latitude
  // console.log("dist to target:", Math.sqrt(x*x+y*y)*69*5280, "feet"); // 0.3387 miles

    if (x>maxX){
      maxX = x;
    } else if (x<minX){
      minX = x;
    }
    if (y>maxY){
      maxY = y;
    } else if (y<minY){
      minY = y
    }
    i+=1;
  }
//  console.log("maxX", maxX, "minX", minX, "maxY", maxY,"minY",minY);
  i=0;

  var scale = (maxY-minY)*1.5;
  if ((maxX-minX)*1.5>scale){
    scale = (maxX-minX)*1.5;
  }
  

  scale = 300/scale;


  scale = Math.floor(scale);
  
  if (scale != canvasScale) {
    canvasScale = scale;
    if (canvasScale > 1){
      canvasScale -= 1;
    }

  }

  document.getElementById("scale").value = canvasScale;


  //drawGrids()
}

function getOrderOfMagnitude(n) {
    var order = Math.floor(Math.log(n) / Math.LN10
                       + 0.000000001); // because float math sucks like that
    return order;
}



function coordToCtx(lat, long){
  var longCorrection = Math.cos(latitude*Math.PI/180);
  // longCorrection=1;

  var x = canvasScale*(long - longitude)*longCorrection + 300;
  var y = 600-(canvasScale*(lat - latitude)+300);

  return [x,y];
}

function ctxToCoord(x, y){
  var longCorrection = Math.cos(latitude*Math.PI/180);

  var long = (canvasScale*longitude*longCorrection+x-300)/(canvasScale*longCorrection);

  var lat = (canvasScale*latitude-y+300)/canvasScale;

  return [lat,long];
}
// 1 degree of Longitude = cosine (latitude in decimal degrees) * length of degree (miles) at equator



function clickEvent(event){
  var clickedCoords = ctxToCoord(event.offsetX, event.offsetY);
  console.log(clickedCoords);
}


function drawDestinations(ctx){
  a = [40.4216702, -86.9184231]//coordToCtx(40,-80);
  console.log("convert", coordToCtx(a[0],a[1]));
  var i = 0;
    // console.log(targetPositions);


    n = 0
    while (n<destinations.length){
      c = destinations[n]
      console.log(c)
      coords = coordToCtx(c[0], c[1]);
    
      var x = coords[0];
      var y = coords[1];
     
      

      color = "black";
      ptSize = 4;

        
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      console.log(color +" made size: " + ptSize);

      ctx.beginPath();
      ctx.arc(x, y, ptSize, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      
      ctx.font = '25px serif';
      ctx.fillText(n+1, x-5, y-10);
      ctx.stroke();
      
      console.log("drew at", x,y)
      n+=1;
    }

  while (i<destinations.length){

    var coords = coordToCtx(destinations[i][0], destinations[i][1]);

    i+=1;
  }
 
  ctx.fillStyle = "black";
  ctx.strokeStyle = "black";


}

function drawRobot(ctx){
    ctx.beginPath();
    ctx.fillStyle = "red";
    var coords = coordToCtx(latitude, longitude);
    var x = coords[0]
    var y = coords[1]
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
   // console.log(x,y);

    ctx.fill();
    ctx.stroke();


    // real heading
    ctx.strokeStyle = "red";
    ctx.beginPath();
    ctx.moveTo(x,y);

    var otherX = x+Math.cos(heading*Math.PI/180-Math.PI/2)*30;
    var otherY = y+Math.sin(heading*Math.PI/180-Math.PI/2)*30
    ctx.lineTo(otherX, otherY);
  //  console.log(otherX, otherY);
    ctx.stroke();



    // target heading
    ctx.beginPath();
    ctx.moveTo(x,y);
    ctx.strokeStyle = "blue";

    otherX = x+Math.cos((targetHeading)*Math.PI/180-Math.PI/2)*30;
    otherY = y+Math.sin((targetHeading)*Math.PI/180-Math.PI/2)*30
    ctx.lineTo(otherX, otherY);
    // console.log(otherX, otherY);
    ctx.stroke();

    ctx.strokeStyle = "black";    
    ctx.fillStyle = "black";

}

function toggleOverride(){
  if (document.getElementById("observe").checked){
    overriding = false;
  } else {
    overriding = true;
  }
}

function begin(){

  drawBoard();
 // updateDestinations();
  // processInfo("-40.00a0.00w93h40.421669x-86.91911y[40.421779, -86.91931],[40.421806, -86.919074],[40.421824, -86.918487],[40.421653, -86.918739],[40.421674, -86.919232]c-65.05924190668195t")
  var requestInt = setInterval(sendInfo, 200);

}


