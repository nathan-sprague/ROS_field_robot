var desSteer=0;
var desSpeed=0;

var realAngle = 0;
var realSpeed = 0;

var latitude = 0;//40.421779;
var longitude = 0;//-86.919310;

var pointsList = {};
var pointsListVersion = -1;
// var pointsOnMap = {"black": [10, [40.421779, -86.919310], [40.421806, -86.919074], [40.421824, -86.918487], [40.421653, -86.918739], [40.421674, -86.919232]]};
var allPoints = [];
var stop = 0;

var targPosApplied = false;

var heading = 0;
var targetHeading = 0;

var overriding = false;



elements = [1,2,3,4];
var subPoints = {}

console.log("Starting");


ready = true;
function sendInfo(){  

 if (ready){
   // ready = false;
     var xhttp = new XMLHttpRequest();
     xhttp.onreadystatechange = function() {
       if (this.readyState == 4 && this.status == 200) {
          info = JSON.parse(this.responseText);
          console.log(info);
          latitude = info["coords"][0];
          longitude = info["coords"][1];

          wheelSpeed = info["wheelSpeed"];
          

          realAngle = info["realAngle"];
      
          heading = info["heading"];
          targetHeading = info["targetHeading"];

          if ("pointsList" in info) {
            

            pointsList = info["pointsList"];
            console.log("points list", pointsList);
          //  pointsListVersion = info["pointsListVersion"];
            // console.log(pointsListVersion);
        //    k = Object.keys(pointsList);

            j=0;
            allPoints = [];
            while (j<pointsList.length){
              allPoints.push(pointsList[j]["coord"]);
              j+=1;
            }
            console.log("all points", allPoints);


         //   updateDestinations();

          }

          document.getElementById("speed").innerHTML = wheelSpeed;
          document.getElementById("angle").innerHTML = realAngle;

          document.getElementById("lat").innerHTML = latitude;
          document.getElementById("long").innerHTML = longitude;

          document.getElementById("targetHeading").innerHTML = targetHeading;
          document.getElementById("heading").innerHTML = heading;

          if (!overriding){
              desSpeed = info["targetSpeed"];
              desSteer = info["targetAngle"];

              document.getElementById("desSpeed").innerHTML = desSpeed;
              document.getElementById("desAngle").innerHTML = desSteer;
          }
          
          
          setScale();
          drawBoard();



       } else if (this.status == 0){
         // ready = false;
        }
     };

      //displaySpeed();


    shouldOverride = 1;
    if (overriding){
      argList = "?override=1";
     if (!stop){

        argList += "&angle=" + desSteer + "&speed=" + desSpeed;
      }
    } else {

      argList = "?override=0"
    }
    if (stop){
      argList += "&s=1"
    }

    // if (coordListVersion == -2){
    //   argList += "&targetPositions=" + targetPositions.toString()
    // }


    argList += "&pointsListVersion=" + pointsListVersion;
     


    console.log("arg list", argList);
     xhttp.open('GET', '/_info' + argList, true);
     xhttp.send();
 } else {
    console.log("not ready");
 }
}



//-40.00a0.00w93h0.0x0.0y[40.421779, -86.91931],[40.421806, -86.919074],[40.421824, -86.918487],[40.421653, -86.918739],[40.421674, -86.919232]c-65.05924190668195t

function changeSpeed(val){
    if (overriding){
      if ((val == 87 || val == 38) && desSpeed<15) { // forward
       desSpeed+=1;
      } else if ((val == 83 || val == 40) && desSpeed>-15) {  // backward
       desSpeed-=1;

      } else if ((val == 65 || val == 37) && desSteer>-40) { // left
       desSteer-=1;
      } else if ((val == 68 || val == 39) && desSteer<40) { //right
       desSteer+=1;
      }
      console.log(desSteer,desSpeed);
      
      if ((val == 32)){ // stop
        desSteer=0;
        desSpeed=0;
        sendInfo();
      }
      displaySpeed();
    }
}


function displaySpeed(){

     document.getElementById("desSpeed").innerHTML = desSpeed;
      document.getElementById("desAngle").innerHTML = desSteer;
    }

function stopNow(){
  if (document.getElementById("stop").innerHTML == "STOP"){
    stop = 1;
    changeSpeed(32);
    document.getElementById("stop").innerHTML = "GO";
  } else {
    stop = 0;
    document.getElementById("stop").innerHTML = "STOP";
  }
}






// var targetPositions = [[47.607533, -122.217883], [26.024640, -81.102921], [45.782204, -66.304858]]; // washington, Florida, Maine

// var targetPositions = [[40.424080, -86.925006], [40.423919, -86.913810], [40.418749, -86.913530], [40.419322, -86.924389]];
function drawBoard(){
      var c = document.getElementById("mainCanvas");
      var ctx = c.getContext("2d");
      ctx.clearRect(0, 0, 1000, 1000);
      if (canvasScale==0){
        setScale();
      }
      drawDestinations(ctx);
      drawRobot(ctx);


      // ctx.beginPath();
      // ctx.moveTo(x, y);
      // ctx.lineTo(x, y);
      // ctx.stroke();
}

var canvasScale = 0;
// drawBoard();

function setScale(){
  console.log("set scale");
//  console.log(allPoints);
    longitude0=0
  latitude0 =0
  var longCorrection = Math.cos(latitude*Math.PI/180);
  console.log("all points", allPoints);
  if (allPoints.length==0){
    canvasScale = 1;
    document.getElementById("scale").value = Math.floor(canvasScale);
    return;
  }

  var tp =  [...allPoints];
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
    // if (canvasScale > 9000000){
    //   canvasScale = 9000000;
      //console.log("scale", canvasScale);
      
      
    // }
  }

  document.getElementById("scale").value = canvasScale;


  //drawGrids()
}

function getOrderOfMagnitude(n) {
    var order = Math.floor(Math.log(n) / Math.LN10
                       + 0.000000001); // because float math sucks like that
    return order;
}

function drawGrids(){


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
  document.getElementById("desLatInput").value = clickedCoords[0];
  document.getElementById("desLongInput").value = clickedCoords[1];

}


function drawDestinations(ctx){
  a = [40.4216702, -86.9184231]//coordToCtx(40,-80);
  console.log("convert", coordToCtx(a[0],a[1]));
  var i = 0;
    // console.log(targetPositions);


    n = 0
    while (n<pointsList.length){
      c = pointsList[n]["coord"]
      console.log(c)
      coords = coordToCtx(c[0], c[1]);
    
      var x = coords[0];
      var y = coords[1];
     
      

      color = "black";
      ptSize = 2;

      if (pointsList[n]["destType"]=="point"){
        color = "black";
        ptSize = 5;

      } else if (pointsList[n]["destType"]=="beginRow"){
        color = "green";
        ptSize = 6;

      } else if (pointsList[n]["destType"]=="endRow"){
        color = "blue";
        ptSize = 6;

      } else if (pointsList[n]["destType"]=="subPoint"){
        color = "yellow";
        ptSize = 2;
      }
        
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      console.log(color +" made size: " + ptSize);

      ctx.beginPath();
      ctx.arc(x, y, ptSize, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      if (pointsList["destType"]=="point"){
        ctx.font = '25px serif';
        ctx.fillText(n+1, x-5, y-10);
        ctx.stroke();
      }
      console.log("drew at", x,y)
      n+=1;
    }

  while (i<allPoints.length){

    var coords = coordToCtx(allPoints[i][0], allPoints[i][1]);

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

    ctx.strokeStyle = "red";
    ctx.beginPath();
    ctx.moveTo(x,y);


    var otherX = x+Math.cos(heading*Math.PI/180-Math.PI/2)*30;
    var otherY = y+Math.sin(heading*Math.PI/180-Math.PI/2)*30
    ctx.lineTo(otherX, otherY);
  //  console.log(otherX, otherY);
    ctx.stroke();


    ctx.beginPath();
    ctx.moveTo(x,y);
    ctx.strokeStyle = "green";
    console.log(heading+realAngle)

    otherX = x+Math.cos((heading+realAngle)*Math.PI/180-Math.PI/2)*30;
    otherY = y+Math.sin((heading+realAngle)*Math.PI/180-Math.PI/2)*30
    ctx.lineTo(otherX, otherY);
    // console.log(otherX, otherY);
    ctx.stroke();

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

  displaySpeed();
  drawBoard();
 // updateDestinations();
  // processInfo("-40.00a0.00w93h40.421669x-86.91911y[40.421779, -86.91931],[40.421806, -86.919074],[40.421824, -86.918487],[40.421653, -86.918739],[40.421674, -86.919232]c-65.05924190668195t")
  var requestInt = setInterval(sendInfo, 500);

}


