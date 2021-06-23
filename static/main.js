var desSteer=0;
var desSpeed=0;

var realAngle = 0;
var realSpeed = 0;

var latitude = 0;//40.421779;
var longitude = 0;//-86.919310;

var targetPositions = [];
// var targetPositions = [[40.421779, -86.919310], [40.421806, -86.919074], [40.421824, -86.918487], [40.421653, -86.918739], [40.421674, -86.919232]];

var stop = 0;

var targPosApplied = false;

var heading = 0;
var targetHeading = 0;

var overriding = false;

var coordListVersion = -1

elements = [1,2,3,4];
var subPoints = []



ready = true;
function sendInfo(){  
  console.log(targetPositions);
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

          if ("coordList" in info) {
            console.log("updating destinations");
            targetPositions = info["coordList"];
            coordListVersion = info["coordListVersion"];
            console.log(coordListVersion);
            updateDestinations();

          } if ("subPoints" in info){
              subPoints = info["subPoints"];
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

    if (coordListVersion == -2){
      argList += "&targetPositions=" + targetPositions.toString()
    }


    argList += "&coordListVersion=" + coordListVersion;
     


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

document.onkeydown = function(evt) {
    evt = evt || window.event;
    // console.log(evt.keyCode);

    changeSpeed(evt.keyCode)
    // sendInfo();
    //w 87 38
    //a 65 37
    //s 83 40
    //d 68 39
    //space 32
    
};

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
drawBoard();

function setScale(){
  console.log("set scale");
    longitude0=0
  latitude0 =0
  var longCorrection = Math.cos(latitude*Math.PI/180);
  
  if (targetPositions.length==0){
    canvasScale = 1;
    document.getElementById("scale").value = Math.floor(canvasScale);
    return;
  }

  var tp =  [...targetPositions];
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

  var scale = maxY-minY;
  if (maxX-minX>scale){
    scale = maxX-minX;
  }
  

  scale = 300/scale;


  canvasScale = Math.floor(scale);
  if (canvasScale > 1){
    canvasScale -= 1;
  }
  //console.log("scale", canvasScale);

  document.getElementById("scale").value = canvasScale;
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
  var i = 0;
    // console.log(targetPositions);


  while (i<targetPositions.length){

    var coords = coordToCtx(targetPositions[i][0], targetPositions[i][1]);
    var x = coords[0];
    var y = coords[1];
    // console.log(coords);
    // console.log(targetPositions[i], coords);
    // console.log(targetPositions[i][1]+1);

    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.font = '25px serif';
    ctx.fillText(i+1, x-5, y-10);
    ctx.stroke();
    i+=1;
  }
  i=0;
  while (i<subPoints.length){
    ctx.fillStyle = "orange";
    ctx.strokeStyle = "orange";
    var coords = coordToCtx(subPoints[i][0], subPoints[i][1]);
    var x = coords[0];
    var y = coords[1];
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

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


function updatePriority(amount){

  var select = document.getElementById("targets");
  var s = select.selectedIndex;
  // console.log(s);
  if (s+amount<0 || s+amount>targetPositions.length || s ==-1){
    console.log("At end, index would be ", s+amount);
    return;
  }
  value = targetPositions[s];


  targetPositions.splice(s, 1);
  if (amount!=0){
    s+=amount;
    targetPositions.splice(s, 0, value);
  }
  coordListVersion = -2;
  updateDestinations();
}

function addDest(x,y){


  targetPositions.splice(0, 0, [parseFloat(x),parseFloat(y)]);
  coordListVersion = -2;
  updateDestinations();

}

function updateDestinations(){
  var select = document.getElementById("targets");

  var length = select.options.length;
  for (i = length-1; i >= 0; i--) {
    select.options[i] = null;
  }

  var i = 0;
  while (i<targetPositions.length){
    var option = document.createElement("option");
    option.text = targetPositions[i][0] + ", " + targetPositions[i][1];
    
    select.add(option);
    i+=1;
  }
  select.size = i;
  drawBoard();
}



function begin(){

  displaySpeed();
  drawBoard();
  updateDestinations();
  // processInfo("-40.00a0.00w93h40.421669x-86.91911y[40.421779, -86.91931],[40.421806, -86.919074],[40.421824, -86.918487],[40.421653, -86.918739],[40.421674, -86.919232]c-65.05924190668195t")
  var requestInt = setInterval(sendInfo, 200);

}


