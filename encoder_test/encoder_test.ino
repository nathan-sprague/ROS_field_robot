float wheelDiameter = 9;
int numHoles = 20;

int touches = 0;
int lastRead = 0;
unsigned long lastHitTime = 0;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(5, INPUT);
}


void loop() {
  int  x = digitalRead(5);
  // put your main code here, to run repeatedly:
//  Serial.println(x);
  if (lastRead != x && x == 0){
    Serial.println("hit hole: " + String(touches++));
  //  delay(150);
    int timeBetweenHoles = millis()-lastHitTime;
    float rpm = 60.0/numHoles/(timeBetweenHoles/1000.0);
    float wheelSpeed = (wheelDiameter/12.0)/numHoles/(timeBetweenHoles/1000.0);
    
//    Serial.println("vehicle speed: " + String(wheelSpeed));
    
    Serial.println("rpm: " + String(rpm));
    lastHitTime = millis();
  }
  lastRead = x;

  delay(10);
}
