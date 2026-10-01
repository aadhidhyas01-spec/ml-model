/*
 * ==============================================================================
 * ESP32 DevKit - 100% Wireless Smart Waste Robot Receiver & Arm Controller
 * ==============================================================================
 * - Base Movement: 4 DC Motors (2x L298N Drivers via GPIO 25, 26, 32, 33, 4, 5)
 * - 5-DOF Robotic Arm: PCA9685 16-Channel 12-Bit I2C PWM Driver (GPIO 21 SDA, 22 SCL)
 * - Wireless Protocol: Wi-Fi UDP (Port 8888)
 * ==============================================================================
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ------------------------------------------------------------------------------
// 1. WI-FI CONFIGURATION
// ------------------------------------------------------------------------------
const char* ssid     = "HP";
const char* password = "12345678";
const unsigned int localUdpPort = 8888;

WiFiUDP udp;
char packetBuffer[255];

// ------------------------------------------------------------------------------
// 2. PCA9685 SERVO DRIVER CONFIGURATION (Address 0x40)
// ------------------------------------------------------------------------------
Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

#define SERVO_FREQ 50  // 50 Hz analog servos
#define SERVOMIN  102  // 0.5 ms (~0 deg)
#define SERVOMAX  512  // 2.5 ms (~180 deg)

// Servo Channels on PCA9685
#define CH_BASE     0  // MG996R (Base rotation)
#define CH_SHOULDER 1  // MG996R (Shoulder reach)
#define CH_ELBOW    2  // MG996R (Elbow pitch)
#define CH_WRIST    3  // MG90S  (Wrist tilt)
#define CH_GRIPPER  4  // MG90S  (Claw clamp)

// Helper function to map degrees to PCA9685 pulse
void setServoAngle(uint8_t channel, int angle) {
  angle = constrain(angle, 0, 180);
  int pulse = map(angle, 0, 180, SERVOMIN, SERVOMAX);
  pca.setPWM(channel, 0, pulse);
}

// ------------------------------------------------------------------------------
// 3. MOTOR DRIVER PIN ASSIGNMENTS (2x L298N for 4WD)
// ------------------------------------------------------------------------------
#define MOTOR_L_IN1  25
#define MOTOR_L_IN2  26
#define MOTOR_L_PWM  4

#define MOTOR_R_IN3  32
#define MOTOR_R_IN4  33
#define MOTOR_R_PWM  5

int motorSpeed = 190; // Default drive speed (0 - 255)

// Safety Watchdog
unsigned long lastPacketTime = 0;
const unsigned long SAFETY_TIMEOUT_MS = 800; // Stop motors if signal lost for 800ms

// ==============================================================================
// MOTOR CONTROL FUNCTIONS
// ==============================================================================
void moveForward() {
  digitalWrite(MOTOR_L_IN1, HIGH);
  digitalWrite(MOTOR_L_IN2, LOW);
  analogWrite(MOTOR_L_PWM, motorSpeed);

  digitalWrite(MOTOR_R_IN3, HIGH);
  digitalWrite(MOTOR_R_IN4, LOW);
  analogWrite(MOTOR_R_PWM, motorSpeed);
}

void moveBackward() {
  digitalWrite(MOTOR_L_IN1, LOW);
  digitalWrite(MOTOR_L_IN2, HIGH);
  analogWrite(MOTOR_L_PWM, motorSpeed);

  digitalWrite(MOTOR_R_IN3, LOW);
  digitalWrite(MOTOR_R_IN4, HIGH);
  analogWrite(MOTOR_R_PWM, motorSpeed);
}

void steerLeft() {
  digitalWrite(MOTOR_L_IN1, LOW);
  digitalWrite(MOTOR_L_IN2, HIGH);
  analogWrite(MOTOR_L_PWM, (int)(motorSpeed * 0.75));

  digitalWrite(MOTOR_R_IN3, HIGH);
  digitalWrite(MOTOR_R_IN4, LOW);
  analogWrite(MOTOR_R_PWM, (int)(motorSpeed * 0.75));
}

void steerRight() {
  digitalWrite(MOTOR_L_IN1, HIGH);
  digitalWrite(MOTOR_L_IN2, LOW);
  analogWrite(MOTOR_L_PWM, (int)(motorSpeed * 0.75));

  digitalWrite(MOTOR_R_IN3, LOW);
  digitalWrite(MOTOR_R_IN4, HIGH);
  analogWrite(MOTOR_R_PWM, (int)(motorSpeed * 0.75));
}

void stopMotors() {
  digitalWrite(MOTOR_L_IN1, LOW);
  digitalWrite(MOTOR_L_IN2, LOW);
  analogWrite(MOTOR_L_PWM, 0);

  digitalWrite(MOTOR_R_IN3, LOW);
  digitalWrite(MOTOR_R_IN4, LOW);
  analogWrite(MOTOR_R_PWM, 0);
}

// ==============================================================================
// 5-DOF ROBOTIC ARM SEQUENCES (PCA9685)
// ==============================================================================
void armHome() {
  setServoAngle(CH_BASE, 90);
  setServoAngle(CH_SHOULDER, 100);
  setServoAngle(CH_ELBOW, 90);
  setServoAngle(CH_WRIST, 90);
  setServoAngle(CH_GRIPPER, 30); // Open claw
  delay(400);
}

void executePickSequence(String wasteClass) {
  stopMotors();
  Serial.print("🤖 Executing Arm Pick for: ");
  Serial.println(wasteClass);

  // 1. Open Gripper & Reach forward down
  setServoAngle(CH_GRIPPER, 30); // Open claw
  delay(300);
  setServoAngle(CH_SHOULDER, 50);
  setServoAngle(CH_ELBOW, 60);
  setServoAngle(CH_WRIST, 70);
  delay(800);

  // 2. Clamp Gripper to pick up item
  setServoAngle(CH_GRIPPER, 95); // Grip item
  delay(600);

  // 3. Lift Arm Up
  setServoAngle(CH_SHOULDER, 110);
  setServoAngle(CH_ELBOW, 100);
  delay(800);

  // 4. Rotate Base towards sorting bin
  if (wasteClass.indexOf("GLASS") >= 0) {
    setServoAngle(CH_BASE, 0);   // Bin 3: Glass / General (0 deg)
  } else if (wasteClass.indexOf("PLASTIC") >= 0 || wasteClass.indexOf("BOTTLE") >= 0 || wasteClass.indexOf("CUP") >= 0 || wasteClass.indexOf("CAP") >= 0) {
    setServoAngle(CH_BASE, 45);  // Bin 1: Plastics, Cups & Caps (45 deg)
  } else if (wasteClass.indexOf("CAN") >= 0 || wasteClass.indexOf("BATTERY") >= 0) {
    setServoAngle(CH_BASE, 135); // Bin 2: Metals (135 deg)
  } else {
    setServoAngle(CH_BASE, 0);   // Bin 3: General Recyclable (0 deg)
  }
  delay(800);

  // 5. Open Gripper to release item in bin
  setServoAngle(CH_GRIPPER, 30); // Drop
  delay(500);

  // 6. Return to Home position
  armHome();
}

// ==============================================================================
// SETUP
// ==============================================================================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n🚀 Initializing Smart Waste Robot DevKit Controller...");

  // Setup Motor Pins
  pinMode(MOTOR_L_IN1, OUTPUT);
  pinMode(MOTOR_L_IN2, OUTPUT);
  pinMode(MOTOR_L_PWM, OUTPUT);
  pinMode(MOTOR_R_IN3, OUTPUT);
  pinMode(MOTOR_R_IN4, OUTPUT);
  pinMode(MOTOR_R_PWM, OUTPUT);
  stopMotors();

  // Setup I2C & PCA9685
  Wire.begin(21, 22); // SDA = GPIO 21, SCL = GPIO 22
  pca.begin();
  pca.setPWMFreq(SERVO_FREQ);
  delay(50);
  armHome();
  Serial.println("✅ PCA9685 5-DOF Arm Initialized to Home Position.");

  // Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }

  Serial.println("\n✅ Wi-Fi Connected!");
  Serial.print("📡 ESP32 DevKit IP Address: ");
  Serial.println(WiFi.localIP());

  // Start UDP Listener
  udp.begin(localUdpPort);
  Serial.printf("🎯 Listening on UDP port %d for AI commands...\n", localUdpPort);
}

// ==============================================================================
// MAIN LOOP
// ==============================================================================
void loop() {
  int packetSize = udp.parsePacket();
  
  if (packetSize > 0) {
    int len = udp.read(packetBuffer, 254);
    if (len > 0) {
      packetBuffer[len] = 0; // Null-terminate
    }

    String cmd = String(packetBuffer);
    cmd.trim();
    lastPacketTime = millis();

    // Process Movement Commands
    if (cmd == "CMD:FORWARD") {
      moveForward();
    } else if (cmd == "CMD:BACKWARD") {
      moveBackward();
    } else if (cmd == "CMD:LEFT") {
      steerLeft();
    } else if (cmd == "CMD:RIGHT") {
      steerRight();
    } else if (cmd == "CMD:STOP") {
      stopMotors();
    }
    // Process Robotic Arm Commands
    else if (cmd.startsWith("ARM:PICK:")) {
      String itemType = cmd.substring(9);
      executePickSequence(itemType);
    }
  }

  // Safety Watchdog: If no wireless command received recently, stop motors
  if (millis() - lastPacketTime > SAFETY_TIMEOUT_MS) {
    stopMotors();
  }
}
