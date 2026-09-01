/*
 * pan_tilt_controller.ino — IntelliTrack Arduino/ESP32 firmware
 *
 * Hardware connections:
 *   Pan servo  → Pin 9
 *   Tilt servo → Pin 10
 *
 * Serial protocol (newline-terminated commands):
 *   Receive: "PAN:<int> TILT:<int>\n"
 *   Reply:   "ACK\n"  on success
 *
 * On boot: centres both servos at 90° and prints "READY\n".
 */

#include <Servo.h>

// ---- Configuration -------------------------------------------------------
const int PAN_PIN  = 9;
const int TILT_PIN = 10;
const int BAUD_RATE = 115200;
const int SERVO_MIN = 0;
const int SERVO_MAX = 180;
const int SERVO_CENTER = 90;

// ---- Servo objects -------------------------------------------------------
Servo panServo;
Servo tiltServo;

// ---- Serial input buffer -------------------------------------------------
String inputBuffer = "";

// ---- Setup ---------------------------------------------------------------
void setup() {
  Serial.begin(BAUD_RATE);
  while (!Serial) { ; }  // Wait for serial port (needed on some boards)

  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);

  panServo.write(SERVO_CENTER);
  tiltServo.write(SERVO_CENTER);

  Serial.println("READY");
}

// ---- Main loop -----------------------------------------------------------
void loop() {
  // Read incoming bytes until a newline
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }
}

// ---- Command parser ------------------------------------------------------
void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  // Expected format: "PAN:<int> TILT:<int>"
  int panIndex  = cmd.indexOf("PAN:");
  int tiltIndex = cmd.indexOf("TILT:");

  if (panIndex == -1 || tiltIndex == -1) {
    Serial.println("ERR:INVALID_CMD");
    return;
  }

  // Parse pan angle
  int panStart  = panIndex + 4;
  int panEnd    = tiltIndex - 1;  // space before TILT:
  int panAngle  = cmd.substring(panStart, panEnd).toInt();

  // Parse tilt angle
  int tiltStart = tiltIndex + 5;
  int tiltAngle = cmd.substring(tiltStart).toInt();

  // Clamp to valid range
  panAngle  = constrain(panAngle,  SERVO_MIN, SERVO_MAX);
  tiltAngle = constrain(tiltAngle, SERVO_MIN, SERVO_MAX);

  panServo.write(panAngle);
  tiltServo.write(tiltAngle);

  Serial.println("ACK");
}
