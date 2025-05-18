#include <M5StickCPlus.h>
#include <Wire.h>
#include <Adafruit_MLX90640.h>
#include <WiFi.h>
#include <WebSocketsServer.h>
#include "M5_STHS34PF80.h"

#include <esp_now.h>

// ——— Configuration ————————————————————————————
/*static const char*   WIFI_SSID     = "";
static const char*   WIFI_PASSWORD = "";*/

static const char*   WIFI_SSID     = "ut-public";
static const char*   WIFI_PASSWORD = "";

static const uint8_t WS_PORT       = 81;
static const uint8_t I2C_MUX_ADDR  = 0x70;  // TCA9548A address
const int MIC_PIN = 36;

static const uint8_t MLX1_BUS      = 4;   // first camera
static const uint8_t MLX2_BUS      = 0;   // second camera

static const uint8_t MLX_I2C_ADDR  = 0x33;
static const uint8_t MLX_COLS      = 32;
static const uint8_t MLX_ROWS      = 24;
static const float   FRAME_DELAY   = 100;   // ms between frames

int16_t motionVal        = 0;
int16_t presenceVal      = 0;
float   ambientTemp      = 0.0;
float   distanceCm       = 0;
int micValue    = 0;
float gyroMagnitude = 0.0;
float accelMagnitude = 0.0;

// ——— Globals ————————————————————————————————————————
WebSocketsServer  webSocket(WS_PORT);

// two separate camera objects and two buffers:
Adafruit_MLX90640 mlx1;
Adafruit_MLX90640 mlx2;
float pixels1[MLX_COLS * MLX_ROWS];
float pixels2[MLX_COLS * MLX_ROWS];

M5_STHS34PF80           tmos;

TaskHandle_t micTaskHandle = nullptr;



// ——— Function Prototypes ————————————————————————
void connectToWiFi();
void initDisplay();
void initThermalCameras();
void selectI2CBus(uint8_t bus);
void initOtherSensors();
void handleWebSocketEvent(uint8_t client, WStype_t type, uint8_t* payload, size_t length);
void broadcastThermalFrames();
void readPIR();
void readIMU();

// ——— Setup ——————————————————————————————————————
void setup() {
  M5.begin();
  Serial.begin(115200);

  Wire.begin(/* SDA */ 32, /* SCL */ 33, 400000);

  connectToWiFi();
  initDisplay();
  
  initThermalCameras();
  initOtherSensors();

  analogSetPinAttenuation(MIC_PIN, ADC_11db);  // extend range to full 3.3 V
  xTaskCreatePinnedToCore(
    micTask,             
    "Mic Reader",        
    2048,                
    nullptr,             
    2,                   // priority (higher than your main loop)
    &micTaskHandle,      // handle
    0                    // run on core 0 or 1 as you like
  );

  webSocket.begin();
  webSocket.onEvent(handleWebSocketEvent);
}

// ——— Main Loop ————————————————————————————————————
void loop() {
  webSocket.loop();
  broadcastThermalFrames();
  readPIR();
  readIMU();
  delay(FRAME_DELAY);
}

// ——— Implementations ———————————————————————————

// Connects to WiFi and prints the IP on Serial
void connectToWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(500);
  }
  Serial.println("\nWiFi connected");
  Serial.printf("IP address: %s\n", WiFi.localIP().toString().c_str());
}

// Sets up the LCD to show IP address
void initDisplay() {
  M5.Lcd.setRotation(3);
  M5.Lcd.setTextSize(2);
  M5.Lcd.setCursor(0, 20);
  M5.Lcd.println("IP:");
  M5.Lcd.println(WiFi.localIP());
}

// Initializes the MLX90640 sensor over the I2C multiplexer
void initThermalCameras() {
  // Camera 1
  selectI2CBus(MLX1_BUS);
  if (!mlx1.begin(MLX_I2C_ADDR, &Wire)) {
    Serial.println("Error: MLX90640 #1 not detected!");
    while (1) delay(10);
  }
  mlx1.setMode(MLX90640_CHESS);
  mlx1.setResolution(MLX90640_ADC_18BIT);
  mlx1.setRefreshRate(MLX90640_8_HZ);

  // Camera 2
  selectI2CBus(MLX2_BUS);
  if (!mlx2.begin(MLX_I2C_ADDR, &Wire)) {
    Serial.println("Error: MLX90640 #2 not detected!");
    while (1) delay(10);
  }
  mlx2.setMode(MLX90640_CHESS);
  mlx2.setResolution(MLX90640_ADC_18BIT);
  mlx2.setRefreshRate(MLX90640_8_HZ);
}

void initOtherSensors() {

  M5.IMU.Init();

  // PIR/Temp on bus 3
  selectI2CBus(3);
  tmos.begin();
  tmos.setGainMode(STHS34PF80_GAIN_DEFAULT_MODE);
  tmos.setMotionThreshold(0xFF);
  tmos.setPresenceThreshold(0xC8);
  tmos.setMotionHysteresis(0x32);
  tmos.setPresenceHysteresis(0x32);

}

// Selects which channel on the TCA9548A I2C multiplexer to use
void selectI2CBus(uint8_t bus) {
  Wire.beginTransmission(I2C_MUX_ADDR);
  Wire.write(1 << bus);
  Wire.endTransmission();
}

// WebSocket event handler (we only broadcast, so this is empty)
void handleWebSocketEvent(uint8_t client, WStype_t type, uint8_t* payload, size_t length) {
  // No incoming messages handled
}

void readPIR() {
  selectI2CBus(3);
  tmos.getPresenceValue(&presenceVal);
  tmos.getMotionValue(&motionVal);
  tmos.getTemperatureData(&ambientTemp);
}


void readIMU(){
  float gx, gy, gz;
  float ax, ay, az;

  // Get gyroscope and accelerometer data
  M5.IMU.getGyroData(&gx, &gy, &gz);
  M5.IMU.getAccelData(&ax, &ay, &az);
  
  // Convert gyroscope data to deg/s
  float gyroX = gx / 131.0;
  float gyroY = gy / 131.0;
  float gyroZ = gz / 131.0;

  // Compute gyroscope magnitude
  gyroMagnitude = sqrt(gyroX * gyroX + gyroY * gyroY + gyroZ * gyroZ);

  // Convert accelerometer data to m/s² (optional: only if needed)
  float accelX = ax * 9.81;
  float accelY = ay * 9.81;
  float accelZ = az * 9.81;

  // Compute accelerometer magnitude
  accelMagnitude = sqrt(accelX * accelX + accelY * accelY + accelZ * accelZ);
}





// Reads one frame from the MLX90640, converts it to CSV, and broadcasts
void broadcastThermalFrames() {
  // Read first camera
  selectI2CBus(MLX1_BUS);
  if (mlx1.getFrame(pixels1) != 0) {
    Serial.println("Error: failed to read MLX90640 #1");
    return;
  }
  // Read second camera
  selectI2CBus(MLX2_BUS);
  if (mlx2.getFrame(pixels2) != 0) {
    Serial.println("Error: failed to read MLX90640 #2");
    return;
  }

  // build CSVs
  String csv1, csv2;
  csv1.reserve(MLX_ROWS * MLX_COLS * 6);
  csv2.reserve(MLX_ROWS * MLX_COLS * 6);
  for (int i = 0; i < MLX_ROWS * MLX_COLS; i++) {
    if (i > 0) {
      csv1 += ',';
      csv2 += ',';
    }
    csv1 += String(pixels1[i], 2);
    csv2 += String(pixels2[i], 2);
  }

  // bundle into JSON with your other sensor values
  String out = "{\"thermal1\":\"" + csv1 + "\"" +
               ",\"thermal2\":\"" + csv2 + "\"" +
               ",\"motion\":"   + String(motionVal) +
               ",\"presence\":" + String(presenceVal) +
               ",\"ambient\":"  + String(ambientTemp) +
               ",\"micValue\":" + String(micValue) +
               ",\"gyroMagnitude\":" + String(gyroMagnitude) +
               ",\"accelMagnitude\":" + String(accelMagnitude) +
               "}";
  webSocket.broadcastTXT(out);
}

void micTask(void *pv) {
  const TickType_t sampleInterval = pdMS_TO_TICKS(10);

  for (;;) {
    micValue = analogRead(MIC_PIN);
    vTaskDelay(sampleInterval);
  }
}

