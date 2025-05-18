#include <M5StickC.h>
#include <Wire.h>
#include <Adafruit_MLX90640.h>
#include <WiFi.h>
#include <WebSocketsServer.h>

//———————— WiFi Credentials ——————————
/*static const char* WIFI_SSID     = "";
static const char* WIFI_PASSWORD = "";*/

static const char* WIFI_SSID     = "ut-public";
static const char* WIFI_PASSWORD = "";

//———————— WebSocket Setup ——————————
static const uint8_t WS_PORT = 86;
WebSocketsServer webSocket(WS_PORT);

//———————— MLX90640 Thermal Sensor ——————————
Adafruit_MLX90640 mlx;
float frame[32 * 24];  // MLX90640 resolution

//———————— Function Prototypes ——————————
void connectToWiFi();
void handleWebSocketEvent(uint8_t client, WStype_t type, uint8_t* payload, size_t length);

void setup() {
  M5.begin();
  Wire.begin(0, 26);  // SDA = GPIO0, SCL = GPIO26 for Thermal Hat
  Serial.begin(115200);
  esp_log_level_set("wifi", ESP_LOG_NONE);
  delay(1000);

  M5.Lcd.setRotation(1);  
  M5.Lcd.setTextSize(2);    
  M5.Lcd.fillScreen(BLACK);

  Serial.println("Initializing MLX90640...");
  if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
    Serial.println("Failed to find MLX90640 sensor!");
    while (1) delay(10);
  }

  Serial.println("MLX90640 found!");
  mlx.setMode(MLX90640_CHESS);
  mlx.setResolution(MLX90640_ADC_18BIT);
  mlx.setRefreshRate(MLX90640_4_HZ);  // ~4 FPS

  connectToWiFi(); // connect to WiFi

  webSocket.begin();           // Start WebSocket server
  webSocket.onEvent(handleWebSocketEvent);

  Serial.printf("WebSocket server started on port %d\n", WS_PORT);
}

void loop() {
  webSocket.loop();  // Handle WebSocket events

  static unsigned long lastSend = 0;
  const unsigned long interval = 250;  // Send every 250 ms (~4 FPS)

  if (millis() - lastSend >= interval) {
    if (mlx.getFrame(frame) != 0) {
      Serial.println("Failed to get frame");
      return;
    }

    // Convert thermal frame to CSV string
    String payload = "";
    for (int i = 0; i < 32 * 24; i++) {
      payload += String(frame[i], 2);
      if (i < (32 * 24 - 1)) payload += ",";
    }

    // Send data over WebSocket
    webSocket.broadcastTXT(payload);
    IPAddress ip = WiFi.localIP();
    Serial.printf("IP address: %s\n", ip.toString().c_str());

    lastSend = millis();
  }
}

void connectToWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println("\nWiFi connected");
  delay(500);  // Allow time for M5.Lcd to initialize
}

void handleWebSocketEvent(uint8_t client, WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      Serial.printf("Client %u connected\n", client);
      break;
    case WStype_DISCONNECTED:
      Serial.printf("Client %u disconnected\n", client);
      break;
    default:
      break;
  }
}

