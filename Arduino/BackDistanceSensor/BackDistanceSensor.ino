#include <WiFi.h>
#include <WebSocketsServer.h>
#include <M5StickCPlus.h>
#include <Unit_Sonic.h>

//——— Wi‑Fi credentials ———————————————————
/*static const char* WIFI_SSID     = "";
static const char* WIFI_PASSWORD = "";*/

static const char* WIFI_SSID     = "ut-public";
static const char* WIFI_PASSWORD = "";

//——— WebSocket server info —————————————————
static const uint8_t WS_PORT = 84;
WebSocketsServer webSocket(WS_PORT);

//——— Distance sensor (Unit_Sonic) ————————————————
SONIC_I2C sonar;
float distanceCm = -1;

//——— Function prototypes ———————————————————
void connectToWiFi();
void handleWebSocketEvent(uint8_t client, WStype_t type, uint8_t* payload, size_t length);

void setup() {
  M5.begin();

  M5.Lcd.setRotation(1);  
  M5.Lcd.setTextSize(2);    
  M5.Lcd.fillScreen(BLACK);

  Serial.begin(115200);
  sonar.begin();

  // Connect to Wi‑Fi
  connectToWiFi();

  // Setup WebSocket server
  webSocket.begin();
  webSocket.onEvent(handleWebSocketEvent);

  Serial.printf("WebSocket server started on port %d\n", WS_PORT);
}

void loop() {
  webSocket.loop();

  static unsigned long lastSend = 0;
  const unsigned long interval = 100; // send every 100 ms

  if (millis() - lastSend >= interval) {
    // Read distance in millimeters
    float mm = sonar.getDistance();

    // Convert to cm if within sensor range, else flag as -1
    distanceCm = mm / 10.0;
    

    // Prepare payload (e.g., JSON or plain text)
    String payload = String(distanceCm, 2);  // two decimal places

    // Broadcast distance to all connected clients
    webSocket.broadcastTXT(payload);

    Serial.printf("→ Broadcast distance: %.2f cm\n", distanceCm);
    lastSend = millis();
  }
}

// Connects to WiFi and prints the IP
void connectToWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(500);
  }
  Serial.println("\nWiFi connected");

  IPAddress ip = WiFi.localIP();
  Serial.printf("IP address: %s\n", ip.toString().c_str());

  // Display on screen
  M5.Lcd.setCursor(0, 10);
  M5.Lcd.printf("IP: %d.%d.%d.%d\n",
                ip[0], ip[1], ip[2], ip[3]);
  M5.Lcd.printf("Port: %d", WS_PORT);
}

// WebSocket event handler (not handling incoming messages)
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


