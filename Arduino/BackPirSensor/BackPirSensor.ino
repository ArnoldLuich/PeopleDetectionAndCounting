#include <WiFi.h>
#include <WebSocketsServer.h>

//——— Wi‑Fi credentials ———————————————————
/*static const char* WIFI_SSID     = "";
static const char* WIFI_PASSWORD = "";*/

static const char* WIFI_SSID     = "ut-public";
static const char* WIFI_PASSWORD = "";

//——— WebSocket server info —————————————————
static const uint8_t WS_PORT = 82;
WebSocketsServer webSocket(WS_PORT);

//——— PIR sensor pin —————————————————————
const int PIR_PIN = 32;

//——— Function prototypes ———————————————————
void connectToWiFi();
void handleWebSocketEvent(uint8_t client, WStype_t type, uint8_t* payload, size_t length);

void setup() {
  Serial.begin(115200);
  pinMode(PIR_PIN, INPUT);

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
  if (millis() - lastSend >= 10) {
    uint8_t pirValue = digitalRead(PIR_PIN) == HIGH ? 1 : 0;

    // Build a String in a local variable…
    String payload = String(pirValue);
    // …and then broadcast it
    webSocket.broadcastTXT(payload);

    Serial.printf("→ Broadcast PIR value: %d\n", pirValue);
    Serial.printf("IP address: %s\n", WiFi.localIP().toString().c_str());
    
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
  Serial.printf("IP address: %s\n", WiFi.localIP().toString().c_str());
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
