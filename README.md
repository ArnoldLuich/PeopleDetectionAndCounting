# Quick Setup Guide

This project connects multiple **M5StickC-PLUS** devices to your Wi‑Fi and sends data to your front-end, where it is possible to count people. Follow the two short sections below.

---

## 1  Flash the Arduino (M5Stick) Sketch

You will need to update three scripts with your Wi-Fi details:

* `MainController`
* `ThermalCamera3`
* `ThermalCamera4`

In each of these scripts, locate the Wi-Fi section and replace the placeholders with **your network name and password**:

```cpp
#define WIFI_SSID     "YourNetworkName"
#define WIFI_PASSWORD "YourPassword"
```

| Step | What to do                                                                                                                                                  |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | **Install Arduino IDE ≥ 2.0** and add the ESP32 board package (Tools ▸ Board Manager ▸ search “esp32 by Espressif”).                                        |
| 2    | Clone/download this repo and open the relevant `.ino` files in the IDE.                                                                                     |
| 3    | Update the Wi-Fi details as shown above in **`MainController`**, **`ThermalCamera3`**, and **`ThermalCamera4`**.                                            |
| 4    | Choose **Board:** `M5Stick-C` (or the model you own) and the correct **COM/USB port**.                                                                      |
| 5    | Click **Upload** (▶). When flashing finishes, open **Serial Monitor** at 115 200 baud to note the **IP address** the device prints (e.g., `192.168.1.123`). |

> **Tip:** The MainController will display the IP address on its screen. For ThermalCamera3 and ThermalCamera4, you need to check the Serial Monitor and look for a line where it uses Serial.printf to print the IP.
> 
> Ports:
> * `MainController: port 81`
> * `ThermalCamera3: port 85`
> * `ThermalCamera4: port 86`

---

## 2  Run the Python Display Script

To visualize the data, use the provided Python script:

1. In the `display.py` file, locate the `URLS` dictionary and replace `IP` with the actual IP address of the devices:

```python
URLS = {
    "cam3": "ws://192.168.1.123:85",
    "cam4": "ws://192.168.1.123:86",
    "thermal": "ws://192.168.1.123:81"
}
```

2. Make sure your PC is connected to the same Wi-Fi network as the M5Stick devices.
3. Run the script

---

## Troubleshooting

* **Can’t reach the IP?** Ensure your PC and the M5Stick devices are on the same Wi-Fi network. Firewalls or corporate guest networks can block local traffic.
* **Wrong network details:** Re-flash with the correct SSID/PW.
* **Still stuck?** Open the Serial Monitor to check for connection errors and verify the printed IP.

