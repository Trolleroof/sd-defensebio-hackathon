#include <WiFi.h>

// Match the router credentials and channel in imu_mesh.ino.
constexpr char kSsid[] = "DBP_NO_ROUTER";
constexpr char kPassword[] = "unusedPass";
constexpr int kChannel = 6;

void setup() {
  Serial.begin(115200);
  if (!WiFi.softAP(kSsid, kPassword, kChannel)) {
    Serial.println("{\"type\":\"router_error\"}");
    return;
  }
  Serial.println("{\"type\":\"router_ready\"}");
}

void loop() {
  delay(1000);
}
