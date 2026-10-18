#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_LSM6DSOX.h>

#include <esp_event.h>
#include <esp_mac.h>
#include <esp_mesh.h>
#include <esp_netif.h>
#include <esp_wifi.h>
#include <nvs_flash.h>
#include <freertos/semphr.h>
#include <string.h>

// Build once as the command root, then as a wearable for both sensor boards.
#define ROLE_SENSOR 0
#define ROLE_ROOT 1
#ifndef NODE_ROLE
#define NODE_ROLE ROLE_SENSOR
#endif
#ifndef POLAR_ID
#define POLAR_ID 0
#endif
#ifndef HR_SOURCE_SPARKFUN
#define HR_SOURCE_SPARKFUN 0
#endif
#if NODE_ROLE == ROLE_SENSOR && HR_SOURCE_SPARKFUN
#include <SparkFun_Bio_Sensor_Hub_Library.h>
#elif NODE_ROLE == ROLE_SENSOR
#include <BLEDevice.h>
#endif

namespace {
constexpr uint32_t kPacketMagic = 0x494D5531;  // "IMU1"
constexpr uint8_t kSchemaVersion = 2;
constexpr uint8_t kMaxRelayHops = 1;
constexpr uint8_t kMeshChannel = 6;
constexpr uint32_t kSamplePeriodMs = 200;
constexpr uint32_t kSensorStatusPeriodMs = 5000;
constexpr int kMeshRxTimeoutMs = 1000;

// These values must be identical on all three boards.
constexpr uint8_t kMeshId[6] = {0x44, 0x42, 0x50, 0x48, 0x59, 0x31};
constexpr char kMeshPassword[] = "demoMeshPass";
constexpr char kRouterSsid[] = "DBP_NO_ROUTER";
constexpr char kRouterPassword[] = "unusedPass";

struct __attribute__((packed)) MeshPacket {
  uint32_t magic;
  uint8_t schema_version;
  uint8_t hop_count;
  uint8_t sample_valid;
  uint8_t wake;
  uint32_t sequence;
  uint32_t uptime_ms;
  uint8_t node_mac[6];
  float accel_m_s2[3];
  float gyro_rad_s[3];
  float yaw_delta_rad;
  float imu_temp_c;
  uint32_t steps;
  int16_t hr_bpm;
  uint8_t hr_valid;
};

static_assert(sizeof(MeshPacket) < MESH_MTU, "Mesh packet exceeds ESP-WIFI-MESH MTU");

Adafruit_LSM6DSOX imu;
bool imuReady = false;
float yawSincePacket = 0;
uint32_t lastImuReadMs = 0;
sensors_event_t latestAccel{}, latestGyro{}, latestTemp{};
bool latestImuValid = false;
portMUX_TYPE parentMux = portMUX_INITIALIZER_UNLOCKED;
SemaphoreHandle_t meshSendMutex = nullptr;
bool parentReady = false;
mesh_addr_t parentAddress{};
uint8_t nodeMac[6]{};
uint32_t sequenceNumber = 0;
uint32_t lastSensorStatusMs = 0;
uint32_t lastSensorDiagMs = 0;
#if NODE_ROLE == ROLE_SENSOR
int16_t latestHr = -1;
uint32_t lastHrMs = 0;
#if HR_SOURCE_SPARKFUN
SparkFun_Bio_Sensor_Hub bioHub(D4, D5);  // RESET, MFIO; both are 3.3 V GPIO.
bool sparkfunReady = false;
uint32_t lastBioReadMs = 0;
#else
portMUX_TYPE hrMux = portMUX_INITIALIZER_UNLOCKED;
volatile bool polarConnected = false;

void onHeartRate(BLERemoteCharacteristic *, uint8_t *data, size_t length, bool) {
  if (length < 2 || ((data[0] & 1) && length < 3)) return;
  int16_t bpm = (data[0] & 1) ? (data[1] | (data[2] << 8)) : data[1];
  portENTER_CRITICAL(&hrMux);
  latestHr = bpm;
  lastHrMs = millis();
  portEXIT_CRITICAL(&hrMux);
}

class PolarCallbacks : public BLEClientCallbacks {
  void onConnect(BLEClient *) override { polarConnected = true; }
  void onDisconnect(BLEClient *) override {
    polarConnected = false;
    portENTER_CRITICAL(&hrMux);
    latestHr = -1;
    lastHrMs = 0;
    portEXIT_CRITICAL(&hrMux);
    Serial.println("{\"type\":\"polar_disconnected\"}");
  }
};

void polarTask(void *) {
  BLEDevice::init("");
  BLEScan *scan = BLEDevice::getScan();
  scan->setActiveScan(true);
  scan->setInterval(160);
  scan->setWindow(80);
  BLEClient *client = BLEDevice::createClient();
  static PolarCallbacks callbacks;
  client->setClientCallbacks(&callbacks);
  const BLEUUID hrService((uint16_t)0x180D);
  const BLEUUID hrMeasurement((uint16_t)0x2A37);
  while (true) {
    BLEScanResults results = scan->start(4, false);
    for (int i = 0; i < results.getCount(); ++i) {
      BLEAdvertisedDevice device = results.getDevice(i);
      if (!device.haveName()) continue;
      const std::string name = device.getName();
      const size_t idStart = name.find_last_of(' ');
      if (name.rfind("Polar", 0) != 0 || idStart == std::string::npos ||
          strtoul(name.c_str() + idStart + 1, nullptr, 16) != POLAR_ID) continue;
      if (client->connect(&device)) {
        BLERemoteService *service = client->getService(hrService);
        BLERemoteCharacteristic *measurement = service ? service->getCharacteristic(hrMeasurement) : nullptr;
        if (measurement && measurement->canNotify()) {
          const uint32_t subscribedAt = millis();
          measurement->registerForNotify(onHeartRate);
          Serial.printf("{\"type\":\"polar_connected\",\"id\":\"%08lX\"}\n", static_cast<unsigned long>(POLAR_ID));
          while (client->isConnected()) {
            vTaskDelay(pdMS_TO_TICKS(500));
            portENTER_CRITICAL(&hrMux);
            const uint32_t last = lastHrMs;
            portEXIT_CRITICAL(&hrMux);
            if (static_cast<uint32_t>(millis() - (last ? last : subscribedAt)) > 10000) {
              Serial.println("{\"type\":\"polar_notifications_stale\"}");
              break;
            }
          }
        }
        client->disconnect();
      }
      break;
    }
    scan->clearResults();
    vTaskDelay(pdMS_TO_TICKS(2000));
  }
}
#endif
#endif

void checkInit(const char *stage, esp_err_t err) {
  if (err == ESP_OK) return;

  Serial.printf("{\"type\":\"init_error\",\"stage\":\"%s\",\"err\":%d}\n",
                stage, static_cast<int>(err));
  Serial.flush();
  while (true) delay(1000);
}

void handleMeshEvent(void *, esp_event_base_t, int32_t eventId, void *eventData) {
  if (eventId == MESH_EVENT_PARENT_CONNECTED) {
    auto *connected = static_cast<mesh_event_connected_t *>(eventData);
    portENTER_CRITICAL(&parentMux);
    memcpy(parentAddress.addr, connected->connected.bssid, sizeof(parentAddress.addr));
    parentReady = true;
    portEXIT_CRITICAL(&parentMux);
    Serial.println("{\"type\":\"mesh_parent_connected\"}");
  } else if (eventId == MESH_EVENT_PARENT_DISCONNECTED) {
    auto *disconnected = static_cast<mesh_event_disconnected_t *>(eventData);
    portENTER_CRITICAL(&parentMux);
    parentReady = false;
    portEXIT_CRITICAL(&parentMux);
    // Reason 200 = beacon timeout (radio contention/range), 2/4 = auth/assoc expire.
    Serial.printf("{\"type\":\"mesh_parent_disconnected\",\"reason\":%u}\n", disconnected->reason);
  } else if (eventId == MESH_EVENT_NO_PARENT_FOUND) {
    Serial.println("{\"type\":\"mesh_no_parent_found\"}");
  } else if (eventId == MESH_EVENT_CHILD_CONNECTED) {
    Serial.println("{\"type\":\"mesh_child_connected\"}");
  } else if (eventId == MESH_EVENT_CHILD_DISCONNECTED) {
    Serial.println("{\"type\":\"mesh_child_disconnected\"}");
  }
}

void initNvs() {
  esp_err_t err = nvs_flash_init();
  if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    checkInit("nvs_erase", nvs_flash_erase());
    err = nvs_flash_init();
  }
  checkInit("nvs_init", err);
}

void initMesh() {
  esp_err_t netifErr = esp_netif_init();
  if (netifErr != ESP_OK && netifErr != ESP_ERR_INVALID_STATE) {
    checkInit("netif_init", netifErr);
  }
  esp_err_t eventLoopErr = esp_event_loop_create_default();
  if (eventLoopErr != ESP_OK && eventLoopErr != ESP_ERR_INVALID_STATE) {
    checkInit("event_loop_init", eventLoopErr);
  }

  static esp_netif_t *meshStaNetif = nullptr;
  checkInit("mesh_netifs_create",
            esp_netif_create_default_wifi_mesh_netifs(&meshStaNetif, nullptr));

  wifi_init_config_t wifiConfig = WIFI_INIT_CONFIG_DEFAULT();
  checkInit("wifi_init", esp_wifi_init(&wifiConfig));
  checkInit("wifi_storage", esp_wifi_set_storage(WIFI_STORAGE_FLASH));
  checkInit("wifi_start", esp_wifi_start());

  checkInit("mesh_init", esp_mesh_init());
  checkInit("mesh_event_register", esp_event_handler_register(
      MESH_EVENT, ESP_EVENT_ANY_ID, &handleMeshEvent, nullptr));
  checkInit("mesh_topology", esp_mesh_set_topology(MESH_TOPO_TREE));
  checkInit("mesh_max_layer", esp_mesh_set_max_layer(2));
  checkInit("mesh_fix_root", esp_mesh_fix_root(true));
  checkInit("mesh_type", esp_mesh_set_type(NODE_ROLE == ROLE_ROOT ? MESH_ROOT : MESH_IDLE));
  checkInit("mesh_authmode", esp_mesh_set_ap_authmode(WIFI_AUTH_WPA2_PSK));

  mesh_cfg_t meshConfig = MESH_INIT_CONFIG_DEFAULT();
  memcpy(meshConfig.mesh_id.addr, kMeshId, sizeof(kMeshId));
  meshConfig.channel = kMeshChannel;
  // This SDK rejects an empty router SSID even with a fixed root and no router.
  meshConfig.router.ssid_len = sizeof(kRouterSsid) - 1;
  memcpy(meshConfig.router.ssid, kRouterSsid, meshConfig.router.ssid_len);
  memcpy(meshConfig.router.password, kRouterPassword, sizeof(kRouterPassword) - 1);
  // Star: both wearables join the root directly; layer 2 is a leaf, so no parent swapping.
  meshConfig.mesh_ap.max_connection = 2;
  meshConfig.mesh_ap.nonmesh_max_connection = 0;
  memcpy(meshConfig.mesh_ap.password, kMeshPassword, sizeof(kMeshPassword) - 1);
  checkInit("mesh_config", esp_mesh_set_config(&meshConfig));
  checkInit("mesh_start", esp_mesh_start());
}

bool copyParent(mesh_addr_t &out) {
  portENTER_CRITICAL(&parentMux);
  bool ready = parentReady;
  if (ready) memcpy(out.addr, parentAddress.addr, sizeof(out.addr));
  portEXIT_CRITICAL(&parentMux);
  return ready;
}

esp_err_t sendToParent(MeshPacket &packet) {
  mesh_addr_t parent{};
  if (!copyParent(parent)) return ESP_ERR_MESH_DISCONNECTED;

  mesh_data_t data{};
  data.data = reinterpret_cast<uint8_t *>(&packet);
  data.size = sizeof(packet);
  data.proto = MESH_PROTO_BIN;
  data.tos = MESH_TOS_P2P;
  xSemaphoreTake(meshSendMutex, portMAX_DELAY);
  esp_err_t err = esp_mesh_send(&parent, &data, MESH_DATA_P2P, nullptr, 0);
  xSemaphoreGive(meshSendMutex);
  return err;
}

void printMac(const uint8_t mac[6], char out[13]) {
  snprintf(out, 13, "%02X%02X%02X%02X%02X%02X",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

void printTelemetry(const MeshPacket &packet) {
  char nodeId[13];
  printMac(packet.node_mac, nodeId);

  Serial.printf(
      "{\"schema_version\":%u,\"type\":\"telemetry\",\"node_id\":\"%s\","
      "\"sequence\":%lu,\"uptime_ms\":%lu,\"received_ms\":%lu,\"hop_count\":%u,"
      "\"sensor_ok\":%s,",
      packet.schema_version, nodeId,
      static_cast<unsigned long>(packet.sequence),
      static_cast<unsigned long>(packet.uptime_ms),
      static_cast<unsigned long>(millis()), packet.hop_count,
      packet.sample_valid ? "true" : "false");

  if (packet.sample_valid) {
    Serial.printf(
        "\"accel_m_s2\":[%.3f,%.3f,%.3f],\"gyro_rad_s\":[%.3f,%.3f,%.3f],\"yaw_delta_rad\":%.4f,"
        "\"imu_temp_c\":%.2f,\"steps\":%lu,\"wake\":%s,",
        packet.accel_m_s2[0], packet.accel_m_s2[1], packet.accel_m_s2[2],
        packet.gyro_rad_s[0], packet.gyro_rad_s[1], packet.gyro_rad_s[2],
        packet.yaw_delta_rad,
        packet.imu_temp_c, static_cast<unsigned long>(packet.steps),
        packet.wake ? "true" : "false");
  } else {
    Serial.print(
        "\"accel_m_s2\":null,\"gyro_rad_s\":null,\"yaw_delta_rad\":null,\"imu_temp_c\":null,"
        "\"steps\":null,\"wake\":null,");
  }

  if (packet.hr_valid) Serial.printf("\"hr_bpm\":%d}\n", packet.hr_bpm);
  else Serial.println("\"hr_bpm\":null}");
}

void meshReceiveTask(void *) {
  uint8_t receiveBuffer[sizeof(MeshPacket)];
  mesh_addr_t from{};

  while (true) {
    mesh_data_t data{};
    data.data = receiveBuffer;
    data.size = sizeof(receiveBuffer);
    data.proto = MESH_PROTO_BIN;
    data.tos = MESH_TOS_P2P;
    int flags = 0;

    esp_err_t err = esp_mesh_recv(
        &from, &data, kMeshRxTimeoutMs, &flags, nullptr, 0);
    if (err != ESP_OK || data.size != sizeof(MeshPacket)) continue;

    MeshPacket packet{};
    memcpy(&packet, receiveBuffer, sizeof(packet));
    if (packet.magic != kPacketMagic || packet.schema_version != kSchemaVersion ||
        packet.hop_count > kMaxRelayHops || packet.sample_valid > 1 ||
        packet.wake > 1 || packet.hr_valid > 1) {
      continue;
    }

    if (esp_mesh_is_root()) {
      printTelemetry(packet);
    } else if (packet.hop_count < kMaxRelayHops) {
      ++packet.hop_count;
      sendToParent(packet);
    }
  }
}

void initImu() {
  Wire.begin();
  imuReady = imu.begin_I2C(0x6A) || imu.begin_I2C(0x6B);
  if (!imuReady) return;

  imu.setAccelDataRate(LSM6DS_RATE_104_HZ);
  imu.setAccelRange(LSM6DS_ACCEL_RANGE_4_G);
  imu.setGyroDataRate(LSM6DS_RATE_104_HZ);
  imu.setGyroRange(LSM6DS_GYRO_RANGE_500_DPS);
  imu.enablePedometer(true);
  imu.enableWakeup(true, 1, 12);
}

void sendImuSample() {
  MeshPacket packet{};
  packet.magic = kPacketMagic;
  packet.schema_version = kSchemaVersion;
  packet.sequence = sequenceNumber++;
  packet.uptime_ms = millis();
  memcpy(packet.node_mac, nodeMac, sizeof(nodeMac));
  packet.hr_bpm = -1;
#if NODE_ROLE == ROLE_SENSOR
#if HR_SOURCE_SPARKFUN
  if (sparkfunReady && static_cast<uint32_t>(millis() - lastHrMs) < 5000 && latestHr > 0) {
    packet.hr_bpm = latestHr;
    packet.hr_valid = 1;
  }
#else
  portENTER_CRITICAL(&hrMux);
  if (polarConnected && static_cast<uint32_t>(millis() - lastHrMs) < 5000 && latestHr > 0) {
    packet.hr_bpm = latestHr;
    packet.hr_valid = 1;
  }
  portEXIT_CRITICAL(&hrMux);
#endif
#endif

  if (imuReady) {
    packet.sample_valid = latestImuValid;
    if (packet.sample_valid) {
      packet.accel_m_s2[0] = latestAccel.acceleration.x;
      packet.accel_m_s2[1] = latestAccel.acceleration.y;
      packet.accel_m_s2[2] = latestAccel.acceleration.z;
      packet.gyro_rad_s[0] = latestGyro.gyro.x;
      packet.gyro_rad_s[1] = latestGyro.gyro.y;
      packet.gyro_rad_s[2] = latestGyro.gyro.z;
      packet.yaw_delta_rad = yawSincePacket;
      yawSincePacket = 0;
      packet.imu_temp_c = latestTemp.temperature;
      packet.steps = imu.readPedometer();
      packet.wake = imu.shake();
    }
  }

  sendToParent(packet);
}
}  // namespace

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("{\"type\":\"boot\",\"stage\":\"start\"}");

  esp_err_t macErr = esp_read_mac(nodeMac, ESP_MAC_WIFI_STA);
  checkInit("read_mac", macErr);
  initNvs();

  meshSendMutex = xSemaphoreCreateMutex();
  if (meshSendMutex == nullptr) {
    Serial.println("Could not create mesh send mutex.");
    while (true) delay(1000);
  }

#if NODE_ROLE == ROLE_SENSOR
  initImu();
#if HR_SOURCE_SPARKFUN
  const uint8_t beginStatus = bioHub.begin();
  const uint8_t configStatus = beginStatus ? beginStatus : bioHub.configBpm(MODE_ONE);
  sparkfunReady = configStatus == 0;
  Serial.printf("{\"type\":\"sparkfun_hr_status\",\"ready\":%s,\"error\":%u}\n",
                sparkfunReady ? "true" : "false", configStatus);
#endif
#endif

  initMesh();
  if (xTaskCreate(meshReceiveTask, "mesh_rx", 12288, nullptr, 5, nullptr) != pdPASS) {
    Serial.println("Could not start mesh receive task.");
    while (true) delay(1000);
  }
#if NODE_ROLE == ROLE_SENSOR
#if !HR_SOURCE_SPARKFUN
  if (POLAR_ID && xTaskCreate(polarTask, "polar", 8192, nullptr, 2, nullptr) != pdPASS) {
    Serial.println("{\"type\":\"init_error\",\"stage\":\"polar_task\"}");
  }
#endif
#endif
}

void loop() {
#if NODE_ROLE == ROLE_SENSOR
  const uint32_t now = millis();
#if HR_SOURCE_SPARKFUN
  if (sparkfunReady && static_cast<uint32_t>(now - lastBioReadMs) >= 1000) {
    lastBioReadMs = now;
    const bioData pulse = bioHub.readBpm();
    // The hub reports status 3 only when it detects a finger.
    latestHr = pulse.status == 3 && pulse.confidence > 0 && pulse.heartRate >= 30 && pulse.heartRate <= 240 ? pulse.heartRate : -1;
    lastHrMs = latestHr > 0 ? millis() : 0;
  }
#endif
  if (imuReady && static_cast<uint32_t>(now - lastImuReadMs) >= 20) {
    sensors_event_t accel, gyro, temp;
    latestImuValid = imu.getEvent(&accel, &gyro, &temp);
    if (latestImuValid) {
      if (lastImuReadMs && static_cast<uint32_t>(now - lastImuReadMs) < 200)
        yawSincePacket += gyro.gyro.z * (now - lastImuReadMs) / 1000.0f;
      latestAccel = accel;
      latestGyro = gyro;
      latestTemp = temp;
    }
    lastImuReadMs = now;
  }
  const uint32_t interval = imuReady ? kSamplePeriodMs : kSensorStatusPeriodMs;
  if (static_cast<uint32_t>(now - lastSensorStatusMs) >= interval) {
    lastSensorStatusMs = now;
    sendImuSample();
  }
  if (static_cast<uint32_t>(now - lastSensorDiagMs) >= 5000) {
    lastSensorDiagMs = now;
    mesh_addr_t parent{};
    Serial.printf(
        "{\"type\":\"sensor_status\",\"imu_ok\":%s,\"parent_connected\":%s}\n",
        imuReady ? "true" : "false", copyParent(parent) ? "true" : "false");
  }
  delay(20);
#else
  static uint32_t lastStatusMs = 0;
  const uint32_t now = millis();
  if (static_cast<uint32_t>(now - lastStatusMs) >= 5000) {
    lastStatusMs = now;
    Serial.printf("{\"type\":\"mesh_status\",\"role\":\"root\",\"node_count\":%d}\n",
                  esp_mesh_get_total_node_num());
  }
  delay(20);
#endif
}
