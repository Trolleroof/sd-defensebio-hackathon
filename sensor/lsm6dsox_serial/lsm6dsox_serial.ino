// Streams an Adafruit LSM6DSOX (STEMMA QT / I2C) over USB serial for dashboard/server.py --imu.
// Needs the "Adafruit LSM6DS" library. One line per sample at 52 Hz:
//   ax_mg,ay_mg,az_mg,gx_dps,gy_dps,gz_dps,temp_c
// Lines starting with '#' are status messages and are ignored by the gateway.

#include <Adafruit_LSM6DSOX.h>

Adafruit_LSM6DSOX imu;
const float MS2_TO_MG = 1000.0 / 9.80665;
const float RAD_TO_DPS = 57.29578;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  if (!imu.begin_I2C()) {
    Serial.println("# LSM6DSOX not found");
    while (1) delay(1000);
  }
  imu.setAccelRange(LSM6DS_ACCEL_RANGE_4_G);
  imu.setGyroRange(LSM6DS_GYRO_RANGE_2000_DPS);
  imu.setAccelDataRate(LSM6DS_RATE_52_HZ);
  imu.setGyroDataRate(LSM6DS_RATE_52_HZ);
  Serial.println("# LSM6DSOX ready: 52 Hz, +-4 g, +-2000 dps");
}

void loop() {
  static uint32_t next = 0;
  if ((int32_t)(millis() - next) < 0) return;
  next = millis() + 19;  // ~52 Hz
  sensors_event_t a, g, t;
  imu.getEvent(&a, &g, &t);
  Serial.print(a.acceleration.x * MS2_TO_MG, 1); Serial.print(',');
  Serial.print(a.acceleration.y * MS2_TO_MG, 1); Serial.print(',');
  Serial.print(a.acceleration.z * MS2_TO_MG, 1); Serial.print(',');
  Serial.print(g.gyro.x * RAD_TO_DPS, 2); Serial.print(',');
  Serial.print(g.gyro.y * RAD_TO_DPS, 2); Serial.print(',');
  Serial.print(g.gyro.z * RAD_TO_DPS, 2); Serial.print(',');
  Serial.println(t.temperature, 2);
}
