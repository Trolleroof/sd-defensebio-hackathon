# Nano ESP32 IMU mesh prototype

Two wearable Nano ESP32 boards sample their LSM6DSOX and either a Polar Verity Sense or a wired SparkFun SEN-15219 pulse oximeter and heart rate sensor. They send packets through ESP-WIFI-MESH to a Nano USB serial root. A fourth Nano runs `../mesh_router/mesh_router.ino` as the local Wi-Fi access point required by this SDK's mesh configuration. The root prints one JSON telemetry record per received sample.

## Optional wired heart-rate sensor

Wire the **SparkFun SEN-15219** to a **wearable Nano ESP32**, sharing the IMU's I²C bus. Keep the IMU's existing A4/A5 connections. Plug a short Qwiic/STEMMA QT cable from the IMU breakout's free connector into the SparkFun board; that cable extends 3.3 V, ground, SDA, and SCL to the second sensor. The SparkFun board still needs separate `RESET` and `MFIO` wires. If the IMU breakout has no free connector, split the same four wires in parallel with a Qwiic hub or header junction.

```text
Nano A4/A5, 3V3, GND ── IMU ── Qwiic/STEMMA QT cable ── SparkFun SEN-15219
Nano D4 ──────────────────────────────────────────────── RESET
Nano D5 ──────────────────────────────────────────────── MFIO
```

| SparkFun SEN-15219 | Nano ESP32 | Existing IMU connection |
| --- | --- | --- |
| `3V3` | `3V3` | share 3.3 V |
| `GND` | `GND` | share ground |
| `SDA` | existing `A4 / SDA` bus | through IMU's free connector or parallel split |
| `SCL` | existing `A5 / SCL` bus | through IMU's free connector or parallel split |
| `RESET` | `D4` | separate control wire |
| `MFIO` | `D5` | separate control wire |

The SparkFun hub is at I²C address `0x55`; the IMU is at `0x6A` or `0x6B`. Do not connect the sensor to 5 V or leave `RESET`/`MFIO` unconnected. The optical sensor needs light, steady finger contact; verify usable readings in that placement before putting it on a moving wearer.

Pin references: [SparkFun hookup and required control pins](https://learn.sparkfun.com/tutorials/sparkfun-pulse-oximeter-and-heart-rate-monitor-hookup-guide/hardware-hookup), [Nano ESP32 A4/A5 and 3V3 pinout](https://docs.arduino.cc/resources/pinouts/ABX00083-full-pinout.pdf).

Install **SparkFun Bio Sensor Hub Library** (tested with 1.1.0). Compile that wearable with `compiler.cpp.extra_flags=-DHR_SOURCE_SPARKFUN=1`. This build reads heart rate from the wired sensor and never starts the Polar BLE task. The default build uses Polar when a `POLAR_ID` is set. The mesh packet still sends only heart rate from either source; SpO₂ is not yet in the packet or dashboard. If the wired sensor fails or loses finger contact, heart rate is missing rather than silently switching to Polar.

```sh
arduino-cli compile -b arduino:esp32:nano_nora firmware/imu_mesh --build-property compiler.cpp.extra_flags=-DHR_SOURCE_SPARKFUN=1
```

## Flash and run

1. Flash the router sketch to the AP Nano and power it.
2. Compile `imu_mesh.ino` with `compiler.cpp.extra_flags=-DNODE_ROLE=1` for the root Nano.
3. Compile each Polar wearable with `compiler.cpp.extra_flags=-DPOLAR_ID=0x12345678`, replacing the example hex ID with the trailing Polar ID in that wearer's advertised name. Compile each wired-SparkFun wearable with `compiler.cpp.extra_flags=-DHR_SOURCE_SPARKFUN=1` instead. The default build has BLE disabled until a Polar ID is assigned.
4. Select **Arduino Nano ESP32** in Arduino IDE. Install **Adafruit LSM6DS**, **Adafruit Unified Sensor**, and **Adafruit BusIO**; install the SparkFun Bio Sensor Hub Library for the wired variant.
5. Connect the root Nano to the laptop over USB and open its serial monitor at **115200 baud**. Power the root, then the first wearable, then the second.
6. Keep the mesh ID, channel, and password identical on all three boards. The prototype uses a fixed demo password in the sketch.

The sketch keeps the supplied LSM6DSOX configuration: I²C address `0x6A` with `0x6B` fallback, 104 Hz accel/gyro, ±4 g, ±500 dps, pedometer, and wake/shake event. It reads the IMU about every 20 ms to accumulate Z-axis rotation and sends a packet every 200 ms with the mean acceleration from those reads. If the sensor is not found, it sends a five-second status packet with `sensor_ok: false`.

## Packet model

Field-by-field contracts for the packet, serial lines, and gateway API are in [`docs/MESH_DATA_CONTRACTS.md`](../../docs/MESH_DATA_CONTRACTS.md).

The mesh packet is a 61-byte versioned binary struct shared by the Nano boards. Reflash both wearables and the root together for schema version 2. Both wearables connect directly to the root (star, max layer 2), so packets arrive with `hop_count: 0`. The relay path that increments `hop_count` is kept but unused.

The root emits a `mesh_status` JSON line every five seconds with the total mesh node count (including the root), emits join/leave events for direct children, and converts each received packet to a telemetry JSON line. Example telemetry:

```json
{"schema_version":2,"type":"telemetry","node_id":"A1B2C3D4E5F6","sequence":12,"uptime_ms":1200,"received_ms":1360,"hop_count":1,"sensor_ok":true,"accel_m_s2":[0.12,0.05,9.77],"gyro_rad_s":[0.01,0.02,0.00],"yaw_delta_rad":0.03,"imu_temp_c":25.4,"steps":3,"wake":false,"hr_bpm":null}
```

`node_id` is the source board's Wi-Fi MAC; `sequence` is per node; `uptime_ms` is the source timestamp; `received_ms` is the root timestamp. Acceleration is m/s² and gyro is rad/s, matching the Adafruit sensor events. If the IMU is unavailable, sensor fields are `null`. Root boot and ESP-IDF log messages may also appear on the serial port, so the laptop reader should accept only lines that parse as JSON telemetry.

The sensor firmware also emits a `sensor_status` JSON line every five seconds with IMU and parent-link state; a node with no parent emits `mesh_no_parent_found`. The BLE client selects the configured Polar ID, uses the standard heart-rate service, and reconnects after a disconnect or ten seconds without HR notifications. HR becomes null after five seconds without a notification. The binary payload is for matching Nano ESP32 firmware only. There is no offline queue: packets sent while a parent is unavailable are dropped. After 30 seconds motionless at startup, the dashboard estimates velocity and position from calibrated acceleration and gyroscope turns. This drifts and is not GPS. The fixed mesh credentials are for a hackathon bench setup only.

### Diagnosing demo dropouts

The root's child join/leave events include the child's MAC. A wearable's `sensor_status` includes `parent_rssi_dbm`, the longest `esp_mesh_send` call since the previous status (`mesh_send_max_ms`), and the last send error code. On the October 18 bench test, child disconnect events belonged to wearable 2 (`206EF1316838`), while wearable 1 stayed associated. With wearable 2 powered off, wearable 1's Polar build still delivered only 16 packets in 45 seconds: reliable mesh sends repeatedly blocked for 2.8–5 seconds despite −16 to −17 dBm Wi-Fi RSSI. A build with Polar BLE disabled delivered 175 packets in 45 seconds with a maximum 0.6-second receive gap. This isolates BLE/Wi-Fi coexistence on the wearable as the main source of wearer 1's packet gaps in that setup. The no-BLE build was only a diagnostic; the Polar build was restored. The `200 ms` send interval above is a target, not a measured rate when both radios run.

If a wearable sees no mesh parent for 20 seconds, it logs `mesh_rebooting` and restarts itself so mesh and Polar acquisition can reconnect without a manual reset. An in-progress mesh send can delay that timer. The root is not restarted by this watchdog, and the packet stream still has gaps while the root is absent.
