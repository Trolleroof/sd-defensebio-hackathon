# Nano ESP32 IMU mesh prototype

Two wearable Nano ESP32 boards sample their LSM6DSOX and Polar Verity Sense and send packets through ESP-WIFI-MESH to a Nano USB serial root. A fourth Nano runs `../mesh_router/mesh_router.ino` as the local Wi-Fi access point required by this SDK's mesh configuration. The root prints one JSON telemetry record per received sample.

## Flash and run

1. Flash the router sketch to the AP Nano and power it.
2. Compile `imu_mesh.ino` with `compiler.cpp.extra_flags=-DNODE_ROLE=1` for the root Nano.
3. Compile the sensor sketch separately for each wearable with `compiler.cpp.extra_flags=-DPOLAR_ID=0x12345678`, replacing the example hex ID with the trailing Polar ID in that wearer's advertised name. The default build has BLE disabled until an ID is assigned.
3. Select **Arduino Nano ESP32** in Arduino IDE. Install **Adafruit LSM6DS**, **Adafruit Unified Sensor**, and **Adafruit BusIO**.
4. Connect the root Nano to the laptop over USB and open its serial monitor at **115200 baud**. Power the root, then the first wearable, then the second.
5. Keep the mesh ID, channel, and password identical on all three boards. The prototype uses a fixed demo password in the sketch.

The sketch keeps the supplied LSM6DSOX configuration: I²C address `0x6A` with `0x6B` fallback, 104 Hz accel/gyro, ±4 g, ±500 dps, pedometer, and wake/shake event. It reads the IMU about every 20 ms to accumulate Z-axis rotation and sends a packet every 200 ms. If the sensor is not found, it sends a five-second status packet with `sensor_ok: false`.

## Packet model

Field-by-field contracts for the packet, serial lines, and gateway API are in [`docs/MESH_DATA_CONTRACTS.md`](../../docs/MESH_DATA_CONTRACTS.md).

The mesh packet is a 61-byte versioned binary struct shared by the Nano boards. Reflash both wearables and the root together for schema version 2. Both wearables connect directly to the root (star, max layer 2), so packets arrive with `hop_count: 0`. The relay path that increments `hop_count` is kept but unused.

The root emits a `mesh_status` JSON line every five seconds with the total mesh node count (including the root), emits join/leave events for direct children, and converts each received packet to a telemetry JSON line. Example telemetry:

```json
{"schema_version":2,"type":"telemetry","node_id":"A1B2C3D4E5F6","sequence":12,"uptime_ms":1200,"received_ms":1360,"hop_count":1,"sensor_ok":true,"accel_m_s2":[0.12,0.05,9.77],"gyro_rad_s":[0.01,0.02,0.00],"yaw_delta_rad":0.03,"imu_temp_c":25.4,"steps":3,"wake":false,"hr_bpm":null}
```

`node_id` is the source board's Wi-Fi MAC; `sequence` is per node; `uptime_ms` is the source timestamp; `received_ms` is the root timestamp. Acceleration is m/s² and gyro is rad/s, matching the Adafruit sensor events. If the IMU is unavailable, sensor fields are `null`. Root boot and ESP-IDF log messages may also appear on the serial port, so the laptop reader should accept only lines that parse as JSON telemetry.

The sensor firmware also emits a `sensor_status` JSON line every five seconds with IMU and parent-link state; a node with no parent emits `mesh_no_parent_found`. The BLE client selects the configured Polar ID, uses the standard heart-rate service, and reconnects after a disconnect. HR becomes null when no notification arrives for five seconds. The binary payload is for matching Nano ESP32 firmware only. There is no offline queue: packets sent while a parent is unavailable are dropped. The dashboard estimates position from steps and turn increments after origin and heading setup; this drifts and is not GPS. The fixed mesh credentials are for a hackathon bench setup only.
