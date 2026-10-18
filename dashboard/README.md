# Operator dashboard

Streams team heart rate, movement, and HRV to a browser and runs the check loop from PLAN.md §1.4–1.5: an alert appears, the operator requests a check, acknowledges it, and resolves it with a note.

```bash
sensor/.venv/bin/pip install -r dashboard/requirements.txt
sensor/.venv/bin/python dashboard/server.py            # replay sensor/test4 + 5 simulated teammates
sensor/.venv/bin/python dashboard/server.py --live     # add the Polar Verity Sense as a live person
sensor/.venv/bin/python dashboard/server.py --imu /dev/cu.usbmodem1101   # add the LSM6DSOX IMU
```

Open http://127.0.0.1:8765.

For the Nano mesh on Windows, run these in separate terminals:

```powershell
uv venv .venv --python 3.13
uv pip install --python .venv\Scripts\python.exe -r dashboard\requirements.txt
.venv\Scripts\python.exe dashboard\mesh_api.py --serial-port COM11
.venv\Scripts\python.exe dashboard\server.py --mesh-api http://127.0.0.1:8766/api/nodes --mesh-node Wearer1=206EF1316774 --mesh-node Wearer2=206EF1316838 --replay= --synthetic 0
```

The gateway serves [the latest data for both nodes](http://127.0.0.1:8766/api/nodes). It reports `serial_connected` and `last_event` from the root; each node has `online`, `age_s`, `received_at`, and `latest` fields. `latest` is the root telemetry packet or `null` until one arrives. It marks a node offline after 10 seconds without telemetry and reconnects if COM11 disappears. Only the gateway opens COM11, so close a serial monitor before starting it. The gateway and the dashboard's `--mesh-api` mode do not record a session. Use the 12-digit Wi-Fi `node_id` from root telemetry for each wearable; colons are accepted in `--mesh-node`. Full field definitions: [`docs/MESH_DATA_CONTRACTS.md`](../docs/MESH_DATA_CONTRACTS.md).

Every person is labeled with where their data comes from: **LIVE** (Polar over BLE), **REPLAY** (a recorded `sensor/<prefix>_*.csv` session, looped), or **SYNTHETIC** (simulated). The simulated team has scripted events for the demo: a distress press (Okafor, ~75 s), HR above expected (Alvarez, ~75 s), and a sensor dropout (Patel, 120–150 s).

For live Polar heart rate, the dashboard shows the average of the last five readings. It holds that display value through gaps of up to 30 seconds and blends new readings into the same average when the sensor reconnects. Sensor freshness and attention items still use the actual arrival time; after 10 seconds without heart-rate data, quality reports `no data` even while the display holds its last value.

**Attention items**

| Type | Trigger |
| --- | --- |
| Distress | Button press (simulated) or "Mark distress" |
| Physiology | HR more than 20 bpm from the current activity baseline for 30 s on IMU wearers; the existing expected-HR model applies to other sources |
| Sensor | No samples for 10 s, or skin contact lost. Marked "cleared" when the signal returns; it stays queued until the operator resolves it |

**IMU (LSM6DSOX)** is a 6-DoF chip: a 3-axis accelerometer (±2–16 g), a 3-axis gyroscope (±125–2000 °/s), and a die temperature sensor (256 LSB/°C, 0 = 25 °C). It does not measure heart rate. The Adafruit breakout uses I2C, so a microcontroller runs `sensor/lsm6dsox_serial/lsm6dsox_serial.ino` and streams `ax_mg,ay_mg,az_mg,gx_dps,gy_dps,gz_dps,temp_c` at 52 Hz over USB serial. Pass `--imu PORT` to add it. With `--live`, it attaches to the Polar wearer, and its accelerometer replaces the Polar's for movement. Without `--live`, it is a person of its own. `--imu` also accepts a saved `imu_*.csv` to replay. The dashboard shows 2-second movement variation, mean angular speed, and the latest acceleration and gyro X/Y/Z axes. Gyro values are rotation rates, not an absolute orientation. It also shows **IMU temp**, the chip's temperature, not skin or body temperature. Simulated people have a simulated IMU. The Polar replay has no gyroscope. Decoding and register constants are in `sensor/lsm6dsox.py`.

**Map** uses the dashboard browser's geolocation for the command origin when available; otherwise enter latitude and longitude in the map header. Both mesh wearers start at that origin. Their positions advance by the LSM6DSOX accelerometer pedometer's step count (assumed 0.7 m per step) and accumulated gyroscope turns. Mount each IMU level with +Z up and +X forward, then enter its initial clockwise-from-north heading in the individual view. Until then, north is assumed and the estimate is labeled uncalibrated. The map labels subsequent positions as low-confidence IMU estimates, and a click or drag correction is labeled separately until the next step. The estimate drifts and is not reliable geolocation; recalibrate and correct it during use. Both wearable and root boards need firmware schema version 2. Simulated people move around `--base` (default: downtown San Diego). Non-mesh live and replayed people need operator placement. Switch to satellite view with the layers button. Tiles come from OpenStreetMap and Esri, so the map needs internet.

**Activity and HR baselines** use each wearer's recent acceleration variation and gyro rotation to label still, walking or running. A label is unknown when motion data is missing or ambiguous. After 30 seconds in one activity, 20 fresh, good-quality HR readings establish that wearer's median baseline for the activity; the baseline then stays fixed for the dashboard session. Until a baseline is ready, no activity-based physiology alert is raised. The activity label, learning progress and three baselines appear in the wearer detail view. Thresholds (35/150/250 mG acceleration variation and 25/80°/s rotation) are provisional and need on-body calibration across mounting positions and users. Baselines are not persisted in mesh mode, so they relearn after a restart. This is an operator cue, not a clinical assessment.

Sources without an IMU retain the first-order kinetic expected-HR model. Its resting HR comes from the first 20 readings; gain and time constants are fixed defaults and unvalidated.

Recorded modes save `samples.csv` (1 Hz per person, including the activity label), `imu_<person>.csv` (every IMU line, when `--imu` is used), and `events.jsonl` (alerts and operator actions) under `dashboard/sessions/<timestamp>/`. Mesh API mode keeps only current data and in-memory dashboard history.
