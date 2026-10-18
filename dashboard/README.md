# Operator dashboard

Streams team heart rate, movement, and HRV to a browser and runs the check loop from PLAN.md §1.4–1.5: an alert appears, the operator requests a check, acknowledges it, and resolves it with a note.

```bash
sensor/.venv/bin/pip install -r dashboard/requirements.txt
sensor/.venv/bin/python dashboard/server.py            # replay sensor/test4 + 5 simulated teammates
sensor/.venv/bin/python dashboard/server.py --live     # add the Polar Verity Sense as a live person
sensor/.venv/bin/python dashboard/server.py --imu /dev/cu.usbmodem1101   # add the LSM6DSOX IMU
```

Open http://127.0.0.1:8765.

For the Nano mesh on Windows, run:

```powershell
python dashboard/server.py --mesh-port COM11 --mesh-node Wearer1=206EF1316774 --mesh-node Wearer2=SECOND_NODE_ID --replay '' --synthetic 0
```

Use the 12-digit Wi-Fi `node_id` from root telemetry for each wearable; colons are accepted. Close the serial monitor before starting the dashboard because only one program can open COM11 at a time. Accepted telemetry is saved to `mesh.jsonl` in the session directory. Mesh reconnects resume automatically.

Every person is labeled with where their data comes from: **LIVE** (Polar over BLE), **REPLAY** (a recorded `sensor/<prefix>_*.csv` session, looped), or **SYNTHETIC** (simulated). The simulated team has scripted events for the demo: a distress press (Okafor, ~75 s), HR above expected (Alvarez, ~75 s), and a sensor dropout (Patel, 120–150 s).

**Attention items**

| Type | Trigger |
| --- | --- |
| Distress | Button press (simulated) or "Mark distress" |
| Physiology | HR more than 20 bpm from expected for 30 s |
| Sensor | No samples for 10 s, or skin contact lost. Marked "cleared" when the signal returns; it stays queued until the operator resolves it |

**IMU (LSM6DSOX)** is a 6-DoF chip: a 3-axis accelerometer (±2–16 g), a 3-axis gyroscope (±125–2000 °/s), and a die temperature sensor (256 LSB/°C, 0 = 25 °C). It does not measure heart rate. The Adafruit breakout uses I2C, so a microcontroller runs `sensor/lsm6dsox_serial/lsm6dsox_serial.ino` and streams `ax_mg,ay_mg,az_mg,gx_dps,gy_dps,gz_dps,temp_c` at 52 Hz over USB serial. Pass `--imu PORT` to add it. With `--live`, it attaches to the Polar wearer, and its accelerometer replaces the Polar's for movement. Without `--live`, it is a person of its own. `--imu` also accepts a saved `imu_*.csv` to replay. The dashboard shows **Rotation** (mean angular speed over ~2 s, °/s) and **IMU temp**. IMU temp is the chip's temperature and follows the board and the air around it. It is not skin or body temperature. Simulated people have a simulated IMU. The Polar replay has no gyroscope. Decoding and register constants are in `sensor/lsm6dsox.py`.

**Map** shows where each person is and their status color. There is no GPS. Simulated people move around `--base` (default: downtown San Diego). Live and replayed people have no position until the operator clicks "Place on map"; their pins are square and can be dragged. Switch to satellite view with the layers button. Tiles come from OpenStreetMap and Esri, so the map needs internet.

**Expected HR** comes from a first-order kinetic model driven by movement. Resting HR is taken from the first 20 readings, and gain and time constants are fixed defaults. The model is not fitted or validated. Swap in a reviewed predictor in `Kinetic.step`.

Each run saves `samples.csv` (1 Hz per person), `imu_<person>.csv` (every IMU line, when `--imu` is used), and `events.jsonl` (alerts and operator actions) under `dashboard/sessions/<timestamp>/`.
