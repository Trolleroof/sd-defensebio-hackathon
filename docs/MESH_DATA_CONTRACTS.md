# Mesh data contracts

Data flows through three layers. Each is a contract; change one side only together with the other.

```
wearable --(1) binary mesh packet--> root --(2) JSON lines on USB serial--> mesh_api.py --(3) HTTP JSON--> dashboard/server.py
```

Source of truth: `firmware/imu_mesh/imu_mesh.ino` (1, 2), `dashboard/mesh_api.py` (3), `MeshApiSource` in `dashboard/server.py` (consumer of 3).

## Timing

| What | Who | Period |
|---|---|---|
| Telemetry packet | each wearable, IMU found | 1000 ms |
| Telemetry packet (`sensor_ok: false`) | each wearable, no IMU | 5000 ms |
| `sensor_status` line | each wearable (own USB only) | 5000 ms |
| `mesh_status` line | root | 5000 ms |
| API poll | dashboard | 100 ms |
| Node marked offline | API | no telemetry for > 10 s |
| HR marked missing | wearable | no Polar notification for 5 s |

Topology: star. Both wearables join the root directly (tree, max layer 2, root takes 2 children), so `hop_count` is always 0.

## 1. Mesh packet (wearable → root)

ESP-WIFI-MESH `MESH_PROTO_BIN`, `MESH_TOS_P2P`, sent to the parent's address. Packed C struct, little-endian, **57 bytes**, no padding.

| Offset | Size | Type | Field | Meaning |
|---:|---:|---|---|---|
| 0 | 4 | `uint32` | `magic` | `0x494D5531` ("IMU1") |
| 4 | 1 | `uint8` | `schema_version` | `1` |
| 5 | 1 | `uint8` | `hop_count` | relay count, source sends 0 |
| 6 | 1 | `uint8` | `sample_valid` | 1 if IMU fields hold a sample, else 0 |
| 7 | 1 | `uint8` | `wake` | LSM6DSOX shake/wake flag, 0 or 1 |
| 8 | 4 | `uint32` | `sequence` | per-node counter from 0 at boot; increments on every send attempt, so gaps = dropped packets |
| 12 | 4 | `uint32` | `uptime_ms` | source `millis()` at sample time |
| 16 | 6 | `uint8[6]` | `node_mac` | source Wi-Fi STA MAC |
| 22 | 12 | `float32[3]` | `accel_m_s2` | x, y, z in m/s² |
| 34 | 12 | `float32[3]` | `gyro_rad_s` | x, y, z in rad/s |
| 46 | 4 | `float32` | `imu_temp_c` | IMU die temperature, °C (not body temperature) |
| 50 | 4 | `uint32` | `steps` | LSM6DSOX pedometer count since boot |
| 54 | 2 | `int16` | `hr_bpm` | heart rate, -1 when invalid |
| 56 | 1 | `uint8` | `hr_valid` | 1 if `hr_bpm` is a fresh Polar reading |

Root drops a packet unless all hold: size is exactly 57, `magic` matches, `schema_version == 1`, `hop_count <= 1`, and `sample_valid`, `wake`, `hr_valid` are each 0 or 1. When `sample_valid` is 0, the IMU fields are zero and must be ignored. Any layout change must bump `schema_version`.

## 2. Serial lines (root USB, 115200 baud)

One JSON object per line, `\n`-terminated. ESP-IDF logs and other non-JSON text can also appear; readers must skip lines that do not parse as a JSON object. The `type` field identifies the line.

### `telemetry`: one per accepted mesh packet

```json
{"schema_version":1,"type":"telemetry","node_id":"206EF1316838","sequence":89,"uptime_ms":95694,"received_ms":107679,"hop_count":0,"sensor_ok":true,"accel_m_s2":[0.144,0.294,9.977],"gyro_rad_s":[0.016,-0.003,-0.418],"imu_temp_c":19.88,"steps":0,"wake":false,"hr_bpm":72}
```

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | packet `schema_version` |
| `type` | `"telemetry"` | |
| `node_id` | string | 12 uppercase hex chars, source MAC without colons |
| `sequence` | int | packet `sequence` |
| `uptime_ms` | int | source clock |
| `received_ms` | int | root `millis()` at receipt; the two clocks are not synchronised |
| `hop_count` | int | packet `hop_count` |
| `sensor_ok` | bool | packet `sample_valid` |
| `accel_m_s2` | [float, float, float] or null | 3 decimals; null when `sensor_ok` is false |
| `gyro_rad_s` | [float, float, float] or null | 3 decimals; null when `sensor_ok` is false |
| `imu_temp_c` | float or null | 2 decimals; null when `sensor_ok` is false |
| `steps` | int or null | null when `sensor_ok` is false |
| `wake` | bool or null | null when `sensor_ok` is false |
| `hr_bpm` | int or null | null unless `hr_valid` |

`(node_id, uptime_ms, sequence)` identifies a sample; a reboot resets `uptime_ms` and `sequence`.

### Status and event lines

| `type` | Extra fields | Emitted by | When |
|---|---|---|---|
| `boot` | `stage: "start"` | all | start of `setup()` |
| `init_error` | `stage: string`, `err: int` (ESP error code; absent for `polar_task`) | all | init failure; except `polar_task`, the board halts |
| `mesh_status` | `role: "root"`, `node_count: int` (whole mesh, including root) | root | every 5 s |
| `mesh_parent_connected` | none | wearable | joined a parent |
| `mesh_parent_disconnected` | `reason: int` (Wi-Fi reason; 200 = beacon timeout, 2/4/8 = auth/assoc/leave) | wearable | lost parent |
| `mesh_no_parent_found` | none | wearable | scan found no parent |
| `mesh_child_connected` | none | root | a wearable joined |
| `mesh_child_disconnected` | none | root | a wearable left |
| `sensor_status` | `imu_ok: bool`, `parent_connected: bool` | wearable | every 5 s |
| `polar_connected` | `id: string` (8 hex chars) | wearable | HR notifications subscribed |

Wearable-only lines appear on that wearable's own USB port, never on the root's.

## 3. Gateway API (`mesh_api.py`)

`GET http://127.0.0.1:8766/api/nodes`, `Content-Type: application/json`, `Cache-Control: no-store`. Any other path returns 404. The gateway is the only process that opens the root's serial port.

```json
{
  "serial_connected": true,
  "last_event": {"received_at": 1792301557.15, "data": {"type": "mesh_status", "role": "root", "node_count": 3}},
  "nodes": {
    "206EF1316774": {"online": true, "age_s": 0.42, "received_at": 1792301556.73, "latest": {"type": "telemetry", "...": "..."}},
    "206EF1316838": {"online": false, "age_s": null, "received_at": null, "latest": null}
  }
}
```

| Field | Type | Meaning |
|---|---|---|
| `serial_connected` | bool | root serial port is open |
| `last_event` | object or null | most recent JSON line of any `type` from the root |
| `last_event.received_at` | float | laptop Unix time, seconds |
| `last_event.data` | object | the line as parsed, unchanged |
| `nodes` | object | always exactly the keys in `NODE_IDS` (`206EF1316774`, `206EF1316838`); telemetry from other IDs is dropped |
| `nodes[id].online` | bool | `serial_connected` and last telemetry ≤ 10 s ago |
| `nodes[id].age_s` | float or null | seconds since last telemetry, 2 decimals |
| `nodes[id].received_at` | float or null | laptop Unix time of last telemetry |
| `nodes[id].latest` | object or null | last `telemetry` line for that node, unchanged |

`latest` stays populated after a node goes offline; check `online` before treating it as current.

### Dashboard consumption (`MeshApiSource`)

- Maps each `--mesh-node Name=NODE_ID` to a person. `connected = online`.
- Processes `latest` only when `(uptime_ms, sequence)` changed since the last poll.
- Uses `hr_bpm` only when it is a number > 0; otherwise HR is cleared.
- Uses IMU only when `sensor_ok` is true and both vectors are 3 numbers. Converts accel to mG (`× 1000 / 9.80665`) and gyro to dps (`× 180 / π`).
- Movement and rotation need more than 5 samples. At 1 Hz that takes 6 s after a node appears, and the 104-sample window covers about 104 s.
