#!/usr/bin/env python3
"""Operator dashboard gateway: streams team physiology to the browser over a WebSocket.

Sources (each person is labeled with where their data comes from):
  LIVE       a Polar Verity Sense over BLE (reuses the decoders in sensor/polar_stream.py),
             and/or an LSM6DSOX IMU streamed over USB serial (sensor/lsm6dsox.py)
  REPLAY     a recorded session from sensor/<prefix>_{hr,ppi,acc}.csv, looped in real time
  SYNTHETIC  simulated teammates so the multi-person views have something to show
  MESH       latest packets from the local mesh API, without session recording

Every tick the gateway computes movement, HRV, an expected HR from a first-order kinetic
model, and attention items. Operator actions come back over the same socket and are saved
with the samples under dashboard/sessions/<timestamp>/ in recorded modes.

Usage:
  python dashboard/server.py                          # replay sensor/test4 + synthetic team
  python dashboard/server.py --live                   # add the Polar sensor as a live person
  python dashboard/server.py --live --synthetic 0     # live sensor only
  python dashboard/server.py --replay sensor/test1    # different recording
  python dashboard/server.py --imu /dev/cu.usbmodem1101   # add the LSM6DSOX (gyro, accel, die temp)
  python dashboard/server.py --mesh-api http://127.0.0.1:8766/api/nodes --mesh-node Wearer1=206EF1316774 --mesh-node Wearer2=206EF1316838 --replay= --synthetic 0
  python dashboard/server.py --base 32.7157,-117.1611 # map center for simulated positions

Mesh locations start at the browser-provided command origin and advance as low-confidence
step-and-turn estimates. The operator can correct locations on the map. Every position
carries its source.
"""

import argparse
import asyncio
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

ROOT = Path(__file__).resolve().parent
TICK_S = 1.0
HISTORY_S = 600          # seconds of history kept per person and sent on connect
STALE_S = 10             # no samples for this long -> "no data"
HR_HOLD_S = 30           # live Polar display hold outside mesh mode
DEVIATION_BPM = 20       # |measured - expected| above this ...
DEVIATION_HOLD_S = 30    # ... for this long -> attention item
MODEL_VERSION = "activity baselines for IMU; kinetic-v0 otherwise (not validated)"


# ---------------------------------------------------------------- per-person state

@dataclass
class Kinetic:
    """HR_ss = rest + gain * movement;  dHR/dt = (HR_ss - HR) / tau  (tau_up rising, tau_down falling)."""
    rest: float | None = None
    gain: float = 0.12        # bpm per mG of movement (std of |a| over 1 s)
    tau_up: float = 30.0
    tau_down: float = 60.0
    state: float | None = None

    def step(self, movement: float | None, dt: float) -> float | None:
        if self.rest is None:
            return None
        target = self.rest + self.gain * min(movement or 0.0, 800.0)
        if self.state is None:
            self.state = target
        d = target - self.state
        self.state += dt / (self.tau_up if d > 0 else self.tau_down) * d
        return self.state


@dataclass
class Person:
    id: str
    name: str
    role: str
    source: str                       # LIVE | REPLAY | SYNTHETIC
    device: str = ""
    hr: float | None = None
    hr_t: float = 0.0
    hr_window: deque = field(default_factory=lambda: deque(maxlen=5))
    display: dict = field(default_factory=dict)
    display_t: float = 0.0
    last_sample: float = 0.0          # wall time of the last sample from the device
    connected: bool = False
    contact: bool | None = None       # skin contact flag, when the sensor reports it
    acc: deque = field(default_factory=lambda: deque(maxlen=52 * 2))   # (time, |a| in mG)
    acc_xyz: tuple | None = None
    rr: deque = field(default_factory=lambda: deque(maxlen=60))         # recent good intervals, ms
    ppi_seen: deque = field(default_factory=lambda: deque(maxlen=20))   # 1 = rejected
    model: Kinetic = field(default_factory=Kinetic)
    early_hr: list = field(default_factory=list)
    dev_since: float | None = None
    activity: str | None = None
    activity_since: float = 0.0
    activity_hr_t: float = 0.0
    activity_samples: dict = field(default_factory=lambda: {"still": [], "walking": [], "running": []})
    activity_baselines: dict = field(default_factory=dict)
    history: deque = field(default_factory=lambda: deque(maxlen=int(HISTORY_S / TICK_S)))
    lat: float | None = None
    lon: float | None = None
    loc: str | None = None            # "simulated" | "origin" | "estimated" | "corrected"; None = no location
    loc_t: float | None = None
    heading: float = 0.0             # degrees clockwise from north; north until calibrated
    heading_set: bool = False
    steps_seen: int | None = None
    uptime_seen: int | None = None
    calibration_start_ms: int | None = None
    calibration_samples: int = 0
    calibration_acc_sum: tuple = (0.0, 0.0)
    calibration_yaw_sum: float = 0.0
    calibration_dt: float = 0.0
    calibration_reference: tuple | None = None
    accel_bias: tuple | None = None
    yaw_bias: float = 0.0
    velocity_n: float = 0.0
    velocity_e: float = 0.0
    still_since_ms: int | None = None
    imu: bool = False                 # has a gyroscope/IMU source
    gyro: deque = field(default_factory=lambda: deque(maxlen=52 * 2))  # (time, |w| in dps)
    gyro_xyz: tuple | None = None
    imu_temp: float | None = None     # IMU die temperature, °C (not body temperature)
    imu_t: float = 0.0                # wall time of the last IMU sample

    def add_hr(self, bpm: float, t: float | None = None):
        sample_t = t or time.time()
        if self.source == "LIVE":
            if not self.id.startswith("mesh_") and sample_t - self.hr_t > HR_HOLD_S:
                self.hr_window.clear()
            self.hr_window.append(bpm)
        self.hr = bpm
        self.hr_t = sample_t
        self.last_sample = sample_t
        if self.model.rest is None:
            self.early_hr.append(bpm)
            if len(self.early_hr) >= 20:  # resting level from the first readings
                self.model.rest = statistics.quantiles(self.early_hr, n=5)[0]

    def add_acc(self, x: float, y: float, z: float):
        self.acc_xyz = (x, y, z)
        self.acc.append((time.time(), math.sqrt(x * x + y * y + z * z)))
        self.last_sample = time.time()

    def add_imu(self, acc: tuple, gyro: tuple, temp: float | None):
        """One LSM6DSOX sample: acc in mG, gyro in dps, die temperature in °C."""
        self.add_acc(*acc)
        self.gyro_xyz = gyro
        self.gyro.append((time.time(), math.sqrt(sum(g * g for g in gyro))))
        if temp is not None:
            self.imu_temp = temp
        self.imu_t = self.last_sample

    def advance(self, steps: int, uptime_ms: int, yaw_delta: float, acc: tuple, gyro: tuple):
        """Estimate horizontal velocity from calibrated acceleration, then integrate position."""
        if self.uptime_seen is not None and uptime_ms < self.uptime_seen:
            self.accel_bias = None
            self.calibration_start_ms = None
            self.still_since_ms = None
        dt = (uptime_ms - self.uptime_seen) / 1000 if self.uptime_seen is not None else 0
        continuous = 0 < dt <= 0.6
        delta = (steps - self.steps_seen) % 65536 if continuous and self.steps_seen is not None else 0
        self.steps_seen, self.uptime_seen = steps, uptime_ms
        if not continuous:
            self.velocity_n = self.velocity_e = 0.0
            if self.accel_bias is None:
                self.calibration_start_ms = None

        if self.accel_bias is None:
            stationary = (continuous and delta == 0 and 8.8 < math.sqrt(sum(a * a for a in acc)) < 10.8
                          and math.sqrt(sum(g * g for g in gyro)) < 0.08
                          and (self.calibration_reference is None or math.dist(acc, self.calibration_reference) < 0.25))
            if not stationary:
                self.calibration_start_ms = None
                self.calibration_samples = 0
                self.calibration_acc_sum = (0.0, 0.0)
                self.calibration_yaw_sum = self.calibration_dt = 0.0
                self.calibration_reference = None
                return
            if self.calibration_start_ms is None:
                self.calibration_start_ms = uptime_ms
                self.calibration_reference = acc
            self.calibration_samples += 1
            self.calibration_acc_sum = (self.calibration_acc_sum[0] + acc[0], self.calibration_acc_sum[1] + acc[1])
            self.calibration_yaw_sum += yaw_delta
            self.calibration_dt += dt
            if uptime_ms - self.calibration_start_ms < 30000 or self.calibration_samples < 100:
                return
            self.accel_bias = tuple(value / self.calibration_samples for value in self.calibration_acc_sum)
            self.yaw_bias = self.calibration_yaw_sum / self.calibration_dt
            self.velocity_n = self.velocity_e = 0.0
            self.still_since_ms = uptime_ms
            return

        if not continuous:
            return
        turn = yaw_delta - self.yaw_bias * dt
        heading_mid = math.radians(self.heading - math.degrees(turn) / 2)
        self.heading = (self.heading - math.degrees(turn)) % 360
        # shortcut: this assumes a level mount; add attitude/gravity compensation if the wearable tilts in use.
        ax, ay = acc[0] - self.accel_bias[0], acc[1] - self.accel_bias[1]
        ax = ax if abs(ax) >= 0.08 else 0.0
        ay = ay if abs(ay) >= 0.08 else 0.0
        if delta == 0 and abs(ax) < 0.15 and abs(ay) < 0.15 and abs(turn / dt) < 0.05:
            self.still_since_ms = self.still_since_ms or uptime_ms
        else:
            self.still_since_ms = None
        if self.still_since_ms is not None and uptime_ms - self.still_since_ms >= 2000:
            self.velocity_n = self.velocity_e = 0.0
            return
        north_a = ax * math.cos(heading_mid) + ay * math.sin(heading_mid)
        east_a = ax * math.sin(heading_mid) - ay * math.cos(heading_mid)
        north = self.velocity_n * dt + 0.5 * north_a * dt * dt
        east = self.velocity_e * dt + 0.5 * east_a * dt * dt
        self.velocity_n += north_a * dt
        self.velocity_e += east_a * dt
        if not self.loc:
            return
        self.lat += math.degrees(north / 6371000)
        self.lon += math.degrees(east / (6371000 * max(0.01, math.cos(math.radians(self.lat)))))
        self.lon = (self.lon + 180) % 360 - 180
        if abs(north) + abs(east) >= 0.001:
            self.loc, self.loc_t = "estimated", time.time()

    def imu_fresh(self, now: float | None = None) -> bool:
        return self.imu and (now or time.time()) - self.imu_t <= (STALE_S if self.id.startswith("mesh_") else 2)

    def add_ppi(self, ppi: float, blocker: int, contact: int):
        self.contact = bool(contact)
        bad = blocker or not 250 < ppi < 2000
        if not bad and len(self.rr) >= 3:
            med = statistics.median(list(self.rr)[-7:])
            bad = abs(ppi - med) > 0.2 * med
        self.ppi_seen.append(1 if bad else 0)
        if not bad:
            self.rr.append(ppi)
        self.last_sample = time.time()

    def movement(self) -> float | None:
        recent = list(self.acc)[-2:] if self.id.startswith("mesh_") else self.acc
        samples = [value for t, value in recent if time.time() - t <= (STALE_S if self.id.startswith("mesh_") else 2)]
        return statistics.pstdev(samples) if len(samples) >= 2 else None

    def rotation(self) -> float | None:
        """Mean angular speed over the last 2 s, dps."""
        recent = list(self.gyro)[-2:] if self.id.startswith("mesh_") else self.gyro
        samples = [value for t, value in recent if time.time() - t <= (STALE_S if self.id.startswith("mesh_") else 2)]
        return statistics.fmean(samples) if samples else None

    def motion_activity(self, now: float, movement: float | None) -> str | None:
        if not self.imu_fresh(now) or movement is None:
            return None
        rotation = self.rotation()
        if movement < 35 and rotation is not None and rotation < 25:
            return "still"
        if movement >= 250 or (movement >= 150 and rotation is not None and rotation >= 80):
            return "running"
        if movement >= 35:
            return "walking"
        return None

    def rmssd(self) -> float | None:
        rr = list(self.rr)[-30:]
        if len(rr) < 5:
            return None
        return math.sqrt(statistics.fmean((b - a) ** 2 for a, b in zip(rr, rr[1:])))

    def quality(self, now: float) -> str:
        if not self.connected or now - self.last_sample > STALE_S:
            return "no data"
        if (self.id == "live" or self.id.startswith("mesh_")) and now - self.hr_t > STALE_S:
            return "no data"
        if self.contact is False:
            return "no contact"
        if len(self.ppi_seen) >= 8 and sum(self.ppi_seen) / len(self.ppi_seen) > 0.5:
            return "motion"
        return "good"


# ---------------------------------------------------------------- sources

class SyntheticSource:
    """Activity blocks with HR driven by a hidden per-person response. Scripted events for the demo."""

    PLAN = [("rest", 60), ("walk", 90), ("run", 60), ("rest", 90), ("walk", 60)]
    LEVEL = {"rest": 12, "walk": 140, "run": 420}

    def __init__(self, person: Person, seed: int, script: str | None = None):
        self.p, self.rng, self.script = person, random.Random(seed), script
        self.p.imu = True  # simulated IMU: gyro and die temperature
        self.rest = self.rng.uniform(56, 72)
        self.gain = self.rng.uniform(0.09, 0.15)
        self.hr = self.rest
        self.offset = self.rng.uniform(0, 300)
        # Position in metres east/north of the base point; heading drifts, speed follows activity
        self.e, self.n = self.rng.uniform(-150, 150), self.rng.uniform(-150, 150)
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.die_t = self.rng.uniform(29, 31)

    SPEED = {"rest": 0.0, "walk": 1.4, "run": 3.2}  # m/s
    ROT = {"rest": 3, "walk": 45, "run": 130}        # typical mean |w| of a torso-worn IMU, dps

    def move(self, act: str, base: tuple[float, float]):
        self.heading += self.rng.gauss(0, 0.25)
        if math.hypot(self.e, self.n) > 350:  # turn back toward the base
            self.heading = math.atan2(-self.n, -self.e) + self.rng.gauss(0, 0.3)
        v = self.SPEED[act]
        self.e += v * math.cos(self.heading)
        self.n += v * math.sin(self.heading)
        lat0, lon0 = base
        self.p.lat = lat0 + self.n / 111_320
        self.p.lon = lon0 + self.e / (111_320 * math.cos(math.radians(lat0)))
        self.p.loc, self.p.loc_t = "simulated", time.time()

    def activity(self, t: float) -> str:
        cycle = sum(d for _, d in self.PLAN)
        k = (t + self.offset) % cycle
        for name, d in self.PLAN:
            if k < d:
                return name
            k -= d
        return "rest"

    async def run(self, team):
        self.p.connected = True
        t0 = time.time()
        while True:
            t = time.time() - t0
            # Scripted demo events
            if self.script == "dropout" and 120 < t % 300 < 150:
                self.p.connected = False
                await asyncio.sleep(1); continue
            self.p.connected = True
            extra = 38 if self.script == "elevated" and t > 45 else 0
            if self.script == "distress" and abs(t - 75) < 0.5:
                team.raise_distress(self.p.id, "button")

            act = self.activity(t)
            self.move(act, team.base)
            level = self.LEVEL[act] * self.rng.uniform(0.8, 1.2)
            self.die_t += (29.5 + 0.006 * level - self.die_t) / 120 + self.rng.gauss(0, 0.02)
            rot = self.ROT[act] * self.rng.uniform(0.8, 1.2)
            for _ in range(52):
                a = 1000 + self.rng.gauss(0, level)
                self.p.add_imu((0.0, 0.0, a), tuple(abs(self.rng.gauss(rot, rot * 0.3)) / math.sqrt(3) for _ in range(3)), self.die_t)
            target = self.rest + self.gain * level + extra
            self.hr += (target - self.hr) / (25 if target > self.hr else 55) + self.rng.gauss(0, 0.6)
            self.p.add_hr(round(self.hr))
            mean_rr = 60000 / self.hr
            for _ in range(max(1, round(self.hr / 60))):
                self.p.add_ppi(mean_rr + self.rng.gauss(0, 900 / self.hr), 0, 1)
            await asyncio.sleep(1)


class ReplaySource:
    """Replays a recorded session in real time, looping. Original timing is preserved."""

    def __init__(self, person: Person, prefix: Path):
        self.p = person
        self.events = []  # (t_rel, kind, values)
        base = None
        for kind in ("hr", "ppi", "acc"):
            path = prefix.parent / f"{prefix.name}_{kind}.csv"
            if not path.exists():
                continue
            with open(path) as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                t = float(r["t"])
                base = t if base is None else min(base, t)
                self.events.append((t, kind, r))
        if not self.events:
            raise SystemExit(f"No recording found at {prefix}_*.csv")
        self.events = sorted(((t - base, k, r) for t, k, r in self.events), key=lambda e: e[0])
        self.has_hr = any(k == "hr" for _, k, _ in self.events)

    async def run(self, team):
        self.p.connected = True
        while True:
            t0 = time.time()
            i = 0
            while i < len(self.events):
                now_rel = time.time() - t0
                while i < len(self.events) and self.events[i][0] <= now_rel:
                    _, kind, r = self.events[i]
                    if kind == "hr" and int(r["bpm"]) > 0:
                        self.p.add_hr(int(r["bpm"]))
                    elif kind == "acc":
                        self.p.add_acc(float(r["x_mg"]), float(r["y_mg"]), float(r["z_mg"]))
                    elif kind == "ppi":
                        self.p.add_ppi(float(r["ppi_ms"]), int(r["blocker"]), int(r["skin_contact"]))
                        if not self.has_hr and self.p.ppi_seen[-1] == 0:  # no HR file: derive HR from intervals
                            self.p.add_hr(round(60000 / statistics.fmean(list(self.p.rr)[-5:])))
                    i += 1
                await asyncio.sleep(0.05)


class LiveSource:
    """Polar Verity Sense over BLE. Reconnects when the sensor drops."""

    def __init__(self, person: Person, address: str | None):
        self.p, self.address = person, address
        self.ble_t = 0.0

    async def run(self, team):
        sys.path.insert(0, str(ROOT.parent / "sensor"))
        import polar_stream as ps
        from bleak import BleakClient

        def on_hr(_, data):
            bpm, rr = ps.parse_hr(data)
            if bpm:
                self.ble_t = time.time()
                self.p.add_hr(bpm)

        def on_pmd(_, data):
            kind, samples = ps.decode_pmd(data)
            if samples:
                self.ble_t = time.time()
            if kind == ps.PMD_ACC and not self.p.imu_fresh():  # the IMU, when present, supplies movement
                for x, y, z in samples:
                    self.p.add_acc(x, y, z)
            elif kind == ps.PMD_PPI:
                for bpm, ppi, _err, blocker, contact in samples:
                    self.p.add_ppi(ppi, blocker, contact)
                    if self.p.ppi_seen[-1] == 0:
                        self.p.add_hr(bpm or round(60000 / statistics.fmean(list(self.p.rr)[-5:])))

        last_error = None
        while True:
            try:
                dev = await ps.find_device(self.address)
                self.p.device = f"{dev.name} {dev.address}"
                team.log("system", self.p.id, f"Connecting to {dev.name}")
                async with BleakClient(dev) as client:
                    self.ble_t = time.time()
                    await client.start_notify(ps.HR_MEASUREMENT, on_hr)
                    try:
                        await client.start_notify(ps.PMD_CONTROL, lambda *_: None)
                        await client.start_notify(ps.PMD_DATA, on_pmd)
                        for k in (ps.PMD_ACC, ps.PMD_PPI):
                            try:
                                await client.write_gatt_char(ps.PMD_CONTROL, ps.START_CMD[k], response=True)
                                await asyncio.sleep(0.5)
                            except Exception as error:
                                team.log("system", self.p.id, f"Optional Polar stream {k} unavailable: {error}")
                    except Exception as error:
                        team.log("system", self.p.id, f"Optional Polar motion streams unavailable: {error}")
                    self.p.connected = True
                    team.log("system", self.p.id, "Sensor connected")
                    while client.is_connected:
                        await asyncio.sleep(0.5)
                        if time.time() - self.ble_t > STALE_S:
                            team.log("system", self.p.id, "No Polar notifications; reconnecting")
                            break
            except (Exception, SystemExit) as e:  # find_device raises SystemExit when nothing is found
                self.p.connected = False
                msg = f"Sensor unavailable: {e}"
                if msg != last_error:  # don't repeat the same failure every retry
                    team.log("system", self.p.id, msg)
                last_error = msg
                await asyncio.sleep(3)
                continue
            last_error = None
            self.p.connected = False
            team.log("system", self.p.id, "Sensor disconnected")
            await asyncio.sleep(3)


class ImuSource:
    """LSM6DSOX lines from a microcontroller over USB serial, or a saved log replayed at 52 Hz.

    Every line is saved to the session as imu_<person>.csv. With a serial port the source
    reconnects when the board is unplugged.
    """

    def __init__(self, person: Person, target: str, session_dir: Path, owns_connection: bool, baud: int = 115200):
        self.p, self.target, self.baud, self.owns = person, target, baud, owns_connection
        self.n = 0  # lines logged
        self.p.imu = True
        sys.path.insert(0, str(ROOT.parent / "sensor"))
        from lsm6dsox import parse_line
        self.parse = parse_line
        self.out = open(session_dir / f"imu_{person.id}.csv", "w")
        self.out.write("t,ax_mg,ay_mg,az_mg,gx_dps,gy_dps,gz_dps,temp_c\n")

    def take(self, line: str):
        s = self.parse(line)
        if s is None:
            return
        acc, gyro, temp = s
        self.p.add_imu(acc, gyro, temp)
        if self.owns:
            self.p.connected = True
        self.out.write(f"{time.time():.3f},{line.strip()}\n")
        self.n += 1
        if self.n % 52 == 0:
            self.out.flush()

    async def run(self, team):
        if Path(self.target).is_file():
            lines = [l for l in Path(self.target).read_text().splitlines() if l and not l.startswith(("#", "t,"))]
            lines = [l.split(",", 1)[1] if l.count(",") == 7 else l for l in lines]  # accept saved imu_*.csv
            while True:
                for l in lines:
                    self.take(l)
                    await asyncio.sleep(1 / 52)
        import serial
        last_error = None
        while True:
            try:
                port = await asyncio.to_thread(serial.Serial, self.target, self.baud, timeout=1)
                team.log("system", self.p.id, f"IMU connected on {self.target}")
                last_error = None
                with port:
                    while True:
                        raw = await asyncio.to_thread(port.readline)
                        if raw:
                            self.take(raw.decode(errors="ignore"))
            except Exception as e:
                if self.owns:
                    self.p.connected = False
                msg = f"IMU unavailable: {e}"
                if msg != last_error:
                    team.log("system", self.p.id, msg)
                last_error = msg
                await asyncio.sleep(3)


class MeshApiSource:
    """Poll the local mesh API; the gateway alone owns the serial port."""

    def __init__(self, people: dict[str, Person], url: str):
        self.people, self.url = people, url
        self.seen = {}

    async def run(self, team):
        last_error = None
        async with ClientSession(timeout=ClientTimeout(total=2)) as client:
            while True:
                try:
                    async with client.get(self.url) as response:
                        response.raise_for_status()
                        nodes = (await response.json())["nodes"]
                    for node_id, p in self.people.items():
                        entry = nodes.get(node_id, {})
                        p.connected = bool(entry.get("online"))
                        data = entry.get("latest")
                        if not isinstance(data, dict):
                            continue
                        marker = (data.get("uptime_ms"), data.get("sequence"))
                        if self.seen.get(node_id) == marker:
                            continue
                        self.seen[node_id] = marker
                        p.last_sample = entry["received_at"]
                        bpm = data.get("hr_bpm")
                        if type(bpm) in (int, float) and math.isfinite(bpm) and bpm > 0:
                            p.add_hr(bpm, entry["received_at"])
                        # Keep the last real HR for display; hr_t still tracks its actual age.
                        acc, gyro = data.get("accel_m_s2"), data.get("gyro_rad_s")
                        if data.get("sensor_ok") and isinstance(acc, list) and len(acc) == 3 and isinstance(gyro, list) and len(gyro) == 3 and all(type(v) in (int, float) and math.isfinite(v) for v in acc + gyro):
                            temp = data.get("imu_temp_c")
                            p.add_imu(tuple(v * 1000 / 9.80665 for v in acc), tuple(v * 180 / math.pi for v in gyro),
                                      temp if type(temp) in (int, float) and math.isfinite(temp) else None)
                            p.imu_t = entry["received_at"]
                            p.acc[-1] = (p.imu_t, p.acc[-1][1])
                            p.gyro[-1] = (p.imu_t, p.gyro[-1][1])
                            if type(data.get("steps")) is int and 0 <= data["steps"] < 65536 and type(data.get("uptime_ms")) is int and data["uptime_ms"] >= 0 and type(data.get("yaw_delta_rad")) in (int, float) and math.isfinite(data["yaw_delta_rad"]):
                                p.advance(data["steps"], data["uptime_ms"], data["yaw_delta_rad"], tuple(acc), tuple(gyro))
                        p.last_sample = entry["received_at"]
                    last_error = None
                except Exception as error:
                    for p in self.people.values():
                        p.connected = False
                    if str(error) != last_error:
                        team.log("system", None, f"Mesh API unavailable: {error}")
                        last_error = str(error)
                await asyncio.sleep(0.1)


# ---------------------------------------------------------------- team, attention, actions

class Team:
    def __init__(self, session_dir: Path | None, base: tuple[float, float]):
        self.base = base
        self.origin: tuple[float, float] | None = None
        self.origin_source: str | None = None
        self.people: dict[str, Person] = {}
        self.alerts: dict[str, dict] = {}
        self.events: list[dict] = []
        self.clients: set[web.WebSocketResponse] = set()
        self.started = time.time()
        self.session_dir = session_dir
        self.samples = self.samples_w = self.events_f = None
        if session_dir:
            session_dir.mkdir(parents=True, exist_ok=True)
            self.samples = open(session_dir / "samples.csv", "w", newline="")
            self.samples_w = csv.writer(self.samples)
            self.samples_w.writerow(["t", "person", "source", "hr", "expected", "movement_mg", "rmssd_ms", "rot_dps", "imu_temp_c", "quality", "status", "lat", "lon", "loc_source", "activity"])
            self.events_f = open(session_dir / "events.jsonl", "a")
        self._next = 1

    def add(self, p: Person):
        self.people[p.id] = p

    def log(self, kind: str, person: str | None, text: str, alert: str | None = None) -> dict:
        ev = {"t": time.time(), "kind": kind, "person": person, "text": text, "alert": alert}
        self.events.append(ev)
        if self.events_f:
            self.events_f.write(json.dumps(ev) + "\n"); self.events_f.flush()
        return ev

    def open_alert(self, person: str, kind: str, reason: str, severity: str):
        if any(a["person"] == person and a["kind"] == kind and a["state"] != "resolved" for a in self.alerts.values()):
            return
        aid = f"A{self._next}"; self._next += 1
        self.alerts[aid] = {"id": aid, "person": person, "kind": kind, "reason": reason, "severity": severity,
                            "state": "open", "opened": time.time(), "updated": time.time()}
        self.log("alert", person, reason, aid)

    def place(self, person: str, lat: float, lon: float):
        p = self.people[person]
        if p.source == "SYNTHETIC" or not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            return
        first = p.loc is None
        p.lat, p.lon, p.loc, p.loc_t = lat, lon, "corrected", time.time()
        p.velocity_n = p.velocity_e = 0.0
        self.log("action", person, f"{p.name} {'placed' if first else 'moved'} on map by operator")

    def set_origin(self, lat: float, lon: float, source: str):
        if source not in ("browser", "manual") or not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            return
        self.origin, self.origin_source = (lat, lon), source
        self.base = self.origin
        for p in self.people.values():
            if p.id.startswith("mesh_") and p.loc is None:
                p.lat, p.lon, p.loc, p.loc_t = lat, lon, "origin", time.time()
                p.velocity_n = p.velocity_e = 0.0
        self.log("action", None, f"Command origin set from {source}")

    def raise_distress(self, person: str, how: str):
        label = "Distress button pressed" if how == "button" else "Distress marked by operator"
        self.open_alert(person, "distress", label, "critical")

    def act(self, aid: str, action: str, note: str = ""):
        a = self.alerts.get(aid)
        if not a:
            return
        state = {"request_check": "check requested", "acknowledge": "acknowledged", "resolve": "resolved"}.get(action)
        if not state:
            return
        a["state"], a["updated"] = state, time.time()
        if note:
            a["note"] = note
        name = self.people[a["person"]].name if a["person"] in self.people else a["person"]
        text = {"request_check": f"Check requested for {name}", "acknowledge": f"{name} acknowledged",
                "resolve": f"Resolved: {note or a['reason']}"}[action]
        self.log("action", a["person"], text, aid)

    def status(self, p: Person, now: float, dev: float | None) -> str:
        if any(a["person"] == p.id and a["kind"] == "distress" and a["state"] != "resolved" for a in self.alerts.values()):
            return "distress"
        q = p.quality(now)
        if q == "no contact" or (q == "no data" and not p.id.startswith("mesh_")):
            return q
        if dev is not None and p.dev_since and now - p.dev_since >= DEVIATION_HOLD_S:
            return "deviation"
        return "ok"

    def tick(self) -> dict:
        now = time.time()
        rows = []
        for p in self.people.values():
            mv = p.movement()
            fresh = now - p.last_sample <= STALE_S and p.connected
            hr_fresh = fresh and now - p.hr_t <= STALE_S
            q = p.quality(now)
            activity = p.motion_activity(now, mv) if fresh and p.imu else p.activity if p.id.startswith("mesh_") else None
            if activity is None and p.id.startswith("mesh_"):
                activity = p.activity
            if p.imu:
                if activity != p.activity:
                    p.activity, p.activity_since, p.dev_since = activity, now, None
                stable_s = now - p.activity_since
                if activity and stable_s >= 30 and q == "good" and hr_fresh and p.hr is not None and p.hr_t != p.activity_hr_t and activity not in p.activity_baselines:
                    p.activity_hr_t = p.hr_t
                    samples = p.activity_samples[activity]
                    samples.append(p.hr)
                    if len(samples) == 20:
                        # shortcut: first stable 20 readings set the baseline; validate activity thresholds and HR range on hardware.
                        p.activity_baselines[activity] = statistics.median(samples)
                exp = p.activity_baselines.get(activity) if activity and stable_s >= 30 else None
            else:
                exp = p.model.step(mv, TICK_S) if fresh else p.model.state
            dev = None
            if q == "good" and hr_fresh and p.hr is not None and exp is not None:
                dev = p.hr - exp
                if abs(dev) > DEVIATION_BPM:
                    p.dev_since = p.dev_since or now
                else:
                    p.dev_since = None
            else:
                p.dev_since = None
            if q == "no data" and p.last_sample and not p.id.startswith("mesh_"):
                self.open_alert(p.id, "stale", "No data from sensor", "device")
            elif q == "no contact":
                self.open_alert(p.id, "contact", "Sensor lost skin contact", "device")
            for a in self.alerts.values():  # sensor problems that went away stay queued, marked cleared
                if a["person"] == p.id and a["state"] != "resolved" and not a.get("cleared") and \
                        (a["kind"] == "stale" and q != "no data" or a["kind"] == "contact" and q not in ("no data", "no contact")):
                    a["cleared"] = now
                    self.log("system", p.id, f"{p.name}: data resumed" if a["kind"] == "stale" else f"{p.name}: contact restored", a["id"])
            if p.dev_since and now - p.dev_since >= DEVIATION_HOLD_S:
                word = "above" if dev > 0 else "below"
                reference = f"{activity} baseline" if p.imu else "expected"
                self.open_alert(p.id, "deviation", f"HR {abs(dev):.0f} bpm {word} {reference} for {DEVIATION_HOLD_S}+ s", "warning")
            st = self.status(p, now, dev)
            display_hr = p.hr if hr_fresh else None
            if p.source == "LIVE":
                display_hr = round(statistics.fmean(p.hr_window), 1) if p.hr_window and (p.id.startswith("mesh_") or now - p.hr_t <= HR_HOLD_S) else None
            row = {"t": round(now, 1), "hr": display_hr, "exp": round(exp, 1) if exp else None,
                   "mv": round(mv) if mv is not None and fresh else None,
                   "rmssd": round(p.rmssd(), 1) if p.rmssd() and fresh else None}
            if p.imu:
                row.update(activity=activity, baselines={k: round(v, 1) for k, v in p.activity_baselines.items()},
                           baseline_samples=len(p.activity_samples[activity]) if activity else 0)
            if p.acc and (p.id.startswith("mesh_") or now - p.acc[-1][0] <= 2):
                row["acc_xyz"] = [round(v) for v in p.acc_xyz]
                row["acc_mag"] = round(p.acc[-1][1])
            if p.imu_fresh(now) or (p.id.startswith("mesh_") and p.gyro_xyz):
                rot = p.rotation()
                row.update(rot=round(rot) if rot is not None else None,
                           gyro_xyz=[round(v, 1) for v in p.gyro_xyz] if p.gyro_xyz else None,
                           temp=round(p.imu_temp, 1) if p.imu_temp is not None else None)
            if p.loc:
                row.update(lat=round(p.lat, 6), lon=round(p.lon, 6))
            if p.id.startswith("mesh_"):
                row["speed_m_s"] = round(math.hypot(p.velocity_n, p.velocity_e), 2) if p.accel_bias is not None and fresh else None
                row["location_calibration"] = "ready" if p.accel_bias is not None else "calibrating" if p.calibration_start_ms is not None and fresh else "hold still"
                row["location_calibration_remaining_s"] = max(0, math.ceil((30000 - (p.uptime_seen - p.calibration_start_ms)) / 1000)) if p.accel_bias is None and p.calibration_start_ms is not None and p.uptime_seen is not None else None
                for key in ("hr", "mv", "rmssd", "activity", "acc_xyz", "acc_mag", "rot", "gyro_xyz", "temp"):
                    value = row.get(key)
                    previous = p.display.get(key)
                    if value is None:
                        if previous is not None:
                            row[key] = previous
                    elif p.last_sample != p.display_t:
                        if key != "hr" and isinstance(value, (int, float)) and isinstance(previous, (int, float)):
                            value = round((value + previous) / 2, 1)
                        elif isinstance(value, list) and isinstance(previous, list):
                            value = [round((a + b) / 2, 1) for a, b in zip(value, previous)]
                        row[key] = p.display[key] = value
                p.display_t = p.last_sample
            p.history.append(row)
            if self.samples_w:
                self.samples_w.writerow([row["t"], p.id, p.source, row["hr"], row["exp"], row["mv"], row["rmssd"], row.get("rot"), row.get("temp"), q, st,
                                         row.get("lat"), row.get("lon"), p.loc, activity])
            signal = None
            if p.id.startswith("mesh_") and p.last_sample:
                age = max(now - p.last_sample, now - p.hr_t) if p.hr_t else 35
                signal = max(0, min(100, round(100 * (1 - max(0, age - 5) / 30))))
            rows.append({"id": p.id, **row, "quality": q, "status": st, "loc": p.loc, "loc_t": p.loc_t,
                         "heading": round(p.heading) if p.heading_set else None,
                         "loc_quality": "low" if p.loc == "estimated" and p.heading_set else "uncalibrated" if p.loc == "estimated" else None,
                         "age": round(now - p.last_sample, 1) if p.last_sample else None,
                         "signal": signal})
        if self.samples:
            self.samples.flush()
        return {"type": "tick", "t": now, "people": rows, "alerts": list(self.alerts.values()),
                "origin": self.origin, "origin_source": self.origin_source}

    def snapshot(self) -> dict:
        return {
            "type": "snapshot", "t": time.time(), "started": self.started, "model": MODEL_VERSION,
            "session": self.session_dir.name if self.session_dir else None, "base": self.base, "origin": self.origin,
            "origin_source": self.origin_source, "stale_s": STALE_S, "deviation_bpm": DEVIATION_BPM,
            "people": [{"id": p.id, "name": p.name, "role": p.role, "source": p.source, "device": p.device, "imu": p.imu,
                        "loc": p.loc, "loc_t": p.loc_t, "loc_quality": "low" if p.loc == "estimated" and p.heading_set else "uncalibrated" if p.loc == "estimated" else None,
                        "heading": p.heading if p.heading_set else None,
                        "history": list(p.history)} for p in self.people.values()],
            "alerts": list(self.alerts.values()), "events": self.events[-200:],
        }


# ---------------------------------------------------------------- web

async def index(_):
    return web.FileResponse(ROOT / "static" / "index.html")


async def ws_handler(request):
    team: Team = request.app["team"]
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    team.clients.add(ws)
    await ws.send_json(team.snapshot())
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            m = json.loads(msg.data)
            if m.get("type") == "action":
                team.act(m["alert"], m["action"], m.get("note", "")[:200])
            elif m.get("type") in ("place", "origin", "heading"):
                try:
                    if m["type"] == "place" and m.get("person") in team.people:
                        team.place(m["person"], float(m["lat"]), float(m["lon"]))
                    elif m["type"] == "origin":
                        team.set_origin(float(m["lat"]), float(m["lon"]), m.get("source"))
                    elif m["type"] == "heading" and m.get("person") in team.people and m["person"].startswith("mesh_"):
                        heading = float(m["degrees"])
                        if math.isfinite(heading) and 0 <= heading < 360:
                            p = team.people[m["person"]]
                            p.heading, p.heading_set = heading, True
                            p.velocity_n = p.velocity_e = 0.0
                            team.log("action", p.id, f"{p.name} heading calibrated to {heading:.0f}°")
                except (KeyError, TypeError, ValueError):
                    continue
            elif m.get("type") == "distress" and m.get("person") in team.people:
                team.raise_distress(m["person"], "operator")
    finally:
        team.clients.discard(ws)
    return ws


async def broadcaster(app):
    team: Team = app["team"]
    sent = len(team.events)
    while True:
        await asyncio.sleep(TICK_S)
        msg = team.tick()
        msg["events"] = team.events[sent:]
        sent = len(team.events)
        for ws in list(team.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                team.clients.discard(ws)


def build(args) -> web.Application:
    lat, lon = (float(x) for x in args.base.split(","))
    if args.mesh_api and args.imu:
        raise SystemExit("--mesh-api cannot be combined with --imu")
    team = Team(None if args.mesh_api else ROOT / "sessions" / time.strftime("%Y%m%d-%H%M%S"), (lat, lon))
    sources = []
    if args.live:
        p = Person("live", args.live_name, "Wearer", "LIVE", device="Polar Verity Sense")
        team.add(p); sources.append(LiveSource(p, args.address))
    if args.replay:
        prefix = Path(args.replay)
        p = Person("replay", "Recorded session", "Wearer", "REPLAY", device=f"{prefix.name} (Polar Verity Sense)")
        team.add(p); sources.append(ReplaySource(p, prefix))
    if args.imu:
        if "live" in team.people:  # same wearer as the Polar: one person, two devices
            p, owns = team.people["live"], False
            p.device += " + LSM6DSOX"
        else:
            src = "REPLAY" if Path(args.imu).is_file() else "LIVE"
            p, owns = Person("imu", args.live_name, "Wearer", src, device=f"LSM6DSOX ({Path(args.imu).name})"), True
            team.add(p)
        sources.append(ImuSource(p, args.imu, team.session_dir, owns))
    if args.mesh_api:
        mesh_people = {}
        for spec in args.mesh_node:
            name, sep, node_id = spec.partition("=")
            node_id = node_id.upper().replace(":", "")
            if not sep or not name or len(node_id) != 12 or any(c not in "0123456789ABCDEF" for c in node_id) or node_id in mesh_people:
                raise SystemExit(f"Invalid --mesh-node {spec!r}; expected NAME=12_HEX_DIGIT_NODE_ID")
            p = Person(f"mesh_{node_id}", name, "Wearer", "LIVE", device=f"Mesh node {node_id}", imu=True)
            team.add(p)
            mesh_people[node_id] = p
        if not mesh_people:
            raise SystemExit("--mesh-api requires at least one --mesh-node NAME=NODE_ID")
        sources.append(MeshApiSource(mesh_people, args.mesh_api))
    roster = [("Alvarez", "Rifleman", "elevated"), ("Chen", "Medic", None), ("Okafor", "Radio", "distress"),
              ("Patel", "Team lead", "dropout"), ("Brooks", "Rifleman", None), ("Novak", "Grenadier", None)]
    for i, (name, role, script) in enumerate(roster[:args.synthetic]):
        p = Person(f"s{i + 1}", name, role, "SYNTHETIC", device="simulated")
        team.add(p); sources.append(SyntheticSource(p, seed=i + 7, script=script))

    app = web.Application()
    app["team"] = team
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)

    async def start(app):
        app["tasks"] = [asyncio.create_task(s.run(team)) for s in sources] + [asyncio.create_task(broadcaster(app))]

    async def stop(app):
        for t in app["tasks"]:
            t.cancel()
        if team.samples:
            team.samples.close()
        if team.events_f:
            team.events_f.close()

    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    if team.session_dir:
        print(f"Recording to {team.session_dir}")
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true", help="stream a Polar Verity Sense as a live person")
    p.add_argument("--live-name", default="Live wearer")
    p.add_argument("--imu", help="LSM6DSOX serial port (e.g. /dev/cu.usbmodem1101) or a saved imu_*.csv to replay")
    p.add_argument("--mesh-api", help="local mesh gateway URL, e.g. http://127.0.0.1:8766/api/nodes; disables session recording")
    p.add_argument("--mesh-node", action="append", default=[], help="wearer and Wi-Fi node ID, e.g. Alex=206EF1316774; repeat per node")
    p.add_argument("--address", help="BLE address of the sensor (default: first 'Polar…' found)")
    default_replay = ROOT.parent / "sensor" / "test4"
    p.add_argument("--replay", default=str(default_replay) if (default_replay.parent / "test4_hr.csv").exists() else None,
                   help="recording prefix to replay, e.g. sensor/test4 ('' to disable)")
    p.add_argument("--synthetic", type=int, default=5, help="number of simulated teammates (0-6)")
    p.add_argument("--base", default="32.7157,-117.1611", help="map center as LAT,LON (default: San Diego)")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    web.run_app(build(args), host="127.0.0.1", port=args.port, print=lambda *_: print(f"Dashboard: http://127.0.0.1:{args.port}"))


if __name__ == "__main__":
    main()
