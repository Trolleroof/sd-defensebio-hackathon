#!/usr/bin/env python3
"""Operator dashboard gateway: streams team physiology to the browser over a WebSocket.

Sources (each person is labeled with where their data comes from):
  LIVE       a Polar Verity Sense over BLE (reuses the decoders in sensor/polar_stream.py),
             and/or an LSM6DSOX IMU streamed over USB serial (sensor/lsm6dsox.py)
  REPLAY     a recorded session from sensor/<prefix>_{hr,ppi,acc}.csv, looped in real time
  SYNTHETIC  simulated teammates so the multi-person views have something to show

Every tick the gateway computes movement, HRV, an expected HR from a first-order kinetic
model, and attention items. Operator actions come back over the same socket and are saved
with the samples under dashboard/sessions/<timestamp>/.

Usage:
  python dashboard/server.py                          # replay sensor/test4 + synthetic team
  python dashboard/server.py --live                   # add the Polar sensor as a live person
  python dashboard/server.py --live --synthetic 0     # live sensor only
  python dashboard/server.py --replay sensor/test1    # different recording
  python dashboard/server.py --imu /dev/cu.usbmodem1101   # add the LSM6DSOX (gyro, accel, die temp)
  python dashboard/server.py --base 32.7157,-117.1611 # map center for simulated positions

Locations are never GPS here: simulated people move around --base, and the operator can place
LIVE/REPLAY people on the map by hand. Every position carries its source.
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

from aiohttp import WSMsgType, web

ROOT = Path(__file__).resolve().parent
TICK_S = 1.0
HISTORY_S = 600          # seconds of history kept per person and sent on connect
STALE_S = 10             # no samples for this long -> "no data"
DEVIATION_BPM = 20       # |measured - expected| above this ...
DEVIATION_HOLD_S = 30    # ... for this long -> attention item
MODEL_VERSION = "kinetic-v0 (not validated)"


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
    last_sample: float = 0.0          # wall time of the last sample from the device
    connected: bool = False
    contact: bool | None = None       # skin contact flag, when the sensor reports it
    acc: deque = field(default_factory=lambda: deque(maxlen=52 * 2))   # recent |a| in mG
    rr: deque = field(default_factory=lambda: deque(maxlen=60))         # recent good intervals, ms
    ppi_seen: deque = field(default_factory=lambda: deque(maxlen=20))   # 1 = rejected
    model: Kinetic = field(default_factory=Kinetic)
    early_hr: list = field(default_factory=list)
    dev_since: float | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=int(HISTORY_S / TICK_S)))
    lat: float | None = None
    lon: float | None = None
    loc: str | None = None            # "simulated" | "placed" (by the operator); None = no location
    loc_t: float | None = None
    imu: bool = False                 # has a gyroscope/IMU source
    gyro: deque = field(default_factory=lambda: deque(maxlen=52 * 2))  # recent |w| in dps
    imu_temp: float | None = None     # IMU die temperature, °C (not body temperature)
    imu_t: float = 0.0                # wall time of the last IMU sample

    def add_hr(self, bpm: float, t: float | None = None):
        self.hr = bpm
        self.last_sample = t or time.time()
        if self.model.rest is None:
            self.early_hr.append(bpm)
            if len(self.early_hr) >= 20:  # resting level from the first readings
                self.model.rest = statistics.quantiles(self.early_hr, n=5)[0]

    def add_acc(self, x: float, y: float, z: float):
        self.acc.append(math.sqrt(x * x + y * y + z * z))
        self.last_sample = time.time()

    def add_imu(self, acc: tuple, gyro: tuple, temp: float | None):
        """One LSM6DSOX sample: acc in mG, gyro in dps, die temperature in °C."""
        self.add_acc(*acc)
        self.gyro.append(math.sqrt(sum(g * g for g in gyro)))
        if temp is not None:
            self.imu_temp = temp
        self.imu_t = self.last_sample

    def imu_fresh(self, now: float | None = None) -> bool:
        return self.imu and (now or time.time()) - self.imu_t <= STALE_S

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
        return statistics.pstdev(self.acc) if len(self.acc) > 5 else None

    def rotation(self) -> float | None:
        """Mean angular speed over the last ~2 s, dps."""
        return statistics.fmean(self.gyro) if len(self.gyro) > 5 else None

    def rmssd(self) -> float | None:
        rr = list(self.rr)[-30:]
        if len(rr) < 5:
            return None
        return math.sqrt(statistics.fmean((b - a) ** 2 for a, b in zip(rr, rr[1:])))

    def quality(self, now: float) -> str:
        if not self.connected or now - self.last_sample > STALE_S:
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

    async def run(self, team):
        sys.path.insert(0, str(ROOT.parent / "sensor"))
        import polar_stream as ps
        from bleak import BleakClient

        def on_hr(_, data):
            bpm, rr = ps.parse_hr(data)
            if bpm:
                self.p.add_hr(bpm)

        def on_pmd(_, data):
            kind, samples = ps.decode_pmd(data)
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
                    await client.start_notify(ps.HR_MEASUREMENT, on_hr)
                    await client.start_notify(ps.PMD_DATA, on_pmd)
                    for k in (ps.PMD_ACC, ps.PMD_PPI):
                        await client.write_gatt_char(ps.PMD_CONTROL, ps.START_CMD[k], response=True)
                        await asyncio.sleep(0.5)
                    self.p.connected = True
                    team.log("system", self.p.id, "Sensor connected")
                    while client.is_connected:
                        await asyncio.sleep(0.5)
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


class MeshSource:
    """Read the root Nano's JSON lines and save the original telemetry for replay."""

    def __init__(self, people: dict[str, Person], port: str, session_dir: Path):
        self.people, self.port = people, port
        self.out = open(session_dir / "mesh.jsonl", "a")

    def take(self, line: str):
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                return
            self.out.write(json.dumps({"t": time.time(), **data}) + "\n")
            self.out.flush()
            if data.get("type") != "telemetry":
                return
            p = self.people.get(data.get("node_id"))
            if p is None:
                return
            p.connected = True
            p.last_sample = time.time()
            bpm = data.get("hr_bpm")
            if isinstance(bpm, (int, float)) and bpm > 0:
                p.add_hr(bpm)
            else:
                p.hr = None
            acc = data.get("accel_m_s2")
            gyro = data.get("gyro_rad_s")
            if data.get("sensor_ok") and isinstance(acc, list) and len(acc) == 3 and isinstance(gyro, list) and len(gyro) == 3 and all(isinstance(v, (int, float)) for v in acc + gyro):
                p.add_imu(tuple(v * 1000 / 9.80665 for v in acc), tuple(v * 180 / math.pi for v in gyro), data.get("imu_temp_c"))
        except (ValueError, TypeError, KeyError):
            return

    async def run(self, team):
        import serial
        last_error = None
        while True:
            try:
                port = await asyncio.to_thread(serial.Serial, self.port, 115200, timeout=1)
                team.log("system", None, f"Mesh connected on {self.port}")
                last_error = None
                with port:
                    while True:
                        raw = await asyncio.to_thread(port.readline)
                        if raw:
                            self.take(raw.decode(errors="ignore"))
            except Exception as e:
                for p in self.people.values():
                    p.connected = False
                msg = f"Mesh unavailable: {e}"
                if msg != last_error:
                    team.log("system", None, msg)
                last_error = msg
                await asyncio.sleep(3)


# ---------------------------------------------------------------- team, attention, actions

class Team:
    def __init__(self, session_dir: Path, base: tuple[float, float]):
        self.base = base
        self.people: dict[str, Person] = {}
        self.alerts: dict[str, dict] = {}
        self.events: list[dict] = []
        self.clients: set[web.WebSocketResponse] = set()
        self.started = time.time()
        self.session_dir = session_dir
        session_dir.mkdir(parents=True, exist_ok=True)
        self.samples = open(session_dir / "samples.csv", "w", newline="")
        self.samples_w = csv.writer(self.samples)
        self.samples_w.writerow(["t", "person", "source", "hr", "expected", "movement_mg", "rmssd_ms", "rot_dps", "imu_temp_c", "quality", "status", "lat", "lon", "loc_source"])
        self.events_f = open(session_dir / "events.jsonl", "a")
        self._next = 1

    def add(self, p: Person):
        self.people[p.id] = p

    def log(self, kind: str, person: str | None, text: str, alert: str | None = None) -> dict:
        ev = {"t": time.time(), "kind": kind, "person": person, "text": text, "alert": alert}
        self.events.append(ev)
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
        if p.source == "SYNTHETIC" or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return
        first = p.loc is None
        p.lat, p.lon, p.loc, p.loc_t = lat, lon, "placed", time.time()
        self.log("action", person, f"{p.name} {'placed' if first else 'moved'} on map by operator")

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
        if q in ("no data", "no contact"):
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
            exp = p.model.step(mv, TICK_S) if fresh else p.model.state
            dev = None
            if fresh and p.hr is not None and exp is not None:
                dev = p.hr - exp
                if abs(dev) > DEVIATION_BPM:
                    p.dev_since = p.dev_since or now
                else:
                    p.dev_since = None
            else:
                p.dev_since = None
            q = p.quality(now)
            if q == "no data" and p.last_sample:
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
                self.open_alert(p.id, "deviation", f"HR {abs(dev):.0f} bpm {word} expected for {DEVIATION_HOLD_S}+ s", "warning")
            st = self.status(p, now, dev)
            row = {"t": round(now, 1), "hr": p.hr if fresh else None, "exp": round(exp, 1) if exp else None,
                   "mv": round(mv) if mv is not None and fresh else None,
                   "rmssd": round(p.rmssd(), 1) if p.rmssd() and fresh else None}
            if p.imu_fresh(now):
                rot = p.rotation()
                row.update(rot=round(rot) if rot is not None else None,
                           temp=round(p.imu_temp, 1) if p.imu_temp is not None else None)
            if p.loc:
                row.update(lat=round(p.lat, 6), lon=round(p.lon, 6))
            p.history.append(row)
            self.samples_w.writerow([row["t"], p.id, p.source, row["hr"], row["exp"], row["mv"], row["rmssd"], row.get("rot"), row.get("temp"), q, st,
                                     row.get("lat"), row.get("lon"), p.loc])
            rows.append({"id": p.id, **row, "quality": q, "status": st, "loc": p.loc, "loc_t": p.loc_t,
                         "age": round(now - p.last_sample, 1) if p.last_sample else None})
        self.samples.flush()
        return {"type": "tick", "t": now, "people": rows, "alerts": list(self.alerts.values())}

    def snapshot(self) -> dict:
        return {
            "type": "snapshot", "t": time.time(), "started": self.started, "model": MODEL_VERSION,
            "session": self.session_dir.name, "base": self.base, "stale_s": STALE_S, "deviation_bpm": DEVIATION_BPM,
            "people": [{"id": p.id, "name": p.name, "role": p.role, "source": p.source, "device": p.device, "imu": p.imu,
                        "loc": p.loc, "loc_t": p.loc_t, "history": list(p.history)} for p in self.people.values()],
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
            elif m.get("type") == "place" and m.get("person") in team.people:
                team.place(m["person"], float(m["lat"]), float(m["lon"]))
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
    team = Team(ROOT / "sessions" / time.strftime("%Y%m%d-%H%M%S"), (lat, lon))
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
    if args.mesh_port:
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
            raise SystemExit("--mesh-port requires at least one --mesh-node NAME=NODE_ID")
        sources.append(MeshSource(mesh_people, args.mesh_port, team.session_dir))
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
        for s in sources:
            if isinstance(s, MeshSource):
                s.out.close()
        team.samples.close(); team.events_f.close()

    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    print(f"Recording to {team.session_dir}")
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true", help="stream a Polar Verity Sense as a live person")
    p.add_argument("--live-name", default="Live wearer")
    p.add_argument("--imu", help="LSM6DSOX serial port (e.g. /dev/cu.usbmodem1101) or a saved imu_*.csv to replay")
    p.add_argument("--mesh-port", help="root Nano serial port, e.g. COM11")
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
