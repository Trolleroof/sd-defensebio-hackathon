#!/usr/bin/env python3
"""Operator dashboard gateway: streams team physiology to the browser over a WebSocket.

Sources (each person is labeled with where their data comes from):
  LIVE       a Polar Verity Sense over BLE (reuses the decoders in sensor/polar_stream.py)
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
        self.rest = self.rng.uniform(56, 72)
        self.gain = self.rng.uniform(0.09, 0.15)
        self.hr = self.rest
        self.offset = self.rng.uniform(0, 300)

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
            level = self.LEVEL[act] * self.rng.uniform(0.8, 1.2)
            for _ in range(52):
                self.p.acc.append(1000 + self.rng.gauss(0, level))
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
            if kind == ps.PMD_ACC:
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


# ---------------------------------------------------------------- team, attention, actions

class Team:
    def __init__(self, session_dir: Path):
        self.people: dict[str, Person] = {}
        self.alerts: dict[str, dict] = {}
        self.events: list[dict] = []
        self.clients: set[web.WebSocketResponse] = set()
        self.started = time.time()
        self.session_dir = session_dir
        session_dir.mkdir(parents=True, exist_ok=True)
        self.samples = open(session_dir / "samples.csv", "w", newline="")
        self.samples_w = csv.writer(self.samples)
        self.samples_w.writerow(["t", "person", "source", "hr", "expected", "movement_mg", "rmssd_ms", "quality", "status"])
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
            p.history.append(row)
            self.samples_w.writerow([row["t"], p.id, p.source, row["hr"], row["exp"], row["mv"], row["rmssd"], q, st])
            rows.append({"id": p.id, **row, "quality": q, "status": st,
                         "age": round(now - p.last_sample, 1) if p.last_sample else None})
        self.samples.flush()
        return {"type": "tick", "t": now, "people": rows, "alerts": list(self.alerts.values())}

    def snapshot(self) -> dict:
        return {
            "type": "snapshot", "t": time.time(), "started": self.started, "model": MODEL_VERSION,
            "session": self.session_dir.name, "stale_s": STALE_S, "deviation_bpm": DEVIATION_BPM,
            "people": [{"id": p.id, "name": p.name, "role": p.role, "source": p.source, "device": p.device,
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
    team = Team(ROOT / "sessions" / time.strftime("%Y%m%d-%H%M%S"))
    sources = []
    if args.live:
        p = Person("live", args.live_name, "Wearer", "LIVE", device="Polar Verity Sense")
        team.add(p); sources.append(LiveSource(p, args.address))
    if args.replay:
        prefix = Path(args.replay)
        p = Person("replay", "Recorded session", "Wearer", "REPLAY", device=f"{prefix.name} (Polar Verity Sense)")
        team.add(p); sources.append(ReplaySource(p, prefix))
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
        team.samples.close(); team.events_f.close()

    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    print(f"Recording to {team.session_dir}")
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true", help="stream a Polar Verity Sense as a live person")
    p.add_argument("--live-name", default="Live wearer")
    p.add_argument("--address", help="BLE address of the sensor (default: first 'Polar…' found)")
    default_replay = ROOT.parent / "sensor" / "test4"
    p.add_argument("--replay", default=str(default_replay) if (default_replay.parent / "test4_hr.csv").exists() else None,
                   help="recording prefix to replay, e.g. sensor/test4 ('' to disable)")
    p.add_argument("--synthetic", type=int, default=5, help="number of simulated teammates (0-6)")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    web.run_app(build(args), host="127.0.0.1", port=args.port, print=lambda *_: print(f"Dashboard: http://127.0.0.1:{args.port}"))


if __name__ == "__main__":
    main()
