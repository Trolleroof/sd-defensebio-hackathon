"""Durable best-effort outbox and Foundry stream publisher for mesh data."""

import json
import math
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


MAX_AGE_S = 3600
MAX_BYTES = 100_000_000


def finite(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def telemetry_record(packet, name, received_at):
    """Flatten one validated mesh packet to the published stream schema."""
    if (packet.get("schema_version") != 2 or type(packet.get("sequence")) is not int
            or type(packet.get("uptime_ms")) is not int or packet["sequence"] < 0 or packet["uptime_ms"] < 0):
        return None
    acc, gyro = packet.get("accel_m_s2"), packet.get("gyro_rad_s")
    acc = acc if isinstance(acc, list) and len(acc) == 3 else [None] * 3
    gyro = gyro if isinstance(gyro, list) and len(gyro) == 3 else [None] * 3
    hr = finite(packet.get("hr_bpm"))
    return {
        "node_id": packet["node_id"], "wearer_name": name, "received_at_ms": round(received_at * 1000),
        "schema_version": 2, "sequence": packet["sequence"], "uptime_ms": packet["uptime_ms"],
        "root_received_ms": packet.get("received_ms") if type(packet.get("received_ms")) is int and packet["received_ms"] >= 0 else None,
        "sensor_ok": packet.get("sensor_ok") is True, "hr_valid": hr is not None and hr > 0,
        "hr_bpm": hr if hr is not None and hr > 0 else None,
        "accel_x_m_s2": finite(acc[0]), "accel_y_m_s2": finite(acc[1]), "accel_z_m_s2": finite(acc[2]),
        "gyro_x_rad_s": finite(gyro[0]), "gyro_y_rad_s": finite(gyro[1]), "gyro_z_rad_s": finite(gyro[2]),
        "yaw_delta_rad": finite(packet.get("yaw_delta_rad")), "imu_temp_c": finite(packet.get("imu_temp_c")),
        "steps": packet.get("steps") if type(packet.get("steps")) is int and packet["steps"] >= 0 else None,
        "hop_count": packet.get("hop_count") if type(packet.get("hop_count")) is int and packet["hop_count"] >= 0 else None,
        "wake": packet.get("wake") if type(packet.get("wake")) is bool else None,
    }


class Outbox:
    def __init__(self, path, max_age_s=MAX_AGE_S, max_bytes=MAX_BYTES):
        self.path = Path(path)
        self.max_age_s, self.max_bytes = max_age_s, max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT NOT NULL, created REAL NOT NULL, payload TEXT NOT NULL, size INTEGER NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS records_order ON records(created)")
            db.execute("CREATE TABLE IF NOT EXISTS counters (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
            db.execute("INSERT OR IGNORE INTO counters VALUES ('dropped', 0)")
            db.execute("INSERT OR IGNORE INTO counters VALUES ('pending_bytes', 0)")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=0.25)

    @contextmanager
    def _db(self):
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    def enqueue(self, kind, record):
        if kind not in ("telemetry", "events"):
            raise ValueError("unknown stream kind")
        event_id = uuid.uuid4().hex
        payload = json.dumps({**record, "event_id": event_id}, allow_nan=False, separators=(",", ":"))
        size = len(payload.encode("utf-8"))
        with self._db() as db:
            db.execute("INSERT INTO records VALUES (?, ?, ?, ?, ?)", (event_id, kind, time.time(), payload, size))
            db.execute("UPDATE counters SET value = value + ? WHERE key = 'pending_bytes'", (size,))
            self._prune(db)
        return event_id

    def _prune(self, db):
        cutoff = time.time() - self.max_age_s
        expired, expired_bytes = db.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM records WHERE created < ?", (cutoff,)).fetchone()
        db.execute("DELETE FROM records WHERE created < ?", (cutoff,))
        excess = db.execute("SELECT value FROM counters WHERE key = 'pending_bytes'").fetchone()[0] - expired_bytes - self.max_bytes
        ids, capacity_bytes = [], 0
        if excess > 0:
            for event_id, size in db.execute("SELECT id, size FROM records ORDER BY created, rowid"):
                ids.append((event_id,))
                capacity_bytes += size
                excess -= size
                if excess <= 0:
                    break
            db.executemany("DELETE FROM records WHERE id = ?", ids)
        db.execute("UPDATE counters SET value = value - ? WHERE key = 'pending_bytes'", (expired_bytes + capacity_bytes,))
        db.execute("UPDATE counters SET value = value + ? WHERE key = 'dropped'", (expired + len(ids),))

    def prune(self):
        with self._db() as db:
            self._prune(db)

    def pending(self, kind, limit=100):
        with self._db() as db:
            return [(event_id, json.loads(payload)) for event_id, payload in db.execute(
                "SELECT id, payload FROM records WHERE kind = ? ORDER BY created, rowid LIMIT ?", (kind, limit))]

    def ack(self, ids):
        if not ids:
            return
        with self._db() as db:
            marks = ",".join("?" for _ in ids)
            size = db.execute(f"SELECT COALESCE(SUM(size), 0) FROM records WHERE id IN ({marks})", ids).fetchone()[0]
            db.executemany("DELETE FROM records WHERE id = ?", [(event_id,) for event_id in ids])
            db.execute("UPDATE counters SET value = value - ? WHERE key = 'pending_bytes'", (size,))

    def stats(self):
        with self._db() as db:
            count, oldest = db.execute("SELECT COUNT(*), MIN(created) FROM records").fetchone()
            size = db.execute("SELECT value FROM counters WHERE key = 'pending_bytes'").fetchone()[0]
            dropped = db.execute("SELECT value FROM counters WHERE key = 'dropped'").fetchone()[0]
        return {"queued": count, "queued_bytes": size, "oldest_age_s": round(time.time() - oldest, 1) if oldest else None, "dropped": dropped}


class Publisher:
    def __init__(self, outbox, host, telemetry_rid, events_rid, client_id, client_secret, scope="api:use-streams-write"):
        parsed = urlparse(host)
        if (not parsed.hostname or parsed.path not in ("", "/") or parsed.query or parsed.fragment
                or (parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1")))):
            raise ValueError("Foundry host must be an HTTPS origin (HTTP is allowed for a localhost mock)")
        self.outbox, self.host = outbox, host.rstrip("/")
        self.rids = {"telemetry": telemetry_rid, "events": events_rid}
        self.client_id, self.client_secret, self.scope = client_id, client_secret, scope
        self.token, self.expires_at = None, 0
        self.last_success, self.error = None, None

    def _token(self):
        if self.token and time.time() < self.expires_at - 30:
            return self.token
        body = urlencode({"grant_type": "client_credentials", "client_id": self.client_id,
                          "client_secret": self.client_secret, "scope": self.scope}).encode()
        request = Request(self.host + "/multipass/api/oauth2/token", data=body,
                          headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urlopen(request, timeout=5) as response:
            result = json.load(response)
        self.token = result["access_token"]
        self.expires_at = time.time() + float(result["expires_in"])
        return self.token

    def step(self):
        self.outbox.prune()
        for kind, rid in self.rids.items():
            batch = self.outbox.pending(kind)
            if not batch:
                continue
            body = json.dumps({"records": [record for _, record in batch]}, allow_nan=False).encode()
            url = f"{self.host}/api/v2/highScale/streams/datasets/{rid}/streams/master/publishRecords"
            for attempt in range(2):
                request = Request(url, data=body, headers={"Content-Type": "application/json",
                                                          "Authorization": "Bearer " + self._token()})
                try:
                    with urlopen(request, timeout=5):
                        pass
                    self.outbox.ack([event_id for event_id, _ in batch])
                    self.last_success, self.error = time.time(), None
                    break
                except HTTPError as error:
                    if error.code == 401 and attempt == 0:
                        self.token = None
                        continue
                    raise

    def run(self, stop):
        delay = 1
        while not stop.is_set():
            try:
                self.step()
                self.error = None
                delay = 1
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"
                delay = min(delay * 2, 60)
            stop.wait(delay)
