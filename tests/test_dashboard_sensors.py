import asyncio
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dashboard.server import LiveSource, MeshApiSource, Person, Team
from sensor.lsm6dsox import parse_line
from sensor import polar_stream


def test_motion_uses_recent_samples_and_exposes_axes():
    person = Person("wearer", "Wearer", "Wearer", "LIVE", imu=True, connected=True)
    team = Team(None, (0, 0))
    team.add(person)
    person.add_imu((0, 0, 1000), (0, 0, 10), 26)
    person.add_imu((0, 0, 1100), (0, 0, -20), 26)
    row = team.tick()["people"][0]
    assert row["mv"] == 50
    assert row["rot"] == 15
    assert row["acc_xyz"] == [0, 0, 1100]
    assert row["gyro_xyz"] == [0, 0, -20]

    person.acc = type(person.acc)((t - 3, value) for t, value in person.acc)
    person.gyro = type(person.gyro)((t - 3, value) for t, value in person.gyro)
    person.imu_t = time.time() - 3
    row = team.tick()["people"][0]
    assert row["mv"] is None
    assert "rot" not in row
    assert "acc_xyz" not in row


def test_live_heart_rate_smooths_short_dropouts_and_expires():
    person = Person("live", "Wearer", "Wearer", "LIVE", imu=True, connected=True)
    team = Team(None, (0, 0))
    team.add(person)
    person.add_hr(80)
    person.hr_t -= 11
    person.add_imu((0, 0, 1000), (0, 0, 0), 26)
    row = team.tick()["people"][0]
    assert row["hr"] == 80
    assert row["quality"] == "no data"
    person.add_hr(120)
    assert team.tick()["people"][0]["hr"] == 100
    person.hr_t -= 31
    assert team.tick()["people"][0]["hr"] is None
    person.add_hr(90)
    assert team.tick()["people"][0]["hr"] == 90


def test_mesh_display_holds_values_while_signal_fades():
    person = Person("mesh_206EF1316774", "Wearer", "Wearer", "LIVE", imu=True, connected=True)
    team = Team(None, (0, 0))
    team.add(person)
    person.add_hr(80)
    person.add_imu((0, 0, 1000), (0, 0, 10), 26)
    first = team.tick()["people"][0]
    assert first["hr"] == 80 and first["acc_xyz"] == [0, 0, 1000]
    assert first["signal"] == 100

    person.connected = False
    person.hr_t -= 40
    person.last_sample -= 40
    person.imu_t -= 40
    held = team.tick()["people"][0]
    assert held["hr"] == 80 and held["acc_xyz"] == [0, 0, 1000]
    assert held["signal"] == 0 and held["status"] == "ok"
    assert not team.alerts

    person.connected = True
    person.add_hr(120)
    person.add_imu((0, 0, 1100), (0, 0, 20), 27)
    resumed = team.tick()["people"][0]
    assert resumed["hr"] == 100
    assert resumed["acc_xyz"] == [0.0, 0.0, 1050.0]
    assert resumed["signal"] == 100


def test_polar_reconnects_when_imu_continues_without_ble():
    class FakeClient:
        is_connected = True

        def __init__(self, _):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            self.is_connected = False

        async def start_notify(self, *_):
            pass

        async def write_gatt_char(self, *_, **__):
            pass

    async def find_device(_):
        return SimpleNamespace(name="Polar test", address="test")

    async def check():
        person = Person("live", "Wearer", "Wearer", "LIVE", imu=True)
        team = Team(None, (0, 0))
        source = LiveSource(person, None)
        task = asyncio.create_task(source.run(team))
        try:
            for _ in range(34):
                person.add_imu((0, 0, 1000), (0, 0, 0), 26)
                await asyncio.sleep(0.05)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert any("No Polar notifications" in event["text"] for event in team.events)

    with patch.dict(sys.modules, {"polar_stream": polar_stream}), \
         patch("bleak.BleakClient", FakeClient), \
         patch("dashboard.server.STALE_S", 0.1), \
         patch.object(polar_stream, "find_device", find_device):
        asyncio.run(check())


def test_nonfinite_imu_samples_are_ignored():
    assert parse_line("0,0,nan,0,0,0,25") is None
    assert parse_line("0,0,1000,0,inf,0,25") is None

    async def check():
        node_id = "206EF1316774"
        person = Person(f"mesh_{node_id}", "Wearer", "Wearer", "LIVE", imu=True)
        team = Team(None, (0, 0))
        team.add(person)
        packet = {"sequence": 1, "uptime_ms": 1000, "sensor_ok": True,
                  "accel_m_s2": [0, 0, float("nan")], "gyro_rad_s": [0, 0, 0],
                  "imu_temp_c": 25, "hr_bpm": float("inf")}

        async def nodes(_):
            return web.json_response({"nodes": {node_id: {"online": packet["sequence"] != 3,
                                                        "received_at": time.time() - (40 if packet["sequence"] == 3 else 0),
                                                        "latest": packet}}})

        app = web.Application()
        app.router.add_get("/api/nodes", nodes)
        async with TestClient(TestServer(app)) as client:
            source = MeshApiSource({node_id: person}, str(client.make_url("/api/nodes")))
            task = asyncio.create_task(source.run(team))
            try:
                await asyncio.sleep(0.2)
                assert person.hr is None and not person.acc
                packet = {**packet, "sequence": 2, "accel_m_s2": [0, 0, 9.80665],
                          "imu_temp_c": float("nan"), "hr_bpm": 80}
                for _ in range(10):
                    if person.hr == 80 and person.acc:
                        break
                    await asyncio.sleep(0.05)
                assert person.hr == 80 and person.acc and person.imu_temp is None
                packet = {**packet, "sequence": 3, "hr_bpm": 90}
                for _ in range(10):
                    if person.hr == 90:
                        break
                    await asyncio.sleep(0.05)
                row = team.tick()["people"][0]
                assert row["hr"] == 85 and row["signal"] == 0
                assert row["acc_xyz"] == [0, 0, 1000]
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    asyncio.run(check())
