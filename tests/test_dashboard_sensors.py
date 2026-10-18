import asyncio
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from dashboard.server import LiveSource, Person, Team
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


def test_heart_rate_expires_while_imu_keeps_reporting():
    person = Person("live", "Wearer", "Wearer", "LIVE", imu=True, connected=True)
    team = Team(None, (0, 0))
    team.add(person)
    person.add_hr(80)
    person.hr_t -= 11
    person.add_imu((0, 0, 1000), (0, 0, 0), 26)
    row = team.tick()["people"][0]
    assert row["hr"] is None
    assert row["quality"] == "no data"


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
