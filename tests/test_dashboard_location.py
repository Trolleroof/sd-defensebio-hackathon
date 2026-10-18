import asyncio
from contextlib import suppress
import math
import time
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dashboard.server import MeshApiSource, Person, Team, ws_handler


class DashboardLocationTest(unittest.TestCase):
    def test_activity_baselines_and_sustained_alert(self):
        team = Team(None, (0, 0))
        person = Person("mesh_1", "Wearer", "Wearer", "LIVE", imu=True, connected=True)
        team.add(person)
        now = [1000.0]

        def sample(acceleration, rotation, heart_rate):
            for z in acceleration:
                person.add_imu((0, 0, z), (rotation, 0, 0), 25)
            person.add_hr(heart_rate)
            row = team.tick()["people"][0]
            now[0] += 1
            return row

        with patch("dashboard.server.time.time", side_effect=lambda: now[0]):
            for _ in range(52):
                row = sample((1000, 1000), 2, 70)
            self.assertEqual((row["activity"], row["baselines"]["still"], row["exp"]), ("still", 70, 70))
            for _ in range(52):
                row = sample((850, 1150), 40, 95)
            self.assertEqual((row["activity"], row["baselines"]["walking"], row["exp"]), ("walking", 95, 95))
            for _ in range(52):
                row = sample((600, 1400), 100, 125)
            self.assertEqual((row["activity"], row["baselines"]["running"], row["exp"]), ("running", 125, 125))
            for _ in range(31):
                row = sample((600, 1400), 100, 155)
            self.assertEqual(row["status"], "deviation")
            self.assertEqual(len([a for a in team.alerts.values() if a["kind"] == "deviation"]), 1)
            self.assertEqual(person.activity_baselines["running"], 125)
            for _ in range(3):
                row = sample((1000, 1000), 2, 70)
            self.assertIsNone(row["exp"])
            self.assertEqual(row["status"], "ok")

    def test_origin_steps_turn_and_correction(self):
        team = Team(None, (32.7157, -117.1611))
        person = Person("mesh_1", "Wearer", "Wearer", "LIVE", imu=True)
        team.add(person)
        team.set_origin(32.0, -117.0, "browser")
        self.assertEqual((person.lat, person.lon, person.loc), (32.0, -117.0, "origin"))

        person.heading, person.heading_set = 90, True
        person.advance(100, 1000, 0)
        person.advance(102, 2000, 0)
        self.assertEqual(person.loc, "estimated")
        self.assertAlmostEqual(person.lat, 32.0, places=6)
        self.assertGreater(person.lon, -117.0)

        person.advance(104, 3000, math.pi / 2)
        self.assertGreater(person.lat, 32.0)
        team.place(person.id, 32.1, -117.1)
        self.assertEqual(person.loc, "corrected")
        person.advance(105, 4000, 0)
        self.assertEqual(person.loc, "estimated")
        self.assertGreater(person.lat, 32.1)

    def test_reboot_gap_and_invalid_origin(self):
        team = Team(None, (0, 0))
        person = Person("mesh_2", "Wearer", "Wearer", "LIVE", imu=True)
        team.add(person)
        team.set_origin(float("nan"), 10, "browser")
        self.assertIsNone(team.origin)
        team.set_origin(0, 0, "manual")
        person.advance(65535, 1000, 0)
        person.advance(1, 2000, 0)
        self.assertAlmostEqual(person.lat, math.degrees(1.4 / 6371000))
        before = person.lat
        person.advance(500, 10000, 0)
        person.advance(1, 1000, 0)
        self.assertEqual(person.lat, before)


class DashboardOriginApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_browser_origin_and_heading_messages(self):
        team = Team(None, (0, 0))
        person = Person("mesh_1", "Wearer", "Wearer", "LIVE", imu=True)
        team.add(person)
        app = web.Application()
        app["team"] = team
        app.router.add_get("/ws", ws_handler)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/ws")
            self.assertEqual((await ws.receive_json())["origin"], None)
            await ws.send_json({"type": "origin", "source": "browser", "lat": 32.0, "lon": -117.0})
            await ws.send_json({"type": "heading", "person": person.id, "degrees": 90})
            await ws.send_json({"type": "origin", "source": "manual", "lat": 999, "lon": 0})
            await ws.close()
        self.assertEqual(team.origin, (32.0, -117.0))
        self.assertEqual((person.loc, person.heading, person.heading_set), ("origin", 90, True))

    async def test_mesh_api_packets_advance_from_browser_origin(self):
        node_id = "206EF1316774"
        team = Team(None, (0, 0))
        person = Person(f"mesh_{node_id}", "Wearer", "Wearer", "LIVE", imu=True)
        team.add(person)
        packet = {"sequence": 1, "uptime_ms": 1000, "sensor_ok": True,
                  "accel_m_s2": [0, 0, 9.80665], "gyro_rad_s": [0, 0, 0],
                  "yaw_delta_rad": 0, "steps": 10, "hr_bpm": 80}

        async def nodes(_):
            return web.json_response({"nodes": {node_id: {"online": True, "received_at": time.time(), "latest": packet}}})

        app = web.Application()
        app["team"] = team
        app.router.add_get("/ws", ws_handler)
        app.router.add_get("/api/nodes", nodes)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()
            await ws.send_json({"type": "origin", "source": "browser", "lat": 32, "lon": -117})
            await ws.send_json({"type": "heading", "person": person.id, "degrees": 90})
            source = MeshApiSource({node_id: person}, str(client.make_url("/api/nodes")))
            task = asyncio.create_task(source.run(team))
            try:
                for _ in range(40):
                    if person.steps_seen == 10:
                        break
                    await asyncio.sleep(0.05)
                self.assertEqual(person.steps_seen, 10)
                packet = {**packet, "sequence": 2, "uptime_ms": 2000, "steps": 12}
                for _ in range(40):
                    if person.loc == "estimated":
                        break
                    await asyncio.sleep(0.05)
                row = team.tick()["people"][0]
                self.assertEqual((row["loc"], row["loc_quality"]), ("estimated", "low"))
                self.assertGreater(row["lon"], -117)
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                await ws.close()


if __name__ == "__main__":
    unittest.main()
