import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from dashboard import mesh_api
from dashboard.foundry_bridge import Outbox, Publisher
from dashboard.server import Person, Team


class MockFoundry(BaseHTTPRequestHandler):
    tokens = 0
    received = []
    fail_code = None

    def log_message(self, *_):
        pass

    def do_POST(self):
        if self.path == "/multipass/api/oauth2/token":
            self.__class__.tokens += 1
            self.send_response(200)
            body = json.dumps({"access_token": f"token-{self.tokens}", "expires_in": 3600}).encode()
        else:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.__class__.received.append((self.path, self.headers["Authorization"], body["records"]))
            status = self.__class__.fail_code or 200
            self.__class__.fail_code = None
            self.send_response(status)
            body = b"{}"
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FoundryBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "outbox.sqlite3"
        self.outbox = Outbox(self.path)

    def test_every_packet_is_queued_before_latest_snapshot_and_survives_restart(self):
        packet = {"type": "telemetry", "node_id": mesh_api.NODE_IDS[0], "schema_version": 2, "uptime_ms": 1000,
                  "sequence": 0, "sensor_ok": True, "hr_bpm": 80, "accel_m_s2": [0, 0, 9.8],
                  "gyro_rad_s": [0, 0, 0], "steps": 0, "received_ms": 1100, "wake": False}
        with patch.object(mesh_api, "outbox", self.outbox):
            for sequence in range(100):  # 10 seconds at two wearables x 5 packets/s
                node_id = mesh_api.NODE_IDS[sequence % 2]
                mesh_api.accept_packet({**packet, "node_id": node_id, "sequence": sequence // 2}, time.time())
            self.assertEqual([mesh_api.latest[node_id][1]["sequence"] for node_id in mesh_api.NODE_IDS], [49, 49])
        records = Outbox(self.path).pending("telemetry")
        self.assertEqual(len(records), 100)
        self.assertEqual([record["sequence"] for _, record in records], [n // 2 for n in range(100)])
        self.assertEqual(len({record["event_id"] for _, record in records}), 100)
        self.assertEqual(records[0][1]["wearer_name"], "Wearer1")
        self.assertEqual((records[0][1]["root_received_ms"], records[0][1]["hr_valid"], records[0][1]["wake"]), (1100, True, False))

    def test_gateway_status_reports_queue_and_publishing_state(self):
        self.outbox.enqueue("telemetry", {"node_id": "A"})
        with patch.object(mesh_api, "outbox", self.outbox), patch.object(mesh_api, "publisher", None):
            server = ThreadingHTTPServer(("127.0.0.1", 0), mesh_api.Handler)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with urlopen(f"http://127.0.0.1:{server.server_port}/api/nodes") as response:
                    status = json.load(response)["foundry"]
            finally:
                server.shutdown()
                server.server_close()
        self.assertEqual((status["queued"], status["publishing"], status["dropped"]), (1, False, 0))

    def test_dashboard_events_include_identity_location_and_alert_state(self):
        team = Team(None, (0, 0), self.outbox)
        person = Person("mesh_206EF1316774", "Alex", "Wearer", "LIVE", imu=True)
        team.add(person)
        team.set_origin(32.0, -117.0, "manual")
        team.place(person.id, 32.1, -117.1)
        team.open_alert(person.id, "distress", "Check wearer", "critical")
        team.act("A1", "acknowledge")
        team.act("A1", "resolve", "Checked")
        events = [record for _, record in self.outbox.pending("events")]
        self.assertEqual([event["event_type"] for event in events], ["location", "location", "alert", "alert", "alert"])
        self.assertEqual(events[1]["location_source"], "corrected")
        self.assertEqual((events[1]["wearer_name"], events[1]["lat"], events[1]["lon"]), ("Alex", 32.1, -117.1))
        self.assertEqual([event["alert_state"] for event in events[2:]], ["open", "acknowledged", "resolved"])

    def test_activity_and_learned_baseline_are_emitted_once(self):
        team = Team(None, (0, 0), self.outbox)
        person = Person("mesh_206EF1316774", "Alex", "Wearer", "LIVE", imu=True, connected=True)
        team.add(person)
        clock = [1000.0]
        with patch("dashboard.server.time.time", side_effect=lambda: clock[0]):
            for _ in range(52):
                person.add_imu((0, 0, 1000), (1, 0, 0), 25)
                person.add_imu((0, 0, 1000), (1, 0, 0), 25)
                person.add_hr(72)
                team.tick()
                clock[0] += 1
        events = [record for _, record in self.outbox.pending("events")]
        self.assertEqual([(event["event_type"], event["activity"]) for event in events],
                         [("activity", "still"), ("baseline", "still")])
        self.assertEqual(events[1]["baseline_bpm"], 72)

    def test_age_and_capacity_expire_oldest_records_and_report_drops(self):
        clock = [1000.0]
        with patch("dashboard.foundry_bridge.time.time", side_effect=lambda: clock[0]):
            small = Outbox(Path(self.temp.name) / "small.sqlite3", max_bytes=210)
            first = small.enqueue("events", {"value": "a" * 100})
            second = small.enqueue("events", {"value": "b" * 100})
            self.assertNotEqual(first, second)
            self.assertEqual([event_id for event_id, _ in small.pending("events")], [second])
            self.assertEqual(small.stats()["dropped"], 1)
            clock[0] += 3601
            small.prune()
            self.assertEqual(small.stats()["queued"], 0)
            self.assertEqual(small.stats()["dropped"], 2)

    def test_publish_retries_with_stable_ids_refreshes_token_and_keeps_rejections(self):
        MockFoundry.tokens, MockFoundry.received, MockFoundry.fail_code = 0, [], 500
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockFoundry)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        publisher = Publisher(self.outbox, f"http://127.0.0.1:{server.server_port}", "telemetry-rid", "events-rid", "id", "secret")
        packet = {"type": "telemetry", "node_id": mesh_api.NODE_IDS[0], "schema_version": 2,
                  "uptime_ms": 1000, "sequence": 1, "sensor_ok": True, "hr_bpm": 80}
        with patch.object(mesh_api, "outbox", self.outbox):
            mesh_api.accept_packet(packet, time.time())
        team = Team(None, (0, 0), self.outbox)
        person = Person("mesh_206EF1316774", "Alex", "Wearer", "LIVE")
        team.add(person)
        team.open_alert(person.id, "distress", "Check wearer", "critical")
        telemetry_id = self.outbox.pending("telemetry")[0][0]
        event_id = self.outbox.pending("events")[0][0]
        with self.assertRaises(HTTPError):
            publisher.step()
        self.assertEqual(self.outbox.stats()["queued"], 2)
        publisher.step()
        self.assertEqual(self.outbox.stats()["queued"], 0)
        self.assertEqual(MockFoundry.received[0][2][0]["event_id"], telemetry_id)
        self.assertEqual(MockFoundry.received[1][2][0]["event_id"], telemetry_id)
        self.assertEqual(MockFoundry.received[2][2][0]["event_id"], event_id)
        MockFoundry.fail_code = 401
        self.outbox.enqueue("events", {"event_type": "location"})
        publisher.step()
        self.assertEqual(MockFoundry.tokens, 2)
        publisher.expires_at = 0
        MockFoundry.fail_code = 400
        self.outbox.enqueue("events", {"event_type": "baseline"})
        with self.assertRaises(HTTPError):
            publisher.step()
        self.assertEqual(self.outbox.stats()["queued"], 1)
        self.assertEqual(MockFoundry.tokens, 3)

    def test_network_failure_does_not_block_local_alert(self):
        publisher = Publisher(self.outbox, "http://127.0.0.1:1", "telemetry-rid", "events-rid", "id", "secret")
        team = Team(None, (0, 0), self.outbox)
        person = Person("mesh_206EF1316774", "Alex", "Wearer", "LIVE")
        team.add(person)
        team.open_alert(person.id, "distress", "Check wearer", "critical")
        self.assertIn("A1", team.alerts)
        with self.assertRaises(URLError):
            publisher.step()
        self.assertEqual(self.outbox.stats()["queued"], 1)


if __name__ == "__main__":
    unittest.main()
