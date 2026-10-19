#!/usr/bin/env python3
"""Expose the latest telemetry from both Nano wearables without recording it."""

import argparse
import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import serial

if __package__:
    from .foundry_bridge import Outbox, Publisher, telemetry_record
else:
    from foundry_bridge import Outbox, Publisher, telemetry_record

NODE_IDS = ("206EF1316774", "206EF1316838")
STALE_S = 10
latest = {node_id: None for node_id in NODE_IDS}
lock = threading.Lock()
serial_connected = False
last_event = None
outbox = None
publisher = None
node_names = {NODE_IDS[0]: "Wearer1", NODE_IDS[1]: "Wearer2"}
bridge_error = None
write_failures = 0


def accept_packet(data, received_at):
    """Queue every valid packet before replacing the latest gateway snapshot."""
    global last_event, bridge_error, write_failures
    with lock:
        last_event = (received_at, data)
    if data.get("type") != "telemetry" or not isinstance(data.get("node_id"), str) or data["node_id"] not in latest:
        return
    node_id = data["node_id"]
    if outbox:
        record = telemetry_record(data, node_names[node_id], received_at)
        if record:
            try:
                outbox.enqueue("telemetry", record)
                bridge_error = None
            except (OSError, ValueError, sqlite3.Error) as error:
                write_failures += 1
                bridge_error = f"Local Foundry queue unavailable: {error}"
    with lock:
        latest[node_id] = (received_at, data)


def read_serial(port_name):
    global serial_connected, last_event
    last_error = None
    while True:
        try:
            with serial.Serial(port_name, 115200, timeout=1) as port:
                with lock:
                    serial_connected = True
                print(f"Reading {port_name} at 115200 baud", flush=True)
                last_error = None
                while True:
                    raw = port.readline()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if not isinstance(data, dict):
                        continue
                    accept_packet(data, time.time())
        except (serial.SerialException, OSError) as error:
            with lock:
                serial_connected = False
            if str(error) != last_error:
                print(f"Waiting for {port_name}: {error}", flush=True)
                last_error = str(error)
            time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path != "/api/nodes":
            self.send_error(404)
            return
        now = time.time()
        with lock:
            connected = serial_connected
            root_event = {"received_at": last_event[0], "data": last_event[1]} if last_event else None
            nodes = {
                node_id: {
                    "online": connected and entry is not None and now - entry[0] <= STALE_S,
                    "age_s": round(now - entry[0], 2) if entry else None,
                    "received_at": entry[0] if entry else None,
                    "latest": entry[1] if entry else None,
                }
                for node_id, entry in latest.items()
            }
        foundry = None
        if outbox:
            try:
                foundry = {**outbox.stats(), "publishing": publisher is not None,
                           "last_success": publisher.last_success if publisher else None,
                           "write_failures": write_failures,
                           "error": bridge_error or (publisher.error if publisher else None)}
            except (OSError, sqlite3.Error) as error:
                foundry = {"error": f"Local Foundry queue unavailable: {error}"}
        body = json.dumps({"serial_connected": connected, "last_event": root_event, "nodes": nodes,
                           "foundry": foundry}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global outbox, publisher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-port", default="COM11")
    parser.add_argument("--http-port", type=int, default=8766)
    parser.add_argument("--mesh-node", action="append", default=[], help="NAME=NODE_ID; match dashboard names")
    parser.add_argument("--foundry-outbox", help="shared SQLite outbox path; enables local packet capture")
    args = parser.parse_args()
    for spec in args.mesh_node:
        name, sep, node_id = spec.partition("=")
        node_id = node_id.upper().replace(":", "")
        if not sep or not name or node_id not in node_names:
            parser.error(f"Invalid --mesh-node {spec!r}; expected NAME=one of {NODE_IDS}")
        node_names[node_id] = name
    if args.foundry_outbox:
        outbox = Outbox(args.foundry_outbox)
        config = {key: os.environ.get(key) for key in ("FOUNDRY_HOST", "FOUNDRY_TELEMETRY_RID", "FOUNDRY_EVENTS_RID",
                                                         "FOUNDRY_CLIENT_ID", "FOUNDRY_CLIENT_SECRET")}
        if any(config.values()) and not all(config.values()):
            parser.error("Foundry publishing needs all FOUNDRY_HOST, FOUNDRY_TELEMETRY_RID, FOUNDRY_EVENTS_RID, FOUNDRY_CLIENT_ID, and FOUNDRY_CLIENT_SECRET")
        if all(config.values()):
            publisher = Publisher(outbox, config["FOUNDRY_HOST"], config["FOUNDRY_TELEMETRY_RID"],
                                  config["FOUNDRY_EVENTS_RID"], config["FOUNDRY_CLIENT_ID"], config["FOUNDRY_CLIENT_SECRET"],
                                  scope=os.environ.get("FOUNDRY_SCOPE", "api:use-streams-write"))

        def publish_loop():
            global bridge_error
            if publisher:
                publisher.run(threading.Event())
            else:
                while True:
                    try:
                        outbox.prune()
                    except (OSError, sqlite3.Error) as error:
                        bridge_error = f"Local Foundry queue unavailable: {error}"
                    time.sleep(5)

        threading.Thread(target=publish_loop, daemon=True).start()
    threading.Thread(target=read_serial, args=(args.serial_port,), daemon=True).start()
    with ThreadingHTTPServer(("127.0.0.1", args.http_port), Handler) as server:
        print(f"Mesh API: http://127.0.0.1:{args.http_port}/api/nodes", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
