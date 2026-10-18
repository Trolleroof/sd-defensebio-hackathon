#!/usr/bin/env python3
"""Expose the latest telemetry from both Nano wearables without recording it."""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import serial

NODE_IDS = ("206EF1316774", "206EF1316838")
STALE_S = 10
latest = {node_id: None for node_id in NODE_IDS}
lock = threading.Lock()
serial_connected = False


def read_serial(port_name):
    global serial_connected
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
                    if not isinstance(data, dict) or data.get("type") != "telemetry":
                        continue
                    node_id = data.get("node_id")
                    if not isinstance(node_id, str) or node_id not in latest:
                        continue
                    with lock:
                        latest[node_id] = (time.time(), data)
        except (serial.SerialException, OSError) as error:
            with lock:
                serial_connected = False
            if str(error) != last_error:
                print(f"Waiting for {port_name}: {error}", flush=True)
                last_error = str(error)
            time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/api/nodes":
            self.send_error(404)
            return
        now = time.time()
        with lock:
            nodes = {
                node_id: {
                    "online": serial_connected and entry is not None and now - entry[0] <= STALE_S,
                    "age_s": round(now - entry[0], 2) if entry else None,
                    "received_at": entry[0] if entry else None,
                    "latest": entry[1] if entry else None,
                }
                for node_id, entry in latest.items()
            }
        body = json.dumps({"nodes": nodes}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-port", default="COM11")
    parser.add_argument("--http-port", type=int, default=8766)
    args = parser.parse_args()
    threading.Thread(target=read_serial, args=(args.serial_port,), daemon=True).start()
    with ThreadingHTTPServer(("127.0.0.1", args.http_port), Handler) as server:
        print(f"Mesh API: http://127.0.0.1:{args.http_port}/api/nodes", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
