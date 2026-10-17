#!/usr/bin/env python3
"""Stream a Polar Verity Sense over Bluetooth LE and print live statistics.

Streams:
  - Heart rate (+ PP/RR intervals when the sensor sends them) via the standard BLE HR service
  - Raw PPG (55 Hz, 3 optical channels + ambient) via Polar's PMD service -- opt-in with --ppg,
    because the sensor stops reporting its own heart rate while raw PPG is streaming
  - Accelerometer (52 Hz, x/y/z in mG) via Polar's PMD service
  - Beat-to-beat (PP) intervals via Polar's PMD service -> RMSSD / SDNN / pNN50.
    The sensor needs ~25 s of steady wear before the first interval arrives.

Usage:
  python polar_stream.py                      # find the first Polar Sense nearby, stream HR + accelerometer
  python polar_stream.py --ppg                # raw PPG instead of the sensor's own HR
  python polar_stream.py --csv run1           # also write run1_hr.csv / run1_ppg.csv / run1_acc.csv
  python polar_stream.py --no-acc --no-ppi    # heart rate only
  python polar_stream.py --scan               # just list nearby Polar devices
"""

import argparse
import asyncio
import csv
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field

from bleak import BleakClient, BleakScanner
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

HR_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"
BATTERY_LEVEL = "00002a19-0000-1000-8000-00805f9b34fb"
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"

PMD_PPG = 0x01
PMD_ACC = 0x02
PMD_PPI = 0x03

# Start-measurement commands (op 0x02). Settings are TLV: type, count, value (little-endian u16).
#   0x00 sample rate, 0x01 resolution, 0x02 range, 0x04 channels
START_PPG = bytes([0x02, PMD_PPG, 0x00, 0x01, 0x37, 0x00, 0x01, 0x01, 0x16, 0x00, 0x04, 0x01, 0x04])
START_ACC = bytes([0x02, PMD_ACC, 0x00, 0x01, 0x34, 0x00, 0x01, 0x01, 0x10, 0x00,
                   0x02, 0x01, 0x08, 0x00, 0x04, 0x01, 0x03])
START_PPI = bytes([0x02, PMD_PPI])  # PPI takes no settings
START_CMD = {PMD_PPG: START_PPG, PMD_ACC: START_ACC, PMD_PPI: START_PPI}
STOP = lambda kind: bytes([0x03, kind])

# (channels, bits per channel in the reference sample) for each PMD stream
PMD_LAYOUT = {PMD_PPG: (4, 22), PMD_ACC: (3, 16)}


# ---------------------------------------------------------------- decoding

def to_signed(value: int, bits: int) -> int:
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def parse_hr(data: bytearray) -> tuple[int, list[float]]:
    """Standard BLE Heart Rate Measurement -> (bpm, [rr_ms...])."""
    flags = data[0]
    i = 1
    if flags & 0x01:
        hr = int.from_bytes(data[i:i + 2], "little"); i += 2
    else:
        hr = data[i]; i += 1
    if flags & 0x08:  # energy expended present
        i += 2
    rr = []
    if flags & 0x10:
        while i + 1 < len(data):
            rr.append(int.from_bytes(data[i:i + 2], "little") / 1024 * 1000)
            i += 2
    return hr, rr


def decode_delta_frame(payload: bytes, channels: int, bits: int) -> list[list[int]]:
    """Polar PMD compressed (delta) frame: one reference sample, then bit-packed delta blocks."""
    ref_bytes = math.ceil(bits / 8)
    ref = [to_signed(int.from_bytes(payload[c * ref_bytes:(c + 1) * ref_bytes], "little"), ref_bytes * 8)
           for c in range(channels)]
    samples = [ref]
    i = channels * ref_bytes
    while i + 2 <= len(payload):
        delta_bits, count = payload[i], payload[i + 1]
        i += 2
        nbytes = math.ceil(delta_bits * channels * count / 8)
        chunk = int.from_bytes(payload[i:i + nbytes], "little")
        i += nbytes
        pos = 0
        mask = (1 << delta_bits) - 1
        for _ in range(count):
            prev = samples[-1]
            sample = []
            for c in range(channels):
                d = to_signed((chunk >> pos) & mask, delta_bits) if delta_bits else 0
                pos += delta_bits
                sample.append(prev[c] + d)
            samples.append(sample)
    return samples


def decode_ppi(payload: bytes) -> list[list[int]]:
    """PPI samples, 6 bytes each -> [hr_bpm, interval_ms, error_estimate_ms, blocker, skin_contact]."""
    out = []
    for off in range(0, len(payload) - 5, 6):
        flags = payload[off + 5]
        out.append([payload[off],
                    int.from_bytes(payload[off + 1:off + 3], "little"),
                    int.from_bytes(payload[off + 3:off + 5], "little"),
                    flags & 0x01,                 # 1 = interval unreliable (motion etc.)
                    (flags >> 1) & 0x01])         # 1 = skin contact detected
    return out


def decode_pmd(data: bytearray) -> tuple[int, list[list[int]]]:
    kind, frame_type = data[0], data[9]
    payload = bytes(data[10:])
    if kind == PMD_PPI:
        return kind, decode_ppi(payload)
    channels, bits = PMD_LAYOUT.get(kind, (0, 0))
    if not channels:
        return kind, []
    if frame_type & 0x80:
        return kind, decode_delta_frame(payload, channels, bits)
    # Uncompressed: fixed-width little-endian signed samples
    width = math.ceil(bits / 8)
    step = width * channels
    out = []
    for off in range(0, len(payload) - step + 1, step):
        out.append([to_signed(int.from_bytes(payload[off + c * width:off + (c + 1) * width], "little"), width * 8)
                    for c in range(channels)])
    return kind, out


# ---------------------------------------------------------------- stats

@dataclass
class Stats:
    started: float = field(default_factory=time.time)
    hr: deque = field(default_factory=lambda: deque(maxlen=60))       # last 60 HR readings (~60 s)
    hr_all_min: int = 999
    hr_all_max: int = 0
    rr: deque = field(default_factory=lambda: deque(maxlen=120))      # last 120 beat intervals
    ppg: deque = field(default_factory=lambda: deque(maxlen=55 * 10))  # 10 s of PPG
    ppg_count: int = 0
    ppg_first: float | None = None
    acc: deque = field(default_factory=lambda: deque(maxlen=52 * 5))  # 5 s of ACC
    acc_count: int = 0
    acc_first: float | None = None
    ppi_received: int = 0
    ppi_raw: deque = field(default_factory=lambda: deque(maxlen=7))  # recent unflagged intervals
    ppi_rejected: int = 0
    ppi_error: deque = field(default_factory=lambda: deque(maxlen=30))
    battery: int | None = None

    @staticmethod
    def rate(n: int, first: float | None) -> float:
        dt = time.time() - first if first else 0
        return n / dt if dt > 1 else 0.0


def fmt(x, spec=".1f", unit=""):
    return "—" if x is None else f"{x:{spec}}{unit}"


def render(s: Stats, name: str) -> Group:
    elapsed = int(time.time() - s.started)
    hdr = f"[bold]{name}[/]  ·  {elapsed // 60:02d}:{elapsed % 60:02d} elapsed  ·  battery {fmt(s.battery, 'd', '%')}"

    hr = Table(title="Heart rate", expand=True, show_header=False)
    hr.add_column(style="cyan"); hr.add_column(justify="right", style="bold")
    cur = s.hr[-1] if s.hr else None
    hr.add_row("Current", fmt(cur, "d", " bpm"))
    hr.add_row("Mean (60 s)", fmt(statistics.fmean(s.hr) if s.hr else None, ".1f", " bpm"))
    hr.add_row("Min / Max (session)", f"{s.hr_all_min if s.hr else '—'} / {s.hr_all_max if s.hr else '—'} bpm")
    hr.add_row("Std dev (60 s)", fmt(statistics.pstdev(s.hr) if len(s.hr) > 1 else None, ".2f", " bpm"))

    hrv = Table(title="Beat intervals / HRV", expand=True, show_header=False)
    hrv.add_column(style="cyan"); hrv.add_column(justify="right", style="bold")
    rr = list(s.rr)
    if len(rr) >= 3:
        diffs = [b - a for a, b in zip(rr, rr[1:])]
        rmssd = math.sqrt(statistics.fmean(d * d for d in diffs))
        sdnn = statistics.pstdev(rr)
        pnn50 = 100 * sum(abs(d) > 50 for d in diffs) / len(diffs)
        hrv.add_row("Last interval", fmt(rr[-1], ".0f", " ms"))
        hrv.add_row("Mean interval", fmt(statistics.fmean(rr), ".0f", " ms"))
        hrv.add_row("RMSSD", fmt(rmssd, ".1f", " ms"))
        hrv.add_row("SDNN", fmt(sdnn, ".1f", " ms"))
        hrv.add_row("pNN50", fmt(pnn50, ".0f", " %"))
        hrv.add_row("Window", f"last {len(rr)} beats")
        if s.ppi_error:
            hrv.add_row("Avg error estimate", fmt(statistics.fmean(s.ppi_error), ".0f", " ms"))
        hrv.add_row("Rejected (motion)", f"{s.ppi_rejected} / {s.ppi_received}")
    else:
        hrv.add_row("Waiting for intervals", f"{len(rr)} so far")
        hrv.add_row("", "[dim]sensor needs ~25 s still[/]")

    ppg = Table(title="PPG (raw optical)", expand=True, show_header=False)
    ppg.add_column(style="cyan"); ppg.add_column(justify="right", style="bold")
    if s.ppg:
        green = [p[0] for p in s.ppg]
        amb = [p[3] for p in s.ppg]
        ppg.add_row("Sample rate", fmt(s.rate(s.ppg_count, s.ppg_first), ".1f", " Hz"))
        ppg.add_row("Ch0 latest", fmt(green[-1], ",d"))
        ppg.add_row("Ch0 pulse amplitude (10 s)", fmt(max(green) - min(green), ",d"))
        ppg.add_row("Ambient mean", fmt(statistics.fmean(amb), ",.0f"))
        ppg.add_row("Samples received", fmt(s.ppg_count, ",d"))
    else:
        ppg.add_row("Status", "off (use --ppg)")

    acc = Table(title="Accelerometer", expand=True, show_header=False)
    acc.add_column(style="cyan"); acc.add_column(justify="right", style="bold")
    if s.acc:
        mags = [math.sqrt(x * x + y * y + z * z) for x, y, z in s.acc]
        x, y, z = s.acc[-1]
        acc.add_row("Sample rate", fmt(s.rate(s.acc_count, s.acc_first), ".1f", " Hz"))
        acc.add_row("x / y / z", f"{x} / {y} / {z} mG")
        acc.add_row("|a| mean (5 s)", fmt(statistics.fmean(mags), ".0f", " mG"))
        acc.add_row("Movement (std |a|)", fmt(statistics.pstdev(mags), ".0f", " mG"))
        acc.add_row("Activity", "still" if statistics.pstdev(mags) < 30 else "moving")
    else:
        acc.add_row("Status", "no data")

    grid = Table.grid(expand=True, padding=1)
    grid.add_column(ratio=1); grid.add_column(ratio=1)
    grid.add_row(hr, hrv)
    grid.add_row(ppg, acc)
    return Group(Panel(hdr, style="green"), grid, "[dim]Ctrl+C to stop[/]")


# ---------------------------------------------------------------- BLE

async def find_device(address: str | None, timeout: float = 10.0):
    if address:
        dev = await BleakScanner.find_device_by_address(address, timeout=timeout)
    else:
        dev = await BleakScanner.find_device_by_filter(
            lambda d, _: bool(d.name) and d.name.startswith("Polar"), timeout=timeout)
    if dev is None:
        raise SystemExit("No Polar device found. Wake the sensor (press the button / put it on) and retry.")
    return dev


async def scan():
    print("Scanning for 8 s…")
    for d in await BleakScanner.discover(timeout=8.0):
        if d.name and "Polar" in d.name:
            print(f"  {d.name:30s} {d.address}")


async def run(args):
    dev = await find_device(args.address)
    print(f"Connecting to {dev.name} ({dev.address})…")
    stats = Stats()

    writers = {}
    files = []
    if args.csv:
        for key, cols in {"hr": ["t", "bpm", "rr_ms"],
                          "ppg": ["t", "ch0", "ch1", "ch2", "ambient"],
                          "acc": ["t", "x_mg", "y_mg", "z_mg"],
                          "ppi": ["t", "hr_bpm", "ppi_ms", "error_ms", "blocker", "skin_contact"]}.items():
            f = open(f"{args.csv}_{key}.csv", "w", newline="")
            files.append(f)
            writers[key] = csv.writer(f)
            writers[key].writerow(cols)

    def on_hr(_, data: bytearray):
        bpm, rr = parse_hr(data)
        if bpm == 0:
            return
        stats.hr.append(bpm)
        stats.hr_all_min = min(stats.hr_all_min, bpm)
        stats.hr_all_max = max(stats.hr_all_max, bpm)
        stats.rr.extend(r for r in rr if 250 < r < 2000)  # drop obvious artifacts
        if "hr" in writers:
            writers["hr"].writerow([f"{time.time():.3f}", bpm, ";".join(f"{r:.1f}" for r in rr)])

    def on_pmd(_, data: bytearray):
        kind, samples = decode_pmd(data)
        now = f"{time.time():.3f}"
        if kind == PMD_PPG:
            stats.ppg_first = stats.ppg_first or time.time()
            stats.ppg.extend(samples); stats.ppg_count += len(samples)
            if "ppg" in writers:
                writers["ppg"].writerows([now, *s] for s in samples)
        elif kind == PMD_ACC:
            stats.acc_first = stats.acc_first or time.time()
            stats.acc.extend(samples); stats.acc_count += len(samples)
            if "acc" in writers:
                writers["acc"].writerows([now, *s] for s in samples)
        elif kind == PMD_PPI:
            for bpm, ppi, err, blocker, _ in samples:
                stats.ppi_received += 1
                if blocker or not 250 < ppi < 2000:
                    stats.ppi_rejected += 1
                    continue
                # Artifact filter: drop intervals >20% off the median of recent unflagged beats
                stats.ppi_raw.append(ppi)
                med = statistics.median(stats.ppi_raw)
                if len(stats.ppi_raw) < 3 or abs(ppi - med) > 0.2 * med:
                    stats.ppi_rejected += 1
                    continue
                stats.rr.append(ppi)
                stats.ppi_error.append(err)
                # The HR service goes quiet once PPI is running (and the PPI HR field is often 0),
                # so derive HR from the last few good intervals.
                bpm = bpm or round(60000 / statistics.fmean(list(stats.rr)[-5:]))
                stats.hr.append(bpm)
                stats.hr_all_min = min(stats.hr_all_min, bpm)
                stats.hr_all_max = max(stats.hr_all_max, bpm)
            if "ppi" in writers:
                writers["ppi"].writerows([now, *s] for s in samples)

    def on_ctrl(_, data: bytearray):
        # Response: F0, op, type, status(0 = ok)
        if len(data) >= 4 and data[0] == 0xF0 and data[3] != 0:
            print(f"[warn] PMD op {data[1]:#x} for stream {data[2]} failed with status {data[3]}")

    async with BleakClient(dev) as client:
        try:
            stats.battery = (await client.read_gatt_char(BATTERY_LEVEL))[0]
        except Exception:
            pass

        await client.start_notify(HR_MEASUREMENT, on_hr)

        pmd_kinds = [k for k, on in ((PMD_PPG, args.ppg), (PMD_ACC, not args.no_acc), (PMD_PPI, not args.no_ppi)) if on]
        if pmd_kinds:
            await client.start_notify(PMD_CONTROL, on_ctrl)
            await client.start_notify(PMD_DATA, on_pmd)
            for k in pmd_kinds:
                await client.write_gatt_char(PMD_CONTROL, START_CMD[k], response=True)
                await asyncio.sleep(0.5)

        try:
            with Live(render(stats, dev.name), refresh_per_second=4, screen=False) as live:
                while client.is_connected:
                    await asyncio.sleep(0.25)
                    live.update(render(stats, dev.name))
                    for f in files:
                        f.flush()
            print("Sensor disconnected.")
        finally:
            for k in pmd_kinds:
                try:
                    await client.write_gatt_char(PMD_CONTROL, STOP(k), response=True)
                except Exception:
                    pass
            for f in files:
                f.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--address", help="BLE address/UUID of the sensor (default: first 'Polar…' found)")
    p.add_argument("--csv", metavar="PREFIX", help="also log raw streams to PREFIX_{hr,ppg,acc,ppi}.csv")
    p.add_argument("--ppg", action="store_true",
                   help="stream raw PPG (the sensor's own HR output stops while this is on)")
    p.add_argument("--no-acc", action="store_true", help="don't stream accelerometer")
    p.add_argument("--no-ppi", action="store_true", help="don't stream beat-to-beat intervals")
    p.add_argument("--scan", action="store_true", help="list nearby Polar devices and exit")
    args = p.parse_args()
    try:
        asyncio.run(scan() if args.scan else run(args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
