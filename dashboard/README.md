# Operator dashboard

Streams team heart rate, movement, and HRV to a browser and runs the check loop from PLAN.md §1.4–1.5: an alert appears, the operator requests a check, acknowledges it, and resolves it with a note.

```bash
sensor/.venv/bin/pip install -r dashboard/requirements.txt
sensor/.venv/bin/python dashboard/server.py            # replay sensor/test4 + 5 simulated teammates
sensor/.venv/bin/python dashboard/server.py --live     # add the Polar Verity Sense as a live person
```

Open http://127.0.0.1:8765.

Every person is labeled with where their data comes from: **LIVE** (Polar over BLE), **REPLAY** (a recorded `sensor/<prefix>_*.csv` session, looped), or **SYNTHETIC** (simulated). The simulated team has scripted events for the demo: a distress press (Okafor, ~75 s), HR above expected (Alvarez, ~75 s), and a sensor dropout (Patel, 120–150 s).

**Attention items**

| Type | Trigger |
| --- | --- |
| Distress | Button press (simulated) or "Mark distress" |
| Physiology | HR more than 20 bpm from expected for 30 s |
| Sensor | No samples for 10 s, or skin contact lost. Marked "cleared" when the signal returns; it stays queued until the operator resolves it |

**Map** shows where each person is and their status color. There is no GPS. Simulated people move around `--base` (default: downtown San Diego). Live and replayed people have no position until the operator clicks "Place on map"; their pins are square and can be dragged. Switch to satellite view with the layers button. Tiles come from OpenStreetMap and Esri, so the map needs internet.

**Expected HR** comes from a first-order kinetic model driven by movement. Resting HR is taken from the first 20 readings, and gain and time constants are fixed defaults. The model is not fitted or validated. Swap in a reviewed predictor in `Kinetic.step`.

Each run saves `samples.csv` (1 Hz per person) and `events.jsonl` (alerts and operator actions) under `dashboard/sessions/<timestamp>/`.
