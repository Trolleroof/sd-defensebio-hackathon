import type { Metadata } from "next";
import { Reveal } from "@/components/Reveal";
import { DASHBOARD_URL } from "@/lib/data";

export const metadata: Metadata = { title: "Dashboard · soliderphys-bench" };

const SOURCES = [
  { label: "Live", body: "Polar Verity Sense over BLE." },
  { label: "Replay", body: "A recorded sensor session, looped." },
  { label: "Synthetic", body: "Simulated teammates with scripted events." },
];

const ALERTS = [
  { label: "Distress", body: "Button press, or marked by the operator." },
  { label: "Physiology", body: "HR more than 20 bpm from expected for 30 s." },
  { label: "Sensor", body: "No samples for 10 s, or skin contact lost." },
];

export default function Dashboard() {
  return (
    <>
      <header className="post-head">
        <div className="wrap">
          <Reveal>
            <span className="eyebrow">Operator dashboard</span>
            <h1>Operator dashboard</h1>
            <p>
              Live heart rate, movement and HRV for a team. Alerts when heart rate drifts from expected. Operators
              request a check, acknowledge it and resolve it with a note.
            </p>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 28 }}>
              <a className="header-cta" style={{ marginLeft: 0 }} href={DASHBOARD_URL}>Open dashboard ↗</a>
              <span className="tag mono">{DASHBOARD_URL}</span>
            </div>
          </Reveal>
        </div>
      </header>

      <section className="section" style={{ borderTop: 0 }}>
        <div className="wrap">
          <Reveal className="section-head">
            <div><span className="eyebrow">Sources</span><h2>Data sources</h2></div>
          </Reveal>
          <Reveal stagger className="grid-3">
            {SOURCES.map((s) => (
              <div key={s.label} className="cell">
                <span className="tag olive">{s.label}</span>
                <p style={{ marginTop: 14 }}>{s.body}</p>
              </div>
            ))}
          </Reveal>
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <Reveal className="section-head">
            <div><span className="eyebrow">Alerts</span><h2>Alert types</h2></div>
            <p>
              Expected heart rate uses a fixed kinetic model that hasn&apos;t been fitted or validated.
            </p>
          </Reveal>
          <Reveal stagger className="grid-3">
            {ALERTS.map((a) => (
              <div key={a.label} className="cell">
                <span className="num">{a.label}</span>
                <p style={{ marginTop: 10 }}>{a.body}</p>
              </div>
            ))}
          </Reveal>
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <Reveal>
            <span className="eyebrow">Run</span>
            <pre className="mono" style={{ margin: "16px 0 0", padding: 20, border: "1px solid var(--line)", overflowX: "auto", fontSize: 13, color: "var(--muted)" }}>
{`sensor/.venv/bin/pip install -r dashboard/requirements.txt
sensor/.venv/bin/python dashboard/server.py          # replay + simulated team
sensor/.venv/bin/python dashboard/server.py --live   # add the Polar sensor`}
            </pre>
          </Reveal>
        </div>
      </section>
    </>
  );
}
