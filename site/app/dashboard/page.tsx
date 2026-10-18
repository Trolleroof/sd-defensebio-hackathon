import type { Metadata } from "next";
import { Reveal } from "@/components/Reveal";
import { DASHBOARD_URL } from "@/lib/data";

export const metadata: Metadata = { title: "Dashboard · PhysBench" };

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
            <h1>Where the predictors end up</h1>
            <p>
              The dashboard streams team heart rate, movement and HRV, flags people whose heart rate drifts from
              expected, and runs the check loop: request, acknowledge, resolve with a note.
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
            <div><span className="eyebrow">Data sources</span><h2>Every person is labelled</h2></div>
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
            <div><span className="eyebrow">Attention items</span><h2>Three alert types</h2></div>
            <p>
              Expected HR comes from a fixed first-order kinetic model. It isn&apos;t fitted or validated yet. A
              predictor that passes PhysBench is what replaces it.
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
            <span className="eyebrow">Run locally</span>
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
