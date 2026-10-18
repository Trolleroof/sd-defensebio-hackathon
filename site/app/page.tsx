import Link from "next/link";
import { Leaderboard } from "@/components/Leaderboard";
import { Reveal } from "@/components/Reveal";
import { Trace } from "@/components/Trace";
import { DASHBOARD_URL, POST_SECTIONS, RELEASE, TASKS } from "@/lib/data";

export default function Home() {
  return (
    <>
      <section className="hero">
        <div className="wrap">
          <div className="rise" style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <span className="tag olive">{RELEASE.version}</span>
            <span className="tag">{RELEASE.status}</span>
          </div>
          <h1 className="rise">
            Personalized <span>heart-rate</span> prediction for agents
          </h1>
          <p className="lede rise">
            Agents fit a heart-rate model for one person from wearable data, then forecast activities that person
            didn&apos;t do in the training data. We score the code on hidden recordings and grade the trace
            separately.
          </p>
          <div className="hero-meta rise">
            <div><span className="eyebrow">Episodes</span><strong>{RELEASE.episodes}</strong></div>
            <div><span className="eyebrow">Held-out subjects</span><strong>6</strong></div>
            <div><span className="eyebrow">Horizon</span><strong>60 s</strong></div>
            <div><span className="eyebrow">Dev calls</span><strong>3</strong></div>
            <div><span className="eyebrow">Updated</span><strong style={{ fontSize: 15 }}>{RELEASE.updated}</strong></div>
          </div>
        </div>
        <div className="wrap"><Trace /></div>
      </section>

      <section className="section" id="leaderboard">
        <div className="wrap">
          <Reveal className="section-head">
            <div>
              <span className="eyebrow">Leaderboard</span>
              <h2>Activity transfer</h2>
            </div>
            <p>Placeholder entries until the held-out pilot runs.</p>
          </Reveal>
          <Leaderboard />
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <Reveal className="section-head">
            <div>
              <span className="eyebrow">Scoring</span>
              <h2>Metrics</h2>
            </div>
          </Reveal>
          <Reveal stagger className="grid-3">
            <div className="cell">
              <span className="num">01</span>
              <h3>Transfer MAE</h3>
              <p>Mean of exertion and recovery error on hidden windows, in bpm.</p>
            </div>
            <div className="cell">
              <span className="num">02</span>
              <h3>Pass@1</h3>
              <p>Valid run, beats the anchor by a frozen margin, no integrity flags.</p>
            </div>
            <div className="cell">
              <span className="num">03</span>
              <h3>Judge score</h3>
              <p>Seven items scored 0–2. The judge doesn&apos;t see the model name or hidden accuracy.</p>
            </div>
          </Reveal>
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <Reveal className="section-head">
            <div>
              <span className="eyebrow">Report</span>
              <h2>Contents</h2>
            </div>
            <Link href="/blog/physbench/" className="tag olive">Read →</Link>
          </Reveal>
          <Reveal stagger className="toc">
            {POST_SECTIONS.map((s) => (
              <Link key={s.id} href={`/blog/physbench/#${s.id}`}>
                <span className="n">{s.n}</span>
                <span className="t">{s.title}<span className="s">{s.sub}</span></span>
                <span className="arrow">→</span>
              </Link>
            ))}
          </Reveal>
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <Reveal className="section-head">
            <div>
              <span className="eyebrow">Tasks</span>
              <h2>{TASKS.length} task families</h2>
            </div>
            <Link href="/tasks/" className="tag">All tasks →</Link>
          </Reveal>
          <Reveal stagger className="grid-3">
            {TASKS.map((t, i) => (
              <Link key={t.id} href={`/tasks/#${t.id}`} className="cell">
                <span className="num">{String(i + 1).padStart(2, "0")} · {t.data}</span>
                <h3>{t.name}</h3>
                <p>{t.summary}</p>
              </Link>
            ))}
          </Reveal>
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <Reveal>
            <a className="panel-link" href={DASHBOARD_URL}>
              <div>
                <span className="eyebrow">Dashboard</span>
                <h3>Operator dashboard</h3>
                <p>Live heart rate, movement and HRV for a team, with alerts when heart rate drifts from expected.</p>
              </div>
              <span className="arrow" style={{ fontSize: 22 }}>↗</span>
            </a>
          </Reveal>
        </div>
      </section>
    </>
  );
}
