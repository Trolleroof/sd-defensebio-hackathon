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
            Can an agent build a <span>personal</span> physiology model?
          </h1>
          <p className="lede rise">
            PhysBench gives a model a sandboxed workspace, real wearable recordings and a fixed budget. It has
            to fit a heart-rate predictor for one person, then forecast activities it has never seen them do.
            Hidden recordings score the code. A separate judge grades the research.
          </p>
          <div className="hero-meta rise">
            <div><span className="eyebrow">Episodes</span><strong>{RELEASE.episodes}</strong></div>
            <div><span className="eyebrow">Held-out subjects</span><strong>6</strong></div>
            <div><span className="eyebrow">Forecast horizon</span><strong>60 s</strong></div>
            <div><span className="eyebrow">Dev feedback</span><strong>3 calls</strong></div>
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
              <h2>Heart rate under activity transfer</h2>
            </div>
            <p>
              Placeholder entries. Model names, scores and costs fill in from <code className="mono">summary.json</code> after
              pilot-0 runs on the held-out split.
            </p>
          </Reveal>
          <Leaderboard />
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <Reveal className="section-head">
            <div>
              <span className="eyebrow">What is scored</span>
              <h2>Code, then conduct</h2>
            </div>
          </Reveal>
          <Reveal stagger className="grid-3">
            <div className="cell">
              <span className="num">01 · Accuracy</span>
              <h3>Phase-balanced transfer MAE</h3>
              <p>Mean of exertion-phase and recovery-phase error on hidden windows. Per-activity and per-horizon breakdowns are kept.</p>
            </div>
            <div className="cell">
              <span className="num">02 · Pass</span>
              <h3>Beat the anchor, cleanly</h3>
              <p>Valid execution, a frozen margin over population kinetics with a personal fit, and no integrity flags from the audit or judge.</p>
            </div>
            <div className="cell">
              <span className="num">03 · Judge</span>
              <h3>Seven-item research rubric</h3>
              <p>Scored 0–2 per item. The judge never sees the model name or its hidden accuracy, and it checks the report against the trace.</p>
            </div>
          </Reveal>
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <Reveal className="section-head">
            <div>
              <span className="eyebrow">Report</span>
              <h2>Read the full write-up</h2>
            </div>
            <Link href="/blog/physbench/" className="tag olive">Open report →</Link>
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
              <span className="eyebrow">Environments</span>
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
                <span className="eyebrow">Field side</span>
                <h3>Operator dashboard</h3>
                <p>
                  The same sensors feed a live team view: heart rate, movement, HRV, expected-HR deviation and the
                  alert-check-resolve loop. Predictors built here are meant to replace its fixed kinetic model.
                </p>
              </div>
              <span className="arrow" style={{ fontSize: 22 }}>↗</span>
            </a>
          </Reveal>
        </div>
      </section>
    </>
  );
}
