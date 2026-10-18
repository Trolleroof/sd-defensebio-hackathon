import type { Metadata } from "next";
import { Reveal } from "@/components/Reveal";
import { TASKS } from "@/lib/data";

export const metadata: Metadata = { title: "Tasks · PhysBench" };

export default function Tasks() {
  return (
    <>
      <header className="post-head">
        <div className="wrap">
          <Reveal>
            <span className="eyebrow">Tasks</span>
            <h1>Tasks</h1>
            <p>
              The current leaderboard uses three measured-data tasks with fixed person splits. Each model runs three agent trials per task, with two coding experiments per episode.
            </p>
          </Reveal>
        </div>
      </header>
      <div className="wrap" style={{ paddingBottom: 96 }}>
        {TASKS.map((t, i) => (
          <Reveal key={t.id} className="task" id={t.id}>
            <div>
              <span className="eyebrow">{String(i + 1).padStart(2, "0")} · {t.family}</span>
              <h3>{t.name}</h3>
            </div>
            <div>
              <p>{t.summary}</p>
              <dl className="kv">
                <dt>Data</dt><dd>{t.data}</dd>
                <dt>Budget</dt><dd>{t.budget}</dd>
              </dl>
              <div className="phases">
                {t.phases.map((p) => <span key={p}>{p}</span>)}
              </div>
            </div>
          </Reveal>
        ))}
      </div>
    </>
  );
}
