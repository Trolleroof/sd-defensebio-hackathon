import Link from "next/link";
import { Leaderboard } from "@/components/Leaderboard";
import { TaskComparison } from "@/components/TaskComparison";
import { TASKS } from "@/lib/data";

const SPEC = "https://github.com/Trolleroof/sd-defensebio-hackathon/blob/main/benchmarks/compact-coding-v1/SPEC.md";

export default function Home() {
  return (
    <div className="wrap research-page">
      <header className="research-header">
        <p className="research-kicker">DefenseBio / Research</p>
        <h1>PhysBench</h1>
        <p className="research-deck">Evaluating coding agents on physiological model development.</p>
        <div className="research-byline"><time dateTime="2026-10-18">October 18, 2026</time><span aria-hidden="true">·</span><span>8 models, 3 tasks, 72 episodes</span></div>
        <nav className="research-links" aria-label="Research resources"><Link href="/blog/physbench/">Read the report <span>↗</span></Link><a href={SPEC}>Methodology <span>↗</span></a><a href="/results/compact-coding-v1.json">Data <span>↓</span></a></nav>
      </header>

      <section className="research-results" id="leaderboard" aria-labelledby="results-heading">
        <div className="research-section-title"><h2 id="results-heading">Results</h2><Link href="/blog/physbench/#methodology">Evaluation protocol ↗</Link></div>
        <Leaderboard />
        <TaskComparison />
      </section>

      <section className="research-context" aria-labelledby="evaluation-heading">
        <h2 id="evaluation-heading">The evaluation</h2>
        <p>Agents train two predictors, evaluate on development data, and select one for hidden replay. Person-balanced error determines prediction Elo; a separate judge assesses the experiment decisions.</p>
        <div className="research-task-list">{TASKS.map((task, i) => <Link key={task.id} href={`/tasks/#${task.id}`}><span className="research-task-number">0{i + 1}</span><span><strong>{task.name}</strong><small>{task.data}</small></span><span aria-hidden="true">↗</span></Link>)}</div>
        <details className="research-notes"><summary>Scope and coverage</summary><p>This retrospective pilot produced 33 valid predictors from 72 attempts, with 65 of 72 process reviews available. Failed attempts remain in the matched outcomes. Pulse ECG references are provisional; fixed cohorts and three agent seeds do not establish population-level ranking confidence.</p></details>
      </section>
    </div>
  );
}
