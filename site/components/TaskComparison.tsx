import { RESULTS } from "@/lib/data";
import { modelFamily } from "@/lib/model-family";

const TASKS = [
  { id: "pulse", label: "Pulse", description: "Heart-rate estimation" },
  { id: "heat", label: "Heat", description: "Core-temperature forecasting" },
  { id: "load", label: "Load", description: "Metabolic-cost prediction" },
] as const;
const scale = Math.ceil(Math.max(...RESULTS.models.flatMap(model => TASKS.map(task => model.tasks[task.id].elo ?? 0))) / 100) * 100;

export function TaskComparison() {
  return <figure className="task-comparison" aria-label="Prediction Elo across the three tasks">
    <figcaption><span className="figure-index">Figure 1</span><strong>Performance across tasks</strong><span>Prediction Elo · higher is better</span></figcaption>
    <div className="task-panels">{TASKS.map(task => {
      const rows = [...RESULTS.models].sort((a, b) => (b.tasks[task.id].elo ?? -1) - (a.tasks[task.id].elo ?? -1) || a.model.localeCompare(b.model));
      return <section className="task-panel" key={task.id} aria-label={`${task.label} prediction Elo`}>
        <h3>{task.label}</h3><p className="task-panel-description">{task.description}</p>
        <ol>{rows.map(model => {
          const score = model.tasks[task.id].elo;
          return <li key={model.model} className={`family-${modelFamily(model.model)}`}>
            <div className="task-score-label"><span title={`${model.model} · ${model.effort} effort`}>{model.model.replace("Claude ", "").replace("DeepSeek V4 Flash 0423", "DeepSeek V4 Flash")}</span><span>{score === null ? "—" : score.toFixed(0)}</span></div>
            <div className="task-score-track" aria-hidden="true"><span style={{ width: `${(score ?? 0) / scale * 100}%` }} /></div>
            <span className="sr-only">{score === null ? "Unranked" : `${score.toFixed(1)} Elo`}; {model.tasks[task.id].valid} of 3 valid attempts; {model.effort} effort.</span>
          </li>;
        })}</ol>
        <div className="task-score-axis" aria-hidden="true"><span>0</span><span>{scale / 2}</span><span>{scale}</span></div>
      </section>;
    })}</div>
    <p className="task-comparison-note">Shared scale; three trials per task. A dash denotes an unranked result. Failed attempts remain in matched outcomes; no confidence intervals estimated.</p>
  </figure>;
}
