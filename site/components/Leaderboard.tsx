"use client";

import Link from "next/link";
import { useState, type CSSProperties } from "react";
import { RESULTS } from "@/lib/data";
import { ModelLogo } from "@/components/ModelLogo";
import { modelFamily } from "@/lib/model-family";

const VIEWS = ["suite", "pulse", "heat", "load", "process"] as const;
type View = (typeof VIEWS)[number];
const CRITERIA = ["evidence", "experiment", "feedback", "repro", "selection"] as const;
type Criterion = (typeof CRITERIA)[number];
type Metric = "elo" | "validity" | "mae";
type Model = (typeof RESULTS.models)[number];
const LABELS = { suite: "Overall", pulse: "Pulse", heat: "Heat", load: "Load", process: "Process" };
const CRITERION_LABELS = { evidence: "Evidence use", experiment: "Experiment choice", feedback: "Feedback integration", repro: "Reproducibility", selection: "Final selection" };
const UNITS = { pulse: "bpm", heat: "°C", load: "W" };


export function Leaderboard() {
  const [view, setView] = useState<View>("suite");
  const [metric, setMetric] = useState<Metric>("elo");
  const [criterion, setCriterion] = useState<Criterion>("evidence");
  const process = view === "process";
  const task = view === "pulse" || view === "heat" || view === "load" ? view : null;
  const denominator = view === "suite" ? 9 : 3;
  const metricLabel = process ? CRITERION_LABELS[criterion] : metric === "elo" ? "Prediction Elo" : metric === "validity" ? "Valid attempts" : `Mean MAE · ${task ? UNITS[task] : ""}`;
  function value(row: Model): number | null {
    if (process) return row.criteria[criterion].mean;
    if (metric === "validity") return (task ? row.tasks[task].valid / 3 : row.valid / 9) * 100;
    if (metric === "mae") return task ? row.tasks[task].mean_loss : null;
    return task ? row.tasks[task].elo : row.overall_elo;
  }
  function format(v: number) {
    if (process) return `${v.toFixed(2)} / 4`;
    if (metric === "validity") return `${v.toFixed(1)}%`;
    if (metric === "mae") return `${v.toFixed(task === "heat" ? 3 : 2)} ${task ? UNITS[task] : ""}`;
    return v.toFixed(1);
  }
  const rows = [...RESULTS.models].sort((a, b) => {
    const av = value(a), bv = value(b);
    if (av === null && bv === null) return a.model.localeCompare(b.model);
    if (av === null) return 1;
    if (bv === null) return -1;
    return (metric === "mae" && !process ? av - bv : bv - av) || a.model.localeCompare(b.model);
  });
  const maximum = Math.max(1, ...rows.map(r => value(r) ?? 0));
  const scale = process ? 4 : metric === "validity" ? 100 : metric === "elo" ? Math.ceil(maximum / 100) * 100 : Number((maximum * 1.1).toPrecision(2));
  const chartDescription = process
    ? `Mean criterion score, 0–4. ${RESULTS.reviews}/${RESULTS.review_target} episode reviews available; n is the observable count.`
    : metric === "validity"
      ? "Executable predictors as a share of scheduled attempts. Validity does not measure accuracy."
      : metric === "mae"
        ? "Lower is better. Error averaged over valid attempts only; failed attempts are excluded from this mean."
        : "Higher is better. Overall Elo pools matched attempts across all three tasks. Valid predictors beat failed attempts; two failures skip.";

  return (
    <div className="research-board">
      <div className="board-controls">
        <div className="board-views" role="group" aria-label="Leaderboard view">
          {VIEWS.map(v => <button key={v} aria-pressed={view === v} onClick={() => { setView(v); setMetric("elo"); }}>{LABELS[v]}</button>)}
        </div>
        <label className="metric-picker">
          <span className="sr-only">{process ? "Process criterion" : "Chart metric"}</span>
          {process ? <select aria-label="Process criterion" value={criterion} onChange={e => setCriterion(e.target.value as Criterion)}>
            {CRITERIA.map(c => <option key={c} value={c}>{CRITERION_LABELS[c]}</option>)}
          </select> : <select aria-label="Chart metric" value={metric} onChange={e => setMetric(e.target.value as Metric)}>
            <option value="elo">Prediction Elo</option><option value="validity">Valid attempts</option>
            {task && <option value="mae">Mean MAE</option>}
          </select>}
        </label>
      </div>

      <figure className="ranking-figure" aria-label={`${LABELS[view]} — ${metricLabel}`}>
        <figcaption className="ranking-heading"><span>Model</span><span>{metricLabel}<small>{!process && metric === "mae" ? "Lower is better" : "Higher is better"}</small></span></figcaption>
        <div className="ranking-rows">
          {rows.map(row => {
            const v = value(row);
            const count = process ? `${row.criteria[criterion].count} observed · ${row.reviews}/9 reviewed` : `${task ? row.tasks[task].valid : row.valid}/${denominator} valid`;
            return <div className={`ranking-row family-${modelFamily(row.model)} ${v === null ? "is-unranked" : ""}`} key={row.model}>
              <div className="ranking-label"><ModelLogo model={row.model} /><span className="ranking-name">{row.model}<small>{row.effort}</small></span><span className="ranking-value">{v === null ? process ? "Unobserved" : metric === "mae" ? "Unavailable" : "Unavailable" : format(v)}</span></div>
              <div className="ranking-track" aria-hidden="true">{v !== null && <div className="ranking-fill" style={{ "--bar-width": `${v / scale * 100}%` } as CSSProperties} />}</div>
              <div className="ranking-evidence">{count}{process && <> · <Link href={`/traces/?model=${encodeURIComponent(row.model)}`}>Read judge traces ↗</Link></>}</div>
            </div>;
          })}
        </div>
        <div className="ranking-axis" aria-hidden="true">{[0, .25, .5, .75, 1].map(t => <span key={t}>{process ? (t * scale).toFixed(0) : metric === "validity" ? `${t * scale}%` : (t * scale).toLocaleString("en-US", {maximumFractionDigits: task === "heat" && metric === "mae" ? 2 : 1})}</span>)}</div>
        <p className="figure-note">{chartDescription} No confidence intervals estimated.</p>
      </figure>

      <details className="score-details"><summary>View exact scores and coverage</summary>
        <div className="results-scroll" tabIndex={0} role="region" aria-label={`${LABELS[view]} exact scores`}>
          <table className="results-table"><thead><tr><th scope="col">Model</th>{process ? <>{CRITERIA.map(c => <th scope="col" key={c}>{CRITERION_LABELS[c]}</th>)}<th scope="col">Reviews</th></> : <><th scope="col">Prediction Elo</th><th scope="col">Valid attempts</th>{task && <th scope="col">Mean MAE · {UNITS[task]}</th>}</>}</tr></thead>
            <tbody>{rows.map(row => <tr key={row.model}><th scope="row">{row.model}<small>{row.effort} effort</small></th>{process ? <>{CRITERIA.map(c => <td key={c}>{row.criteria[c].mean === null ? "Unobserved" : `${row.criteria[c].mean.toFixed(2)} (${row.criteria[c].count})`}</td>)}<td>{row.reviews}/9</td></> : <><td>{(task ? row.tasks[task].elo : row.overall_elo)?.toFixed(1) ?? "Unavailable"}</td><td>{task ? `${row.tasks[task].valid}/3` : `${row.valid}/9`}</td>{task && <td>{row.tasks[task].mean_loss?.toFixed(task === "heat" ? 3 : 2) ?? "Unavailable"}</td>}</>}</tr>)}</tbody>
          </table>
        </div>
      </details>
      <div className="board-source"><span>8 models · 3 trials per task · 18 Oct 2026</span><Link href="/traces/">Judge traces ↗</Link><a href="/results/pooled-task-elo-v1.json" download>Match audit ↗</a><a href="/results/compact-coding-v1.json" download>Download data ↗</a></div>
    </div>
  );
}
