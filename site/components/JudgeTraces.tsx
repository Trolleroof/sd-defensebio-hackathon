"use client";

import { useEffect, useState } from "react";
import { ModelLogo } from "@/components/ModelLogo";
import snapshot from "@/lib/judge-traces.json";

const CRITERIA = { evidence: "Evidence use", experiment: "Experiment choice", feedback: "Feedback integration", repro: "Reproducibility", selection: "Final selection" };
const TASKS = { pulse: "Pulse", heat: "Heat", load: "Load" };
type Task = keyof typeof TASKS;
type Criterion = keyof typeof CRITERIA;
type Evidence = { id: string; kind: string; decision?: { hypothesis?: string; code_diff?: string }; execution_valid?: boolean; primary_loss?: number | null; failure_reason?: string | null; reason?: string; evidence_ids?: string[] };
type Packet = { events: Evidence[]; rubric: string; trajectory_sha256: string };
type Report = { packet_sha256: string; judge_configuration_sha256: string; verdict: { criteria: { id: Criterion; score: number | null; reason: string; evidence_ids: string[] }[] } };
type Episode = { id: string; model: string; effort: string; task: string; seed: number; execution_valid: boolean; status: string; packet: Packet | null; report: Report | null; provenance: { receipt_sha256: string; trajectory_sha256: string; report_sha256: string | null } };
// The exporter validates criterion IDs and evidence schemas before publication.
const DATA = snapshot as Omit<typeof snapshot, "records"> & { records: Episode[] };
const MODELS = [...new Set(DATA.records.map(record => record.model))].sort();

function EvidenceCard({ event, episode }: { event: Evidence; episode: Episode }) {
  const unit = episode.task === "pulse" ? "bpm" : episode.task === "heat" ? "°C" : "W";
  return <section className="trace-event" id={`${episode.id}-${event.id}`}>
    <div className="trace-event-heading"><code>{event.id}</code><span>{event.kind.replaceAll("_", " ")}</span></div>
    {event.decision?.hypothesis && <p>{event.decision.hypothesis}</p>}
    {event.decision?.code_diff && <span className="trace-code-reference">Code reference: {event.decision.code_diff}</span>}
    {event.kind === "development_result" && <>
      <p>{event.execution_valid ? "Valid development predictor" : "Invalid development predictor"}{event.primary_loss != null ? ` · MAE ${event.primary_loss.toFixed(4)} ${unit}` : ""}</p>
      {event.failure_reason && <p className="trace-muted">{event.failure_reason}</p>}
    </>}
    {event.reason && <p>{event.reason}</p>}
    {event.evidence_ids && <span className="trace-code-reference">Selection references: {event.evidence_ids.join(", ")}</span>}
    {event.kind === "failed_before_experiment" && <p>No experiment evidence was available to the judge.</p>}
  </section>;
}

export function JudgeTraces() {
  const [model, setModel] = useState("all");
  const [task, setTask] = useState("all");
  const [seed, setSeed] = useState("all");
  const [status, setStatus] = useState("all");
  const [active, setActive] = useState(DATA.records[0].id);
  useEffect(() => {
    const query = new URLSearchParams(window.location.search);
    const requestedModel = query.get("model");
    if (requestedModel && MODELS.includes(requestedModel)) setModel(requestedModel);
    const episode = DATA.records.find(record => record.id === query.get("episode"));
    if (episode) { setActive(episode.id); setModel(episode.model); }
  }, []);
  const records = DATA.records.filter(record => (model === "all" || record.model === model) && (task === "all" || record.task === task) && (seed === "all" || String(record.seed) === seed) && (status === "all" || record.status === status));
  const selected = records.find(record => record.id === active) ?? records[0];
  return <section className="trace-browser" aria-label="Episode judge trace browser">
    <div className="trace-intro"><p><strong>{DATA.reviewed}/{DATA.episodes} reviews saved</strong> · {DATA.missing} missing. Judge: {DATA.judge.model} · {DATA.judge.effort} effort.</p><p>Five criteria, scored 0–4; null means unobservable. These are the saved explanations and compressed decision/development evidence. The judge did not receive competitor identities, hidden test scores, or full executable code.</p><a href="/results/judge-traces.json" download>Download all judge traces ↗</a><details className="trace-raw"><summary>Judge instructions and grading schema</summary><p>{DATA.judge.instructions}</p><pre>{JSON.stringify(DATA.judge.grade_tool, null, 2)}</pre></details></div>
    <div className="trace-filters">
      <label>Model<select value={model} onChange={e => setModel(e.target.value)}><option value="all">All models</option>{MODELS.map(name => <option key={name}>{name}</option>)}</select></label>
      <label>Task<select value={task} onChange={e => setTask(e.target.value)}><option value="all">All tasks</option>{Object.entries(TASKS).map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label>
      <label>Trial seed<select value={seed} onChange={e => setSeed(e.target.value)}><option value="all">All seeds</option>{[0, 1, 2].map(value => <option key={value}>{value}</option>)}</select></label>
      <label>Review<select value={status} onChange={e => setStatus(e.target.value)}><option value="all">All reviews</option><option value="reviewed">Saved reviews</option><option value="missing">Missing reviews</option></select></label>
    </div>
    <p className="trace-count" role="status">{records.length} matching episodes</p>
    <div className="trace-workspace">
      <div className="trace-episode-list" role="group" aria-label="Choose an episode">{records.map(record => <button key={record.id} data-episode-id={record.id} aria-pressed={selected?.id === record.id} onClick={() => setActive(record.id)}><span className="trace-episode-model"><ModelLogo model={record.model} />{record.model}</span><span>{TASKS[record.task as Task]} · seed {record.seed}<span className="trace-episode-status">{record.status === "reviewed" ? "Reviewed" : "Missing review"}</span></span></button>)}</div>
      {selected ? <article className="trace-inspector" data-episode-id={selected.id} key={selected.id} aria-label="Selected episode review">
        <header className="trace-inspector-header"><h2>{selected.model}</h2><p>{TASKS[selected.task as Task]} · seed {selected.seed} · {selected.effort} effort · {selected.execution_valid ? "Valid predictor" : "Failed attempt"}</p><a href={`/traces/?episode=${encodeURIComponent(selected.id)}`}>Link to this episode ↗</a></header>
        <h3>Judge verdict</h3>
        {selected.report ? <div className="trace-criteria">{selected.report.verdict.criteria.map(criterion => <section className="trace-criterion" key={criterion.id}>
          <div className="trace-criterion-heading"><h4>{CRITERIA[criterion.id]}</h4><span>{criterion.score === null ? "Unobservable" : `${criterion.score} / 4`}</span></div>
          <p>{criterion.reason}</p>
          <div className="trace-citations">{criterion.evidence_ids.map(id => <a key={id} href={`#${selected.id}-${id}`}>{id} ↓</a>)}</div>
        </section>)}</div> : <p className="trace-missing">No validated judge review was saved for this episode. It contributes no judge score. Any available evidence packet is shown below.</p>}
        <h3>Evidence shown to the judge</h3><p className="trace-muted">Recorded in event order. Development errors below are distinct from hidden test results.</p>
        {selected.packet ? selected.packet.events.map(event => <EvidenceCard key={event.id} event={event} episode={selected} />) : <p className="trace-muted">No judge evidence packet was saved.</p>}
        <details className="trace-raw"><summary>Exact packet and verdict JSON</summary><pre>{JSON.stringify({ packet: selected.packet, report: selected.report }, null, 2)}</pre></details>
        <details className="trace-raw"><summary>Evidence provenance</summary><dl><dt>Receipt SHA-256</dt><dd>{selected.provenance.receipt_sha256}</dd><dt>Trajectory SHA-256</dt><dd>{selected.provenance.trajectory_sha256}</dd><dt>Judge packet SHA-256</dt><dd>{selected.report?.packet_sha256 ?? "No validated report"}</dd><dt>Judge configuration SHA-256</dt><dd>{DATA.judge.configuration_sha256}</dd><dt>Report file SHA-256</dt><dd>{selected.provenance.report_sha256 ?? "No validated report"}</dd></dl></details>
      </article> : <p className="trace-empty">No episodes match these filters. Change a filter to see reviews.</p>}
    </div>
  </section>;
}
