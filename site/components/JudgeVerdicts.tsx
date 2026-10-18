import Link from "next/link";
import { ModelLogo } from "@/components/ModelLogo";
import traces from "@/lib/judge-traces.json";

const CRITERIA = [
  { id: "evidence", label: "Evidence use" },
  { id: "experiment", label: "Experiment choice" },
  { id: "feedback", label: "Feedback integration" },
  { id: "repro", label: "Reproducibility" },
  { id: "selection", label: "Final selection" },
];
const FINDINGS = [
  { episode: "openai--gpt-6-astra-medium-heat-seed-2", criterion: "feedback", title: "Successful repair, unsupported diagnosis", context: "The retry succeeded, but its explanation named a different error from the one in the evidence." },
  { episode: "anthropic--claude-opus-5-5-medium-heat-seed-2", criterion: "feedback", title: "Retaining the better baseline", context: "After a regularization change worsened development MAE, the model kept its original candidate." },
  { episode: "google--gemini-4-argon-medium-load-seed-0", criterion: "feedback", title: "Ambitious proposal, identical results", context: "A proposed CV and physics-feature experiment returned the same development loss as the baseline." },
  { episode: "anthropic--claude-sonnet-5-5-medium-load-seed-1", criterion: "selection", title: "Sound selection, failed final attempt", context: "The model correctly selected its valid development candidate; the final predictor still failed. Process and prediction scores capture different outcomes." },
];

export function JudgeVerdicts() {
  const summaries = CRITERIA.map(criterion => {
    const values = traces.records.flatMap(record => record.report?.verdict.criteria.filter(row => row.id === criterion.id && row.score !== null).map(row => row.score as number) ?? []);
    return { ...criterion, count: values.length, mean: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null };
  });
  return <section className="judge-verdicts" id="judge-verdicts" aria-labelledby="judge-verdicts-heading">
    <div className="research-section-title"><h2 id="judge-verdicts-heading">LLM judge verdicts</h2><Link href="/traces/">Browse all episode reviews ↗</Link></div>
    <p className="judge-verdicts-deck">{traces.reviewed} saved reviews · {traces.missing} missing · GPT-6.1 Sol, Medium. Scores assess research decisions separately from prediction Elo.</p>
    <div className="judge-criterion-summary" aria-label="Observed criterion averages">{summaries.map(summary => <div className="judge-summary-item" key={summary.id}>
      <div><span>{summary.label}</span><strong>{summary.mean?.toFixed(2) ?? "—"}<small> / 4</small></strong></div>
      <div className="judge-summary-track" aria-hidden="true"><span style={{ width: `${(summary.mean ?? 0) / 4 * 100}%` }} /></div>
      <span className="judge-summary-count">{summary.count} observable reviews</span>
    </div>)}</div>
    <p className="judge-evidence-scope">Episode-weighted means over observable scores; nulls are excluded. The judge received compressed decision and development evidence, not full code or hidden test results.</p>
    <div className="judge-findings">{FINDINGS.map(finding => {
      const record = traces.records.find(row => row.id === finding.episode)!;
      const verdict = record.report!.verdict.criteria.find(row => row.id === finding.criterion)!;
      const criterion = CRITERIA.find(row => row.id === finding.criterion)!;
      return <article className="judge-finding" key={finding.episode}>
        <div className="judge-finding-model"><ModelLogo model={record.model} /><span>{record.model} · {record.task} · seed {record.seed}</span></div>
        <h3>{finding.title}</h3><p>{finding.context}</p>
        <div className="judge-verdict-label">Judge · {criterion.label}<span>{verdict.score === null ? "Unobservable" : `${verdict.score} / 4`}</span></div>
        <blockquote>{verdict.reason}</blockquote>
        <Link href={`/traces/?episode=${encodeURIComponent(record.id)}`}>Read verdict and cited events ↗</Link>
      </article>;
    })}</div>
  </section>;
}
