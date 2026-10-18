// Every number in this file is a placeholder. Replace from runs/<run-id>/summary.json
// once a held-out pilot has been scored.

export const DASHBOARD_URL =
  process.env.NEXT_PUBLIC_DASHBOARD_URL ?? "http://127.0.0.1:8765";

export const RELEASE = {
  version: "v0.1",
  status: "Placeholder results",
  episodes: 48,
  updated: "Pending pilot-0",
};

export type Entry = {
  model: string;
  effort: string;
  pass: number; // Pass@1, percent
  passCi: [number, number];
  mae: number; // phase-balanced transfer MAE, bpm
  judge: number; // rubric total, 0–14
  cost: number; // USD per episode
  baseline?: boolean;
};

export const ENTRIES: Entry[] = [
  { model: "model-a", effort: "high", pass: 62, passCi: [48, 74], mae: 6.1, judge: 11.4, cost: 4.2 },
  { model: "model-b", effort: "max", pass: 55, passCi: [41, 68], mae: 6.6, judge: 10.8, cost: 7.9 },
  { model: "model-c", effort: "high", pass: 47, passCi: [33, 61], mae: 7.0, judge: 9.6, cost: 2.1 },
  { model: "model-d", effort: "medium", pass: 38, passCi: [25, 52], mae: 7.4, judge: 9.9, cost: 1.3 },
  { model: "model-e", effort: "high", pass: 31, passCi: [19, 45], mae: 7.9, judge: 8.1, cost: 3.4 },
  { model: "model-f", effort: "medium", pass: 22, passCi: [12, 36], mae: 8.6, judge: 7.2, cost: 0.6 },
  { model: "model-g", effort: "low", pass: 14, passCi: [6, 27], mae: 9.3, judge: 5.9, cost: 0.3 },
  { model: "kinetic_pop_calibrated", effort: "anchor", pass: 0, passCi: [0, 0], mae: 8.2, judge: 0, cost: 0, baseline: true },
];

export type MetricKey = "pass" | "mae" | "judge" | "cost";

export const METRICS: Record<
  MetricKey,
  { label: string; unit: string; max: number; lowerIsBetter: boolean; format: (v: number) => string }
> = {
  pass: { label: "Pass@1", unit: "%", max: 100, lowerIsBetter: false, format: (v) => `${v.toFixed(0)}%` },
  mae: { label: "Transfer MAE", unit: "bpm", max: 12, lowerIsBetter: true, format: (v) => v.toFixed(1) },
  judge: { label: "Judge score", unit: "/14", max: 14, lowerIsBetter: false, format: (v) => v.toFixed(1) },
  cost: { label: "Cost / episode", unit: "USD", max: 10, lowerIsBetter: true, format: (v) => `$${v.toFixed(2)}` },
};

// Per-horizon MAE for the top placeholder entry vs the anchor, seconds ahead → bpm.
export const HORIZON = {
  seconds: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
  agent: [2.9, 3.6, 4.2, 4.9, 5.4, 5.9, 6.3, 6.7, 7.0, 7.3, 7.6, 7.9],
  anchor: [3.1, 4.2, 5.3, 6.2, 7.0, 7.8, 8.5, 9.1, 9.6, 10.1, 10.5, 10.9],
};

export const TASKS = [
  {
    id: "hr-transfer",
    name: "Heart rate under activity transfer",
    family: "Estimator improvement",
    data: "PPG-DaLiA · real",
    budget: "3 dev calls · 60 s history · 120 s horizon",
    summary:
      "Train on sitting, driving and walking. Forecast stairs, table soccer, cycling and recovery.",
    phases: ["Build", "Investigate", "Transfer"],
  },
  {
    id: "recovery",
    name: "Personalized recovery forecasting",
    family: "Estimator improvement",
    data: "Local CSV adapter · real or pilot",
    budget: "Persistent action cap · evaluation and acquisition budgets",
    summary:
      "Forecast one participant's heart rate after exercise, with uncertainty.",
    phases: ["Build", "Investigate"],
  },
  {
    id: "campaigns",
    name: "Linked research campaigns",
    family: "Structure and retention",
    data: "Structural fixtures · synthetic",
    budget: "3 acquisition credits · 8 dev calls across 4 rounds",
    summary:
      "Four rounds in one workspace. Each round adds a condition: longer bouts, carried load, a later session. Checkpoints also score earlier conditions.",
    phases: ["Build", "Investigate", "Transfer", "Sustain"],
  },
];

export const POST_SECTIONS = [
  { id: "introduction", n: "01", title: "Introduction", sub: "Motivation" },
  { id: "environment", n: "02", title: "The environment", sub: "Data, phases, contract" },
  { id: "methodology", n: "03", title: "Methodology", sub: "Splits, anchor, judge" },
  { id: "results", n: "04", title: "Results", sub: "Scores by model" },
  { id: "failures", n: "05", title: "Failure analysis", sub: "Observed failures" },
  { id: "limitations", n: "06", title: "Limitations", sub: "Known gaps" },
];
