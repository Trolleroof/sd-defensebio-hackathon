import type { Metadata } from "next";
import { JudgeTraces } from "@/components/JudgeTraces";

export const metadata: Metadata = {
  title: "Judge traces · soliderphys-bench",
  description: "Episode-level LLM judge verdicts, cited events, and the evidence packets behind the process scores.",
};

export default function TracesPage() {
  return <div className="wrap research-page">
    <header className="research-header">
      <p className="research-kicker">Evaluation / Evidence</p>
      <h1>Judge traces</h1>
      <p className="research-deck">Read each verdict alongside the evidence the judge received.</p>
    </header>
    <JudgeTraces />
  </div>;
}
