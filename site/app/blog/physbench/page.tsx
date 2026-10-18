import type { Metadata } from "next";
import { HorizonChart } from "@/components/HorizonChart";
import { Leaderboard } from "@/components/Leaderboard";
import { Reveal } from "@/components/Reveal";
import { TocSpy } from "@/components/TocSpy";
import { POST_SECTIONS, RELEASE } from "@/lib/data";

export const metadata: Metadata = {
  title: "PhysBench report",
  description: "Setup, methodology and results for PhysBench.",
};

function H2({ id }: { id: string }) {
  const s = POST_SECTIONS.find((x) => x.id === id)!;
  return (
    <h2 id={s.id}>
      <span className="n">{s.n}</span>
      {s.title}
    </h2>
  );
}

export default function Post() {
  return (
    <>
      <header className="post-head">
        <div className="wrap">
          <Reveal>
            <span className="eyebrow">Report · {RELEASE.version}</span>
            <h1>PhysBench {RELEASE.version}</h1>
            <p>Setup, methodology and results. Results are placeholders.</p>
            <div className="byline">
              <span>DefenseBio</span>
              <span>{RELEASE.updated}</span>
            </div>
          </Reveal>
        </div>
      </header>

      <div className="wrap article">
        <TocSpy />
        <article className="prose">
          <H2 id="introduction" />
          <p>
            Heart-rate models are usually fit to a population. With weeks of data from one wearer, you can fit a
            model to that person instead. We test whether an agent can do that, and whether its model still works
            on activities missing from the training data.
          </p>
          <p>
            The predictor is scored on recordings the agent never sees. A separate judge reads the trace and
            grades how the agent worked.
          </p>
          <p className="placeholder">Placeholder: comparison to related benchmarks.</p>

          <H2 id="environment" />
          <p>
            Data comes from PPG-DaLiA: 15 participants, ECG-derived heart rate, wrist and chest accelerometers,
            labelled activities. Each episode targets one participant. The agent gets a Python workspace, that
            participant&apos;s support recordings, four reference participants and a fitted baseline.
          </p>
          <table>
            <thead>
              <tr><th>Phase</th><th>Input</th><th>Action</th></tr>
            </thead>
            <tbody>
              <tr><td><strong>Build</strong></td><td>Support data, reference cohort, baseline</td><td>Fit a predictor</td></tr>
              <tr><td><strong>Investigate</strong></td><td>Up to 3 dev-feedback calls</td><td>Revise</td></tr>
              <tr><td><strong>Transfer</strong></td><td>Submit</td><td>Scored on stairs, table soccer, cycling, recovery</td></tr>
            </tbody>
          </table>
          <p>
            Interface: <code>predict(history, future) → (60,)</code> in bpm. <code>history</code> is the last 60 s
            of heart rate and inputs. <code>future</code> is the next 120 s of inputs. Dev feedback, submission and
            writes to <code>submission/</code> need a justification of at least 40 characters.
          </p>

          <H2 id="methodology" />
          <h3>Splits</h3>
          <p>
            4 reference, 4 development, 6 held-out. S3 moved from held-out to development after its baseline
            scores were seen during calibration. S6 is excluded because its recording is truncated.
          </p>
          <h3>Anchor and pass</h3>
          <p>
            The anchor, <code>kinetic_pop_calibrated</code>, uses population heart-rate kinetics with a personal
            offset and gain fit on support data. A run passes if it executes, beats the anchor by the frozen margin,
            submits a report and has no integrity flags. Criteria are hashed before any held-out run.
          </p>
          <h3>Judge</h3>
          <p>Seven items scored 0–2, plus two integrity flags. Model name and hidden accuracy are withheld.</p>

          <H2 id="results" />
          <p className="placeholder">Placeholder: headline result.</p>
          <Reveal className="figure">
            <Leaderboard />
            <figcaption>Figure 1 · Placeholder scores.</figcaption>
          </Reveal>
          <Reveal className="figure">
            <HorizonChart />
            <figcaption>Figure 2 · Placeholder. MAE (bpm) by horizon, top entry vs anchor.</figcaption>
          </Reveal>
          <p className="placeholder">Placeholder: per-activity results and paired differences with bootstrap CIs.</p>

          <H2 id="failures" />
          <p className="placeholder">Placeholder: failure cases with trace excerpts.</p>
          <table>
            <thead>
              <tr><th>Model</th><th>Failure</th><th>Caught by</th></tr>
            </thead>
            <tbody>
              <tr><td className="mono">model-x</td><td><span className="placeholder">Placeholder</span></td><td className="mono">audit</td></tr>
              <tr><td className="mono">model-y</td><td><span className="placeholder">Placeholder</span></td><td className="mono">judge</td></tr>
              <tr><td className="mono">model-z</td><td><span className="placeholder">Placeholder</span></td><td className="mono">hidden score</td></tr>
            </tbody>
          </table>

          <H2 id="limitations" />
          <ul>
            <li>
              <strong>Isolation.</strong> Per-episode users, root-only labels and resource limits are enforced. The
              network block is a Python hook, not a network namespace.
            </li>
            <li><strong>Scope.</strong> One dataset, one target: heart rate from movement.</li>
            <li><strong>Judge.</strong> Agreement with expert reviewers hasn&apos;t been measured.</li>
            <li><strong>Campaigns.</strong> Synthetic fixtures. They don&apos;t model human physiology.</li>
          </ul>
          <p className="placeholder">Placeholder: v0.2 changes.</p>
        </article>
      </div>
    </>
  );
}
