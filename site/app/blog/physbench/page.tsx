import type { Metadata } from "next";
import { HorizonChart } from "@/components/HorizonChart";
import { Leaderboard } from "@/components/Leaderboard";
import { Reveal } from "@/components/Reveal";
import { TocSpy } from "@/components/TocSpy";
import { POST_SECTIONS, RELEASE } from "@/lib/data";

export const metadata: Metadata = {
  title: "PhysBench report",
  description: "How PhysBench measures whether agents can build personalized physiology predictors.",
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
            <h1>Measuring research, not recall, on wearable physiology</h1>
            <p>
              An environment where agents fit, test and submit personalized heart-rate predictors under fixed
              budgets. They're scored on hidden recordings and judged on how they got there.
            </p>
            <div className="byline">
              <span>DefenseBio</span>
              <span>{RELEASE.updated}</span>
              <span>Results below are placeholders</span>
            </div>
          </Reveal>
        </div>
      </header>

      <div className="wrap article">
        <TocSpy />
        <article className="prose">
          <H2 id="introduction" />
          <p>
            Most physiology models are fit once to a population and then applied to everyone. A wearable worn by
            one person for weeks makes a different question possible: can a model learn <em>this</em> person&apos;s
            response to load, and does it hold up when they do something new?
          </p>
          <p>
            That's research work. You inspect data, choose a structure, test it against limited feedback and
            decide when to stop. Existing agent benchmarks score code tasks with a known answer, or ML tasks on
            public data that may be memorized. PhysBench scores the predictor on recordings the agent never sees,
            and scores the reasoning separately.
          </p>
          <p className="placeholder">
            Placeholder: one paragraph positioning PhysBench against long-horizon research benchmarks once pilot
            results are in.
          </p>

          <H2 id="environment" />
          <p>
            Each episode is one target participant from PPG-DaLiA: 15 people, an ECG-derived heart-rate reference,
            wrist and chest accelerometers and labelled activities. The agent gets a Python workspace, the
            target&apos;s support recordings, four reference participants and a fitted conventional baseline.
          </p>
          <table>
            <thead>
              <tr><th>Phase</th><th>Agent has</th><th>Agent does</th></tr>
            </thead>
            <tbody>
              <tr><td><strong>Build</strong></td><td>Support data from familiar activities, reference cohort, baseline</td><td>Inspect, fit a predictor</td></tr>
              <tr><td><strong>Investigate</strong></td><td>≤3 dev-feedback calls on held-out familiar windows</td><td>Revise with evidence</td></tr>
              <tr><td><strong>Transfer</strong></td><td>Submit</td><td>Hidden scoring on stairs, table soccer, cycling and recovery</td></tr>
            </tbody>
          </table>
          <p>
            The contract is <code>predict(history, future) → (60,)</code> bpm. History is the last 60 s of heart
            rate and inputs. Future is the next 120 s of inputs only. Consequential actions are rejected unless
            they come with a written justification of at least 40 characters.
          </p>

          <H2 id="methodology" />
          <h3>Splits</h3>
          <p>
            Four participants form a visible reference cohort, four are used for calibration and freezing the
            criteria, and six are held out. One participant moved from held-out to development after their
            baseline numbers were viewed during calibration. That move is recorded rather than hidden.
          </p>
          <h3>Anchor and pass criteria</h3>
          <p>
            The anchor is <code>kinetic_pop_calibrated</code>: population heart-rate kinetics with a personal
            offset and gain fitted on support. A pass needs a valid run, an improvement over the anchor of at
            least a frozen margin, an explicit submission with a report, and no integrity flags. Criteria are
            hashed into a frozen config before any held-out run, and the held-out runner refuses to start
            without them.
          </p>
          <h3>Judge</h3>
          <p>
            A fixed seven-item rubric, 0–2 per item, plus two integrity flags. The judge never sees the model's
            identity or its hidden accuracy.
          </p>

          <H2 id="results" />
          <p className="placeholder">
            Placeholder: headline finding in two sentences. For example, which models clear the anchor, by how
            much, and whether judge scores track accuracy.
          </p>
          <Reveal className="figure">
            <Leaderboard />
            <figcaption>Figure 1 · Placeholder. Pass@1, transfer MAE, judge score and cost per episode.</figcaption>
          </Reveal>
          <Reveal className="figure">
            <HorizonChart />
            <figcaption>Figure 2 · Placeholder. Transfer MAE (bpm) by forecast horizon for the top entry vs the anchor.</figcaption>
          </Reveal>
          <p className="placeholder">
            Placeholder: per-activity breakdown (stairs vs cycling vs recovery) and paired model differences
            with bootstrap CIs.
          </p>

          <H2 id="failures" />
          <p className="placeholder">
            Placeholder: two or three failure modes observed in traces, each with one linked trace excerpt.
          </p>
          <table>
            <thead>
              <tr><th>Model</th><th>What happened</th><th>Caught by</th></tr>
            </thead>
            <tbody>
              <tr><td className="mono">model-x</td><td><span className="placeholder">Placeholder: failure description</span></td><td className="mono">audit</td></tr>
              <tr><td className="mono">model-y</td><td><span className="placeholder">Placeholder: failure description</span></td><td className="mono">judge</td></tr>
              <tr><td className="mono">model-z</td><td><span className="placeholder">Placeholder: failure description</span></td><td className="mono">hidden score</td></tr>
            </tbody>
          </table>

          <H2 id="limitations" />
          <ul>
            <li>
              <strong>Isolation is partly best-effort.</strong> Per-episode users, root-only labels and resource
              limits are enforced. The network guard isn't kernel-level, because the container can't create
              network namespaces.
            </li>
            <li>
              <strong>One dataset, one target.</strong> Heart rate from movement on PPG-DaLiA. Other channels
              are wired in the adapter but not yet enabled.
            </li>
            <li>
              <strong>The judge is unvalidated.</strong> Its agreement with expert reviewers hasn't been
              measured.
            </li>
            <li>
              <strong>Linked campaigns use synthetic fixtures.</strong> They test structure and retention, not
              human physiology.
            </li>
          </ul>
          <p className="placeholder">Placeholder: what changes in v0.2.</p>
        </article>
      </div>
    </>
  );
}
