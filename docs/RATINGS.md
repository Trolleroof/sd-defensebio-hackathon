# Campaign Elo

Elo is a leaderboard presentation layer over measured campaign outcomes. The raw prediction, uncertainty, retention, execution and research-quality evidence remains available. Ratings do not select training rewards or grant preferred models a rank.

## Frozen v1 rules

- One match is two competitors on the same complete campaign case under the same declared evidence/resource protocol. Forecast samples, individual conditions and successive checkpoints do not become separate matches.
- Prediction Elo compares final hidden condition-balanced MAE. Lower error wins. Uncertainty Elo independently compares the final proper 90% interval score. No weighted mixture of raw units or judge scores is used.
- All competitors start at 1000. A recorded seed shuffles a complete round-robin schedule once, with K=32 and a logistic expectation scale of 400. Update: `expected_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))`; `change = K * (observed_A - expected_A)`; equal/opposite changes apply to both players. There are no model-specific priors or bonuses.
- Differences within an absolute 1e-9 tolerance draw. A valid executable submission beats an invalid run. Two invalid runs are skipped and do not earn draws. A competitor with zero valid cases is ineligible for a displayed rank. Missing runs must be supplied explicitly as failures; incomplete and duplicate case batches are rejected.
- Input row order cannot change ratings: cases and competitors are canonicalized before the seeded schedule. Sequential Elo still depends on schedule, opponent pool, K and case selection; these fields and all pairwise outcomes are exported.

This is our declared benchmark adaptation, not an implementation of FIDE chess eligibility or rating regulations. The underlying actual-versus-expected update pattern is described in the [FIDE rating regulations](https://handbook.fide.com/chapter/B022024).

## Interpretation

A fine-tuned model reaches the top only if its observed campaign outcomes earn that rank. Keep the base model, numerical/search baselines and chosen frozen agents in the same evaluation pool; record exact checkpoint, prompt, decoding and budget versions. Choose checkpoints on development data, then evaluate the frozen selection on an untouched final batch. Do not tune against final Elo and reuse those same cases as held-out proof.

Current ratings are provisional: only three related synthetic cases, fixed conventional competitors, one model family, and no repeated stochastic LLM trials. Ratings have no estimated confidence intervals and should not be presented as a statistically established model leaderboard. The trained model has not been run or added as a fabricated row. Different benchmark versions/panels require fresh pools or an explicit anchoring protocol; Elo values from unrelated pools are not directly comparable.

## Run

```sh
python3 -m benchmark campaign-demo .benchmark-runs/elo-demo --cases 3
python3 -m benchmark rate .benchmark-runs/elo-demo/report.json .benchmark-runs/elo-demo/recomputed-ratings.json
```

Campaign reports retain raw errors and include a prediction Elo table. `ratings.json` contains separate prediction and uncertainty tables, configuration, matched cases, W/D/L counts and every update. The `rate` command also works with an existing complete campaign report; identity-based fine-tune advantages are explicitly regression-tested.

External judge outcomes remain a separate pending panel. Elo does not create research-quality evidence where no judgment has been collected.

## Existing real-data PhysBench runs

```sh
python3 -m benchmark rate-physbench runs/pilot-0 runs/pilot-0/elo.json
```

This offline adapter reads `results.json` and `run_meta.json` from the existing PhysBench runner. It validates a complete subject/repetition/model batch, applies explicit-submission, executable-validity and the existing integrity gate, then compares hidden phase-balanced MAE. Beating the anchor margin is not an additional ranking gate. Any provider/harness failure excludes the entire matched subject/repetition for all competitors, with exclusions exported; those failures do not become model losses. Development and held-out runs retain their split labels and configuration hashes. No rating is produced for an entirely excluded batch.

This path neither launches new paid runs nor invents results from the earlier PR's narrative. Repetitions of the same subject are correlated. PhysBench's current contract has no forecast intervals, so no uncertainty Elo is fabricated for it.
