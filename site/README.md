# soliderphys-bench site

Public leaderboard, report and task pages for soliderphys-bench. Next.js static export, hosted on Cloudflare Pages at https://soliderphys-bench.pages.dev. The previous https://physbench.pages.dev address is retained for existing links.

```sh
cd site
npm install
npm run dev        # http://localhost:3000
npm run deploy     # build + wrangler pages deploy
```

The site publishes the sealed compact-coding-v1 pilot: 72 attempts, 33 valid predictors, 65/72 process reviews. `lib/results.json` is the public data snapshot, also available at `public/results/compact-coding-v1.json`. Keep these byte-identical. No private labels, recordings, full traces or credentials are published. Scores were checked against all receipt and judge-packet hashes before export.

The header links to the operator dashboard at `NEXT_PUBLIC_DASHBOARD_URL` (default `http://127.0.0.1:8765`). Set it at build time to point at a hosted dashboard.

The public charts use `pooled-task-elo-v1`, a post-run reanalysis of those same
outcomes. All model pairs are compared within each task and agent seed. Lower
MAE wins; valid beats failed; two failures skip. Overall Elo pools the nine
matched task/seed slots with initial 1000, K=32, scale=400, order seed 0. All
participants retain a numeric rating, including models with zero valid attempts.
Original complete-suite ratings remain in `suite_elo`; original task values
remain in `original_elo`. MAE and judge criteria are unchanged.

Reproduce without API calls:

```sh
python3 scripts/rate_results.py
python3 -m unittest discover -s scripts -p 'test_*.py'
```

The downloadable `public/results/pooled-task-elo-v1.json` records every scheduled
comparison, skipped match, rating update and final rating. These ratings describe
this fixed opponent pool and protocol, not population-level performance.

Model logos are vendored SVGs from Lobe Icons; source commit and license are in `public/model-logos/`.
