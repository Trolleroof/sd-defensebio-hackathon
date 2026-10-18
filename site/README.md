# PhysBench site

Public leaderboard, report and task pages for PhysBench. Next.js static export, hosted on Cloudflare Pages at https://physbench.pages.dev.

```sh
cd site
npm install
npm run dev        # http://localhost:3000
npm run deploy     # build + wrangler pages deploy
```

The site publishes the sealed compact-coding-v1 pilot: 72 attempts, 33 valid predictors, 65/72 process reviews. `lib/results.json` is the public data snapshot, also available at `public/results/compact-coding-v1.json`. Keep these byte-identical. No private labels, recordings, full traces or credentials are published. Scores were checked against all receipt and judge-packet hashes before export.

The header links to the operator dashboard at `NEXT_PUBLIC_DASHBOARD_URL` (default `http://127.0.0.1:8765`). Set it at build time to point at a hosted dashboard.
