# PhysBench site

Public leaderboard, report and task pages for PhysBench. Next.js static export, hosted on Cloudflare Pages at https://physbench.pages.dev.

```sh
cd site
npm install
npm run dev        # http://localhost:3000
npm run deploy     # build + wrangler pages deploy
```

All leaderboard and chart numbers in `lib/data.ts` are placeholders until a held-out pilot has been scored. Text placeholders in the report are styled with `.placeholder`.

The header links to the operator dashboard at `NEXT_PUBLIC_DASHBOARD_URL` (default `http://127.0.0.1:8765`). Set it at build time to point at a hosted dashboard.
