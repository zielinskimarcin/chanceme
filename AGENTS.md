# Chance Me engineering and SEO contract

This repository powers `https://chance-me.com`, a Bocconi admission calculator. Work in small, testable slices and preserve the distinction between official Bocconi facts, self-reported historical profiles, and model estimates.

## Architecture

- `src/pages`, `src/static`, `src/guides`: authored UI, assets and Markdown guides. These are the active sources; root-level legacy HTML/CSS/CSV are historical and are not published.
- `scripts/build.mjs`: deterministic static build into ignored `public/`, intermediate `dist/` for regression tests. Preview builds are noindex.
- `api/index.py`: sole Flask serverless entry point. `lib/engine/` holds the integrity-checked frozen model and minimal pseudonymous serving data; no training or source sheets at runtime.
- `lib/report_pdf.py`: portable PDF rendering using ReportLab's packaged fonts.
- `vercel.json`: output directory, API routing and headers.
- `docs/PRODUCTION-RELEASE.md`: provenance, measurement contract and rollback.

## Local commands

```bash
npm ci --ignore-scripts
npm run build:production
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/check.py --production
npm run test:js
.venv/bin/python -m unittest discover -s tests -v
PORT=18782 .venv/bin/python -B -m api.index
```

Test the complete application over HTTP. Never serve the repository root or deploy the private research lab. Engine updates must regenerate the manifest through the reviewed packaging workflow and repeat prediction parity checks.

## Data and privacy rules

- Never commit credentials, exports containing personal identifiers, raw analytics exports, or production tokens.
- Owner scope update 2026-10-29: the private first-party product collector may retain explicitly submitted SAT/Bocconi Test score and GPA, program/round, tab visit ID, event time, path, sanitized source and approximate network country/region/city for 30 days. Keep 90-day coarse aggregates backward compatible.
- Never send names, emails, free text, raw IP addresses or credentials to the collector. Detailed history is owner-only, never a public API or committed export.
- Document the source date and methodology before adding or changing historical records.
- Never present the output as a guarantee of admission or as an official Bocconi result.

## SEO operating rules

- Start every SEO change from a dated Search Console baseline: clicks, impressions, CTR, average position, top queries, top pages, and countries.
- Match one primary search intent per indexable page. Avoid thin programmatic pages and invented claims.
- Prefer official Bocconi sources for current admissions rules. Record the source and review date when those rules affect copy or the model.
- Keep canonical URLs, redirects, sitemap, robots.txt, titles, descriptions, structured data, and internal links consistent.
- Measure calculator completion and CTA conversion alongside organic clicks; traffic alone is not the product outcome.
- No link spam, fake recommendations, undisclosed self-promotion, or automated Reddit posting.

## Definition of done

- Tests pass locally and in CI.
- Invalid API input returns a controlled 4xx response.
- No secrets or visitor records enter Git history; telemetry follows the explicit owner scope and private access above.
- User-visible claims have a source or are clearly labeled as estimates.
- SEO releases include a before/after annotation and a rollback path.
