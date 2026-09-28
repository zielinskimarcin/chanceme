# Chance Me engineering and SEO contract

This repository powers `https://chance-me.com`, a Bocconi admission calculator. Work in small, testable slices and preserve the distinction between official Bocconi facts, self-reported historical profiles, and model estimates.

## Architecture

- `index.html`: static frontend, metadata, GA4, Microsoft Clarity, and Vercel Insights.
- `api/index.py`: Flask serverless API and admission heuristics.
- `api/*.csv`: versioned historical input data. Treat it as unverified self-reported data, not official admissions data.
- `vercel.json`: Vercel routing.

## Local commands

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r api/requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python api/index.py
```

Serve `index.html` from the repository root when testing the complete frontend locally; do not open it as a `file://` URL.

## Data and privacy rules

- Never commit credentials, exports containing personal identifiers, raw analytics exports, or production tokens.
- Do not send raw SAT, Bocconi Test, GPA, email, or free-text application data to analytics.
- Analytics events may contain only coarse product dimensions such as test type, program, session, result band, and CTA type.
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
- No secrets or personal data enter Git history or analytics.
- User-visible claims have a source or are clearly labeled as estimates.
- SEO releases include a before/after annotation and a rollback path.
