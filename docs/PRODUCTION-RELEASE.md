# Campus production release — 2026-10-27

## Scope and source of truth

Publishes the approved Chance Me redesign, GPA calculator, English SAT lessons, two substantial guides and separate full report. The existing canonical domain and guide URLs are preserved. Existing historical root HTML/CSV remain in Git for provenance but are excluded from the static output and runtime data loading. Edit `src/`, then build; do not edit `public/` or legacy root HTML.

## Frozen model

`lib/engine/release/manifest.json` records the source hashes, exported model version, hashes of runtime modules and snapshot. The serving snapshot has 378 SAT reports, 93 Bocconi Test reports and 49 BT fit IDs. Only id, score, GPA, course, session, cycle and outcome are exported from historical frames, plus minimal public source-status metadata and numeric model state. No respondent timestamps, raw source sheets, free text, private validation files or training are required at runtime. The loader checks hashes and fails closed. Model probabilities remain experimental estimates from self-reported historical reports, not verified admissions outcomes or official cutoffs.

The production adapter retains the reviewed `/api/estimate`, `/api/options`, `/api/reports`, `/api/report.pdf` contracts. PDF export uses the currently selected programme/round for both estimate and records, not a stale original profile. ReportLab packaged Vera fonts remove the macOS font dependency.

## Technical SEO

Six indexable canonical pages: homepage, GPA calculator, SAT tutoring, two guides and privacy. Reports and GPA source notes carry noindex; private profiles are URL fragments rather than crawlable parameter pages. Static HTML exposes complete guide text, sources, internal links and image captions without client rendering. Production and preview robots are separated. The sitemap includes canonical indexable URLs only. Titles/descriptions are distinct; homepage retains its established search title. WebApplication, Service/Offer, Article and Breadcrumb schemas reflect actual page content. No fabricated ratings or admissions guarantees. OG images use the real interface or the credited campus photograph. Asset URLs have content versions.

## Controlled analytics

Existing GA4 stream G-TQX27XKE8K, Clarity vx5oj467o6 and Vercel Web Analytics are retained. No optional measurement libraries load before affirmative analytics consent; rejection and browser privacy signals disable measurement. Six-month preference, visible withdrawal in footer. GA4 Enhanced Measurement was disabled in its existing stream following Marcin's explicit approval on 2026-10-27 (verified 2026-10-28 05:04 UTC); historical data remains.

GA automatic page views are disabled in code. Manual page locations contain allowlisted paths only: no query, hash or referring profile URL. Events use a strict allowlist; SAT/BT score, GPA, grades, email and free text are discarded. Events cover calculator start/submission/result/failure, GPA calculation/transfer, guide and CTA clicks, PDF download, full report opening and Cal.com successful-booking callback. A click is not a confirmed booking. No booking is created in QA. GA4 custom dimensions/key event configuration is not asserted unless separately verified.

Clarity is excluded from the admission calculator, GPA and report entirely; it may run only on consented public lesson/guide pages without query or fragment. Its dashboard receipt was not verified. Vercel beforeSend sanitizes the URL and suppresses events after withdrawal. No ad-storage consent is granted. Search Console is already linked to this GA4 stream.

## Visible content changes beyond the approved prototype

- Added Privacy & cookies page, optional analytics notice and footer settings.
- Added accurate search/social metadata and structured data; these do not rewrite the accepted marketing sections.
- Corrected accidental sentence ending “using SAT or Bocconi Test guide.” to “using SAT or Bocconi Test.”
- Preserved SAT hero “SAT tutoring for Bocconi. Go for 1500+.” and the exact approved Marcin biography with only “get into Bocconi.” bold.
- Replaced a stale Polish tutoring-site link in the full report’s hidden About section with the English lessons page. No invented testimonials or own SAT score added.

## Search baseline and backlink policy

A dated private Search Console export and audit are retained outside Git. Both existing guide URLs were unknown to Google at audit time; sitemap fetch failed in Search Console. Priority is valid discovery, indexable static content and preserved canonical paths. Search position and traffic are observations, not guarantees. Backlink-provider totals are not a ranking metric: useful GSC referring pages are distinguished from inherited spam/domain-list links. No links were purchased, automated posts sent or disavow file submitted.

## Release and rollback

Existing project `bocconi-chanceme`, team `zielinskimarcins-projects`; no DNS, billing, security protection or secrets changes. Required path: local regression checks, Linux CI, protected preview HTTP/API/PDF checks and visual QA, then merge and production verification. The previous production deployment is `dpl_8Q6rp2yZXUuwN7WXYmHpphbUkryc`, commit `a05b4c305e0b3c457b4630ca617f8b02fe40687a`. Roll back using the official Vercel rollback to that deployment in the same scope, then revert the release merge in Git to keep future deployments consistent. Do not disable deployment protection. Analytics-setting rollback is separate: re-enable the existing stream's prior Enhanced Measurement settings only if restoring automatic collection is expressly intended.

Independent pre-release parity: 216 complete evaluate() responses match the source engine exactly in separate Python 3.9/3.12 processes, across both tests, six programmes, Early/Winter and three score/GPA levels.


## Footer refinement — 2026-10-28

Replaced the loosely spaced footer rows with a brand/short description on the left, a consistent six-link navigation grid on the right, and a quiet legal/privacy row below a divider. All calculators, lessons, guides, privacy, report and404 inherit the same footer. Privacy and cookie controls are authored in the footer templates rather than appended outside the legal row by the build. Existing destinations and consent behaviour remain intact. Rollback baseline: productiondpl_FHoqPpQ2Wb8dCwDoyR5bTfjYMJ1E / merge37853c85a09b70555a83b9fff946f0893827264a.


## Preferred Google site name — 2026-10-28

Following the owner's explicit choice, the homepage's single WebSite node now declares “Chance Me for Bocconi” as name and “Chance Me” as alternateName. Its URL matches the canonical homepage. All generated pages use the same preferred og:site_name. Page titles, descriptions, visible copy and favicon assets/references are unchanged. Four green raster identity concepts are previews awaiting selection and are not published.

This corrects the previously reversed preferred/alternate names. Google chooses the displayed site name after recrawling; deployment does not guarantee an immediate search-result change. The dated Search Console baseline remains in private QA records referenced above; this is a brand-presentation change, not a claim of ranking improvement. Rollback baseline: production dpl_ARiDM4LfUpK9xFVe3Ac7NDjqepjC, main74af3dd1a56881763d198ed42aa242ad8c13e7dd; revert this metadata commit for subsequent builds.


## Approved green percent favicon — 2026-10-28

The owner selected the Percent raster identity and requested installation. Replaced published favicon references with centrally generated PNG96/32/16 and Apple180 links across every page; added root ICO fallback16/32/48. Larger PNG192/512 exports and the original generated master/provenance are retained. Raster design stays faithful to the chosen dark-green/mint/white mark. Icon URLs are stable and revalidate their cache, while other assets keep content versioning and existing cache behaviour.

The preferred WebSite and Open Graph name remains Chance Me for Bocconi, with Chance Me as alternate. Added application-name for saved browser applications. No page titles, marketing text, layout, model or analytics changes. Crawlability and actual image dimensions must be verified on preview and public production. Google can take days to weeks to adopt a favicon/site name and does not guarantee display; request homepage recrawl only after public verification.

Rollback: previous production dpl_BRDvCSzvMUmgheq8QCEignkDb7kh, main0be89e1420c906d845d0f9fd4c8c89147e507464; revert the favicon release for subsequent builds.
