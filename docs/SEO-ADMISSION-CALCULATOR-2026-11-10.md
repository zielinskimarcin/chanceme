# Admission calculator SEO — 2026-11-10

## Baseline and diagnosis

GSC Web, all countries/devices, verified date ranges 2026-11-02–08 vs 2026-10-26–11-01: 82/86 clicks, 184/208 impressions, CTR44.6/41.3%, average position5.7/3.9. All clicks homepage. Exact query `bocconi admission calculator`:3/3 clicks,18/19 impressions, position5.7/4.9; admissions plural4/5,9/9,4.7/3.2. The rolling24h view onNov10 (Nov9–10 local UI, preliminary, last update4.5h) reports13clicks/50impressions; exact singular query0/2/8.0. Private figures retained only as aggregate observations, no visitor exports.

Manual Google desktop sample around12:45–12:50 America/Denver, English, gl=us,pws=0, footer Provo/IP, Results are not personalized: Alexandros1/ChanceMe8 on `bocconi admission calculator`; Alexandros1/ChanceMe3 on `bocconi chance calculator`; ChanceMe1/Alexandros2 on `chance me bocconi`. This is not a global rank. No competitor GSC, exhaustive keyword database or comparable backlink counts were available. Its Reddit references are observable, but their causal ranking contribution is not established.

Homepage title and H1 already match the target. Public canonical, robots, six sitemap entries and guide responses were200 and indexable. GSC homepage inspection: indexed, smartphone crawl2026-10-31 11:44:21 (UI), fetch successful, indexing allowed, Google canonical=inspectedURL. GSC submitted sitemap remained Couldn't fetch / unknown type /0discovered, submittedOct27, no read date. Public sitemap is valid and accessible now; cause of GSC failure is not identified. A stale crawl and sitemap error are maintenance issues, not proof of the ranking cause.

## Small release and hypothesis

Keep homepage as primary URL, established title/H1/description and calculator unchanged. Add a compact below-calculator explanation of the distinction between Bocconi55/45 and the experimental estimate, with relevant SAT/Test/GPA guide links. Spell out the six already-supported programmes in the existing scope answer. Add the missing in-body calculator link in the Bocconi Test guide; SAT guide already had one. These improve input guidance, crawlable context and navigation; no guaranteed ranking effect.

Replace the site-wide fixedOct27 sitemap date with per-page authored modification dates. Only the homepage and Test guide receiveNov10. Test guide Article/OG dateModified agrees with its sitemap entry, while visible official-facts CheckedOct27 remains honest. Untouched pages retainOct27. Do not mass-refresh dates for freshness.

Sources verifiedNov10:
- https://www.unibocconi.it/en/applying-bocconi/bachelor-and-law-programs/application-and-admissions/admissions (55/45, current programmes)
- https://developers.google.com/search/docs/fundamentals/creating-helpful-content
- https://developers.google.com/search/docs/crawling-indexing/links-crawlable
- https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap
- https://developers.google.com/search/docs/appearance/title-link

## Validation and rollout

Local production/preview builds and output gates,87JS/43Python tests, exact protected preview HTTP/content and browser checks, then GitHubCI and production verification. Compare HTML against the tested build; check canonical/noindex separation and per-page dates. Homepage form/model/report/analytics sources must remain unchanged. No new dependencies, services or paid tools.

Rollback: revert this PR; previous production dpl_BAZVvXygjtjpPgWFhzTztHbohRT5 / main d48fee9fac2fb9f07019f025443ed23cb5644f1d. Existing deployment protection stays enabled. Preview builds are noindex and must not be promoted as production without a production rebuild.

After successful release, inspect sitemap and request reprocessing once if the existing error remains. Check actual response; a submission is not indexing or a ranking boost. After recrawl and at least two complete comparable weeks, compare exact singular/plural/chance-calculator query+homepage+country+device, clicks/impressions/CTR, not the whole-domain average. No new automation. UI/browser availability may limit GSC actions; record actual result separately.
