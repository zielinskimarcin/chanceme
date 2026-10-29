# Anonymous product metrics on Vercel Hobby

Prepared 2026-10-29. Implementation is staged; activation requires the dedicated Upstash Free connection and protected preview QA. No new account, integration, credential, paid plan or production setting has been created by this change.

## Purpose and design

The owner requested useful action and score-distribution statistics without a popup. Keep existing Vercel Web Analytics for pageviews/visitors/referrers. Its Hobby plan does not include custom events. A separate first-party Node function accepts small allowlisted batches after successful UI actions and writes independent daily counters to a private Redis database. This is our own report, not a paid Vercel Analytics feature. The Flask estimate/model/PDF API is unchanged and never waits for telemetry.

The proposed storage is a **dedicated Chance Me Upstash Redis Free database** in an EU region, connected only to this Vercel project. The published provider offer is $0,256MB,500K commands/month; verify the actual Marketplace plan before provisioning. Do not use a trial, pay-as-you-go fallback or an unrelated application's database. Install no third-party browser SDK; the server uses native fetch to the Redis REST API. Secrets stay in the Vercel secret store, never the browser, logs, Git, generated report, or commands printed to the user.

## Measured actions

- calculator_completed: explicit main-form calculation with an actual response. Ineligible/unsupported statuses remain distinguishable. Automatic report/filter recalculations do not count as new user calculations.
- calculator_failed: explicit calculation request failed; fixed reason only.
- gpa_calculated/gpa_transferred: explicit GPA interactions, without subject grades.
- report_rendered: full report estimate and comparison actually populated, once per report render. A pageview alone is not this event.
- report_failed: full report estimate or comparison failed, no payload copied.
- pdf_generated/pdf_failed: nonempty application/pdf received, or generation failure observed by the page. Aborted requests do not count as failures.
- pdf_download_initiated: browser download invoked. Does not prove saving to disk.
- cta_clicked: allowlisted local navigation, lesson, booking and WhatsApp clicks, with coarse placement.
- booking_confirmed: Cal.com's bookingSuccessfulV2 callback; no appointment details. Missing callback means unknown, not no booking.

These are anonymous action counts, not unique applicants, a linked user funnel, confirmed leads, verified results or attribution from a specific traffic source. Blocking, connection loss, browser shutdown, sampling/caps and cancelled events can undercount; repeated actions count separately. This release cannot reconstruct historical actions.

## Data minimization and retention

Only calculator_completed may carry test type, programme, round, result status/band and locally generated score/GPA ranges. SAT uses50-point bands with1550–1600 merged; BT uses2-point bands with48–50 merged; GPA uses0.25 bands with9.75–10.00 merged. All exact inputs, emails, names, grades, free text, URLs, referrers and other unexpected fields reject the batch. Browser and server schemas are tested against each other.

Each score/programme/round histogram is stored independently, never as a combined profile. No event rows or visitor timeline is retained. Each action receives an independent in-memory random UUID solely for replay protection; the server stores only its hash in a separate daily dedup key. Hashes are unlinked to histogram bins or event rows and expire within48h. Metrics keys expire at UTC midnight90days after that day's start: at most90days, even with no future traffic. Redis performs expiry; no new cron or agent automation is necessary. Raw request bodies are not written to application logs. Hosting providers still process network requests under their own policies.

No analytics cookies, browser fingerprint, IP-based identifier or analytics local/session storage ID is added. Existing functional drafts and nonidentifying on/off preference remain. DNT/GPC, old refusals and the current opt-out suppress collection. Backend also rejects DNT/GPC signals and cross-origin browser requests. Server does not receive a client opt-out token because the browser sends no event when opted out.

The owner-only report suppresses distribution bins with fewer than5observations and never contains hashes. This is data minimization, not a mathematical anonymity guarantee; raw private aggregate storage still contains small counts. Retention/access and the collection purpose remain documented in privacy.

## No-popup basis and limits

The design is single-site product reliability/usefulness statistics, with immediate aggregation, short retention, no linked identity and an accessible objection mechanism. A privacy-page notice describes the purpose, categories, provider, retention and legitimate-interest basis. The relevant controller assessment is: (1) understand failures and use of the tool, (2) independent aggregate counts and coarse ranges suffice, exact profiles and recording are unnecessary, (3) limited expected impact with opt-out, privacy signals, no marketing/admissions decisions and restricted access. This is an engineering assessment, not a legal certification or an assertion of a universal consent exemption. If deployment jurisdiction or the chosen provider terms require consent for a category, leave that category disabled rather than silently overriding the requirement. Exact applicant histories are outside scope.

Primary sources reviewed2026-10-29:
- https://vercel.com/docs/analytics/limits-and-pricing
- https://vercel.com/docs/logs/runtime (Hobby logs retain1h; not analytics history)
- https://vercel.com/docs/vercel-blob/usage-and-pricing (2,000 included advanced operations: Blob was evaluated and not selected for counters)
- https://upstash.com/pricing/redis
- https://upstash.com/docs/redis/features/restapi
- https://www.garanteprivacy.it/faq/cookie
- https://www.garanteprivacy.it/home/docweb/-/docweb-display/docweb/9677876
- https://www.cnil.fr/fr/node/677 (national variation and conditions matter)

## Reliability and abuse controls

POST /api/product-events accepts JSON <=8KiB and <=16events with strict enum validation. Same-origin browser headers are required; these are not authentication against malicious scripts. Each instance allows20requests/minute. A persisted1,000batch/day limit bounds ordinary accepted writes; same-instance cap and storage-failure circuits avoid repeated work. Atomic Redis compare-and-set retries at most twice; identifiers deduplicate repeated batches. Backend storage requests time out after2.5s; the entire collector is limited to10s. Queues are bounded and sent once after30s or on hide/leave; failures are discarded, not replayed. This intentionally favors the calculator over complete telemetry.

Public endpoints can be abused; origin headers can be forged outside browsers. Process-local rate limiting is not a global account quota guard. Provider Free limits can halt analytics; no paid fallback is authorized. If suspicious traffic appears, disable the metrics integration/endpoint through the release rollback and assess a project-scoped edge rate rule separately. Existing calculator operation must remain independent even when metrics stops.

## Private report

Run scripts/product-metrics-report.mjs through an approved project-scoped environment. It reads only known daily keys over at most90days and creates a local HTML file mode0600 under ignored qa/. No unauthenticated report endpoint or public admin page is introduced. Never copy exports under public/ or Git. Example **without any credentials**:

    node scripts/product-metrics-report.mjs --days 7 --out qa/product-metrics-report.html

With --demo the same command generates clearly labelled synthetic data and needs no storage. Demo counts must never be reported as real visitors. The private report can be opened in Codex; it is a local snapshot, not a live dashboard. Node env run can supply the existing connected project environment after activation; do not paste tokens into argv.

## Activation and rollback

1. Run tests, production/preview build gates, actual protected-preview route/browser checks, payload and failure QA. Review local baseline performance and unchanged API hashes.
2. Final approval must name the new Upstash Free database/integration and its two server-only variables UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN. This standing GitHub/Vercel permission explicitly excludes new integrations and environment/secret changes. No broad OAuth access is required or authorized. Abort if Free is unavailable or binding terms need a user handoff.
3. Connect only the dedicated project, verify Free/no paid fallback, test actual Redis expiry/concurrency using synthetic data without copying credentials. Preview collection remains disabled in production code; use a separate local adapter test against the dedicated preview-only database if approved. Do not send synthetic rows into production counts.
4. Merge only after required CI/preview checks pass; rebuild production (preview HTML is noindex). Verify real action delivery without retaining exact inputs. Mark any controlled own QA clearly and do not call it a customer. Check private report can read and unauthenticated access cannot.
5. Rollback baseline: production dpl_GxZ5FSSU7XCxook18iAGL31pjefx, main9a4a62d36a497071c01c04f918ec5a0f72680a44. Revert this release and redeploy/rollback within the same team. Calculator, model and Vercel pageviews remain on their previous implementation. Removing/rotating the newly approved metrics credentials disables ingestion without affecting the Python API. Do not delete stored data or disconnect unrelated resources during rollback.
