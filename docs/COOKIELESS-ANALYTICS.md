> Historical design, superseded by [independent Vercel and optional GA](ANALYTICS-RESTORE.md) on 2026-10-29. The first-party collector and local dashboard are retired.

# Cookieless measurement — 2026-10-28

## Owner request and scope

Marcin requested removal of the cookie popup and counting visits without cookies. The implementation removes Google Analytics and Microsoft Clarity from the browser runtime, rather than silently enabling them without consent. Existing accounts, Enhanced Measurement settings, Search Console link and historical data are unchanged. No new service, paid plan, integration or credentials.

Existing Vercel team/project verified; team is Hobby, so custom events are not supported. This release measures page views, not calculator completion, lesson CTA conversion or bookings. The old optional product hooks remain a harmless no-op to avoid breaking cached UI. Do not report those conversions as measured after this release. Browser blockers, DNT/GPC and opt-out can still prevent visits from being counted.

## Data contract

- Vercel's existing same-origin script starts automatically for a new production visitor, without analytics cookies or a locally created visitor ID. No identify/group/enableCookie calls. The optional Vercel attribution store is cleared before loading the script.
- beforeSend accepts pageview only and reconstructs the URL from an allowlisted pathname. Queries, fragments, payloads, flags and identity additions are discarded. External referrers containing queries, fragments or userinfo cause suppression rather than possible disclosure. Internal referrers are suppressed by the provider script.
- Scores, GPA, subject grades, names, emails, booking details and shared report fragments are not sent. Calculator session drafts and functional transfer remain intact.
- DNT/GPC disable measurement. An existing explicit opt-out is respected. Unexpired rejection from the old consent preference migrates to an opt-out; prior acceptance does not enable GA/Clarity.
- The privacy page has inline controls, not a popup. Only a non-identifying on/off preference is stored when changed (or when preserving an existing refusal). Turning off stops subsequent callbacks and reloads an already-loaded library.
- Obsolete _ga/_gid/_gat/_clck/_clsk first-party cookies are expired. Necessary third-party calendar processing is described separately; this does not claim Cal.com never uses storage.

## Source review and limits

Official documentation read2026-10-28:
- https://vercel.com/docs/analytics/privacy-policy : aggregate statistics, no cross-site personal identifiers, request-derived visitor hash discarded after24hours, default automatic page views. These provider descriptions do not constitute a blanket legal certification for every jurisdiction.
- https://vercel.com/docs/analytics/package : beforeSend, returning null to suppress, safe URL customization.
- https://vercel.com/docs/analytics/custom-events : custom events are Pro/Enterprise; current Hobby has page views only.
- https://www.garanteprivacy.it/documents/10160/0/GUIDELINES%2BON%2BCOOKIES%2BAND%2BOTHER%2BTRACKING%2BTOOLS%2B-%2BExecutive%2BSummary.pdf/2278a9d4-a0e1-1578-1a09-8291106f4591?version=3.0 : distinguishes minimized single-site aggregate analytics and profiling; information notice remains necessary. We publish the actual processing and opt-out, not an assertion that all analytics products are consent-exempt.

Checked the provider script currently served by this project: no document.cookie or sessionStorage use, optional local identity storage only through identify/group (never called here), optional cookie endpoint only through enableCookie (never called here). Retest provider behaviour when replacing the SDK.

## Validation and rollback

83JavaScript and21Python tests, production output checks: PASS. Tests cover new automatic cookieless visits, no identifier creation, no popup creation, legacy refusal migration, opt-out/re-enable, DNT/GPC/preview suppression, removal of old cookies, no GA/Clarity on any route, URL/data filtering and no paid custom events. Release gate requires privacy/analytics links and rejects obsolete popup controls.

Exact protected preview and public browser/receipt checks must pass before reporting completion. Retain private QA proofs outside Git. Marketing copy, page titles except the privacy page, canonical URLs, model, favicon and Cal.com booking flow stay intact. Visible content changes are limited to Privacy &analytics text, footer Privacy/Analyticssettings links and removal of the automatic popup.

Rollback baseline: productiondpl_CQm8J1QLvNPjcgPYr5ECiUnNzJxH, mainf40d1e40b504bf71aae072fcfceadfdc719b587b. Revert this release for subsequent builds; restoring that production version restores the earlier consent-based loader. Removed old browser cookies cannot be recreated by rollback; functional drafts are preserved.
