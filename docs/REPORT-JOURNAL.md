# Private report request journal

The owner requested durable request tracking on 2026-11-05. The existing Chance Me Redis stores a separate operational journal; no new service, secrets, permissions, public API or marketing subscription is introduced.

Each valid request accepted after profile/address validation and quota checks records its normalized email, creation time, code expiry, environment/test flag, processing timestamps, failed-code count and provider references. Events distinguish code submission, address confirmation, PDF submission and provider submission errors. The HTML snapshot derives expired codes from their deadline and displays provider delivery separately. A provider timeout is an error of the send attempt, not proof that no message was delivered.

Journal keys have no automatic TTL. Confirmation-code hashes and PDF bytes remain exclusively in the existing15-minute pending keys; the journal contains no code, PDF, score, GPA, raw IP or usable request token. The separate rate counters expire after24hours. Records are not copied into analytics, logs, public output or Git. The public privacy page describes this retention and the existing data contact.

## Owner inspection

From the repository, with the existing owner Vercel login:

```sh
/Users/mz/.local/bin/vercel env run -e development --project bocconi-chanceme --scope zielinskimarcins-projects -- .venv/bin/python scripts/report_email_history.py
```

On the owner’s Mac, double-click `scripts/Otworz rejestr raportow.command` in Finder to refresh and open the snapshot using the existing Vercel login. It runs the same command above and does not store credentials.

This writes `qa/report-requests.html`, mode0600. It is a private snapshot, not a live hosted dashboard. Open the file locally. The same command refreshes the data. Emails can be searched locally; tests are hidden by default and excluded from totals. Add `--include-tests` to show the owner QA, or `--no-delivery` to skip the optional read-only Resend lookups. Provider credentials stay only in process memory. The script refuses outputs in repository source/public/build paths. Do not commit or upload snapshots.

Current statuses: request saved, waiting for code, code expired, email verified, PDF submitted, send-attempt error, or locked after five invalid codes. Provider delivery is read on snapshot refresh and is not presented as a precise delivery timestamp. Existing requests in flight during the release can enter the journal upon successful verification, with their original request time shown as unknown. Older completed requests are not backfilled or inferred.

## Storage and recovery

Namespace `chanceme:report-journal:v1:`; record IDs are SHA256 of a random request token. A sorted index orders records by creation time. Pending+journal creation is atomic. Stage updates are atomic and do not downgrade a submitted report. Provider acceptance is written before marking the temporary request complete; a failed journal write leaves the pending state retryable under the existing provider idempotency key. Journal records remain when pending keys expire.

Rollback: production deployment `dpl_4Y1ie7WdpSodAViGj9ZU1NkWWBEY`, main `f98607f9ca7c3d62e12b2b7d6c71f7b5884f7950`. Reverting code stops new journaling but preserves all stored history and email service configuration. Keep the owner reader to inspect existing entries. No deletion of existing data or integration changes are part of this release.

Validation includes mocked-provider lifecycle tests, isolated expiring synthetic Redis QA of actual Lua transitions, real owner-only email delivery, private snapshot review and exact preview/production checks.
