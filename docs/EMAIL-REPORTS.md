# Email report delivery — 2026-11-05

## State

Code ready for QA; outbound delivery is disabled until Resend, its verified sender and `REPORT_EMAIL_ENABLED=true` are configured. No production email addresses have been captured by the previous release: it had no email form or email persistence endpoint. A feature-status endpoint enables the new form only when all required configuration exists. The old direct PDF endpoint remains disabled.

## Flow and limits

The visitor supplies a single email address and a validated calculator profile. The backend prepares a server-owned PDF containing at most ten close profiles, saves a short-lived request in the existing Chance Me Redis, and sends a six-digit confirmation code. Only the verified code can trigger the attachment. No addresses or codes in URLs, browser storage or analytics. Mail confirmation and attachment have separate Resend idempotency keys. A failed provider call never claims successful submission.

Temporary pending email/code hash/PDF expire after15minutes; PDF bytes are removed after provider acceptance. Quotas use HMAC identifiers rather than raw IP/email keys and expire after24hours:40requests/day globally,3per mailbox,8per Vercel-observedIP; five incorrect code attempts invalidate a request. Lua claims and60-second locks avoid concurrent duplicate sends. Resend retains its own delivery records; no mailing list is created. Non-production environments only accept the owner's three pre-authorised test addresses. Origin checks allow the canonical production origins and exact Vercel preview origin. No settings are changed on other products or databases.

## Configuration required (values stay in Vercel)

- Existing `KV_REST_API_URL` and `KV_REST_API_TOKEN` for the Chance Me Redis.
- `RESEND_API_KEY`: sending-only, scoped to the verified sending domain.
- `REPORT_EMAIL_FROM`: `Chance Me <reports@mail.chance-me.com>` after domain verification.
- `REPORT_EMAIL_ENABLED`: `true` only after provider acceptance/inbox QA to an owner's authorised test mailbox. Leave unset/false until then.

Proposed integration: `resend/resend-email`, resource `chance-me-reports`, Free plan, `mail.chance-me.com`, region`eu-west-1`, project`bocconi-chanceme` (`prj_WyyUR0Bo7neO1RgkMZXpuLaGwDfD`), team`zielinskimarcins-projects`, production and development only. Do not attach unrelated projects, enable paid fallback or overwrite existing website/mail DNS. Only add the sending-subdomain records supplied by Resend after specific owner approval.

An attempted official CLI installation was rejected by automatic approval review: it would create a new persistent integration/access/secrets without exact authorisation. No installation occurred. Do not retry through another route. Obtain explicit approval for the exact free integration, sender-subdomain DNS and project secrets first; account/legal acceptance may require the owner in the service UI. Resend panel currently shows login rather than an accessible account.

## QA and release

Local automated coverage includes recipient validation/header injection, ownership/code lockout, expiry behaviour, failed providers, repeat confirmation, PDF scope, origin/configuration gates and frontend transitions. Existing frozen probabilities remain unchanged. The existing real Chance Me Redis was tested under an isolated `chanceme:report-email:qa:<random>:` prefix with fake sender, synthetic mailbox and60-second pendingTTL: atomic quota, exclusive lock, idempotent sent state and PDF removal passed. No real mail was sent. Full browser request→code→accepted flow was tested locally with a fake provider and fake storage; this is not proof of real mail delivery.

Before activation: connect specifically approved provider, verify sender, configure project secrets, run protected exact preview tests, send only to an authorised owner test address and confirm actual inbox receipt/attachment. Then merge green PR and check production. Do not expose email capture while delivery is unconfigured.

Rollback: `REPORT_EMAIL_ENABLED=false` disables all new writes/sends and leaves comparisons working; code rollback returns to main`b26c70759dcd905b7361373c2b5d4898d3519fe3` / production`dpl_9eraYYMH2QRmrx5evCfb6R26Z84Q`. Temporary Redis requests expire without removing pre-existing data.

Official references: [send API](https://resend.com/docs/api-reference/emails/send-email), [attachments](https://resend.com/docs/dashboard/emails/attachments), [idempotency](https://resend.com/docs/dashboard/emails/idempotency-keys), [Redis REST](https://upstash.com/docs/redis/features/restapi), [Vercel trusted request headers](https://vercel.com/docs/headers/request-headers).
