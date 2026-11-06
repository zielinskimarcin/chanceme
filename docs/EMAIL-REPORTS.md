# Email report delivery — 2026-11-05

## State and flow

Resend Free is connected to this project; the sending domain `mail.chance-me.com` is verified. Provider credentials remain in Vercel. This release enables email-only PDFs after address confirmation; the old direct-download endpoint stays disabled.

The visitor supplies one email address and a calculator profile. The server prepares a PDF containing a stable selection of 15–20 close profiles when available, or fewer if the close-data pool is smaller, and sends a six-digit confirmation code. Only a valid confirmation triggers the attachment. Addresses and codes stay out of URLs, browser storage and analytics. No mailing list is created. Provider acceptance is described as submission for delivery, rather than guaranteed inbox arrival.

## Retention and abuse controls

Temporary pending email/code hash/PDF expire after15minutes; PDF bytes are removed after provider acceptance. Quota keys use HMAC identifiers rather than raw IP/email and expire after24hours:40requests/day globally,3per mailbox,8per Vercel-observedIP. Five incorrect codes invalidate a request. Atomic Redis scripts and60-second send locks prevent concurrent duplicates; Resend idempotency keys keep retries safe. Non-production recipients are restricted to the approved owner test accounts. Origin checks allow the canonical site and exact preview/development origins.

## Configuration

The existing Chance Me Redis supplies `KV_REST_API_URL` and `KV_REST_API_TOKEN`. The approved Resend integration supplies `RESEND_API_KEY` and `RESEND_EMAIL_DOMAIN`. Application settings: `REPORT_EMAIL_FROM` uses the verified sending domain; `REPORT_EMAIL_ENABLED=true` activates the feature only after delivery QA. Keep the latter unset or false if the provider is unavailable. Do not log values or pull credentials into files.

Provider-required records were added only under the approved sending subdomain in authoritative IONOS: public DKIM TXT at `resend._domainkey.mail`, SPF TXT at `send.mail` (`v=spf1 include:amazonses.com ~all`) and return-path MX at `send.mail` (`feedback-smtp.eu-west-1.amazonses.com`, priority10). Their authoritative readback matched the provider, which then verified the domain. Existing website routing, nameservers and apex mail records were preserved.

## QA and release

Automated tests cover address/header injection, profile validation, code lockout, provider failure, duplicate confirmation, bounded close-profile PDF scope, origin/configuration gates and frontend transitions. Atomic quotas, TTL, exclusive claims and completed state were also checked against the existing Redis under a separate expiring synthetic QA namespace. The full browser flow passed locally using a fake provider/storage.

Real delivery then passed with the actual provider and Redis, only to an authorised owner test mailbox. The confirmation and PDF were independently found in its inbox. Repeated verification did not send another attachment. Private message identifiers and delivery receipts are retained only in local ignored QA and the owner's project records.

The first real send returned Cloudflare403/1010 for the default Python-urllib signature. An honest `User-Agent: ChanceMe/2.0 (+https://chance-me.com)` fixed compatibility without changing credentials or protections. Regression coverage checks that header.

Release gates: green CI, exact protected preview, provider acceptance and inbox receipt, followed by production verification. The frozen admission model is unchanged.

## Rollback

Set `REPORT_EMAIL_ENABLED=false` to stop new requests while keeping comparisons available. Code rollback for this revision is main `a29eaee1e045f9230702085ff3c1ab596a1f42e5`, deployment `dpl_Havi1ohhEERUitPaffCKz4zMBtVB`: ten-profile comparisons and working plain-text email delivery. Pending Redis requests expire naturally; do not remove existing data or integrations as part of rollback.

Official references: [Resend send API](https://resend.com/docs/api-reference/emails/send-email), [attachments](https://resend.com/docs/dashboard/emails/attachments), [idempotency](https://resend.com/docs/dashboard/emails/idempotency-keys), [Redis REST](https://upstash.com/docs/redis/features/restapi), [Vercel request headers](https://vercel.com/docs/headers/request-headers), [Cloudflare1010 client guidance](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-1xxx-errors/error-1010/).

## Approved email design — 2026-11-05

Simplified inline-style HTML for the confirmation and PDF messages uses the website’s Nunito wordmark and font where supported, a plain white layout, a subtle mint code field and a single-line legal footer. Plain-text alternatives remain available. The confirmation code stays in the message body, outside the preheader. Record limits are not advertised in visitor copy. The selection size is stable for a normalized profile and varies from 15 to20, without padding from distant records. The owner approved the simplified design and requested deployment on 2026-11-05.
