> Current policy approved 2026-11-05: a stable selection of 15–20 close profiles when available, fewer for sparse data. The same normalized profile keeps its selection size on retries. Visitor copy does not advertise a numerical limit. No distant backfill, paging or model changes. Confirmed email delivery is active; see [email delivery](EMAIL-REPORTS.md). The original first-stage notes below are history.

# Report access — 2026-11-04

The owner requested a maximum of ten close historical profiles and email delivery instead of unrestricted PDF downloads. The deployed first stage restricts `/api/reports` and the individual evidence in `/api/estimate` to at most ten profiles inside ±50 SAT or ±4 Bocconi Test points and ±0.5 GPA. Sorting is deterministic by normalised distance and ID; no distant backfill or pagination. Programme and round are still validated. Counts remain aggregate totals; cycle filters narrow the returned selection. `complete: false` explicitly marks a partial dataset. This is a per-comparison limit, not a guarantee against repeated profile queries.

The frozen engine and probabilities are unchanged. Both report views share the cap. `/api/report.pdf` fails closed with503 and no PDF bytes even if an email is supplied. The UI disables export and explains that future reports require email delivery. No addresses are collected and no delivery success is claimed.

Production currently has no mail provider configured (only the existing Redis variables were listed; no secret values were pulled). Recommended next stage: the existing Vercel Python backend plus Resend, confirmed email link before generating the attachment, rate limits backed by existing Redis, idempotent sending, and no email/profile data in analytics. This requires owner authorisation and setup of the new provider, sending-domain verification/DNS and project secrets; these are outside the standing deployment consent. Do not activate email capture until actual delivery works. Supabase is unnecessary for this standalone feature.

Sources: https://resend.com/docs/dashboard/emails/attachments and https://supabase.com/docs/guides/functions/examples/send-emails .

Rollback: revert this change and redeploy the previous main ecfdb65c6700a71e5b9b6ba8dd608ebf98bd05e2; restoring it also reopens unrestricted comparisons and direct PDF downloads. No data, secrets, security settings or integrations are modified by this stage.
