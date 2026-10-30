import { validateBatch, collect, MAX_BODY, requestGeography } from '../lib/product-metrics.mjs';
import { redisStore, configured } from '../lib/product-metrics-store.mjs';

const origins = new Set(['https://chance-me.com', 'https://www.chance-me.com']);
let windowStarted = 0, calls = 0;
export function makeHandler({ store = redisStore, enabled = () => process.env.VERCEL_ENV === 'production' && configured(), clock = Date.now } = {}) {
  let pausedUntil = 0;
  return async function handler(req, res) {
    res.setHeader('Cache-Control', 'no-store'); res.setHeader('X-Robots-Tag', 'noindex, nofollow');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    if (req.method !== 'POST') { res.setHeader('Allow', 'POST'); res.statusCode = 405; return res.end(); }
    // Older browsers can omit Fetch Metadata; the exact Origin allowlist remains mandatory.
    if (!origins.has(req.headers.origin) || (req.headers['sec-fetch-site'] !== undefined && req.headers['sec-fetch-site'] !== 'same-origin')) { res.statusCode = 403; return res.end(); }
    // No IP, user agent, Referer, request bodies or credentials are logged by the application.
    if (req.headers.dnt === '1' || req.headers['sec-gpc'] === '1' || !enabled()) { res.statusCode = 204; return res.end(); }
    if (!/^application\/json(?:;|$)/i.test(req.headers['content-type'] || '')) { res.statusCode = 415; return res.end(); }
    const now = clock();
    if (now < pausedUntil) { res.statusCode = 204; return res.end(); }
    if (now-windowStarted > 60000) { windowStarted = now; calls = 0; }
    if (++calls > 20) { res.statusCode = 429; return res.end(); }
    try {
      let body = req.body;
      if (typeof body !== 'string' && body !== undefined && !Buffer.isBuffer(body)) body = JSON.stringify(body);
      if (body === undefined) {
        const chunks = []; let size = 0;
        for await (const chunk of req) { size += Buffer.byteLength(chunk); if (size > MAX_BODY) { res.statusCode = 413; return res.end(); } chunks.push(Buffer.from(chunk)); }
        body = Buffer.concat(chunks);
      }
      if (Buffer.byteLength(body) > MAX_BODY) { res.statusCode = 413; return res.end(); }
      let events;
      try { events = validateBatch(JSON.parse(body)); } catch { res.statusCode = 400; return res.end(); }
      const update = await collect(store(), events, { now: new Date(now), geography: requestGeography(req.headers) });
      if (update.capped) pausedUntil = Date.parse(new Date(now).toISOString().slice(0,10)+'T00:00:00Z')+86400000;
      res.statusCode = 204; return res.end();
    } catch {
      pausedUntil = clock()+60000;
      // A generic response, with no exception/payload dump or retry instruction.
      res.statusCode = 503; return res.end();
    }
  };
}
export default makeHandler();
