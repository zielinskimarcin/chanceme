import { METRICS_PREFIX, RETENTION_DAYS, freshState } from './product-metrics.mjs';

const DAY = 86400;
const dayOf = date => date.toISOString().slice(0, 10);
// Counts and transient deduplication update atomically. Redis expiry needs no cron.
export const WRITE_LUA = `
local current = redis.call('GET', KEYS[1])
if (current or '') ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[1], ARGV[2], 'EXAT', ARGV[4])
redis.call('SET', KEYS[2], ARGV[3], 'EXAT', ARGV[5])
return 1`;
export const DETAIL_WRITE_LUA = WRITE_LUA.replace('return 1', `redis.call('SET', KEYS[3], ARGV[6], 'EXAT', ARGV[7])\nreturn 1`);

const connection = env => ({ url: env.UPSTASH_REDIS_REST_URL || env.KV_REST_API_URL, token: env.UPSTASH_REDIS_REST_TOKEN || env.KV_REST_API_TOKEN });

export function configured(env = process.env) {
  try {
    const url = new URL(connection(env).url);
    return url.protocol === 'https:' && url.hostname.endsWith('.upstash.io') && url.pathname === '/' && !url.username && !url.password && !url.search && !url.hash && Boolean(connection(env).token);
  } catch { return false; }
}

export function redisStore({ env = process.env, request = fetch, prefix = METRICS_PREFIX } = {}) {
  if (!/^chanceme:(?:product|qa):[a-z0-9:-]+:$/.test(prefix)) throw new Error('Invalid storage namespace');
  async function command(args, maxResponse=2*1024*1024) {
    if (!configured(env)) throw new Error('Storage unavailable');
    const response = await request(connection(env).url, { method: 'POST', redirect: 'error',
      headers: { Authorization: `Bearer ${connection(env).token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(args), signal: AbortSignal.timeout(2500) });
    if (!response.ok) throw new Error('Storage unavailable');
    const text = await response.text();
    if (text.length > maxResponse) throw new Error('Storage response too large');
    const body = JSON.parse(text);
    if (body.error) throw new Error('Storage command failed');
    return body.result;
  }
  return {
    async read(now) {
      const day = dayOf(now), key = prefix+day;
      const [raw, ids, records] = await command(['MGET', key, key+':dedup', key+':records']);
      if (!raw) return null;
      const state = JSON.parse(raw);
      state.dedupDay = day; state.dedup = ids ? JSON.parse(ids) : [];
      if(records) state.records=JSON.parse(records);
      return { state, etag: raw };
    },
    async write(state, etag, now) {
      const day = dayOf(now), key = prefix+day;
      const persistent = { version: 1, days: { [day]: state.days[day] } };
      const midnight = Date.parse(day+'T00:00:00Z')/1000;
      const args=[etag || '', JSON.stringify(persistent), JSON.stringify(state.dedup), midnight+RETENTION_DAYS*DAY, midnight+2*DAY];
      const detailed=Array.isArray(state.records);
      const result = await command(['EVAL', detailed?DETAIL_WRITE_LUA:WRITE_LUA, detailed?3:2, key, key+':dedup', ...(detailed?[key+':records']:[]), ...args, ...(detailed?[JSON.stringify(state.records),midnight+30*DAY]:[])]);
      if (result !== 1) { const error = new Error('Concurrent update'); error.code = 'CONFLICT'; throw error; }
    },
    isConflict(error) { return error?.code === 'CONFLICT'; },
    async detailedHistory(days=7,now=new Date()) {
      const count=Math.min(30,Math.max(1,days));
      const keys=Array.from({length:count},(_,i)=>prefix+dayOf(new Date(now.getTime()-i*DAY*1000))+':records');
      // One bounded MGET per refresh, not 30 billed round trips per minute.
      const values=await command(['MGET',...keys],64*1024*1024);
      return values.flatMap(raw=>raw?JSON.parse(raw):[]);
    },
    async history(days = 7, now = new Date()) {
      const count = Math.min(RETENTION_DAYS, Math.max(1, days));
      const keys = Array.from({ length: count }, (_, i) => prefix+dayOf(new Date(now.getTime()-i*DAY*1000)));
      const records = await command(['MGET', ...keys]);
      const state = freshState();
      for (const record of records) if (record) Object.assign(state.days, JSON.parse(record).days);
      return state;
    },
  };
}
