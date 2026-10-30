/** Version 1 aggregate compatibility and version 2 bounded private visit history. */
import { createHash } from 'node:crypto';

export const METRICS_PREFIX = 'chanceme:product:v1:';
export const MAX_BATCH = 16;
export const MAX_BODY = 8192;
// Bound ordinary ingestion in the proposed free Redis tier. Not an account quota guarantee.
export const MAX_DAILY_WRITES = 1000;
const DAY = 86400000;
export const RETENTION_DAYS = 90;
const kinds = ['SAT', 'Bocconi'];
const programmes = ['Management', 'Finance', 'Management and Computer Science', 'Management for the Arts and Cultures', 'International Politics and Government', 'Mathematical and Computing Sciences for Artificial Intelligence'];
const dimensions = {
  test_type: kinds, program: programmes, round: ['Early', 'Winter'],
  status: ['estimated', 'ineligible', 'invalid', 'unsupported'],
  result_band: ['lower', 'balanced', 'stronger'],
  destination: ['calculator', 'gpa', 'lessons', 'booking', 'whatsapp', 'sat_guide', 'bocconi_test_guide', 'report', 'pdf'],
  position: ['header', 'hero', 'content', 'footer'], reason: ['unavailable'],
};
export const EVENT_DIMENSIONS = {
  page_view: [],
  calculator_completed: ['test_type', 'program', 'round', 'status', 'result_band', 'score_band', 'gpa_band'],
  calculator_failed: ['reason'], gpa_calculated: [], gpa_transferred: ['destination'],
  report_rendered: [], report_failed: ['reason'], pdf_generated: [], pdf_failed: ['reason'], pdf_download_initiated: ['destination'],
  cta_clicked: ['destination', 'position'], booking_confirmed: ['destination'],
};
const satBands = new Set([...Array.from({ length: 23 }, (_, i) => `${400+i*50}-${449+i*50}`), '1550-1600']);
const btBands = new Set([...Array.from({ length: 24 }, (_, i) => `${i*2}-${i*2+1}.99`), '48-50']);
const gpaBands = new Set([...Array.from({ length: 39 }, (_, i) => `${(i/4).toFixed(2)}-${(i/4+.24).toFixed(2)}`), '9.75-10.00']);
const own = (object, key) => Object.hasOwn(object, key);
const plain = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const hash = id => createHash('sha256').update(id).digest('hex').slice(0, 32);
const dayOf = date => date.toISOString().slice(0, 10);

export function validateBatch(body) {
  if (!plain(body) || ![1, 2].includes(body.version) || Object.keys(body).some(k => !['version', 'events'].includes(k)) || !Array.isArray(body.events) || !body.events.length || body.events.length > MAX_BATCH) throw new Error('Invalid batch');
  return body.events.map(item => {
    if (!plain(item) || typeof item.event !== 'string' || !own(EVENT_DIMENSIONS, item.event) || typeof item.id !== 'string' || !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(item.id)) throw new Error('Invalid event');
    const allowed = EVENT_DIMENSIONS[item.event];
    const detailKeys = body.version === 2 ? ['visit_id','occurred_at','page','referrer','device','browser','os','timezone','utc_offset_minutes','source','medium','campaign','content','term','utm_id', ...(item.event === 'calculator_completed' ? ['score','gpa','probability','cycle','applicant_category','gpa_confirmed','gpa_estimated','result'] : [])] : [];
    if (Object.keys(item).some(key => !['id', 'event', ...allowed, ...detailKeys].includes(key))) throw new Error('Unexpected field');
    const clean = { event: item.event, dedup: hash(item.id), dimensions: {} };
    for (const key of allowed) {
      if (!own(item, key)) continue;
      const value = item[key];
      const valid = key === 'score_band' ? (item.test_type === 'SAT' ? satBands : item.test_type === 'Bocconi' ? btBands : new Set()).has(value)
        : key === 'gpa_band' ? gpaBands.has(value) : dimensions[key].includes(value);
      if (!valid) throw new Error('Invalid dimension');
      clean.dimensions[key] = value;
    }
    if (item.event === 'calculator_completed' && !['test_type', 'program', 'round', 'status'].every(k => own(clean.dimensions, k))) throw new Error('Missing calculation dimensions');
    if (body.version === 2) clean.details = validateDetails(item);
    return clean;
  });
}

const uuid = value => typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
function validateDetails(item) {
  if (!uuid(item.visit_id) || typeof item.occurred_at !== 'string' || !Number.isFinite(Date.parse(item.occurred_at)) || !/^\d{4}-\d{2}-\d{2}T/.test(item.occurred_at)) throw Error('Invalid visit');
  if (typeof item.page !== 'string' || !/^\/[a-zA-Z0-9/_%.~-]{0,199}$/.test(item.page) || item.page.startsWith('//')) throw Error('Invalid page');
  const details = {visit_id:item.visit_id,occurred_at:new Date(item.occurred_at).toISOString(),page:item.page};
  if (item.device !== undefined) {if (!['mobile','tablet','desktop'].includes(item.device)) throw Error('Invalid device');details.device=item.device;}
  if (item.referrer !== undefined && item.referrer !== '') {
    let url;try {url=new URL(item.referrer);} catch {throw Error('Invalid referrer');}
    if (!['https:','http:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || item.referrer.length>300) throw Error('Invalid referrer');
    details.referrer=url.origin+url.pathname;
  }
  for (const key of ['source','medium','campaign','content','term','utm_id']) if (item[key] !== undefined) {
    if (typeof item[key] !== 'string' || !/^[a-zA-Z0-9 _.,:~+\/-]{1,80}$/.test(item[key])) throw Error('Invalid attribution');
    details[key]=item[key];
  }
  for (const [key,min,max] of [['score',0,item.test_type==='SAT'?1600:50],['gpa',0,10],['probability',0,100]]) if (item[key] !== undefined) {
    if (typeof item[key] !== 'number' || !Number.isFinite(item[key]) || item[key]<min || item[key]>max) throw Error('Invalid numeric input');
    details[key]=item[key];
  }
  for (const [key,values] of [['browser',['Chrome','Edge','Firefox','Safari','Opera','Samsung Internet','Other']],['os',['Windows','macOS','iOS','Android','Linux','ChromeOS','Other']]]) if (item[key]!==undefined) {
    if(!values.includes(item[key]))throw Error('Invalid platform');details[key]=item[key];
  }
  if(item.timezone!==undefined){
    if(typeof item.timezone!=='string'||item.timezone.length>80)throw Error('Invalid timezone');
    try{new Intl.DateTimeFormat('en',{timeZone:item.timezone});}catch{throw Error('Invalid timezone');}details.timezone=item.timezone;
  }
  if(item.utc_offset_minutes!==undefined){if(!Number.isInteger(item.utc_offset_minutes)||Math.abs(item.utc_offset_minutes)>840)throw Error('Invalid timezone offset');details.utc_offset_minutes=item.utc_offset_minutes;}
  for(const key of ['gpa_confirmed','gpa_estimated'])if(item[key]!==undefined){if(typeof item[key]!=='boolean')throw Error('Invalid flag');details[key]=item[key];}
  if(item.cycle!==undefined){if(item.cycle!=='27-28')throw Error('Invalid cycle');details.cycle=item.cycle;}
  if(item.applicant_category!==undefined){if(item.applicant_category!=='international')throw Error('Invalid applicant category');details.applicant_category=item.applicant_category;}
  if(item.result!==undefined){
    const result=item.result;
    if(!plain(result)||Object.keys(result).some(k=>!['probability','display_percent','status'].includes(k))||typeof result.probability!=='number'||!Number.isFinite(result.probability)||result.probability<0||result.probability>1||typeof result.display_percent!=='string'||!/^(?:<1|>99|100|\d{1,2})%$/.test(result.display_percent)||!dimensions.status.includes(result.status))throw Error('Invalid calculation result');
    details.result={probability:result.probability,display_percent:result.display_percent,status:result.status};
  }
  return details;
}
export function requestGeography(headers) {
  const text = (key,max=100) => {try {const value=decodeURIComponent(String(headers[key]||''));return value.length<=max && /^[\p{L}\p{N} .,'()_-]*$/u.test(value)?value:'';}catch{return '';}};
  const country=text('x-vercel-ip-country',2).toUpperCase();
  return {country:/^[A-Z]{2}$/.test(country)?country:'Unknown',region:text('x-vercel-ip-country-region'),city:text('x-vercel-ip-city')};
}

export function freshState() { return { version: 1, days: {}, dedupDay: '', dedup: [] }; }
const increment = (object, key) => { object[key] = (object[key] || 0) + 1; };

export function aggregate(state, events, now = new Date(), geography = {}) {
  if (!plain(state) || state.version !== 1 || !plain(state.days) || !Array.isArray(state.dedup)) throw new Error('Invalid stored state');
  const next = structuredClone(state), today = dayOf(now);
  // Expiry runs on writes; report reads also filter expiry. Detailed history expires independently in Redis after 30 days.
  const earliest = dayOf(new Date(now.getTime() - (RETENTION_DAYS-1)*DAY));
  for (const date of Object.keys(next.days)) if (date < earliest || date > today) delete next.days[date];
  if (next.dedupDay !== today) { next.dedupDay = today; next.dedup = []; }
  const current = next.days[today] || { batches: 0, events: {}, distributions: {} };
  if (current.batches >= MAX_DAILY_WRITES) return { state: next, changed: false, capped: true, accepted: 0 };
  let accepted = 0;
  for (const event of events) {
    if (next.dedup.includes(event.dedup)) continue;
    next.dedup.push(event.dedup); accepted++;
    increment(current.events, event.event);
    if (event.details) {
      next.records ||= [];
      if (next.records.length < 1000) {
        const occurred=Date.parse(event.details.occurred_at);
        const details={...event.details};
        // Client clock is informational; reject implausible timestamps as a sorting authority.
        if (Math.abs(occurred-now.getTime())>86400000) details.occurred_at=now.toISOString();
        next.records.push({event:event.event,dimensions:event.dimensions,details,received_at:now.toISOString(),...geography});
      } else current.detailCapped=true;
    }
    // Keep the existing aggregate schema compatible with cached version 1 clients.
    for (const [dimension, value] of Object.entries(event.dimensions)) {
      const key = `${event.event}:${dimension}`;
      const histogram = current.distributions[key] ||= {};
      // Separate SAT/BT score scales, not profiles or cross-tabulations.
      increment(histogram, dimension === 'score_band' ? `${event.dimensions.test_type}:${value}` : value);
    }
  }
  if (!accepted) return { state: next, changed: false, capped: false, accepted: 0 };
  current.batches++; next.days[today] = current;
  return { state: next, changed: true, capped: false, accepted };
}

/** Never return deduplication hashes or small demographic/result bins to the report. */
export function summarize(state, { now = new Date(), days = 7, privateDetails = false } = {}) {
  const earliest = dayOf(new Date(now.getTime()-(Math.min(RETENTION_DAYS, Math.max(1, days))-1)*DAY));
  const today = dayOf(now), result = { from: earliest, to: today, days: [], totals: {}, distributions: {}, limited: false };
  for (const [date, value] of Object.entries(state.days).sort()) {
    if (date < earliest || date > today) continue;
    const limited = value.batches >= MAX_DAILY_WRITES || Boolean(value.detailCapped);
    result.days.push({ date, events: value.events, limited }); result.limited ||= limited;
    for (const [key, count] of Object.entries(value.events)) result.totals[key] = (result.totals[key] || 0) + count;
    for (const [key, bins] of Object.entries(value.distributions)) {
      const histogram = result.distributions[key] ||= {};
      for (const [bin, count] of Object.entries(bins)) histogram[bin] = (histogram[bin] || 0) + count;
    }
  }
  const rolling = Object.entries(state.days).filter(([date]) => date >= dayOf(new Date(now.getTime()-29*DAY)) && date <= today).reduce((total, [,day]) => total+day.batches, 0);
  result.budget = { rollingWrites: rolling, dailyLimit: MAX_DAILY_WRITES };
  if (!privateDetails) for (const histogram of Object.values(result.distributions)) for (const [bin, count] of Object.entries(histogram)) if (count < 5) { delete histogram[bin]; histogram['Other / small groups'] = null; }
  return result;
}

/** CAS prevents concurrent increments from overwriting one another; duplicates are idempotent. */
export async function collect(store, events, { now = new Date(), attempts = 2, geography = {} } = {}) {
  for (let attempt = 0; attempt < attempts; attempt++) {
    const snapshot = await store.read(now);
    const update = aggregate(snapshot?.state || freshState(), events, now, geography);
    if (!update.changed) return update;
    try { await store.write(update.state, snapshot?.etag, now); return update; }
    catch (error) { if (!store.isConflict(error) || attempt+1 === attempts) throw error; }
  }
}

/** Test classification follows the visit, including report tabs without their own UTM. */
export function classifyTraffic(records) {
 const tests=new Set(records.filter(r=>/^(?:owner-qa|qa[-_])/i.test(r.details?.source||'')).map(r=>r.details?.visit_id));
 return records.map(r=>({...r,traffic:tests.has(r.details?.visit_id)?'test':'user'}));
}
