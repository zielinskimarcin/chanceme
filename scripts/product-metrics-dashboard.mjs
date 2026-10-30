/** Owner-only loopback dashboard. Inject approved environment at launch; never expose tokens. */
import http from 'node:http';
import {redisStore,configured} from '../lib/product-metrics-store.mjs';
import {summarize} from '../lib/product-metrics.mjs';
import {renderReport} from './product-metrics-report.mjs';
if(!configured()) throw Error('Approved storage environment unavailable');
const port=18791,host=`127.0.0.1:${port}`,store=redisStore();
let cached='',expires=0,pending;
async function report(){
 if(cached&&Date.now()<expires)return cached;
 if(pending)return pending;
 pending=(async()=>{const now=new Date();const [state,records]=await Promise.all([store.history(30,now),store.detailedHistory(30,now)]);cached=renderReport(summarize(state,{days:30,now,privateDetails:true}),{records,now}).replace('</head>','<meta http-equiv="refresh" content="60"></head>').replace('Refreshing this file does not fetch new data; regenerate the private report to update it.','Live local dashboard: refreshes from storage every 60 seconds while this process runs.');expires=Date.now()+30000;return cached;})();
 try{return await pending;}finally{pending=undefined;}
}
http.createServer(async(req,res)=>{
 res.setHeader('Cache-Control','no-store');res.setHeader('X-Content-Type-Options','nosniff');res.setHeader('X-Frame-Options','DENY');res.setHeader('Content-Security-Policy',"default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'");
 if(req.method!=='GET'||req.url!=='/'||req.headers.host!==host||req.headers['sec-fetch-site']==='cross-site'||(req.headers.origin&&req.headers.origin!==`http://${host}`)){res.writeHead(403);return res.end('Forbidden');}
 try{res.setHeader('Content-Type','text/html; charset=utf-8');res.end(await report());}catch{res.writeHead(503);res.end('Storage temporarily unavailable. Refresh to retry.');}
}).listen(port,'127.0.0.1',()=>console.log(`Private live dashboard: http://${host}/`));
