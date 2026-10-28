import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
process.chdir(root);
const production=process.argv.includes('--production')&&process.env.VERCEL_ENV!=='preview';
const host='https://chance-me.com';
const read=p=>fs.readFileSync(p,'utf8');
const escape=s=>s.replaceAll('&','&amp;').replaceAll('"','&quot;');
fs.mkdirSync('dist',{recursive:true});
fs.cpSync('src/static','dist',{recursive:true});
for(const name of fs.readdirSync('src/pages'))fs.copyFileSync('src/pages/'+name,'dist/'+name);
execFileSync(process.execPath,['scripts/build-guides.mjs',...(production?['--production']:[])],{stdio:'inherit'});
const pages=[
 {file:'index.html',url:'/',title:'Bocconi Admission Calculator | SAT, GPA & Bocconi Test',description:'Estimate your Bocconi admission chances using SAT or Bocconi Test scores and GPA. Compare past applicants by programme and round for 2027/28 entry.',type:'WebApplication'},
 {file:'gpa-campus.html',url:'/bocconi-gpa-calculator/',title:'Bocconi GPA Calculator | School Grades to Bocconi GPA',description:'Calculate a Bocconi-scale GPA from school grades, choose your country or curriculum, and transfer your result to the admission calculator.',type:'WebApplication',name:'Bocconi GPA calculator'},
 {file:'sat-lessons.html',url:'/sat-tutoring/',title:'SAT Tutoring for Bocconi | Go for 1500+ | Chance Me',description:'SAT tutoring for Bocconi with Marcin, a third-year Bocconi student. Three years of experience, test preparation and application support. €50 per hour.',type:'Service',name:'SAT tutoring for Bocconi'},
 {file:'bocconi-test-score/index.html',url:'/bocconi-test-score/',name:'Bocconi Test guide'},
 {file:'bocconi-sat-score/index.html',url:'/bocconi-sat-score/',name:'SAT for Bocconi'},
 {file:'full-report.html',url:'/report/',title:'Your Bocconi Admission Report | Chance Me',description:'Explore your model estimate and self-reported historical Bocconi applicant data.',noindex:true,name:'Your report'},
 {file:'gpa-school-sources.html',url:'/gpa-school-sources/',title:'School Grading Systems: Sources | Chance Me',description:'Sources and limitations for the school systems used in the Chance Me GPA calculator.',noindex:true,name:'GPA sources'}
];
const base=read('dist/index.html');
const header=base.match(/<header class="site-header">[\s\S]*?<\/header>/)[0].replace('class="current"','').replaceAll('href="#calculator"','href="/#calculator"').replaceAll('href="#about"','href="/#about"');
const footer=base.match(/<footer class="site-footer">[\s\S]*?<\/footer>/)[0].replaceAll('href="#','href="/#');
const plain=(title,body)=>`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${title}</title><link rel="stylesheet" href="/campus.css"><script src="/navigation.js" defer></script></head><body><div class="site-shell">${header}<main class="plain-copy"><h1>${title}</h1>${body}</main>${footer}</div></body></html>`;
fs.writeFileSync('dist/privacy.html',plain('Privacy & analytics',read('src/privacy.html')));
pages.push({file:'privacy.html',url:'/privacy/',title:'Privacy & Analytics | Chance Me',description:'How Chance Me handles calculator inputs, cookieless traffic statistics, local drafts and lesson bookings.',name:'Privacy & analytics'});
fs.writeFileSync('dist/404.html',plain('Page not found','<p>This page does not exist. You can return to the <a href="/">admission calculator</a>, use the <a href="/bocconi-gpa-calculator/">GPA calculator</a> or browse the guides above.</p>'));
fs.rmSync('public',{recursive:true,force:true});fs.mkdirSync('public');
fs.cpSync('dist/assets','public/assets',{recursive:true});
for(const n of fs.readdirSync('dist'))if(/\.(css|js|svg|ico)$/.test(n))fs.copyFileSync('dist/'+n,'public/'+n);
function structured(p){
 const url=host+p.url;
 const graph=[];
 if(p.type==='WebApplication')graph.push({'@type':'WebApplication','@id':url+'#application',name:p.name||'Bocconi Admission Calculator',url,description:p.description,applicationCategory:'EducationalApplication',operatingSystem:'Any',isAccessibleForFree:true,offers:{'@type':'Offer',price:'0',priceCurrency:'EUR'},publisher:{'@type':'Organization',name:'Chance Me',url:host}});
 if(p.type==='Service')graph.push({'@type':'Service','@id':url+'#service',name:p.name,url,description:p.description,serviceType:'Online SAT tutoring',provider:{'@type':'Person',name:'Marcin Zieliński',url:host+'/#about'},offers:{'@type':'Offer',price:'50',priceCurrency:'EUR',url:url+'#book-lesson',description:'60-minute online SAT lesson. Payment after the lesson.'}});
 if(p.url==='/')graph.push({'@type':'WebSite','@id':host+'/#website',name:'Chance Me for Bocconi',alternateName:'Chance Me',url:host+'/',inLanguage:'en'});
 else if(p.name&&!p.noindex)graph.push({'@type':'BreadcrumbList',itemListElement:[{'@type':'ListItem',position:1,name:'Chance Me',item:host+'/'},{'@type':'ListItem',position:2,name:p.name,item:url}]});
 return graph.length?`<script type="application/ld+json">${JSON.stringify({'@context':'https://schema.org','@graph':graph})}</script>`:'';
}
for(const p of [...pages,{file:'404.html',url:null,noindex:true}]){
 let html=read('dist/'+p.file);
 // One shared favicon set on every generated page, including privacy and 404.
 html=html.replace(/<link\b(?=[^>]*\brel="(?:icon|shortcut icon|apple-touch-icon)")[^>]*>/g,'');
 html=html.replace('</head>','<link rel="icon" href="/assets/favicon-96.png" type="image/png" sizes="96x96"><link rel="icon" href="/assets/favicon-32.png" type="image/png" sizes="32x32"><link rel="icon" href="/assets/favicon-16.png" type="image/png" sizes="16x16"><link rel="apple-touch-icon" href="/assets/apple-touch-icon.png" sizes="180x180"><meta name="application-name" content="Chance Me for Bocconi"></head>');
 if(p.title)html=html.replace(/<title>[\s\S]*?<\/title>/,`<title>${escape(p.title)}</title>`);
 if(p.description){html=html.replace(/<meta name="description"[^>]*>/g,'');html=html.replace('</head>',`<meta name="description" content="${escape(p.description)}"></head>`);}
 html=html.replace(/<meta name="robots"[^>]*>/g,'').replace(/<link rel="canonical"[^>]*>/g,'');
 html=html.replace('</head>',`<meta name="robots" content="${production&&!p.noindex?'index,follow,max-image-preview:large':'noindex,follow'}">${p.url?`<link rel="canonical" href="${host+p.url}">`:''}<link rel="stylesheet" href="/site.css"><script src="/analytics.js" defer></script></head>`);
 if(p.title&&p.url){
  html=html.replace(/<meta (?:property="(?:og:[^"]+|article:modified_time)"|name="twitter:[^"]+")[^>]*>/g,'');
  const image=host+'/assets/social/chance-me.png';
  html=html.replace('</head>',`<meta property="og:type" content="website"><meta property="og:site_name" content="Chance Me for Bocconi"><meta property="og:title" content="${escape(p.title)}"><meta property="og:description" content="${escape(p.description)}"><meta property="og:url" content="${host+p.url}"><meta property="og:image" content="${image}"><meta property="og:image:alt" content="Chance Me for Bocconi"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="${escape(p.title)}"><meta name="twitter:description" content="${escape(p.description)}"><meta name="twitter:image" content="${image}">${structured(p)}</head>`);
 }
 // Only private input/results surfaces are masked. Clarity also excludes calculator pages entirely.
 html=html.replace(/(<(?:form|aside|section)[^>]*\bid="(?:profile-form|result-panel|evidence|gpa-form|book-lesson)"[^>]*)(>)/g,'$1 data-clarity-mask="true"$2');
 html=html.replace(/using SAT or Bocconi Test guide\./g,'using SAT or Bocconi Test.');
 if(!html.includes('</footer>'))html=html.replace('</body>',footer+'</body>');
 // Version actual files, avoiding stale JS/CSS after the redesign.
 html=html.replace(/(src|href)="(\/[^"?#]+)(?:\?[^"#]*)?"/g,(match,attr,url)=>{
   const file=path.join('public',url);
   if(!fs.existsSync(file)||!fs.statSync(file).isFile())return match;
   // Stable icon URLs help crawlers retain one canonical brand asset.
   if(/^\/assets\/(?:favicon-\d+|apple-touch-icon)\.png$/.test(url))return `${attr}="${url}"`;
   return `${attr}="${url}?v=${crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex').slice(0,12)}"`;
 });
 const target=p.url?path.join('public',p.url,'index.html'):'public/404.html';
 fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,html);
}
fs.writeFileSync('public/robots.txt',`User-agent: *\nAllow: /\nDisallow: /api/\nSitemap: ${host}/sitemap.xml\n`);
fs.writeFileSync('public/sitemap.xml',`<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${pages.filter(p=>!p.noindex).map(p=>`<url><loc>${host+p.url}</loc><lastmod>2026-10-27</lastmod></url>`).join('')}</urlset>\n`);
fs.writeFileSync('public/build-manifest.json',JSON.stringify({version:'campus-production-2026-10-27',indexable:production,pages:pages.map(p=>p.url)},null,2));
console.log(`Built ${pages.length} pages (${production?'production':'preview noindex'}), public files only.`);
