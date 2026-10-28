/** Compile authored Markdown guides into standalone, crawlable Chance Me pages.
 * Run with GUIDE_NODE_MODULES pointing to a directory containing marked, or install marked normally.
 * --production changes file-level robots only; the local Flask preview stays noindex.
 */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const require=createRequire(import.meta.url);
const markedPath=require.resolve('marked',{paths:[process.env.GUIDE_NODE_MODULES||root]});
const {marked}=await import(pathToFileURL(markedPath));
const config=JSON.parse(fs.readFileSync(path.join(root,'src/guides/guides.json'),'utf8'));
const production=process.argv.includes('--production');
const htmlEscape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const slug=s=>s.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const home=read('dist/index.html');
const header=home.match(/<header class="site-header">[\s\S]*?<\/header>/)[0].replace(/<a class="current" /g,'<a ').replace(/href="#calculator"/g,'href="/#calculator"').replace(/href="#about"/g,'href="/#about"');
const footer=home.match(/<footer class="site-footer">[\s\S]*?<\/footer>/)[0].replace(/href="#/g,'href="/#');
const credit='<a href="https://commons.wikimedia.org/wiki/File:SDA_Bocconi_new_campus.jpg" target="_blank" rel="noopener noreferrer">SDA Bocconi’s new campus · Photo: Burakuberi</a> · <a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noopener noreferrer">CC BY-SA 4.0</a>. Resized; cropped to fit this page.';
for(const guide of config.guides){
 const md=read(`src/guides/${guide.slug}.md`);
 const pieces=md.split(/^## (.+)$/gm);
 if(pieces[0].trim())throw Error('Guide must begin with an H2 section');
 const sections=[];
 for(let i=1;i<pieces.length;i+=2)sections.push({title:pieces[i].trim(),id:slug(pieces[i]),markdown:pieces[i+1]});
 if(new Set(sections.map(s=>s.id)).size!==sections.length)throw Error('Duplicate guide section ID');
 const toc=sections.map(s=>`<a href="#${s.id}">${htmlEscape(s.title)}</a>`).join('');
 const body=sections.map(section=>{
   let contents=marked.parse(section.markdown,{gfm:true,breaks:false});
   contents=contents.replace(/<table>/g,`<div class="guide-table" role="region" tabindex="0" aria-label="${htmlEscape(section.title)}"><table><caption class="guide-sr-only">${htmlEscape(section.title)}</caption>`).replace(/<\/table>/g,'</table></div>').replace(/<th(?=[ >])/g,'<th scope="col"');
   if(section.title==='Frequently asked questions')contents=contents.replace(/<h3>([\s\S]*?)<\/h3>([\s\S]*?)(?=<h3>|$)/g,(_,q,a)=>`<details class="guide-faq"><summary>${q}<span class="chevron" aria-hidden="true"></span></summary><div>${a}</div></details>`);
   return `<section class="guide-section${section.title==='Official sources'?' guide-sources':''}" id="${section.id}" aria-labelledby="heading-${section.id}"><h2 id="heading-${section.id}">${htmlEscape(section.title)}</h2>${contents}</section>`;
 }).join('\n');
 const url=`${config.canonicalHost}/${guide.slug}/`;
 const other=config.guides.find(g=>g.slug!==guide.slug);
 const image=`${config.canonicalHost}/assets/guides/bocconi-campus.webp`;
 const dateText=new Intl.DateTimeFormat('en-GB',{day:'numeric',month:'long',year:'numeric',timeZone:'UTC'}).format(new Date(config.checked));
 const readTime=Math.ceil(md.replace(/<[^>]+>/g,' ').split(/\s+/).length/220);
 const jsonLd={'@context':'https://schema.org','@graph':[
   {'@type':'Article','@id':`${url}#article`,headline:`${guide.heading}: ${guide.subheading}`,description:guide.description,mainEntityOfPage:{'@type':'WebPage','@id':url},image:[image],dateModified:config.checked,inLanguage:'en',author:{'@type':'Person',name:'Marcin Zieliński',url:`${config.canonicalHost}/#about`},publisher:{'@type':'Organization',name:'Chance Me',url:config.canonicalHost},isAccessibleForFree:true},
   {'@type':'BreadcrumbList',itemListElement:[{'@type':'ListItem',position:1,name:'Chance Me',item:config.canonicalHost+'/'},{'@type':'ListItem',position:2,name:guide.shortTitle,item:url}]}
 ]};
 let activeHeader=header.replace('class="guides-toggle"','class="guides-toggle guide-current"');
 activeHeader=activeHeader.replace(`href="/${guide.slug}/"`,`href="/${guide.slug}/" aria-current="page"`);
 const html=`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="${production?'index,follow,max-image-preview:large':'noindex,nofollow'}"><meta name="theme-color" content="#f8fafc"><title>${htmlEscape(guide.title)}</title><meta name="description" content="${htmlEscape(guide.description)}"><link rel="canonical" href="${url}"><meta property="og:type" content="article"><meta property="og:site_name" content="Chance Me for Bocconi"><meta property="og:title" content="${htmlEscape(guide.title)}"><meta property="og:description" content="${htmlEscape(guide.description)}"><meta property="og:url" content="${url}"><meta property="og:image" content="${image}"><meta property="og:image:width" content="960"><meta property="og:image:height" content="1280"><meta property="og:image:alt" content="SDA Bocconi’s new campus in Milan, photographed by Burakuberi"><meta property="article:modified_time" content="${config.checked}"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="${htmlEscape(guide.title)}"><meta name="twitter:description" content="${htmlEscape(guide.description)}"><meta name="twitter:image" content="${image}"><meta name="author" content="Marcin Zieliński"><link rel="icon" href="/assets/favicon-c-32.png" type="image/png"><link rel="apple-touch-icon" href="/assets/apple-touch-icon-c.png"><link rel="preload" href="/assets/nunito-latin.woff2" as="font" type="font/woff2" crossorigin><link rel="stylesheet" href="/campus.css"><link rel="stylesheet" href="/guides.css"><script src="/guides.js" defer></script><script type="application/ld+json">${JSON.stringify(jsonLd).replaceAll('<','\\u003c')}</script></head>
<body class="guide-page"><a class="skip-link" href="#guide-main">Skip to guide</a><div class="site-shell">${activeHeader}
<main id="guide-main"><div class="guide-hero"><div class="guide-hero-copy"><nav class="guide-breadcrumbs" aria-label="Breadcrumb"><a href="/">Chance Me</a><span aria-hidden="true">/</span><span>${htmlEscape(guide.shortTitle)}</span></nav><p class="guide-cycle">2027/28 entry · International undergraduate applicants</p><h1>${htmlEscape(guide.heading)}<span>${htmlEscape(guide.subheading)}</span></h1><p class="guide-lead">${htmlEscape(guide.lead)}</p><div class="guide-byline"><img src="/assets/marcin.png" width="38" height="38" alt=""><p>By <a href="#guide-author">Marcin Zieliński</a><span>Checked <time datetime="${config.checked}">${dateText}</time> · ${readTime} min read</span></p></div></div><figure class="guide-campus"><img src="/assets/guides/bocconi-campus.webp" srcset="/assets/guides/bocconi-campus-small.webp 480w, /assets/guides/bocconi-campus.webp 960w" sizes="(max-width:680px) 100vw, 360px" width="960" height="1280" alt="SDA Bocconi’s new campus in Milan, with its curved facade and courtyard" fetchpriority="high" decoding="async"><figcaption>${credit}</figcaption></figure></div>
<div class="guide-facts">${guide.facts.map(([v,l])=>`<div><strong>${htmlEscape(v)}</strong><span>${htmlEscape(l)}</span></div>`).join('')}</div>
<div class="guide-layout"><aside class="guide-sidebar"><nav aria-label="On this page"><p>On this page</p>${toc}</nav><a class="guide-sidebar-calculator" href="/#calculator">Check your admission profile <span aria-hidden="true">→</span></a></aside><div class="guide-main-copy"><details class="guide-mobile-contents"><summary>On this page<span class="chevron" aria-hidden="true"></span></summary><nav aria-label="Guide sections">${toc}</nav></details><article>${body}</article>
<section class="guide-author" id="guide-author" aria-labelledby="guide-author-heading"><img src="/assets/marcin.png" alt="Marcin Zieliński, author and Bocconi student" width="96" height="96" loading="lazy" decoding="async"><div><h2 id="guide-author-heading">${htmlEscape(guide.authorCta)}</h2><p>${htmlEscape(guide.authorText)}</p><a class="guide-author-link" href="/sat-tutoring/">${htmlEscape(guide.authorLink)} <span aria-hidden="true">→</span></a></div></section>
<nav class="guide-related" aria-label="Related guides"><a href="/${other.slug}/"><span>Read next</span><strong>${htmlEscape(other.shortTitle)} <span aria-hidden="true">→</span></strong><p>${htmlEscape(other.subheading)}</p></a><a href="/bocconi-gpa-calculator/"><span>Your application</span><strong>Bocconi GPA calculator <span aria-hidden="true">→</span></strong><p>Find the right school years and prepare your grades.</p></a></nav>
<p class="guide-independent">Chance Me is independent and is not affiliated with Bocconi University or College Board. Official sources and your admission offer take precedence.</p></div></div></main>${footer}</div>
<dialog class="guide-image-dialog" aria-labelledby="guide-dialog-title"><div><h2 id="guide-dialog-title"></h2><button type="button" aria-label="Close screenshot">×</button></div><img alt=""></dialog></body></html>`;
 const dest=path.join(root,'dist',guide.slug);fs.mkdirSync(dest,{recursive:true});fs.writeFileSync(path.join(dest,'index.html'),html);
 console.log(`Built /${guide.slug}/ (${readTime}min,${sections.length}sections,${production?'production':'local noindex'})`);
}
for(const name of ['guides.css','guides.js'])fs.copyFileSync(path.join(root,'src',name),path.join(root,'dist',name));
fs.cpSync(path.join(root,'src/assets/guides'),path.join(root,'dist/assets/guides'),{recursive:true});
fs.writeFileSync(path.join(root,'dist/guides-sitemap.xml'),`<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${config.guides.map(g=>`<url><loc>${config.canonicalHost}/${g.slug}/</loc><lastmod>${config.checked}</lastmod></url>`).join('')}</urlset>\n`);
