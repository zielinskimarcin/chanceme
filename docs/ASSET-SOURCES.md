# Assets and authorship

Observed 2026-10-25. This is the local prototype.

- dist/assets/bocconi-wordmark.svg: the wordmark path from the official page https://www.unibocconi.it/en, element #logo-bocconi, viewBox 0 0 188 40, original fill #0046AD. Saved through a read-only DOM inspection. Used only as the small Bocconi word in for Bocconi. The page retains its independent-tool statement. No font file was downloaded; exact letter shapes are supplied by the vector. The brand/font research is recorded in the project's Product, SEO and analytics note.
- dist/assets/marcin.png: existing creator portrait copied without image edits from /Users/mz/Documents/Projekty/Apps/chanceme/photo.png, already used on chance-me.com. Profile links use the existing author block's LinkedIn and Instagram URLs. Student status and SAT tutoring are established context; no current year of study, unverified outcome or fabricated testimonial was added.
- dist/assets/nunito-latin.woff2 and nunito-OFL.txt: existing licensed font assets from the registered SAT project.
- dist/favicon.svg is intentionally transparent while the future logo/favicon is undecided. The header reserves a 30–32px slot without drawing a replacement mark. The old icon was also removed from the empty result panel.


## Country flags — 2026-10-25

Local SVG flags from [flag-icons](https://github.com/lipis/flag-icons), pinned npm7.5.0, MIT licence retained at dist/assets/flags/LICENSE.txt. Only selected4:3assets extracted as bytes, no package code executed. research/flag-assets.json records archive URL/SHA256/file manifest; scripts/vendor_flags.py is the repeatable vendor operation.70country/territoryflags plusEU and an original simple globeSVG. Runtime uses local paths and lazy images; no emoji or external flag request.

## IB identification symbol — 2026-10-25

Local file: dist/assets/flags/ib.svg. Source artwork by International Baccalaureate Organisation, original SVG mirrored at https://upload.wikimedia.org/wikipedia/commons/8/88/International_Baccalaureate_Logo.svg ; description and provenance: https://commons.wikimedia.org/wiki/File:International_Baccalaureate_Logo.svg . The source marks it PD-textlogo with trademark restrictions; this is not part of the MIT flag-icons set. Original artwork preserved, displayed with object-fit:contain for curriculum identification beside the IB Diploma label. No World School/authorization badge or affiliation claim. Official programme identities referenced at https://www.ibo.org/digital-toolkit/logos-and-programme-models/ .


## Bocconi guide visuals — 2026-10-27

- `src/assets/guides/bocconi-campus.webp` (960×1280) and `bocconi-campus-small.webp` (480×640) are byte-for-byte copies of the existing licensed images at the registered Polish SAT source, `src/assets/admissions/bocconi-campus-{960,480}.webp`. Photographer: **Burakuberi**. Original: [SDA Bocconi new campus.jpg](https://commons.wikimedia.org/wiki/File:SDA_Bocconi_new_campus.jpg), photographed 2021-05-02, **CC BY-SA 4.0**. Visible captions on both articles link the source and licence, identify the SDA campus accurately and disclose resizing/CSS cropping. No AI-generated campus image.
- `bocconi-scoring-official.png` (1178×320): actual public Bocconi University test page, expanded scoring panel, captured 2026-10-27 with the browser screenshot API. Shows the current deductions and minimum eligibility conditions. [Source](https://www.unibocconi.it/en/applying-bocconi/bachelor-and-law-programs/application-and-admissions/online-bocconi-test).
- `bocconi-sat-rules-official.png` (1178×202): actual public Bocconi University SAT panel, captured 2026-10-27. Shows superscoring, minimum totals/sections and score-age rule. [Source](https://www.unibocconi.it/en/applying-bocconi/bachelor-and-law-programs/application-and-admissions/sat-and-act).
- The two source screenshots are short attributed excerpts used to explain the relevant rules, with capture date and source link immediately below. No private portal or applicant record. No simulation form, account or application was submitted. No implied university endorsement or general reuse licence for its website content.
- The 55/45 weight bar and arithmetic examples are original HTML explanations, not university output or reported student results.
- `scripts/build-guides.mjs` copies these four source assets unchanged to `dist/assets/guides/`. Existing portrait and self-hosted Nunito are reused. The author's personal SAT score is not shown.


Production social preview: `src/static/assets/social/chance-me.png` is an actual screenshot of the empty Chance Me homepage captured 2026-10-27. No applicant result or third-party UI was fabricated.
