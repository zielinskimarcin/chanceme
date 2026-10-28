# GPA calculator: design and copy — 2026-10-25

## Compact unsupported-scale state — 2026-10-25

Latest user correction replaces disabled grey cards with one short notice: “This scale needs Bocconi’s GPA calculator ↗.” The link opens https://apply.unibocconi.it/ (the personalized calculator lives inside the application); “I already have my GPA →” reveals the existing annual portal-GPA inputs. Before that action, irrelevant year cards, graduation setup, Calculate and conversion footnote are hidden when both years are blocked. A mixed route keeps its supported year and any already-open second-year scale picker. The calculation guard and stored values remain intact. The separate “6–10 scale” pill is removed from the empty right panel, including its unused CSS. No calculation changes.

Browser verified on the local page: compact A-level state has no disabled-field cards; action +8.8/9.2 gives9.00; switching to IB and annual6/7 gives9.60; switching back restores the short notice and clears the old result. JS syntax check passes. Screenshot: qa/gpa-compact-portal-notice.png. Rollback: qa/before-compact-portal-notice/.

## Unsupported grade input is blocked — 2026-10-25

Marcin requested direct A*/A/B entry where a verified conversion is possible, and explicitly authorized blocking otherwise. Rechecked Bocconi’s 2027/28 guide and Cambridge’s qualification/grade descriptions. These establish the grade sets and application workflow, but no verified Bocconi letter-to-GPA mapping was found. A levels, AS and letter-graded GCSE/IGCSE therefore remain portal-only. No invented ordinal or UCAS conversion was introduced. The numeric 9–1 option remains the existing independent estimate.

Final behavior: unsupported non-portal input disables the school-grade fields, mode selector and Calculate GPA; a visible disabled placeholder and “Bocconi GPA required” notice explain what to do. Enter Bocconi GPA results enables only the affected years’ 6–10 inputs, preserving other numeric years and stored school-grade drafts. The country/curriculum picker stays visible so users can change scales again. Unsupported routes do not offer a misleading custom-scale shortcut. This gate applies consistently to all existing portal-only routes. Engine validation still rejects unsupported conversion even if a UI disable were bypassed. Main admissions/report/model behavior is unchanged.

Sources checked 2026-10-25:
- https://www.unibocconi.it/sites/default/files/media/attachments/GPA%20calculator%20guida_new_202728_0.pdf
- https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-advanced/cambridge-international-as-and-a-levels/qualification/
- https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-upper-secondary/cambridge-igcse/grading-choice/

## Current dropdown correction — 2026-10-25

Marcin explicitly rejected separate programme tiles. Keep the existing searchable Country or curriculum dropdown. Its first group is International curricula: IB Diploma, A levels / AS levels, GCSE / IGCSE, US High School, European Baccalaureate. Countries with local SVG flags follow by region; International / another system is last. A direct curriculum selects the relevant scale without a country. A levels/AS and GCSE numeric/letter variants remain separate in the adjacent scale selector. The same component serves the second year. Saved drafts retain the route; IB and national calculations are unchanged. The catalogue now has 92 routes and 74 source entries. Letter-grade routes require portal GPA, not guessed letter spacing. No tiles remain.

## Latest country-first flow — 2026-10-25

This supersedes the generic scale-only selector below. Start with Country or curriculum (searchable, local SVG flags), then Education system. A country with one national system selects it automatically; variants require selection. Compact Pass/Best/source and the relevant transcript note replace a long explanation. Manual bounds are requested only for school-defined scales. Different school or system? opens the same picker for the second year. Source data and all exceptions: GPA-COUNTRY-SYSTEMS-RESEARCH.md.70countries/territories,90routes;61automatic numeric,12custom,17portal. Do not describe all routes as automatic conversion.

## Outcome

One useful flow: choose a scale, enter two annual records, calculate, then use the GPA in the admissions calculator without losing the test score, programme or round. Same no-icon horizontal brand, restored Nunito, white fields, green actions and pale blue decorative panel. English UI. Keep drafts only in this browser tab for30 minutes. No external grade service or new account.

## Primary copy

H1: Bocconi GPA calculator
Intro: Convert your school grades to a GPA estimate, or combine your two results from Bocconi.
Setup: Graduation year / Grading scale / Passing grade / Best possible grade.
Year cards: Third-last year / Second-last year, with the actual school-year dates. Each offers School average / Subject grades / Bocconi GPA. Different grading scale is available for the second year.
School average helper: Use the unweighted average printed on your transcript.
Subject helper: Enter final grades, one per line or separated by semicolons.
Official-value helper: Enter the6–10 result from Bocconi’s application.
Primary form action: Calculate GPA.
Empty panel: Your GPA will appear here. Add both school years to see the result.
Result title: Estimated GPA, or Average Bocconi GPA when both annual values were entered from the portal.
Result action: Use in admission calculator →.
Estimate note: This is an estimate. Confirm your GPA in Bocconi’s application.
Official-values note: Average of the two Bocconi GPA values you entered.
Three short disclosures below: Which grades count? / Where is the official calculator? / How is this estimate calculated?

## Calculation contract

Keep the established independent numeric interpolation, not a claim to reproduce Bocconi’s unpublished algorithm. Each subject/annual average maps passing→6 and best→10; two annual outputs receive equal weight. Reject values beyond the best mark, reject an annual average below passing, and flag individual failing grades treated as passing. Decimal commas accepted. Reverse scales supported. Exact values transfer; formatting does not drive arithmetic. Both portal entries preserve official_portal provenance; any estimate remains a simulation. No GPA confirmation checkbox.

Offer verified numeric scale examples (IB2–7, French10–20), standard numeric templates2–6,4–1,5–10,4–6,50–100 and editable custom numeric bounds. These are school-scale templates, not an official country catalogue. Letter-only or mixed-scale records should use portal GPA rather than a fabricated official mapping. Same-year mixed scales remain outside this simplified estimator.

## Verified sources and limits

Bocconi guide2027/28, checked2026-10-25: portal-only personalized calculator, transcript average versus subjects, unweighted values, final grades, subject exclusions and failing-subject handling. The guide gives IB2–7 and French10–20 examples. It does not publish conversion tables or rounding rules.
https://www.unibocconi.it/sites/default/files/media/attachments/GPA%20calculator%20guida_new_202728_0.pdf

Bocconi GPA/curriculum page, checked2026-10-25: standard graduation2027 uses years ending2025/2026;11-year diplomas use grades10/11, so the UI offers an explicit11-year path. A missing transcript requires a school statement before duplicating an available year; this exceptional procedure stays in official guidance, not an automatic copy shortcut.
https://www.unibocconi.it/en/applying-bocconi/bachelor-and-law-programs/application-and-admissions/gpa-and-high-school-curriculum

Remove application deadlines, competitor comparison, Reddit screenshots, repeated country disclaimers, source acknowledgement checkboxes and promotional filler from the main flow. Retain source links and the distinction between an estimate and a portal result.
