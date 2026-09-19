# NOTES

## Decisions

- **Companies are `hm`, `microsoft`, `amazon`.** The original brief named Delta, but
  `data/raw/` contains no Delta PDFs. Amazon was substituted; its disclosure is the cleanest
  of the three. Update the company list elsewhere in the project to match.
- **`data/did/{company}.csv` is hand-verified, not machine-generated.** Every value was read
  off the cited page and checked as a literal substring of that page's text (Microsoft's chart
  figures were confirmed visually from a rendered page image, see below). `pipeline/draft_did.py`
  remains as a *locator* that finds candidate pages and dumps `{company}_DRAFT_v2.csv` +
  `{company}_REJECTED.txt`, but **nothing downstream reads those files** — only the verified
  `{company}.csv` feeds the scoring stages.
- **Schema is the 7-column spec form**: `year,metric,value,unit,scope,source,page`.
- Values are recorded exactly as printed, with thousands separators removed. No unit conversion,
  no interpolation, no filled gaps, no derived scope splits.

## Coverage

| Company | Rows | Years | Metrics | Reports |
|---|---|---|---|---|
| hm | 70 | 2019–2025 | 13 | 2025, 2024 |
| amazon | 65 | 2019–2025 | 6 | 2025, 2024 |
| microsoft | 18 | 2020–2025 | 10 | 2025, 2024 |

Sources: `hm` 2025 p.41 + p.47, 2024 p.34 · `amazon` 2025 p.46, 2024 p.50 ·
`microsoft` 2025 p.30/p.54/p.62, 2024 p.13/p.21.

## Data-quality issues

### 1. H&M restates prior years heavily — 17 pairs, and some are enormous

The same metric-year carries different values in the 2024 vs 2025 report. **These rows are
deliberately not deduplicated**; the `source` column distinguishes them. Examples:

| Year | Metric | 2024 report | 2025 report |
|---|---|---|---|
| 2024 | scope_1_emissions | 15,102 | 17,002 |
| 2023 | scope_1_emissions | 15,754 | 17,050 |
| 2024 | scope_3_emissions_excl_use_phase | 6,955,000 | **4,865,000** |
| 2023 | scope_3_emissions_excl_use_phase | 6,754,000 | **5,438,000** |
| 2019 | scope_3_emissions_excl_use_phase | 9,115,000 | **7,442,000** |

The Scope 3 restatements move the **2019 baseline** as well as the recent years — i.e. the
yardstick moved at the same time as the measurement. The 2024 report footnotes "we have made
some updates in how we calculate emissions compared with 2023." Stage 6 should treat the most
recent report as the current actual and emit each delta as a drift event.

### 2. Amazon changed its carbon-intensity denominator

- Through the 2024 report: `carbon_intensity_per_gms` — grams CO2e per **$ of gross merchandise sales**
- From the 2025 report: `carbon_intensity_per_revenue` — grams CO2e per **$ of revenue**

Stored under two metric names because they are not comparable (2024: 72.6 vs 2025-report 2024: 109.0
for the same year). Amazon's own footnote: began reporting per-revenue "to better align to
independent reporting standards."

### 3. Amazon also restates Scope 3 and total footprint

Scope 1 and Scope 2 are identical across both reports, but Scope 3 moved for every recent year
(2023: 47.40 → 48.30; 2024: 50.32 → 51.62), and so did the total footprint. Scope 1/2 stability
alongside Scope 3 drift is itself a signal worth surfacing.

### 4. Units differ across companies — do not compare raw values

- `hm` — tCO2e (2025 report) and "tonnes CO2e" (2024 report); same magnitude, different label
- `amazon` — **MMT CO2e**, i.e. millions of metric tons (a 10⁶ factor vs the others)
- `microsoft` — metric tons CO2e

Stage 8's `peer_percentile` must normalize or restrict itself to within-company trends.

### 5. Fiscal vs calendar year

- **microsoft** — fiscal year ends June 30. The `year` column holds the numeric fiscal year
  (`2025` = FY25 ≈ Jul 2024–Jun 2025); the `scope` column spells out `FY25 (fiscal year ends June 30)`.
  Also note Microsoft's file naming is offset from the report's own title: `2025_sustainability_report.pdf`
  is titled the *2026* Environmental Sustainability Report, and `2024_..._report.pdf` is titled *2025*.
  **Trust the data year in the table, never the filename or the publication year.**
- **amazon** — calendar year.
- **hm** — financial year is not calendar-aligned.

### 6. Microsoft coverage is thin, and that is a finding rather than a gap to paper over

Microsoft does not publish a Scope 1/2/3 time series in the report itself. What exists:

- **Total emissions only**, FY20–FY25, and only as a **bar chart image** on 2025 report p.30.
  Its text layer extracts the digits reversed (`000,188,21` ← `12,881,000`), so the values were
  confirmed by rendering the page and reading it visually. Internally consistent:
  20,290,000 / 16,215,000 = 1.25, matching the report's stated "increased 25% year over year."
- **Scope split as percentages only** (FY24 and FY25).
- Absolute Scope 1/2/3 figures live in a separate **"Environmental Data Fact Sheet"** that is
  not in `data/raw/`. Adding it would materially improve Microsoft coverage.

**Scope 1/2/3 absolutes were deliberately NOT synthesised** by multiplying total × percentage —
that would be imputation. Microsoft should surface as low-confidence / `abstain` in Stage 8
rather than being silently filled in.

Worth noting: Microsoft's Scope 2 share jumped from **1.74% (FY24) to 13.34% (FY25)** because it
stopped using non-additional unbundled RECs. Reported emissions rose as a result of a *more*
conservative accounting choice — a useful counter-example for the model, since not every increase
is bad behaviour.

### 7. Microsoft's carbon-removal figures are not comparable year to year

- FY24: `21,927,370` — exact, described as "total metric tons **contracted**"
- FY25: `45,000,000` — recorded from "**more than** 45 million metric tons", i.e. a **lower bound**,
  and described as "**contributed**" rather than "contracted"

Stored under two metric names (`carbon_removal_contracted`, `carbon_removal_contributed`) so nothing
reads them as a clean 22M → 45M trend. The vagueness of "more than 45 million" is itself a
Stage 3 specificity signal.

### 8. Smaller boundary notes

- **amazon** reports a single Scope 2 figure (market-based, per its own methodology note), so no
  market-vs-location comparison is possible. `hm` reports both.
- **hm** has two different "recycled materials share" figures for the same year — 32% for
  commercial products, 35% including packaging. Stored as `recycled_materials_share` and
  `recycled_materials_share_incl_packaging`.
- **hm** total-emissions rows from the 2024 report carry no stated market/location basis; the
  `scope` column says so rather than guessing.
- H&M's 2024 report labels its baseline column across two stacked header lines (`2019` above
  `(Baseline)`). Automated parsing saw only `(Baseline)` and dropped those cells; reading the page
  directly recovered them. The 2019 values from that report are included.

## Available but not yet extracted

- H&M Scope 3 **by category** (15 categories), 2025 report p.41 and 2024 report p.34.
- Amazon Scope 1 breakdown (fossil fuels vs refrigerants), both reports.
- H&M energy consumption / MWh detail, 2025 report p.41.
- H&M reports 2021–2023 (only 2024 and 2025 transcribed so far). Adding them would extend both
  year coverage and the restatement chain.

---

# Pipeline & dashboard

## What runs today (no API key needed)

`make all` → `make demo` → http://localhost:8000

| Stage | Status |
|---|---|
| 0 — contract (`pipeline/models.py`, `schema/output.schema.json`) | done |
| 6 — say-do / goalpost drift (`pipeline/stage6_saydo.py`) | **done, real data** |
| 8 — scoring + abstention (`pipeline/stage8_score.py`) | done |
| Dashboard (`web/index.html`) | done |
| 1,2,3,4,5,7 — ingest, claims, linguistics, sins, evidence, debate | **blocked: no `ANTHROPIC_API_KEY`** |

The dashboard is a single dependency-free HTML file. `npm install` was not viable
(disk at ~290 MB free, and a partial install had already corrupted one package), and a
zero-dependency page is the more reliable demo artefact anyway — it works offline from
`file://` via `web/data.js`, and over HTTP via `web/public/data/*.json`.

## Results (derived, not configured)

| Company | Republished cells | Restated | Median move | Max | Drift score |
|---|---|---|---|---|---|
| H&M Group | 18 | 15 (83%) | 9.21% | 19.48% | **49.8** |
| Amazon | 24 | 5 (21%) | 1.90% | 2.58% | **15.7** |
| Microsoft | 0 | — | — | — | **unmeasurable** |

Overall greenwashing likelihood is **withheld for all three** (`abstain: true`): only 1 of 5
sub-scores is measurable, and `config/weights.yaml` requires 3. An unmeasured component is
`None`, never 0 — scoring it 0 would read as "clean", which is the easiest way for a
greenwashing detector to fail silently in the flattering direction.

## No-leakage discipline

`make check-leakage` greps the scoring pipeline and config for company names and **fails the
build** if any appear. The analyst's prior about relative risk lives in `eval/analyst_prior.md`,
quarantined out of every prompt, weight and lexicon, and is used only to check the output
afterwards. The pipeline independently reproduced the prior's ordering for the two measurable
companies — worth reporting precisely *because* it was not an input. Caveats are in that file:
n=2 is weak evidence, and Microsoft is unmeasurable rather than exonerated.

## Known limitations

1. **4 of 5 sub-scores need an API key.** Vagueness, unsupported claims, sins severity and the
   say-do glidepath are all blocked. The claims list, debate court and eval tab are empty.
2. **Peer percentile on n=2** measurable companies. Indicative only; labelled as such in the UI.
3. **Drift only sees 2 reports per company.** Transcribing H&M 2021–2023 and the equivalent
   Amazon/Microsoft years would lengthen the restatement chain considerably.
4. **Microsoft cannot be scored at all** on this signal — no metric-year appears twice. Adding
   its Environmental Data Fact Sheet to `data/raw/microsoft/` is the highest-value next input.
5. `say_do_gap` is a *partial* implementation: goalpost drift is live, target-vs-glidepath is not.

## 3-minute demo script

1. **Open on H&M** (`#hm`). Headline: *15 of 18 republished figures were restated between
   reports.* Point at the amber 49.8 — then immediately at **ABSTAIN** beside it: the system
   refuses an overall verdict on one component. That is the calibration story.
2. **"How confident are we and why"** — the hatched bars are unmeasured, each labelled with the
   exact blocker. Nothing is quietly defaulted to zero.
3. **Drift evidence tab** — every row is a figure the company published twice, differently, with
   both source PDFs and page numbers. Scope 3 for 2019 moved −18.4%: *the baseline itself moved*,
   which retroactively changes every percentage-against-target the company has ever quoted.
4. **Time machine, the punchline.** Sitting on 2024 in Hindsight: intensity 100. Flip to
   **"As of then"**: it empties. Standing in 2024, H&M looked clean — every restatement arrives
   with the 2025 report. That gap between what was knowable and what is now known is the product.
5. **Switch to Microsoft.** Score is `n/a`, not 0. Explain why that is the honest answer, and
   what document would fix it.
6. **Investor view** — framed as a data-quality overlay on transition risk: a company that
   restates its own history is a company whose reported series you should fit with wider error
   bars. No return claims.
