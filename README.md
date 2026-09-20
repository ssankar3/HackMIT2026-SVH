# Restated — a greenwashing time machine

Corporate sustainability reports get restated, softened, and re-worded from
one year to the next, and almost nobody goes back and checks. Restated reads
a company's sustainability reports the way an analyst would: it extracts
every measurable claim, tracks the same claim and the same numbers across
multiple years of reports, and flags where the story quietly changed —
a number restated, a target dropped, a firm commitment turned into a hedge.

Everything runs **without an API key**: claim extraction, linguistic
analysis, cross-year drift detection, and rule-based "greenwashing sins" are
all deterministic regex/lexicon pipelines, not an LLM. Every score traces
back to the literal words or figures that produced it.

## Quick start

```
make demo
```

Builds the pipeline for the three seeded companies (H&M, Amazon, Microsoft)
and serves the dashboard at `http://localhost:8000/`.

Other useful targets — run `make help` for the full list:

| Target | What it does |
|---|---|
| `make all` | schema + say-do + language + score (no API key needed) |
| `make lang` | Stage 2/3/3b/4 only: claims, linguistics, language drift, sins |
| `make score` | Stage 8: scoring → `out/*.json` + `web/public/data/` |
| `make eval` | compare pipeline output against the held-out analyst prior |
| `make check-leakage` | assert no company name is hardcoded into the scoring logic |

## What the dashboard shows

- **Time machine** — a company's key metrics plotted year over year, sourced
  from *every* report that ever published that number, so a restatement
  between two reports shows up as two different lines for the same year.
- **Score breakdown** — five sub-scores (vagueness, unsupported claims, sins
  severity, say-do gap, goalpost drift), each `None`/abstained rather than
  defaulted to a flattering 0 when the pipeline can't measure it.
- **Drift evidence** — every restated figure and every softened/dropped
  claim, cited back to its exact PDF page.
- **Try a claim** — paste any sentence and see the same rule-based analysis
  applied live, word by word.
- **Try a company** — enter a company name, a year range, and a link (or
  upload) for each year's report PDF, and the same pipeline runs on demand
  to build a new dashboard page. No report is ever guessed or scraped
  automatically — you supply the source. A company added this way correctly
  shows its numeric drift score as "unmeasurable" rather than a fabricated
  number, since that figure depends on hand-verified data the other three
  companies have and a freshly-added company doesn't.

## Project layout

```
data/raw/{company}/{year}_sustainability_report.pdf   source PDFs
data/did/{company}.csv                                hand-verified figures (year,metric,value,unit,scope,source,page)
pipeline/                                              Stage 2-8 scripts (see NOTES.md for the full pipeline map)
pipeline/live_server.py                                local server behind "Try a company"
out/, web/public/data/                                 pipeline output consumed by the dashboard
web/index.html                                         the entire frontend (single static file, no build step)
```

## Known limitations

- Stages that need an LLM (verified-evidence extraction, the debate/judge
  stage) require `ANTHROPIC_API_KEY` and are not run by default.
- Peer percentile is computed across only the companies currently loaded —
  indicative, not a real industry benchmark.
- "Try a company" needs a direct link or upload for each report; it has no
  web-search step, and some sites (e.g. bot-protected CDNs) will refuse a
  scripted download even of a correct URL — upload the PDF in that case.

See `NOTES.md` for the full write-up: data-quality caveats per company, the
no-leakage discipline behind the scoring, and the demo script.
