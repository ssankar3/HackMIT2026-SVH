"""Stage 8: scoring, abstention, and the final /out/{company}.json contract.

Two rules drive everything here:

1. A sub-score that could not be measured is None, never 0. Scoring a missing
   component as zero would read as "clean" and is the single easiest way to
   make a greenwashing detector lie in the safe direction.
2. An overall score is emitted only when at least `min_components` of the five
   sub-scores are live (config/weights.yaml). Otherwise the company abstains
   and the UI must say so.

The as-of / hindsight split is real here, not cosmetic. A restatement is only
knowable once the report that makes it has been published, so "as of 2024"
genuinely cannot see a revision that first appears in the 2025 report. Scrubbing
the time machine backwards makes later-discovered drift disappear.

Usage:
    python3 pipeline/stage8_score.py [--limit N] [company ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import language_merge as LM  # noqa: E402
from models import (  # noqa: E402
    Claim, CompanyOutput, CompanySummary, Confidence, DataCoverage, DidPoint,
    DriftEvent, EvalSummary, SubScores, TimelinePoint, YearScore,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
WEB_DATA = ROOT / "web" / "public" / "data"
CONFIG = ROOT / "config" / "weights.yaml"

DISPLAY = {"hm": "H&M Group", "microsoft": "Microsoft", "amazon": "Amazon", "delta": "Delta Air Lines"}

SUB_SCORE_FIELDS = ["vagueness", "unsupported_claims", "sins_severity", "say_do_gap", "goalpost_drift"]

# Stages that cannot run without an Anthropic API key, and what each one blocks.
LLM_BLOCKED = {
    "stage5_evidence": "verified evidence quotes (and a non-proxy unsupported_claims)",
    "stage7_debate": "judge probability, confidence band",
}


def load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def publish_year_of(source: str) -> Optional[int]:
    head = Path(source).name.split("_", 1)[0]
    return int(head) if head.isdigit() and len(head) == 4 else None


def drift_intensity(events: List[dict]) -> Optional[float]:
    """Scale a set of drift events onto 0-100 by total magnitude moved.
    Saturates at a cumulative 60% so one huge or several large moves both max out."""
    if not events:
        return None
    total = sum(abs(e["magnitude_pct"]) for e in events if e.get("magnitude_pct") is not None)
    total += 15.0 * sum(1 for e in events if e["type"] == "metric_redefined")
    return round(min(total / 60.0, 1.0) * 100, 1)


def blend_overall(subs: Dict[str, Optional[float]], cfg: dict) -> tuple:
    """Weighted blend over measurable sub-scores only, with weights renormalized.
    Returns (overall_or_None, n_measured, reasons)."""
    weights = cfg["weights"]
    live = {k: v for k, v in subs.items() if v is not None}
    reasons: List[str] = []

    missing = [k for k in SUB_SCORE_FIELDS if subs.get(k) is None]
    for m in missing:
        blockers = [s for s, blocks in LLM_BLOCKED.items() if m in blocks]
        if blockers:
            reasons.append(f"{m}: requires {', '.join(blockers)} (no ANTHROPIC_API_KEY set)")
        else:
            reasons.append(f"{m}: not measurable from available data")

    if len(live) < cfg["min_components"]:
        reasons.insert(0, (
            f"only {len(live)} of {len(SUB_SCORE_FIELDS)} sub-scores could be measured; "
            f"config requires at least {cfg['min_components']} before an overall score is emitted"
        ))
        return None, len(live), reasons

    total_w = sum(weights[k] for k in live)
    overall = sum(live[k] * weights[k] for k in live) / total_w
    return round(overall, 1), len(live), reasons


def pick_confidence(n_measured: int, n_verified_quotes: int, cfg: dict) -> tuple:
    c = cfg["confidence"]
    if n_measured >= c["min_components_for_high"] and n_verified_quotes >= c["min_verified_quotes_for_high"]:
        return Confidence.high, "multiple independent sub-scores plus corroborating verified evidence"
    if n_measured >= c["min_components_for_medium"] and n_verified_quotes >= c["min_verified_quotes_for_medium"]:
        return Confidence.medium, "several sub-scores measurable with some verified evidence"
    bits = []
    if n_measured < c["min_components_for_medium"]:
        bits.append(f"only {n_measured} of {len(SUB_SCORE_FIELDS)} sub-scores measurable")
    if n_verified_quotes < c["min_verified_quotes_for_medium"]:
        bits.append(f"{n_verified_quotes} verified evidence quotes (need "
                    f"{c['min_verified_quotes_for_medium']}+ for medium)")
    return Confidence.low, "capped at low: " + "; ".join(bits)


def build(company: str, cfg: dict) -> CompanyOutput:
    s6 = json.loads((OUT_DIR / f"{company}.stage6.json").read_text(encoding="utf-8"))
    events = s6["drift_events"]

    lang = LM.load_language(company)
    ld = LM.load_langdrift(company)
    vagueness = unsupported = None
    lang_basis: dict = {}
    if lang:
        vagueness, unsupported, lang_basis = LM.language_sub_scores(lang)

    sins = LM.load_sins(company)
    subs = SubScores(
        vagueness=vagueness,
        unsupported_claims=unsupported,
        sins_severity=(sins or {}).get("sins_severity"),
        say_do_gap=s6["say_do_gap"],
        goalpost_drift=s6["goalpost_drift"],
    )
    overall, n_measured, reasons = blend_overall(subs.model_dump(), cfg)
    n_verified_quotes = 0  # Stage 5 blocked
    confidence, conf_rationale = pick_confidence(n_measured, n_verified_quotes, cfg)

    # ---- timeline, with a genuine as-of vs hindsight split -----------------
    did_rows = s6["did_rows"]
    data_years = sorted({r["year"] for r in did_rows})

    # the series as currently told (latest report per metric)
    latest_by_metric = s6["series"]
    series_years: Dict[int, List[DidPoint]] = {}
    for metric, pts in latest_by_metric.items():
        for p in pts:
            series_years.setdefault(p["year"], []).append(DidPoint(
                year=p["year"], metric=metric, value=p["value"], unit=p["unit"],
                scope=p.get("scope"), source=p["source"], page=p.get("page"),
            ))

    timeline: List[TimelinePoint] = []
    year_scores: List[YearScore] = []
    for y in data_years:
        for_year = [e for e in events if e["year"] == y]
        # as-of: only drift a reader could have known about by year y, i.e. the
        # restating report was already published
        as_of_events = [e for e in for_year
                        if (py := publish_year_of(e.get("new_source") or "")) and py <= y]
        timeline.append(TimelinePoint(
            year=y,
            said_claim_ids=[],
            did_points=series_years.get(y, []),
            drift_ids=[e["drift_id"] for e in for_year],
            score_as_of=drift_intensity(as_of_events),
            score_hindsight=drift_intensity(for_year),
        ))
        year_scores.append(YearScore(
            year=y,
            sub_scores=SubScores(goalpost_drift=drift_intensity(for_year)),
            overall=None,
            confidence=Confidence.low,
            abstain=True,
            abstain_reasons=reasons,
            n_claims=0,
            n_verified_quotes=0,
            n_drift_events=len(for_year),
        ))

    headline = None
    b = s6["goalpost_drift_basis"]
    if s6["goalpost_drift"] is not None:
        headline = (
            f"{b['n_restatements']} of {b['n_comparable_pairs']} republished figures were restated "
            f"between reports (median move {b['median_abs_restatement_pct']}%, "
            f"max {b['max_abs_restatement_pct']}%)."
        )
    else:
        headline = (
            "No metric-year was published by more than one report, so restatement behaviour "
            "cannot be assessed from the available documents."
        )

    coverage = DataCoverage(
        n_documents=s6["n_reports"],
        n_did_rows=s6["n_did_rows"],
        did_years=s6["did_years"],
        did_metrics=s6["did_metrics"],
        stages_completed=["stage6_saydo", "stage8_score"],
        stages_blocked={k: "no ANTHROPIC_API_KEY in environment" for k in LLM_BLOCKED},
    )

    summary = CompanySummary(
        company=company,
        display_name=DISPLAY.get(company, company),
        overall_score=overall,
        confidence=confidence,
        confidence_rationale=conf_rationale,
        peer_percentile=None,  # filled in once every company is built
        abstain=overall is None,
        abstain_reasons=reasons,
        headline=headline,
        sub_scores=subs,
    )

    claims = [Claim(**c) for c in LM.to_schema_claims(company, LM.load_claim_rows(company))]
    lang_events = [DriftEvent(**e) for e in LM.to_schema_drift(company, ld)] if ld else []
    dcounts = LM.drift_counts(ld) if ld else {}
    cross = LM.cross_signal(events, ld)

    coverage.n_claims = len(claims)
    coverage.stages_completed += ["stage2_claims", "stage3_language", "stage3b_langdrift"]

    summary.headline = (summary.headline or "")
    if dcounts:
        summary.headline += (
            f" Language: {dcounts['substantive']} substantive softening events "
            f"across reports ({dcounts['boilerplate']} recycled boilerplate excluded)."
        )

    out = CompanyOutput(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        summary=summary,
        coverage=coverage,
        timeline=timeline,
        year_scores=year_scores,
        evidence=[],
        drift_events=[DriftEvent(**e) for e in events] + lang_events,
        did_points=[DidPoint(**{k: r[k] for k in ("year", "metric", "value", "unit", "scope", "source", "page")})
                    for r in did_rows],
        claims=claims,
        top_damaging_claim_ids=[c.claim_id for c in claims[:5]],
        eval=EvalSummary(n_gold=0, notes="Stage 4/7 blocked; no LLM predictions to score yet."),
    )
    out.language = {
        "series": (lang or {}).get("series", []),
        "sub_score_basis": lang_basis,
        "drift_counts": dcounts,
        "sins": {k: v for k, v in (sins or {}).items() if k != "tagged"},
        "sins_top": (sins or {}).get("tagged", [])[:25],
        "cross_signal": cross,
    }
    return out


def assign_peer_percentiles(outs: Dict[str, CompanyOutput]) -> None:
    """Percentile on goalpost_drift among companies where it is measurable.
    With a handful of peers this is indicative only, and the UI says so."""
    measurable = {c: o.summary.sub_scores.goalpost_drift for c, o in outs.items()
                  if o.summary.sub_scores.goalpost_drift is not None}
    if len(measurable) < 2:
        return
    for c, o in outs.items():
        v = measurable.get(c)
        if v is None:
            continue
        below = sum(1 for x in measurable.values() if x < v)
        o.summary.peer_percentile = round(below / (len(measurable) - 1) * 100, 1)


def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("companies", nargs="*")
    ap.add_argument("--limit", type=int, default=None, help="cap companies processed")
    args = ap.parse_args(argv)

    cfg = load_cfg()
    companies = args.companies or sorted(p.stem.split(".")[0] for p in OUT_DIR.glob("*.stage6.json"))
    if args.limit:
        companies = companies[: args.limit]

    outs = {c: build(c, cfg) for c in companies}
    assign_peer_percentiles(outs)

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    index = []
    for c, o in outs.items():
        blob = o.model_dump_json(indent=2)
        (OUT_DIR / f"{c}.json").write_text(blob, encoding="utf-8")
        (WEB_DATA / f"{c}.json").write_text(blob, encoding="utf-8")
        index.append({
            "company": c, "display_name": o.summary.display_name,
            "overall_score": o.summary.overall_score, "abstain": o.summary.abstain,
            "goalpost_drift": o.summary.sub_scores.goalpost_drift,
            "confidence": o.summary.confidence.value,
            "peer_percentile": o.summary.peer_percentile,
        })
        s = o.summary
        print(f"\n=== {c} ===", file=sys.stderr)
        print(f"  overall        {s.overall_score}  (abstain={s.abstain})", file=sys.stderr)
        print(f"  goalpost_drift {s.sub_scores.goalpost_drift}", file=sys.stderr)
        print(f"  confidence     {s.confidence.value} -- {s.confidence_rationale}", file=sys.stderr)
        print(f"  percentile     {s.peer_percentile}", file=sys.stderr)
        print(f"  drift events   {len(o.drift_events)}", file=sys.stderr)

    (WEB_DATA / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    # Also emit a plain JS bundle so web/index.html works when opened directly
    # from disk (file:// blocks fetch). Serving over HTTP still picks up the
    # JSON files, so this is a fallback, not the source of truth.
    bundle = {c: json.loads(o.model_dump_json()) for c, o in outs.items()}
    (ROOT / "web" / "data.js").write_text(
        "window.__GREENWASH_DATA__ = " + json.dumps(bundle, indent=1) + ";\n",
        encoding="utf-8",
    )
    print(f"\nwrote {len(outs)} company files to {OUT_DIR} and {WEB_DATA}", file=sys.stderr)
    print(f"wrote offline bundle to {ROOT / 'web' / 'data.js'}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
