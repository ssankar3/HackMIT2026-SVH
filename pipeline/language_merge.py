"""Bridge Stage 3/3b language output into the Stage 8 scoring contract.

Two sub-scores become computable here WITHOUT an API key:

  vagueness           100*(1 - mean specificity). Specificity counts how many of
                      five evidence slots a claim pins down (quantity, baseline,
                      deadline, scope, verification) minus a vague/hedge penalty.
  unsupported_claims  share of claims carrying no third-party verification
                      language.

Both are PROXIES and are labelled as such downstream. In particular
`unsupported_claims` only sees verification wording in the claim's own sentence;
a claim assured elsewhere in the report reads as unsupported here. Fixing that
needs Stage 5 evidence retrieval. The number is still informative as a relative
measure of how often a company attaches its proof to its assertion.

Claims are capped per company (`MAX_CLAIMS`) so the browser bundle stays small;
the full set always remains in out/language/{company}_claim_features.csv.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
LANG_DIR = ROOT / "out" / "language"
DRIFT_DIR = ROOT / "out" / "langdrift"
SINS_DIR = ROOT / "out" / "sins"

MAX_CLAIMS = 220

# Stage 2's vocabulary -> the frozen schema's ClaimType enum.
CLAIM_TYPE_MAP = {
    "target": "target",
    "achievement": "achievement",
    "attribute": "product_label",
    "other": "general",
    "": "general",
}

# Recycled boilerplate is real but weak: republishing identical wording is not
# the same act as weakening a commitment. It is reported separately and kept
# out of the headline drift count.
WEAK_EVENT_TYPES = {"boilerplate_recycled"}


def have_language(company: str) -> bool:
    return (LANG_DIR / f"{company}_language.json").exists()


def load_language(company: str) -> Optional[dict]:
    p = LANG_DIR / f"{company}_language.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_langdrift(company: str) -> Optional[dict]:
    p = DRIFT_DIR / f"{company}_langdrift.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_claim_rows(company: str) -> List[dict]:
    p = LANG_DIR / f"{company}_claim_features.csv"
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def language_sub_scores(lang: dict) -> tuple:
    """(vagueness, unsupported_claims, basis dict). Uses the most recent report
    year, which is the company's current posture."""
    series = [s for s in lang.get("series", []) if s.get("n_claims")]
    if not series:
        return None, None, {"reason": "no claims extracted"}
    latest = series[-1]

    spec = latest.get("mean_specificity")
    verified = latest.get("verified_share")
    vagueness = round((1.0 - spec) * 100, 1) if spec is not None else None
    unsupported = round((1.0 - verified) * 100, 1) if verified is not None else None

    return vagueness, unsupported, {
        "year": latest["year"],
        "n_claims": latest["n_claims"],
        "mean_specificity": spec,
        "quantified_share": latest.get("quantified_share"),
        "verified_share": verified,
        "hedged_share": latest.get("hedged_share"),
        "firm_share": latest.get("firm_share"),
        "say_more_prove_less": latest.get("say_more_prove_less"),
        "caveat": ("both are lexical proxies; unsupported_claims only detects "
                   "verification wording inside the claim sentence itself"),
    }


def to_schema_claims(company: str, rows: List[dict]) -> List[dict]:
    """Map feature rows onto the frozen Claim model. Keeps the most specific
    claims first so the cap drops the least informative ones."""
    scored = sorted(rows, key=lambda r: -_f(r.get("specificity_score"), 0.0))
    # drop pure filler before capping
    meaningful = [r for r in scored if r.get("claim_type") != "other"] or scored
    out = []
    for i, r in enumerate(meaningful[:MAX_CLAIMS]):
        out.append({
            "claim_id": f"{company}-claim-{i:04d}",
            "claim_text": r["sentence"],
            "claim_type": CLAIM_TYPE_MAP.get(r.get("claim_type", ""), "general"),
            "metric": None,
            "canonical_metric": None,
            "target_value": _f(r.get("quantity")),
            "unit": None,
            "baseline_year": int(r["baseline"]) if (r.get("baseline") or "").isdigit() else None,
            "target_year": int(r["deadline"]) if (r.get("deadline") or "").isdigit() else None,
            "scope": r.get("scope") or None,
            "has_third_party_verification": (r.get("verification_present") == "True"),
            "verifier_name": None,
            "chunk_id": f"{Path(r.get('source_file','')).name}#p{r.get('page')}",
            "doc_name": Path(r.get("source_file", "")).name,
            "publish_year": int(r["publish_year"]),
            "page": int(r["page"]) if (r.get("page") or "").isdigit() else None,
            "features": {
                "vague_word_ratio": _f(r.get("vague_ratio")),
                "hedging_ratio": _f(r.get("hedging_ratio")),
                "future_orientation_ratio": _f(r.get("future_ratio")),
                "quantification_present": r.get("quantification_present") == "True",
                "baseline_present": r.get("baseline_present") == "True",
                "date_present": r.get("date_present") == "True",
                "scope_present": r.get("scope_present") == "True",
                "verification_present": r.get("verification_present") == "True",
                "specificity_score": _f(r.get("specificity_score")),
                "readability": _f(r.get("readability")),
                "sentiment": None,
            },
            "sins": [],
            "evidence_ids": [],
            "progress_status": None,
            "debate": None,
        })
    return out


def to_schema_drift(company: str, ld: dict) -> List[dict]:
    """Language drift events in the DriftEvent shape. `year` is the REPORT year
    that introduced the change, which is what makes them joinable against
    numeric restatements (see cross_signal)."""
    out = []
    for i, e in enumerate(ld.get("events", [])):
        out.append({
            "drift_id": e.get("event_id", f"{company}-lang-{i:03d}"),
            "year": int(e["to_year"]),
            "type": "wording_softened",
            "canonical_metric": None,
            "old": e["old_text"][:300],
            "new": e["new_text"][:300],
            "magnitude_pct": None,
            "description": f"[{e['type']}] {e['detail']}",
            "evidence_chunk_ids": [],
            "old_source": e.get("old_source"),
            "new_source": e.get("new_source"),
            "page": e.get("new_page") if isinstance(e.get("new_page"), int) else None,
        })
    return out


def drift_counts(ld: dict) -> dict:
    counts = ld.get("counts", {})
    weak = sum(v for k, v in counts.items() if k in WEAK_EVENT_TYPES)
    strong = sum(v for k, v in counts.items() if k not in WEAK_EVENT_TYPES)
    return {"substantive": strong, "boilerplate": weak, "by_type": counts}


def publish_year_of(source: str) -> Optional[int]:
    head = Path(source or "").name.split("_", 1)[0]
    return int(head) if head.isdigit() and len(head) == 4 else None


def cross_signal(numeric_events: List[dict], ld: Optional[dict]) -> dict:
    """Do the numbers and the words move in the SAME report?

    Both sides are keyed by the report that introduced the change -- for a
    numeric restatement that is the publish year of `new_source`, for a language
    event it is `to_year`. Joining on data year instead would be wrong: a 2025
    report restating 2019 is a 2025 act.
    """
    num_by_report: Counter = Counter()
    for e in numeric_events:
        py = publish_year_of(e.get("new_source", ""))
        if py:
            num_by_report[py] += 1

    lang_by_report: Counter = Counter()
    if ld:
        for e in ld.get("events", []):
            if e["type"] not in WEAK_EVENT_TYPES:
                lang_by_report[int(e["to_year"])] += 1

    years = sorted(set(num_by_report) | set(lang_by_report))
    rows = [{
        "report_year": y,
        "numeric_restatements": num_by_report.get(y, 0),
        "language_softenings": lang_by_report.get(y, 0),
        "both": bool(num_by_report.get(y) and lang_by_report.get(y)),
    } for y in years]

    concurrent = [r for r in rows if r["both"]]
    return {
        "rows": rows,
        "n_years": len(rows),
        "n_concurrent": len(concurrent),
        "concurrent_years": [r["report_year"] for r in concurrent],
        "interpretation": (
            "Report-years where the company both restated prior figures and "
            "softened claim wording. Co-occurrence is suggestive, not causal, "
            "and with this few report-years it is not statistically testable."
        ),
    }


def load_sins(company: str) -> Optional[dict]:
    p = SINS_DIR / f"{company}_sins.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
