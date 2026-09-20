"""Bridge Stage 3/3b language output into the Stage 8 scoring contract.

Sub-scores computed here WITHOUT an API key:

  vagueness           100*(1 - mean specificity) over material claims.
  unsupported_claims  share of quantified target/achievement claims with no
                      sentence-level verification. Discounted when the report
                      itself carries a third-party assurance statement -- a
                      figure on p.41 is not proven by 'limited assurance' on
                      p.80, but the report is also not naked.
  say_do_gap          linked target claims vs the verified DiD series
                      (see claim_link.py).

Claims are capped per company (`MAX_CLAIMS`) so the browser bundle stays small;
the full set always remains in out/language/{company}_claim_features.csv.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
LANG_DIR = ROOT / "out" / "language"
DRIFT_DIR = ROOT / "out" / "langdrift"
SINS_DIR = ROOT / "out" / "sins"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim_link import score_say_do  # noqa: E402

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


MATERIAL = {"target", "achievement", "attribute"}
# A report-level assurance statement is evidence, but it does not sit on the
# claim sentence. We discount unsupported_claims rather than zeroing it.
ASSURANCE_DISCOUNT = 0.55


def _truthy(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def language_sub_scores(lang: dict, rows: Optional[List[dict]] = None) -> tuple:
    """(vagueness, unsupported_claims, basis dict). Uses the most recent report
    year, scored only on material claims (target / achievement / attribute)."""
    series = [s for s in lang.get("series", []) if s.get("n_claims")]
    if not series:
        return None, None, {"reason": "no claims extracted"}
    latest = series[-1]
    year = latest["year"]

    material = [r for r in (rows or [])
                if int(r.get("publish_year") or 0) == year
                and (r.get("claim_type") or "") in MATERIAL]
    if not material:
        spec = latest.get("mean_specificity")
        vagueness = round((1.0 - spec) * 100, 1) if spec is not None else None
        return vagueness, None, {"year": year, "reason": "no material claims in latest year"}

    specs = [_f(r.get("specificity_score"), 0.0) for r in material]
    spec = sum(specs) / len(specs)
    vagueness = round((1.0 - spec) * 100, 1)

    checkable = [r for r in material
                 if _truthy(r.get("quantification_present"))
                 and (r.get("claim_type") or "") in {"target", "achievement"}]
    if checkable:
        naked = sum(1 for r in checkable if not _truthy(r.get("verification_present")))
        unsupported = naked / len(checkable)
        assured = bool(latest.get("report_assured"))
        if assured:
            unsupported *= ASSURANCE_DISCOUNT
        unsupported = round(unsupported * 100, 1)
    else:
        unsupported = None

    return vagueness, unsupported, {
        "year": year,
        "n_claims": latest["n_claims"],
        "n_material": len(material),
        "n_checkable": len(checkable),
        "mean_specificity": round(spec, 4),
        "quantified_share": latest.get("quantified_share"),
        "verified_share": latest.get("verified_share"),
        "hedged_share": latest.get("hedged_share"),
        "firm_share": latest.get("firm_share"),
        "say_more_prove_less": latest.get("say_more_prove_less"),
        "report_assured": bool(latest.get("report_assured")),
        "assurance_auditor": latest.get("assurance_auditor"),
        "assurance_discount_applied": ASSURANCE_DISCOUNT if latest.get("report_assured") else 0,
        "caveat": (
            "vagueness is lexical specificity over material claims; "
            "unsupported_claims is the naked-assertion rate among quantified "
            "targets/achievements, discounted if the report carries a third-party "
            "assurance statement"
        ),
    }


def _parse_highlights(raw) -> list:
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []


def _sins_by_sentence(sins: Optional[dict]) -> dict:
    out: Dict[str, list] = {}
    if not sins:
        return out
    for t in sins.get("tagged") or []:
        out.setdefault(t.get("sentence") or "", []).extend(t.get("sins") or [])
    return out


def _links_by_sentence(say_do: Optional[dict]) -> dict:
    out = {}
    for lk in (say_do or {}).get("links") or []:
        out[lk.get("sentence") or ""] = lk
    return out


def to_schema_claims(company: str, rows: List[dict],
                     sins: Optional[dict] = None,
                     say_do: Optional[dict] = None) -> List[dict]:
    """Map feature rows onto the frozen Claim model. Keeps the most specific
    claims first so the cap drops the least informative ones."""
    scored = sorted(rows, key=lambda r: -_f(r.get("specificity_score"), 0.0))
    meaningful = [r for r in scored if r.get("claim_type") != "other"] or scored
    sin_map = _sins_by_sentence(sins)
    link_map = _links_by_sentence(say_do)
    out = []
    for i, r in enumerate(meaningful[:MAX_CLAIMS]):
        sent = r["sentence"]
        lk = link_map.get(sent) or {}
        qty = _f(r.get("quantity"))
        if qty is None and r.get("quantity"):
            m = re.search(r"[\d,.]+", r["quantity"])
            qty = _f(m.group(0).replace(",", "")) if m else None
        out.append({
            "claim_id": f"{company}-claim-{i:04d}",
            "claim_text": sent,
            "claim_type": CLAIM_TYPE_MAP.get(r.get("claim_type", ""), "general"),
            "metric": lk.get("family"),
            "canonical_metric": lk.get("canonical_metric"),
            "target_value": qty,
            "unit": None,
            "baseline_year": int(r["baseline"]) if (r.get("baseline") or "").isdigit() else None,
            "target_year": int(r["deadline"]) if (r.get("deadline") or "").isdigit() else None,
            "scope": r.get("scope") or None,
            "has_third_party_verification": _truthy(r.get("verification_present")),
            "verifier_name": None,
            "chunk_id": f"{Path(r.get('source_file','')).name}#p{r.get('page')}",
            "doc_name": Path(r.get("source_file", "")).name,
            "publish_year": int(r["publish_year"]),
            "page": int(r["page"]) if (r.get("page") or "").isdigit() else None,
            "features": {
                "vague_word_ratio": _f(r.get("vague_ratio")),
                "hedging_ratio": _f(r.get("hedging_ratio")),
                "future_orientation_ratio": _f(r.get("future_ratio")),
                "quantification_present": _truthy(r.get("quantification_present")),
                "baseline_present": _truthy(r.get("baseline_present")),
                "date_present": _truthy(r.get("date_present")),
                "scope_present": _truthy(r.get("scope_present")),
                "verification_present": _truthy(r.get("verification_present")),
                "specificity_score": _f(r.get("specificity_score")),
                "readability": _f(r.get("readability")),
                "sentiment": _f(r.get("sentiment")),
                "negated": _truthy(r.get("negated")),
                "conditional": _truthy(r.get("conditional")),
                "highlights": _parse_highlights(r.get("highlights_json")),
                "report_assured": _truthy(r.get("report_assured")),
            },
            "sins": sin_map.get(sent, []),
            "evidence_ids": [],
            "progress_status": lk.get("status"),
            "debate": None,
        })
    return out


def compute_say_do(company: str, did_rows: List[dict]) -> dict:
    return score_say_do(load_claim_rows(company), did_rows)


# Stage3b's own event vocabulary -> the frozen schema's DriftType enum. A
# negated firm/hedged commitment IS the commitment being dropped, so it maps
# onto the existing commitment_dropped value rather than adding a new enum
# member. Everything else keeps the prior behaviour (wording_softened).
DRIFT_TYPE_MAP = {"commitment_negated": "commitment_dropped"}


def to_schema_drift(company: str, ld: dict) -> List[dict]:
    """Language drift events in the DriftEvent shape. `year` is the REPORT year
    that introduced the change, which is what makes them joinable against
    numeric restatements (see cross_signal)."""
    out = []
    for i, e in enumerate(ld.get("events", [])):
        out.append({
            "drift_id": e.get("event_id", f"{company}-lang-{i:03d}"),
            "year": int(e["to_year"]),
            "type": DRIFT_TYPE_MAP.get(e["type"], "wording_softened"),
            "canonical_metric": None,
            "old": e["old_text"][:300],
            "new": e["new_text"][:300],
            "magnitude_pct": None,
            "description": f"[{e['type']}] {e['detail']}",
            "evidence_chunk_ids": [],
            "old_source": e.get("old_source"),
            "new_source": e.get("new_source"),
            "page": e.get("new_page") if isinstance(e.get("new_page"), int) else None,
            "match_basis": e.get("matched_via"),
            "matched_slots": e.get("matched_slots", []),
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
