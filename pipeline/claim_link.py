"""Link extracted target claims to the verified DiD series (say vs do).

This is the half of Stage 6 that needs language: a sentence like "30% recycled
materials by 2025" is useless as a greenwashing signal until it is pinned to
the company's own published recycled-share series. Linking is lexicon + regex
over metric families (config/lexicons.yaml) -- no company names, no LLM.

A link is only scored when units are comparable (a % target vs a % series, or
a "% reduce" target vs a computed % change on an emissions series). Anything
that would require guessing a unit conversion is left unmeasurable.

Usage (called from language_merge / stage 8, not standalone):
    from claim_link import score_say_do
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
LEXICONS = ROOT / "config" / "lexicons.yaml"

RE_NUM = re.compile(r"([\d,.]+)")
RE_PCT = re.compile(r"%|per\s*cent", re.I)
RE_REDUCE = re.compile(
    r"\b(reduc\w+|cut|lower\w*|decreas\w+|declin\w+|drop\w*)\b", re.I,
)
# Quantity is about a local project, not the company-wide series.
RE_OFF_TOPIC = re.compile(
    r"\b(food waste|concrete|xbox|hq2|cars? off the road|standby mode|"
    r"this building|this store|this facility)\b", re.I,
)
NEAR = 100  # chars: quantity / target verb must sit near the metric phrase
MIN_MEASURABLE = 3


def _load_families() -> List[Tuple[str, List[re.Pattern]]]:
    d = yaml.safe_load(LEXICONS.read_text(encoding="utf-8"))
    out = []
    for fam, phrases in (d.get("metric_families") or {}).items():
        pats = [re.compile(rf"\b{re.escape(p)}\b", re.I) for p in phrases]
        out.append((fam, pats))
    return out


_FAMILIES: Optional[List[Tuple[str, List[re.Pattern]]]] = None


def families() -> List[Tuple[str, List[re.Pattern]]]:
    global _FAMILIES
    if _FAMILIES is None:
        _FAMILIES = _load_families()
    return _FAMILIES


# DiD column -> family. First match wins; more specific families listed first
# in lexicons.yaml (total_emissions before scope_1).
FAMILY_TO_DID = {
    "total_emissions": lambda m: m.startswith("total_emissions") or m == "scope_1_2_emissions",
    "scope_3": lambda m: m.startswith("scope_3"),
    "scope_2": lambda m: m.startswith("scope_2"),
    "scope_1": lambda m: m == "scope_1_emissions",
    "recycled_share": lambda m: m.startswith("recycled"),
    "renewable_share": lambda m: m.startswith("renewable"),
    "carbon_intensity": lambda m: "intensity" in m,
}

UP_FAMILIES = {"recycled_share", "renewable_share"}
DOWN_FAMILIES = {"total_emissions", "scope_1", "scope_2", "scope_3", "carbon_intensity"}


def parse_quantity(q: str) -> Tuple[Optional[float], bool]:
    """Return (value, is_percent)."""
    if not q:
        return None, False
    m = RE_NUM.search(q)
    if not m:
        return None, False
    try:
        val = float(m.group(1).replace(",", ""))
    except ValueError:
        return None, False
    return val, bool(RE_PCT.search(q))


def match_family(sentence: str) -> Optional[Tuple[str, int]]:
    """Return (family, match_start) for the first family whose phrase sits
    near a number or a reduction verb. Distant mentions ('GHG in the
    appendix') are not a target."""
    if RE_OFF_TOPIC.search(sentence):
        return None
    qty = RE_NUM.search(sentence)
    red = RE_REDUCE.search(sentence)
    anchors = [m.start() for m in (qty, red) if m]
    for fam, pats in families():
        for p in pats:
            m = p.search(sentence)
            if not m:
                continue
            if anchors and not any(abs(m.start() - a) <= NEAR for a in anchors):
                continue
            return fam, m.start()
    return None


def resolve_metric(family: str, available: List[str]) -> Optional[str]:
    pred = FAMILY_TO_DID.get(family)
    if not pred:
        return None
    hits = [m for m in available if pred(m)]
    if not hits:
        return None
    # prefer the shortest / least-qualified name (the headline series)
    hits.sort(key=lambda m: (len(m), m))
    return hits[0]


def latest_by_metric(did_rows: List[dict]) -> Dict[str, List[dict]]:
    by: Dict[str, List[dict]] = defaultdict(list)
    for r in did_rows:
        by[r["metric"]].append(r)
    out = {}
    for metric, rows in by.items():
        # keep the series as told by the newest report that carries each year
        best: Dict[int, dict] = {}
        for r in rows:
            src = str(r.get("source") or "")
            head = Path(src).name.split("_", 1)[0]
            pub = int(head) if head.isdigit() else 0
            year = int(r["year"])
            prev = best.get(year)
            prev_pub = 0
            if prev:
                ph = Path(str(prev.get("source") or "")).name.split("_", 1)[0]
                prev_pub = int(ph) if ph.isdigit() else 0
            if prev is None or pub >= prev_pub:
                best[year] = r
        out[metric] = [best[y] for y in sorted(best)]
    return out


def _year(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def glidepath(family: str, target: float, target_is_pct: bool,
              baseline_year: Optional[int], deadline: Optional[int],
              series: List[dict], as_of_year: int) -> Optional[dict]:
    """Compare a numeric target against the company's own series.

    Share-like families (recycled, renewable): target is a level to reach.
    Emissions-like families: a percent target is a reduction vs baseline;
    an absolute target is left unmeasurable (unit conversion risk).
    """
    if not series:
        return None
    by_year = {int(p["year"]): float(p["value"]) for p in series}
    latest_year = max(y for y in by_year if y <= as_of_year) if any(y <= as_of_year for y in by_year) else max(by_year)
    actual = by_year[latest_year]
    unit = series[-1].get("unit") or ""

    if family in UP_FAMILIES:
        if not target_is_pct and "%" not in unit:
            return None
        t0 = baseline_year if baseline_year in by_year else min(by_year)
        start = by_year[t0]
        if deadline and deadline > t0:
            elapsed = min(1.0, max(0.0, (as_of_year - t0) / (deadline - t0)))
        else:
            elapsed = 1.0
        required = start + (target - start) * elapsed
        gap = actual - required
        status = "on_track" if actual >= required else "behind"
        if deadline and as_of_year >= deadline and actual < target:
            status = "behind"
        return {
            "status": status,
            "actual": actual,
            "required": round(required, 3),
            "target": target,
            "actual_year": latest_year,
            "elapsed": round(elapsed, 3),
            "gap": round(gap, 3),
            "direction": "up",
        }

    if family in DOWN_FAMILIES:
        if not target_is_pct:
            return None  # do not invent a unit conversion
        t0 = baseline_year if baseline_year in by_year else min(by_year)
        start = by_year[t0]
        if not start:
            return None
        if deadline and deadline > t0:
            elapsed = min(1.0, max(0.0, (as_of_year - t0) / (deadline - t0)))
        else:
            elapsed = 1.0
        # target is "reduce X%" -- required level is start * (1 - X/100 * elapsed)
        required = start * (1.0 - (target / 100.0) * elapsed)
        actual_pct = (actual - start) / abs(start) * 100.0
        status = "on_track" if actual <= required else "behind"
        return {
            "status": status,
            "actual": actual,
            "required": round(required, 3),
            "target": target,
            "actual_year": latest_year,
            "elapsed": round(elapsed, 3),
            "actual_pct_vs_baseline": round(actual_pct, 2),
            "direction": "down",
        }
    return None


def score_say_do(claim_rows: List[dict], did_rows: List[dict],
                 as_of_year: Optional[int] = None) -> dict:
    """Return (score or None, basis, per-claim links).

    Score is 0-100, higher = riskier = more linked targets behind glidepath.
    Needs at least 2 measurable links; otherwise None (unmeasurable, not 0).
    """
    series = latest_by_metric(did_rows)
    available = list(series)
    if as_of_year is None and did_rows:
        as_of_year = max(int(r["year"]) for r in did_rows)
    as_of_year = as_of_year or 2025

    links: List[dict] = []
    for r in claim_rows:
        if (r.get("claim_type") or "") != "target":
            continue
        hit = match_family(r.get("sentence") or "")
        if not hit:
            continue
        fam, _ = hit
        metric = resolve_metric(fam, available)
        if not metric:
            continue
        qty, is_pct = parse_quantity(r.get("quantity") or "")
        if qty is None:
            continue
        g = glidepath(
            fam, qty, is_pct,
            _year(r.get("baseline")),
            _year(r.get("deadline")),
            series[metric],
            as_of_year,
        )
        if not g:
            continue
        # a reduction verb on an UP family (or vice versa) is a confused claim;
        # still report it but do not let it flip the status.
        links.append({
            "sentence": r.get("sentence"),
            "page": r.get("page"),
            "publish_year": _year(r.get("publish_year")),
            "source_file": r.get("source_file"),
            "family": fam,
            "canonical_metric": metric,
            "quantity": r.get("quantity"),
            "baseline": r.get("baseline") or None,
            "deadline": r.get("deadline") or None,
            **g,
        })

    measurable = [x for x in links if x["status"] in ("on_track", "behind", "reversed")]
    behind = [x for x in measurable if x["status"] != "on_track"]
    fams = {x["family"] for x in measurable}
    only_easy_shares = fams and fams <= UP_FAMILIES
    if len(measurable) < MIN_MEASURABLE:
        return {
            "say_do_gap": None,
            "n_linked": len(links),
            "n_measurable": len(measurable),
            "n_behind": len(behind),
            "links": links[:40],
            "reason": (
                f"only {len(measurable)} target(s) could be pinned to a verified "
                f"series with comparable units; need {MIN_MEASURABLE}+ to score"
            ),
        }
    if only_easy_shares:
        return {
            "say_do_gap": None,
            "n_linked": len(links),
            "n_measurable": len(measurable),
            "n_behind": len(behind),
            "links": links[:40],
            "reason": (
                "every linked target is a share metric (renewable/recycled). "
                "Scoring that alone would read as clean while absolute emissions "
                "can still be rising, so the say-do gap is left unmeasurable"
            ),
        }

    score = round(100.0 * len(behind) / len(measurable), 1)
    return {
        "say_do_gap": score,
        "n_linked": len(links),
        "n_measurable": len(measurable),
        "n_behind": len(behind),
        "n_on_track": len(measurable) - len(behind),
        "links": links[:40],
        "formula": "100 * (linked targets behind glidepath) / (measurable linked targets)",
        "reason": None,
    }
