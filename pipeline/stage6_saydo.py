"""Stage 6: say-do gap and goalpost drift.

This is the quant core, and the part of the pipeline that needs no LLM: it runs
entirely off the hand-verified `/data/did/{company}.csv` files. Two signals come
out of the data alone:

  metric_restated   the company republished a PRIOR year's actual at a DIFFERENT
                    value in a later report. The measurement moved after the fact.
  metric_redefined  a metric disappears between reports and a differently-named
                    variant appears (e.g. intensity per-GMS -> per-revenue). The
                    denominator moved, so the series is no longer comparable.

Both are computed by comparing reports against each other, with no per-company
configuration anywhere in this file. A company that never restates scores 0.

The target-glidepath half of say-do (claim target vs actual trajectory) needs
Stage 2 claims, so when claims are absent `say_do_gap` is returned as None with
an abstain reason rather than being defaulted to a number.

Usage:
    python3 pipeline/stage6_saydo.py [--limit N] [company ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DID_DIR = ROOT / "data" / "did"
OUT_DIR = ROOT / "out"

# A restatement below this magnitude is treated as a rounding/label artefact
# rather than a substantive move. Applied uniformly to every company.
RESTATEMENT_NOISE_FLOOR_PCT = 0.5


def fmt_val(v: float) -> str:
    """Plain decimal, never scientific notation -- `%g` turns 9_115_000 into
    '9.115e+06', which is unreadable in an evidence table."""
    f = float(v)
    return str(int(f)) if f == int(f) else f"{f:,.4g}".replace(",", "")


def load_did(company: str) -> pd.DataFrame:
    path = DID_DIR / f"{company}.csv"
    if not path.exists():
        raise FileNotFoundError(f"no verified DiD file at {path}")
    df = pd.read_csv(path)
    df["company"] = company
    return df


def publish_year_of(source: str) -> Optional[int]:
    """`<slug>/2025_sustainability_report.pdf` -> 2025. This is the REPORT's
    year, which is not the data year: an issuer on a fiscal calendar publishes
    a report labelled one year that mostly covers the prior one."""
    stem = Path(source).name
    head = stem.split("_", 1)[0]
    return int(head) if head.isdigit() and len(head) == 4 else None


def find_restatements(df: pd.DataFrame) -> List[dict]:
    """Same (data year, metric) published by two different reports at different
    values. Ordered oldest-report -> newest-report so `old`/`new` read correctly."""
    events: List[dict] = []
    for (year, metric), grp in df.groupby(["year", "metric"]):
        grp = grp.copy()
        grp["pub"] = grp["source"].map(publish_year_of)
        grp = grp.sort_values("pub")
        if grp["pub"].nunique() < 2:
            continue
        rows = grp.to_dict("records")
        for older, newer in zip(rows, rows[1:]):
            if older["value"] == newer["value"]:
                continue
            base = abs(float(older["value"]))
            pct = ((float(newer["value"]) - float(older["value"])) / base * 100) if base else None
            if pct is not None and abs(pct) < RESTATEMENT_NOISE_FLOOR_PCT:
                continue
            events.append({
                "year": int(year),
                "type": "metric_restated",
                "canonical_metric": metric,
                "old": fmt_val(older["value"]),
                "new": fmt_val(newer["value"]),
                "magnitude_pct": round(pct, 2) if pct is not None else None,
                "old_source": older["source"],
                "new_source": newer["source"],
                "page": int(newer["page"]) if pd.notna(newer["page"]) else None,
                "description": (
                    f'{metric} for {year} was published as {fmt_val(older["value"])} '
                    f'{older["unit"]} in {Path(older["source"]).name}, then restated to '
                    f'{fmt_val(newer["value"])} {newer["unit"]} in {Path(newer["source"]).name}'
                    + (f" ({pct:+.1f}%)" if pct is not None else "")
                ),
            })
    return events


def find_redefinitions(df: pd.DataFrame) -> List[dict]:
    """A metric present in an older report vanishes while a sibling name (same
    prefix up to the last segment) appears in the newer one."""
    by_pub: Dict[int, set] = defaultdict(set)
    for _, r in df.iterrows():
        pub = publish_year_of(r["source"])
        if pub:
            by_pub[pub].add(r["metric"])

    events: List[dict] = []
    pubs = sorted(by_pub)
    for older_pub, newer_pub in zip(pubs, pubs[1:]):
        dropped = by_pub[older_pub] - by_pub[newer_pub]
        added = by_pub[newer_pub] - by_pub[older_pub]
        for d in sorted(dropped):
            prefix = d.rsplit("_", 1)[0]
            siblings = [a for a in sorted(added) if a.rsplit("_", 1)[0] == prefix and a != d]
            for s in siblings:
                events.append({
                    "year": newer_pub,
                    "type": "metric_redefined",
                    "canonical_metric": s,
                    "old": d,
                    "new": s,
                    "magnitude_pct": None,
                    "old_source": next(
                        (r["source"] for _, r in df.iterrows()
                         if r["metric"] == d and publish_year_of(r["source"]) == older_pub), None),
                    "new_source": next(
                        (r["source"] for _, r in df.iterrows()
                         if r["metric"] == s and publish_year_of(r["source"]) == newer_pub), None),
                    "page": None,
                    "description": (
                        f"metric '{d}' reported through the {older_pub} report was replaced by "
                        f"'{s}' in the {newer_pub} report; the two are not directly comparable"
                    ),
                })
    return events


def comparable_pair_count(df: pd.DataFrame) -> int:
    """How many (year, metric) cells were republished by >1 report at all -- the
    denominator for a restatement RATE. Without this, a company that simply
    publishes more history would look worse."""
    n = 0
    for _, grp in df.groupby(["year", "metric"]):
        if grp["source"].map(publish_year_of).nunique() >= 2:
            n += 1
    return n


def goalpost_drift_score(events: List[dict], n_comparable: int) -> Tuple[Optional[float], dict]:
    """0-100. Blends how OFTEN republished figures move with how FAR they move.

    Both halves are needed: frequent tiny revisions are housekeeping, while one
    enormous revision is material. Returns None when the company never published
    a comparable figure twice -- that is unmeasurable, not clean.
    """
    restatements = [e for e in events if e["type"] == "metric_restated"]
    redefinitions = [e for e in events if e["type"] == "metric_redefined"]

    if n_comparable == 0:
        return None, {
            "reason": "no metric-year was published by more than one report, "
                      "so restatement cannot be measured",
            "n_comparable_pairs": 0,
        }

    rate = len(restatements) / n_comparable
    mags = [abs(e["magnitude_pct"]) for e in restatements if e.get("magnitude_pct") is not None]
    median_mag = float(pd.Series(mags).median()) if mags else 0.0
    max_mag = max(mags) if mags else 0.0

    # rate maps 0..1 -> 0..100; magnitude saturates at 30% (a 30% restatement is
    # already extreme for a reported emissions figure)
    rate_component = rate * 100
    mag_component = min(median_mag / 30.0, 1.0) * 100
    redef_component = min(len(redefinitions) * 25.0, 100.0)

    score = 0.45 * rate_component + 0.40 * mag_component + 0.15 * redef_component
    return round(min(score, 100.0), 1), {
        "n_comparable_pairs": n_comparable,
        "n_restatements": len(restatements),
        "restatement_rate": round(rate, 3),
        "median_abs_restatement_pct": round(median_mag, 2),
        "max_abs_restatement_pct": round(max_mag, 2),
        "n_redefinitions": len(redefinitions),
        "formula": "0.45*rate + 0.40*min(median_mag/30,1) + 0.15*min(25*n_redef,100)",
    }


def latest_series(df: pd.DataFrame) -> Dict[str, List[dict]]:
    """Per metric, the series as told by the MOST RECENT report that carries it.
    This is the 'current official story' the time machine plots."""
    out: Dict[str, List[dict]] = {}
    for metric, grp in df.groupby("metric"):
        grp = grp.copy()
        grp["pub"] = grp["source"].map(publish_year_of)
        newest = grp["pub"].max()
        sub = grp[grp["pub"] == newest].sort_values("year")
        out[metric] = [{
            "year": int(r["year"]), "value": float(r["value"]), "unit": r["unit"],
            "scope": r.get("scope"), "source": r["source"],
            "page": int(r["page"]) if pd.notna(r["page"]) else None,
        } for _, r in sub.iterrows()]
    return out


def analyse(company: str, limit: Optional[int] = None) -> dict:
    df = load_did(company)
    if limit:
        df = df.head(limit)

    events = find_restatements(df) + find_redefinitions(df)
    events.sort(key=lambda e: (e["year"], e["type"], e.get("canonical_metric") or ""))
    for i, e in enumerate(events):
        e["drift_id"] = f"{company}-drift-{i:03d}"

    n_comparable = comparable_pair_count(df)
    score, basis = goalpost_drift_score(events, n_comparable)

    return {
        "company": company,
        "drift_events": events,
        "goalpost_drift": score,
        "goalpost_drift_basis": basis,
        "say_do_gap": None,
        "say_do_gap_basis": {
            "reason": "target-vs-actual glidepath requires Stage 2 claims "
                      "(targets with baseline and target year); no claims available",
        },
        "series": latest_series(df),
        "did_rows": [{
            "year": int(r["year"]), "metric": r["metric"], "value": float(r["value"]),
            "unit": r["unit"], "scope": r.get("scope"), "source": r["source"],
            "page": int(r["page"]) if pd.notna(r["page"]) else None,
        } for _, r in df.iterrows()],
        "n_did_rows": len(df),
        "did_years": sorted(int(y) for y in df["year"].unique()),
        "did_metrics": sorted(df["metric"].unique().tolist()),
        "n_reports": int(df["source"].nunique()),
    }


def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("companies", nargs="*")
    ap.add_argument("--limit", type=int, default=None, help="only use the first N CSV rows")
    args = ap.parse_args(argv)

    companies = args.companies or sorted(p.stem for p in DID_DIR.glob("*.csv") if "_" not in p.stem)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for c in companies:
        res = analyse(c, args.limit)
        path = OUT_DIR / f"{c}.stage6.json"
        path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        b = res["goalpost_drift_basis"]
        print(f"\n=== {c} ===", file=sys.stderr)
        print(f"  did rows          {res['n_did_rows']} across {res['n_reports']} reports", file=sys.stderr)
        print(f"  comparable cells  {b.get('n_comparable_pairs')}", file=sys.stderr)
        print(f"  restatements      {b.get('n_restatements')} "
              f"(rate {b.get('restatement_rate')}, median {b.get('median_abs_restatement_pct')}%, "
              f"max {b.get('max_abs_restatement_pct')}%)", file=sys.stderr)
        print(f"  redefinitions     {b.get('n_redefinitions')}", file=sys.stderr)
        print(f"  goalpost_drift    {res['goalpost_drift']}", file=sys.stderr)
        print(f"  -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
