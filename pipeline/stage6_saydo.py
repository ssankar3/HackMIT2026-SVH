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
import yaml

ROOT = Path(__file__).resolve().parent.parent
DID_DIR = ROOT / "data" / "did"
OUT_DIR = ROOT / "out"
CONFIG = ROOT / "config" / "weights.yaml"

# Rows whose metric carries this prefix are FORWARD-LOOKING TARGETS, not
# historical actuals. The `year` column holds the target's deadline year and
# `value` holds the committed floor expressed so that HIGHER IS MORE AMBITIOUS
# (a % reduction promised, or a % share to be reached); a stated range like
# "45-50%" is recorded as 45, because the floor is what was actually promised.
# A target that cannot be encoded on that scale -- an intensity ceiling such as
# "methane intensity below 0.20%", where lower is better -- is deliberately
# left out rather than encoded ambiguously.
TARGET_PREFIX = "target_"

DRIFT_DEFAULTS = {
    "restatement_noise_floor_pct": 0.5,
    "magnitude_saturation_pct": 30.0,
    "rate_weight": 0.45,
    "magnitude_weight": 0.40,
    "redefinition_weight": 0.15,
    "target_cut_saturation_pct": 50.0,
    "target_rate_weight": 0.60,
    "target_magnitude_weight": 0.40,
    "actuals_block_weight": 0.65,
    "targets_block_weight": 0.35,
}


def load_drift_cfg() -> dict:
    """Read the `drift:` block from config/weights.yaml. Every constant used by
    this stage lives there so the blend is auditable in one place rather than
    buried in the code."""
    cfg = dict(DRIFT_DEFAULTS)
    try:
        blob = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
        cfg.update({k: v for k, v in (blob.get("drift") or {}).items() if k in DRIFT_DEFAULTS})
    except (OSError, yaml.YAMLError):
        pass
    return cfg


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


def split_actuals_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Historical actuals vs forward-looking targets. Everything downstream of
    this treats them separately: restating last year's number and cutting next
    decade's promise are different acts and must not share a denominator."""
    is_target = df["metric"].astype(str).str.startswith(TARGET_PREFIX)
    return df[~is_target].copy(), df[is_target].copy()


def find_restatements(df: pd.DataFrame, noise_floor_pct: float = 0.5) -> List[dict]:
    """Same (data year, metric) published by two different reports at different
    values. Ordered oldest-report -> newest-report so `old`/`new` read correctly."""
    RESTATEMENT_NOISE_FLOOR_PCT = noise_floor_pct
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


def find_target_drift(targets: pd.DataFrame) -> List[dict]:
    """Goalpost moves on FORWARD commitments, which `metric_restated` cannot
    see: the measurement stays put and the promise moves instead.

      target_value_changed   the same deadline-year target was republished at a
                             different level. `weakened` records the direction;
                             only weakenings feed the risk score, but a
                             strengthening is still emitted as evidence so the
                             record reads both ways.
      commitment_dropped     a target that was still live (deadline later than
                             the report that dropped it) simply stopped being
                             published. The deadline guard matters: a target
                             vanishing AFTER its deadline has passed is a
                             concluded commitment, not an abandoned one.
    """
    events: List[dict] = []
    if targets.empty:
        return events

    targets = targets.copy()
    targets["pub"] = targets["source"].map(publish_year_of)
    targets = targets[targets["pub"].notna()]
    if targets.empty:
        return events
    newest_pub = int(targets["pub"].max())

    for (year, metric), grp in targets.groupby(["year", "metric"]):
        rows = grp.sort_values("pub").to_dict("records")

        for older, newer in zip(rows, rows[1:]):
            if float(older["value"]) == float(newer["value"]):
                continue
            base = abs(float(older["value"]))
            pct = ((float(newer["value"]) - float(older["value"])) / base * 100) if base else None
            weakened = float(newer["value"]) < float(older["value"])
            events.append({
                "year": int(year),
                "type": "target_value_changed",
                "canonical_metric": metric,
                "old": fmt_val(older["value"]),
                "new": fmt_val(newer["value"]),
                "magnitude_pct": round(pct, 2) if pct is not None else None,
                "weakened": bool(weakened),
                "old_source": older["source"],
                "new_source": newer["source"],
                "page": int(newer["page"]) if pd.notna(newer["page"]) else None,
                "description": (
                    f'the {year} target for {metric.removeprefix(TARGET_PREFIX)} was published as '
                    f'{fmt_val(older["value"])}{older["unit"]} in {Path(older["source"]).name}, then '
                    f'{"cut to" if weakened else "raised to"} {fmt_val(newer["value"])}{newer["unit"]} '
                    f'in {Path(newer["source"]).name}'
                    + (f" ({pct:+.1f}%)" if pct is not None else "")
                ),
            })

        # Still-live commitment that stopped being published entirely.
        last = rows[-1]
        last_pub = int(last["pub"])
        if last_pub < newest_pub and int(year) > last_pub:
            newer_sources = targets[targets["pub"] == newest_pub]["source"]
            events.append({
                "year": int(year),
                "type": "commitment_dropped",
                "canonical_metric": metric,
                "old": fmt_val(last["value"]),
                "new": None,
                "magnitude_pct": -100.0,
                "weakened": True,
                "old_source": last["source"],
                "new_source": newer_sources.iloc[0] if len(newer_sources) else None,
                "page": None,
                "description": (
                    f'the {year} target for {metric.removeprefix(TARGET_PREFIX)} '
                    f'({fmt_val(last["value"])}{last["unit"]}) was last published in '
                    f'{Path(last["source"]).name} and no longer appears as of the '
                    f'{newest_pub} report, with its deadline still in the future'
                ),
            })
    return events


def comparable_target_count(targets: pd.DataFrame) -> int:
    """Deadline-year targets that were published by more than one report, or
    published once and then dropped while still live -- the denominator for a
    target-weakening RATE, so a company that simply publishes more targets does
    not look worse for it."""
    if targets.empty:
        return 0
    t = targets.copy()
    t["pub"] = t["source"].map(publish_year_of)
    t = t[t["pub"].notna()]
    if t.empty:
        return 0
    newest_pub = int(t["pub"].max())
    n = 0
    for (year, _metric), grp in t.groupby(["year", "metric"]):
        pubs = grp["pub"].nunique()
        last_pub = int(grp["pub"].max())
        if pubs >= 2 or (last_pub < newest_pub and int(year) > last_pub):
            n += 1
    return n


def comparable_pair_count(df: pd.DataFrame) -> int:
    """How many (year, metric) cells were republished by >1 report at all -- the
    denominator for a restatement RATE. Without this, a company that simply
    publishes more history would look worse."""
    n = 0
    for _, grp in df.groupby(["year", "metric"]):
        if grp["source"].map(publish_year_of).nunique() >= 2:
            n += 1
    return n


def goalpost_drift_score(events: List[dict], n_comparable: int,
                         n_comparable_targets: int = 0,
                         cfg: Optional[dict] = None) -> Tuple[Optional[float], dict]:
    """0-100, blended from two independently-measurable blocks.

    ACTUALS  how often republished figures move, and how far. Frequent tiny
             revisions are housekeeping; one enormous revision is material.
    TARGETS  how often a published forward commitment was cut or abandoned,
             and how deeply. Moving the goal is the more literal reading of
             "goalpost drift" than moving the measurement, which is why the
             target block carries the heavier weight of the two.

    Each block is scored only if it is measurable, and the blend is renormalized
    over whichever blocks exist. A company with no target rows therefore scores
    exactly what it scored before targets were modelled at all -- an absent
    block is never silently treated as a clean zero.
    """
    cfg = cfg or dict(DRIFT_DEFAULTS)
    restatements = [e for e in events if e["type"] == "metric_restated"]
    redefinitions = [e for e in events if e["type"] == "metric_redefined"]
    tgt_changes = [e for e in events if e["type"] == "target_value_changed"]
    tgt_dropped = [e for e in events if e["type"] == "commitment_dropped"]
    tgt_weakened = [e for e in tgt_changes if e.get("weakened")] + tgt_dropped

    basis: dict = {
        "n_comparable_pairs": n_comparable,
        "n_restatements": len(restatements),
        "n_redefinitions": len(redefinitions),
        "n_comparable_targets": n_comparable_targets,
        "n_targets_weakened": len(tgt_weakened),
        "n_targets_dropped": len(tgt_dropped),
        "n_targets_strengthened": len([e for e in tgt_changes if not e.get("weakened")]),
    }

    if n_comparable == 0 and n_comparable_targets == 0:
        basis["reason"] = ("no metric-year was published by more than one report and no "
                           "forward target was republished, so goalpost drift cannot be measured")
        return None, basis

    blocks: List[Tuple[float, float]] = []   # (component, weight)

    if n_comparable > 0:
        rate = len(restatements) / n_comparable
        mags = [abs(e["magnitude_pct"]) for e in restatements if e.get("magnitude_pct") is not None]
        median_mag = float(pd.Series(mags).median()) if mags else 0.0
        actuals_component = (
            cfg["rate_weight"] * (rate * 100)
            + cfg["magnitude_weight"] * min(median_mag / cfg["magnitude_saturation_pct"], 1.0) * 100
            + cfg["redefinition_weight"] * min(len(redefinitions) * 25.0, 100.0)
        )
        blocks.append((actuals_component, cfg["actuals_block_weight"]))
        basis.update({
            "restatement_rate": round(rate, 3),
            "median_abs_restatement_pct": round(median_mag, 2),
            "max_abs_restatement_pct": round(max(mags), 2) if mags else 0.0,
            "actuals_component": round(actuals_component, 1),
        })

    if n_comparable_targets > 0:
        t_rate = len(tgt_weakened) / n_comparable_targets
        cuts = [abs(e["magnitude_pct"]) for e in tgt_weakened if e.get("magnitude_pct") is not None]
        median_cut = float(pd.Series(cuts).median()) if cuts else 0.0
        target_component = (
            cfg["target_rate_weight"] * (t_rate * 100)
            + cfg["target_magnitude_weight"] * min(median_cut / cfg["target_cut_saturation_pct"], 1.0) * 100
        )
        blocks.append((target_component, cfg["targets_block_weight"]))
        basis.update({
            "target_weakening_rate": round(t_rate, 3),
            "median_target_cut_pct": round(median_cut, 2),
            "target_component": round(target_component, 1),
        })

    total_w = sum(w for _, w in blocks)
    score = sum(c * w for c, w in blocks) / total_w
    basis["formula"] = (
        "actuals = 0.45*rate + 0.40*min(median_mag/30,1) + 0.15*min(25*n_redef,100); "
        "targets = 0.60*weakening_rate + 0.40*min(median_cut/50,1); "
        "blended 0.65/0.35 over whichever blocks are measurable"
    )
    return round(min(score, 100.0), 1), basis


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

    cfg = load_drift_cfg()
    actuals, targets = split_actuals_targets(df)

    events = (find_restatements(actuals, cfg["restatement_noise_floor_pct"])
              + find_redefinitions(actuals)
              + find_target_drift(targets))
    events.sort(key=lambda e: (e["year"], e["type"], e.get("canonical_metric") or ""))
    for i, e in enumerate(events):
        e["drift_id"] = f"{company}-drift-{i:03d}"

    n_comparable = comparable_pair_count(actuals)
    n_comparable_targets = comparable_target_count(targets)
    score, basis = goalpost_drift_score(events, n_comparable, n_comparable_targets, cfg)

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
        # Actuals only: a 2030 target is not a measured data point and must not
        # be drawn on the time machine as though it were one.
        "series": latest_series(actuals),
        "targets": latest_series(targets),
        "did_rows": [{
            "year": int(r["year"]), "metric": r["metric"], "value": float(r["value"]),
            "unit": r["unit"], "scope": r.get("scope"), "source": r["source"],
            "page": int(r["page"]) if pd.notna(r["page"]) else None,
        } for _, r in actuals.iterrows()],
        "n_did_rows": len(actuals),
        "did_years": sorted(int(y) for y in actuals["year"].unique()),
        "did_metrics": sorted(actuals["metric"].unique().tolist()),
        "n_target_rows": len(targets),
        "target_metrics": sorted(targets["metric"].unique().tolist()),
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
