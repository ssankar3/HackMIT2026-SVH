"""Stage 3: linguistic analysis of extracted claims. No LLM, fully deterministic.

Every feature here is a counting rule over a lexicon, so each score decomposes
back to the exact words that produced it. That is the point: a quant desk can
audit "why is this claim vague?" down to the token, which an LLM verdict cannot
offer.

Per claim:
  vague / hedging / future-orientation ratios, quantification, baseline, date,
  scope and verification presence, a composite specificity score, Flesch
  reading ease, passive-voice rate, and modal strength.

Per company-year:
  the above averaged, plus two aggregate signals --

  say_more_prove_less   positive-tone words per QUANTIFIED claim. Rising means
                        the company is adding promotional language faster than
                        it is adding evidence.
  commitment_mix        share of claims that are firm vs hedged. A book of
                        commitments drifting from 'will' to 'aim to' is
                        softening without restating a single number.

Usage:
    python3 pipeline/stage3_language.py --company hm
    python3 pipeline/stage3_language.py --company hm --max-pages 40 --limit 2
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_claims_v1 import (  # noqa: E402
    Lex, run as extract_claims,
)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
LEXICONS = ROOT / "config" / "lexicons.yaml"
OUT_DIR = ROOT / "out" / "language"

RE_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
RE_SENT_END = re.compile(r"[.!?]")
# "were reduced", "has been achieved", "is being phased" -- agency-free phrasing
RE_PASSIVE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has been|have been|had been)\s+"
    r"(?:\w+ly\s+)?\w+(?:ed|en)\b", re.I,
)
RE_MODAL_FIRM = re.compile(r"\b(?:will|shall|must)\b", re.I)
RE_MODAL_SOFT = re.compile(r"\b(?:may|might|could|should|would)\b", re.I)
RE_FUTURE_YEAR = re.compile(r"\bby\s+(?:19|20)\d{2}\b", re.I)


def words(t: str) -> List[str]:
    return RE_WORD.findall(t)


def syllables(w: str) -> int:
    """Vowel-group heuristic. Approximate, but stable across documents, which
    is all a comparative readability trend needs."""
    w = w.lower().rstrip("e")
    groups = re.findall(r"[aeiouy]+", w)
    return max(1, len(groups))


def flesch_reading_ease(text: str) -> Optional[float]:
    ws = words(text)
    n_sent = max(1, len(RE_SENT_END.findall(text)))
    if len(ws) < 3:
        return None
    syl = sum(syllables(w) for w in ws)
    return round(206.835 - 1.015 * (len(ws) / n_sent) - 84.6 * (syl / len(ws)), 1)


def ratio(text_low: str, terms: List[str], n_words: int) -> float:
    if not n_words:
        return 0.0
    hits = sum(len(re.findall(rf"\b{re.escape(t)}", text_low)) for t in terms)
    return round(hits / n_words, 4)


class Lexicons:
    def __init__(self, path: Path):
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
        low = lambda k: [w.lower() for w in d.get(k, [])]  # noqa: E731
        self.vague = low("vague")
        self.hedging = low("hedging")
        self.verification = low("verification")
        self.positive = low("positive_tone")
        self.future = ["will", "aim", "target", "goal", "ambition", "plan",
                       "by 2030", "by 2040", "by 2050", "future", "intend", "commit"]


def claim_features(row: dict, lx: Lexicons) -> dict:
    s = row["sentence"]
    low = s.lower()
    ws = words(s)
    n = len(ws)

    vague = ratio(low, lx.vague, n)
    hedge = ratio(low, lx.hedging, n)
    future = ratio(low, lx.future, n) + (0.02 if RE_FUTURE_YEAR.search(s) else 0.0)

    quantified = bool(row.get("quantity"))
    has_baseline = bool(row.get("baseline"))
    has_deadline = bool(row.get("deadline"))
    has_scope = bool(row.get("scope"))
    verified = any(re.search(rf"\b{re.escape(t)}", low) for t in lx.verification)

    # Specificity rewards checkability and penalises softening language.
    # Weights are uniform across the five evidence slots on purpose: no slot is
    # privileged, so the score stays explainable as "how many ways is this
    # claim pinned down".
    present = sum([quantified, has_baseline, has_deadline, has_scope, verified])
    specificity = present / 5.0
    specificity -= min(0.3, (vague + hedge) * 4)
    specificity = round(max(0.0, min(1.0, specificity)), 3)

    return {
        **row,
        "n_words": n,
        "vague_ratio": vague,
        "hedging_ratio": hedge,
        "future_ratio": round(future, 4),
        "quantification_present": quantified,
        "baseline_present": has_baseline,
        "date_present": has_deadline,
        "scope_present": has_scope,
        "verification_present": verified,
        "specificity_score": specificity,
        "readability": flesch_reading_ease(s),
        "passive_voice": bool(RE_PASSIVE.search(s)),
        "modal_firm": bool(RE_MODAL_FIRM.search(s)),
        "modal_soft": bool(RE_MODAL_SOFT.search(s)),
        "positive_tone_words": sum(len(re.findall(rf"\b{re.escape(t)}", low)) for t in lx.positive),
    }


def aggregate(year: int, rows: List[dict]) -> dict:
    n = len(rows)
    mean = lambda k: round(statistics.mean([r[k] for r in rows]), 4) if n else None  # noqa: E731
    share = lambda k: round(sum(1 for r in rows if r[k]) / n, 3) if n else None  # noqa: E731

    n_quant = sum(1 for r in rows if r["quantification_present"])
    tone = sum(r["positive_tone_words"] for r in rows)
    firm = sum(1 for r in rows if r["strength"] == "firm")
    hedged = sum(1 for r in rows if r["strength"] == "hedged")

    reads = [r["readability"] for r in rows if r["readability"] is not None]
    return {
        "year": year,
        "n_claims": n,
        "mean_specificity": mean("specificity_score"),
        "vague_ratio": mean("vague_ratio"),
        "hedging_ratio": mean("hedging_ratio"),
        "future_ratio": mean("future_ratio"),
        "quantified_share": share("quantification_present"),
        "verified_share": share("verification_present"),
        "scope_share": share("scope_present"),
        "passive_share": share("passive_voice"),
        "readability": round(statistics.mean(reads), 1) if reads else None,
        # headline aggregates
        "say_more_prove_less": round(tone / n_quant, 2) if n_quant else None,
        "firm_share": round(firm / n, 3) if n else None,
        "hedged_share": round(hedged / n, 3) if n else None,
        "commitment_ratio": round(firm / hedged, 2) if hedged else None,
    }


def run_company(company: str, max_pages: int, limit: Optional[int]) -> dict:
    pdfs = sorted((RAW / company).glob("*.pdf"))
    if limit:
        pdfs = pdfs[-limit:]
    lx = Lexicons(LEXICONS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    per_year: Dict[int, List[dict]] = {}
    for pdf in pdfs:
        year = int(pdf.name.split("_", 1)[0])
        rows, st, _ = extract_claims(pdf, max_pages, OUT_DIR / "_claims")
        feats = [claim_features({**r, "company": company, "publish_year": year}, lx)
                 for r in rows]
        per_year[year] = feats
        all_rows.extend(feats)
        print(f"  {pdf.name:<36} {len(feats):>4} claims  "
              f"({st.sentences} sentences scanned)", file=sys.stderr)

    series = [aggregate(y, per_year[y]) for y in sorted(per_year)]

    out_csv = OUT_DIR / f"{company}_claim_features.csv"
    if all_rows:
        cols = list(all_rows[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(all_rows)

    result = {"company": company, "series": series, "n_claims": len(all_rows)}
    (OUT_DIR / f"{company}_language.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def report(res: dict) -> None:
    p = lambda *a: print(*a, file=sys.stderr)  # noqa: E731
    s = res["series"]
    p("\n" + "=" * 96)
    p(f"LANGUAGE PROFILE BY REPORT YEAR  --  {res['company']}  ({res['n_claims']} claims)")
    p("=" * 96)
    hdr = (f"{'year':<6}{'claims':>7}{'specif':>8}{'quant%':>8}{'verif%':>8}"
           f"{'vague':>8}{'hedge':>8}{'future':>8}{'passive':>9}{'read':>7}"
           f"{'firm%':>7}{'hedg%':>7}{'say>prove':>11}")
    p(hdr)
    p("-" * 96)
    pc = lambda v: f"{v*100:.0f}" if v is not None else "-"  # noqa: E731
    nv = lambda v, d=3: f"{v:.{d}f}" if v is not None else "-"  # noqa: E731
    for r in s:
        p(f"{r['year']:<6}{r['n_claims']:>7}{nv(r['mean_specificity']):>8}"
          f"{pc(r['quantified_share']):>8}{pc(r['verified_share']):>8}"
          f"{nv(r['vague_ratio'],4):>8}{nv(r['hedging_ratio'],4):>8}"
          f"{nv(r['future_ratio'],4):>8}{pc(r['passive_share']):>9}"
          f"{nv(r['readability'],1):>7}{pc(r['firm_share']):>7}"
          f"{pc(r['hedged_share']):>7}{nv(r['say_more_prove_less'],2):>11}")

    if len(s) >= 2:
        a, b = s[0], s[-1]
        p("\nTREND  " + f"{a['year']} -> {b['year']}")
        def delta(label, k, better_when_up=True, pct=False):
            va, vb = a.get(k), b.get(k)
            if va is None or vb is None:
                p(f"  {label:<26} n/a"); return
            d = vb - va
            direction = "improving" if (d > 0) == better_when_up else "deteriorating"
            if abs(d) < 1e-9:
                direction = "flat"
            fmt = (lambda v: f"{v*100:.0f}%") if pct else (lambda v: f"{v:.3f}")
            p(f"  {label:<26} {fmt(va):>8} -> {fmt(vb):>8}   {d:+.3f}  {direction}")
        delta("mean specificity", "mean_specificity", True)
        delta("quantified share", "quantified_share", True, pct=True)
        delta("verified share", "verified_share", True, pct=True)
        delta("vagueness", "vague_ratio", False)
        delta("hedging", "hedging_ratio", False)
        delta("say-more-prove-less", "say_more_prove_less", False)
        delta("firm share", "firm_share", True, pct=True)


def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--company", required=True)
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None, help="only the N most recent reports")
    args = ap.parse_args(argv)
    res = run_company(args.company, args.max_pages, args.limit)
    report(res)
    print(f"\nwrote {OUT_DIR}/{args.company}_language.json "
          f"and {args.company}_claim_features.csv", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
