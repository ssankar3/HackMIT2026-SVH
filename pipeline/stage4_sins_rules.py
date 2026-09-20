"""Stage 4: TerraChoice Seven Sins, derived by rule. No LLM.

Each sin below fires on signals already computed in Stage 3 (specificity,
verification, quantification) or on the verified DiD CSV. Every tag carries the
exact reason it fired, so a reviewer can overturn it by looking at one field --
which an LLM severity score does not permit.

IMPLEMENTED (4 of 7)
  vagueness         claim is unquantified/unscoped and leans on vague or
                    hedging vocabulary
  no_proof          a measurable assertion with no third-party verification
                    language anywhere in it
  false_labels      invokes certification/standards language without naming the
                    body that certifies it
  hidden_trade_off  touts a metric in a year when the company's OWN verified
                    data shows total emissions rose

NOT IMPLEMENTED (3 of 7) -- and deliberately left empty rather than guessed:
  irrelevance          needs to know whether a claimed practice is already
                       legally mandated in that jurisdiction. No such source
                       is in the corpus.
  lesser_of_two_evils  needs a category-level judgment ("efficient SUV") that
                       no lexicon in this repo can ground.
  fibbing              requires matching a claim's number to the right DiD
                       metric and asserting the company is WRONG. That is the
                       most serious accusation in the set and the metric-
                       matching here is fuzzy; a false positive is defamatory.
                       Left for the LLM normalisation step in Stage 6.

Usage:
    python3 pipeline/stage4_sins_rules.py --company hm
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
LANG_DIR = ROOT / "out" / "language"
DID_DIR = ROOT / "data" / "did"
OUT_DIR = ROOT / "out" / "sins"

VAGUE_SPECIFICITY_MAX = 0.25    # below this a claim pins almost nothing down
NO_PROOF_SPECIFICITY_MIN = 0.20  # above this it asserts enough to need backing

# Certification vocabulary vs the bodies that actually certify. A claim using
# the former without the latter is the "false labels" pattern: the LOOK of
# third-party endorsement without an endorser.
RE_CERT_LANGUAGE = re.compile(
    r"\b(certifi\w*|accredit\w*|standard|verified|approved|compliant|label|seal|"
    r"audited|assured)\b", re.I)
RE_CERT_BODY = re.compile(
    r"\b(SBTi|Science[- ]Based Targets|GOTS|FSC|PEFC|OEKO|RWS|RMS|GRS|RCS|BCI|"
    r"Cradle to Cradle|ISO\s?\d{4,5}|GHG Protocol|CDP|GRI|TCFD|SASB|Bluesign|"
    r"Fairtrade|Rainforest Alliance|B Corp|EPEAT|Energy Star|LEED|BREEAM|"
    r"Textile Exchange|Higg|ZDHC|RE100|Deloitte|EY|KPMG|PwC|Ernst|Bureau Veritas|"
    r"SGS|DNV|TÜV|Intertek)\b", re.I)

RE_MEASURABLE = re.compile(r"\d")

# A hidden trade-off is a claim that ADVERTISES an improvement while a bigger
# number moves the wrong way. Merely stating a figure in a bad year is not the
# sin -- without this gate the rule tagged every quantified sentence in the year
# and told you nothing.
RE_IMPROVEMENT = re.compile(
    r"\b(reduc\w+|cut|lower\w*|decreas\w+|improv\w+|increas\w+ (?:the )?(?:share|use|"
    r"proportion)|saved?|avoided|eliminat\w+|phased? out|switch\w+ to|transition\w+ to|"
    r"achiev\w+|progress|milestone|on track)\b", re.I)


def load_claims(company: str) -> List[dict]:
    p = LANG_DIR / f"{company}_claim_features.csv"
    if not p.exists():
        sys.exit(f"missing {p}; run: make lang COMPANY={company}")
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def total_emissions_trend(company: str) -> Dict[int, float]:
    """Year -> total emissions, as told by the most recent report that carries
    it. Used by hidden_trade_off: a green claim lands differently in a year the
    company's own footprint grew."""
    p = DID_DIR / f"{company}.csv"
    if not p.exists():
        return {}
    rows = []
    with p.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["metric"].startswith("total_emissions")]
    if not rows:
        return {}
    newest = max(rows, key=lambda r: r["source"])["source"]
    out: Dict[int, float] = {}
    for r in rows:
        if r["source"] == newest:
            try:
                out[int(r["year"])] = float(r["value"])
            except ValueError:
                continue
    return out


def _b(v) -> bool:
    return str(v).strip().lower() == "true"


def _f(v, d=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def sins_for(claim: dict, worsening_years: set) -> List[dict]:
    text = claim["sentence"]
    spec = _f(claim.get("specificity_score"))
    vague = _f(claim.get("vague_ratio"))
    hedge = _f(claim.get("hedging_ratio"))
    verified = _b(claim.get("verification_present"))
    quantified = _b(claim.get("quantification_present"))
    scoped = _b(claim.get("scope_present"))
    year = int(claim["publish_year"])
    out: List[dict] = []

    # --- vagueness -------------------------------------------------------
    if spec <= VAGUE_SPECIFICITY_MAX and (vague + hedge) > 0 and not quantified:
        sev = min(1.0, 0.30 + min(vague + hedge, 0.08) * 5.0 + (VAGUE_SPECIFICITY_MAX - spec))
        out.append({
            "sin": "vagueness",
            "rationale": (f"no quantification and specificity {spec:.2f}; relies on vague/hedging "
                          f"vocabulary (ratio {vague + hedge:.3f})"),
            "severity": round(sev, 2),
            "evidence_needed": "a number, a boundary/scope, and a baseline year",
        })

    # --- no proof --------------------------------------------------------
    if not verified and quantified and spec >= NO_PROOF_SPECIFICITY_MIN:
        sev = 0.35 + (0.2 if scoped else 0.0) + (0.15 if _b(claim.get("date_present")) else 0.0)
        out.append({
            "sin": "no_proof",
            "rationale": ("states a measurable outcome but names no assurance, audit, "
                          "standard or third-party verifier"),
            "severity": round(min(sev, 1.0), 2),
            "evidence_needed": "an assurance statement or named third-party verifier for this figure",
        })

    # --- false labels ----------------------------------------------------
    if RE_CERT_LANGUAGE.search(text) and not RE_CERT_BODY.search(text):
        m = RE_CERT_LANGUAGE.search(text)
        out.append({
            "sin": "false_labels",
            "rationale": (f"uses certification-style wording ('{m.group(0)}') without naming "
                          f"the certifying body or standard"),
            "severity": 0.45 if verified else 0.6,
            "evidence_needed": "the name of the certifying organisation and the standard applied",
        })

    # --- hidden trade-off ------------------------------------------------
    if year in worsening_years and quantified and not _b(claim.get("negated")) and RE_IMPROVEMENT.search(text):
        m = RE_IMPROVEMENT.search(text)
        out.append({
            "sin": "hidden_trade_off",
            "rationale": (f"advertises an improvement ('{m.group(0)}') in a report year when the "
                          f"company's own verified data shows TOTAL emissions rose"),
            "severity": 0.5,
            "evidence_needed": "total-footprint context presented alongside the highlighted metric",
        })
    return out


def analyse(company: str) -> dict:
    claims = load_claims(company)
    totals = total_emissions_trend(company)
    worsening = {y for y in totals if (y - 1) in totals and totals[y] > totals[y - 1]}

    tagged, counter = [], Counter()
    for i, c in enumerate(claims):
        s = sins_for(c, worsening)
        if not s:
            continue
        for tag in s:
            counter[tag["sin"]] += 1
        tagged.append({
            "claim_index": i,
            "sentence": c["sentence"],
            "page": c.get("page"),
            "publish_year": int(c["publish_year"]),
            "source_file": c.get("source_file"),
            "specificity_score": _f(c.get("specificity_score")),
            "sins": s,
            "max_severity": round(max(t["severity"] for t in s), 2),
        })

    tagged.sort(key=lambda t: -t["max_severity"])
    sev_all = [t["max_severity"] for t in tagged]
    # 0-100 sub-score: how much of the claim book carries a sin, weighted by severity
    sins_severity = round(
        (sum(sev_all) / len(claims)) * 100, 1) if claims else None

    return {
        "company": company,
        "n_claims": len(claims),
        "n_tagged": len(tagged),
        "counts": dict(counter.most_common()),
        "sins_severity": sins_severity,
        "worsening_years": sorted(worsening),
        "not_implemented": ["irrelevance", "lesser_of_two_evils", "fibbing"],
        "tagged": tagged,
    }


def report(res: dict, show: int) -> None:
    p = lambda *a: print(*a, file=sys.stderr)  # noqa: E731
    p("\n" + "=" * 92)
    p(f"SEVEN SINS (rule-based)  --  {res['company']}")
    p("=" * 92)
    p(f"  claims                {res['n_claims']}")
    p(f"  claims with >=1 sin   {res['n_tagged']}  "
      f"({res['n_tagged']/res['n_claims']*100:.0f}%)" if res["n_claims"] else "")
    p(f"  sins_severity score   {res['sins_severity']}")
    if res["worsening_years"]:
        p(f"  years total emissions rose  {res['worsening_years']}")
    p("")
    for s, n in res["counts"].items():
        p(f"    {s:<20} {n:>4}")
    p(f"  not implemented: {', '.join(res['not_implemented'])}")

    p("\n" + "=" * 92)
    p(f"TOP {show} BY SEVERITY")
    p("=" * 92)
    for t in res["tagged"][:show]:
        p(f"\n[{t['max_severity']}]  {t['publish_year']} p.{t['page']}  "
          f"spec={t['specificity_score']:.2f}")
        p(f"  \"{t['sentence'][:200]}\"")
        for s in t["sins"]:
            p(f"   -> {s['sin']} ({s['severity']}): {s['rationale']}")
            p(f"      needs: {s['evidence_needed']}")


def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--company", required=True)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args(argv)

    res = analyse(args.company)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.company}_sins.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    report(res, args.show)
    print(f"\nwrote {OUT_DIR}/{args.company}_sins.json", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
