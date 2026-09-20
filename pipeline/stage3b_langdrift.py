"""Stage 3b: cross-year LANGUAGE drift. No LLM, no embeddings.

Stage 6 catches a company restating a NUMBER. This catches it softening a
SENTENCE -- which happens earlier, leaves no trace in any data table, and is
invisible to anyone reading a single report.

Claims are matched between consecutive report years with TF-IDF cosine over
character n-grams, then matched pairs are diffed for:

  commitment_softened   firm -> hedged ('will reduce' -> 'aims to reduce')
  quantity_dropped      a number present last year is gone this year
  deadline_dropped      a target year present last year is gone
  deadline_pushed       the target year moved later
  scope_narrowed        a scope qualifier was removed
  specificity_fell      composite specificity dropped materially
  boilerplate_recycled  near-identical wording republished verbatim

TF-IDF is used deliberately over embeddings: matches stay explainable (you can
read off the shared terms), it needs no model download, and it is deterministic,
so the same corpus always yields the same evidence.

Usage:
    python3 pipeline/stage3b_langdrift.py --company hm
    python3 pipeline/stage3b_langdrift.py --company hm --threshold 0.55
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim_link import match_family  # noqa: E402

LANG_DIR = ROOT / "out" / "language"
OUT_DIR = ROOT / "out" / "langdrift"

MATCH_THRESHOLD = 0.55     # below this, two sentences are not the same claim
BOILERPLATE_THRESHOLD = 0.93
SPECIFICITY_DROP = 0.10
# Commitment scale ONLY. "achievement" is deliberately absent: it is a
# different speech act (a thing done, not a thing promised), so treating
# firm -> achievement as a downgrade is a category error -- that pattern is
# a commitment being FULFILLED. Comparisons are skipped unless both sides sit
# on this scale.
STRENGTH_RANK = {"firm": 2, "hedged": 1, "": 0}


def _truthy(v) -> bool:
    return str(v).strip().lower() == "true"


def load_claims(company: str) -> Dict[int, List[dict]]:
    path = LANG_DIR / f"{company}_claim_features.csv"
    if not path.exists():
        sys.exit(f"missing {path}; run stage3_language.py --company {company} first")
    by_year: Dict[int, List[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["publish_year"] = int(r["publish_year"])
            r["specificity_score"] = float(r["specificity_score"] or 0)
            by_year[r["publish_year"]].append(r)
    return dict(by_year)


SLOT_WEIGHTS = {"metric_family": 0.40, "scope": 0.20, "deadline": 0.20, "claim_type": 0.20}
SLOT_MATCH_THRESHOLD = 0.60   # a pair qualifies via slots alone at/above this
DEADLINE_CLOSE_YEARS = 1


def _row_slots(row: dict) -> dict:
    """Precompute the structured slots used for matching, once per row, so
    the O(n*m) pairing loop doesn't re-run match_family()'s regex scan
    per (i, j) pair."""
    return {
        "family": match_family(row.get("sentence") or ""),
        "scope": (row.get("scope") or "").strip(),
        "deadline": deadline_of(row),
        "claim_type": row.get("claim_type") or "",
    }


def slot_overlap(sa: dict, sb: dict) -> Tuple[float, List[str]]:
    """Structured similarity from already-extracted Stage-2 slots, independent
    of wording. Returns (score in [0,1], list of slot names that agreed) so a
    match is explainable by WHICH slots agreed, not just a similarity float.
    No single slot clears SLOT_MATCH_THRESHOLD alone (max weight 0.40), so a
    slots-only match always needs at least two independently-agreeing slots --
    that's what stops two unrelated claim_type='target' claims from matching
    on that field by itself."""
    score, agreed = 0.0, []
    if sa["family"] and sb["family"] and sa["family"][0] == sb["family"][0]:
        score += SLOT_WEIGHTS["metric_family"]
        agreed.append(f"metric_family={sa['family'][0]}")
    if sa["scope"] and sa["scope"] == sb["scope"]:
        score += SLOT_WEIGHTS["scope"]
        agreed.append(f"scope={sa['scope']}")
    da, db = sa["deadline"], sb["deadline"]
    if da and db and da.isdigit() and db.isdigit():
        if da == db:
            score += SLOT_WEIGHTS["deadline"]
            agreed.append(f"deadline={da}")
        elif abs(int(da) - int(db)) <= DEADLINE_CLOSE_YEARS:
            score += SLOT_WEIGHTS["deadline"] * 0.5
            agreed.append(f"deadline~{da}/{db}")
    if sa["claim_type"] and sa["claim_type"] == sb["claim_type"]:
        score += SLOT_WEIGHTS["claim_type"]
        agreed.append(f"claim_type={sa['claim_type']}")
    return round(score, 3), agreed


def match_years(a: List[dict], b: List[dict], threshold: float
                ) -> List[Tuple[dict, dict, float, float, List[str]]]:
    """Greedy best-first 1-to-1 matching between two years' claims.

    Character n-grams rather than words: report language is heavily templated,
    and char n-grams stay robust to the small morphological edits ('reduce' ->
    'reducing') that are exactly what softening looks like. This alone is
    blind to paraphrase ('cut absolute emissions 50% by 2030' vs 'reduce our
    carbon footprint by half within the decade'), so a pair also qualifies if
    its STRUCTURED slots (metric family / scope / deadline / claim type)
    agree strongly enough, even when the wording similarity is low."""
    if not a or not b:
        return []
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), min_df=1)
    X = vec.fit_transform([r["sentence"] for r in a] + [r["sentence"] for r in b])
    sim = cosine_similarity(X[: len(a)], X[len(a):])

    slots_a = [_row_slots(r) for r in a]
    slots_b = [_row_slots(r) for r in b]

    pairs: List[Tuple[int, int, float, float, List[str], float]] = []
    for i in range(len(a)):
        for j in range(len(b)):
            tfidf_sim = float(sim[i, j])
            slot_score, agreed = slot_overlap(slots_a[i], slots_b[j])
            qualifies_tfidf = tfidf_sim >= threshold
            qualifies_slots = slot_score >= SLOT_MATCH_THRESHOLD
            if not (qualifies_tfidf or qualifies_slots):
                continue
            # Rank by whichever signal cleared its own bar; if both cleared,
            # the stronger of the two wins the rank so a confident textual
            # match is never displaced by a weaker corroborating slot match,
            # and vice versa.
            rank = max(tfidf_sim if qualifies_tfidf else 0.0,
                       slot_score if qualifies_slots else 0.0)
            pairs.append((i, j, tfidf_sim, slot_score, agreed, rank))

    pairs.sort(key=lambda t: -t[5])
    used_a: set = set()
    used_b: set = set()
    out = []
    for i, j, tfidf_sim, slot_score, agreed, _rank in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((a[i], b[j], round(tfidf_sim, 3), round(slot_score, 3), agreed))
    return out


RE_LEADING_YEAR = re.compile(r"^\s*((?:19|20)\d{2})\s*[:\u2013-]")


def deadline_of(row: dict) -> str:
    """Stage 2's deadline regex wants 'by <year>'. Reports also head a target
    with the year itself ('2030: Reduce absolute scope 1, 2 and 3...'), which
    otherwise reads as the deadline vanishing."""
    d = row.get("deadline", "")
    if d:
        return d
    m = RE_LEADING_YEAR.match(row.get("sentence", ""))
    return m.group(1) if m else ""


def diff_pair(old: dict, new: dict, sim: float) -> List[dict]:
    """What got weaker between two versions of the same claim."""
    ev: List[dict] = []
    add = lambda t, d: ev.append({"type": t, "detail": d})  # noqa: E731

    if sim >= BOILERPLATE_THRESHOLD:
        add("boilerplate_recycled",
            "near-identical wording republished; no change in commitment or evidence")

    o_s, n_s = old.get("strength", ""), new.get("strength", "")
    o_neg, n_neg = _truthy(old.get("negated")), _truthy(new.get("negated"))

    if not o_neg and n_neg and o_s in ("firm", "hedged") and n_s in ("firm", "hedged"):
        # The tier string is unchanged (e.g. firm -> firm), so STRENGTH_RANK
        # alone sees no move -- but the claim now says the OPPOSITE of what it
        # said before ("we will X" -> "we will NOT X"). A much stronger say-do
        # signal than ordinary softening.
        add("commitment_negated",
            f"commitment now negated: previously '{o_s}', now '{n_s}' but negated")
    elif o_neg or n_neg:
        # Either side is negated but this isn't the real->negated transition
        # above (e.g. both negated, or negated->real). Comparing tiers here
        # would be a category error, so no strength-based event fires.
        pass
    elif o_s in STRENGTH_RANK and n_s in STRENGTH_RANK and o_s:
        if STRENGTH_RANK[n_s] < STRENGTH_RANK[o_s]:
            add("commitment_softened", f"commitment strength '{o_s}' -> '{n_s or 'none'}'")
    elif o_s == "achievement" and n_s == "hedged":
        # a stated accomplishment reverting to an aspiration is a real reversal
        add("achievement_reverted_to_aspiration",
            "previously reported as achieved, now framed as an aim")

    if old.get("quantity") and not new.get("quantity"):
        add("quantity_dropped", f"quantity '{old['quantity']}' no longer stated")

    od, nd = deadline_of(old), deadline_of(new)
    if od and not nd:
        add("deadline_dropped", f"target year {od} no longer stated")
    elif od and nd and nd.isdigit() and od.isdigit() and int(nd) > int(od):
        add("deadline_pushed", f"target year moved {od} -> {nd}")

    if old.get("scope") and not new.get("scope"):
        add("scope_narrowed", f"scope qualifier '{old['scope']}' removed")

    drop = old["specificity_score"] - new["specificity_score"]
    if drop >= SPECIFICITY_DROP:
        add("specificity_fell",
            f"specificity {old['specificity_score']:.2f} -> {new['specificity_score']:.2f} ({-drop:+.2f})")
    return ev


def analyse(company: str, threshold: float) -> dict:
    by_year = load_claims(company)
    years = sorted(by_year)
    events: List[dict] = []

    for ya, yb in zip(years, years[1:]):
        for old, new, tfidf_sim, slot_score, agreed in match_years(by_year[ya], by_year[yb], threshold):
            matched_via = (
                "both" if tfidf_sim >= threshold and slot_score >= SLOT_MATCH_THRESHOLD
                else "text" if tfidf_sim >= threshold
                else "slots"
            )
            # diff_pair's BOILERPLATE_THRESHOLD check is about near-identical
            # WORDING, a text concept -- it must keep using the TF-IDF score,
            # never the slot score.
            for e in diff_pair(old, new, tfidf_sim):
                events.append({
                    **e,
                    "from_year": ya, "to_year": yb,
                    "similarity": tfidf_sim,  # kept for backward compatibility
                    "tfidf_similarity": tfidf_sim,
                    "slot_similarity": slot_score,
                    "matched_via": matched_via,
                    "matched_slots": agreed,
                    "old_text": old["sentence"], "new_text": new["sentence"],
                    "old_page": old.get("page"), "new_page": new.get("page"),
                    "old_source": old.get("source_file"), "new_source": new.get("source_file"),
                    "old_negated": _truthy(old.get("negated")), "new_negated": _truthy(new.get("negated")),
                    "old_conditional": _truthy(old.get("conditional")), "new_conditional": _truthy(new.get("conditional")),
                })

    for i, e in enumerate(events):
        e["event_id"] = f"{company}-lang-{i:03d}"

    counts = dict(sorted(
        {t: sum(1 for e in events if e["type"] == t) for t in {e["type"] for e in events}}.items(),
        key=lambda kv: -kv[1]))
    return {"company": company, "years": years, "n_events": len(events),
            "counts": counts, "events": events}


def report(res: dict, show: int) -> None:
    p = lambda *a: print(*a, file=sys.stderr)  # noqa: E731
    p("\n" + "=" * 94)
    p(f"CROSS-YEAR LANGUAGE DRIFT  --  {res['company']}   "
      f"{res['years'][0]}-{res['years'][-1]}   {res['n_events']} events")
    p("=" * 94)
    if not res["counts"]:
        p("  no matched claims crossed the similarity threshold")
        return
    for t, n in res["counts"].items():
        p(f"  {t:<24} {n:>4}")

    # Lead with the events that carry real signal, not recycled boilerplate.
    priority = ["commitment_softened", "achievement_reverted_to_aspiration",
                "commitment_negated",
                "deadline_dropped", "deadline_pushed",
                "quantity_dropped", "scope_narrowed", "specificity_fell",
                "boilerplate_recycled"]
    ranked = sorted(res["events"], key=lambda e: priority.index(e["type"]))
    p("\n" + "=" * 94)
    p(f"TOP {show} DRIFT EVENTS")
    p("=" * 94)
    for e in ranked[:show]:
        p(f"\n[{e['type']}]  {e['from_year']} -> {e['to_year']}  (sim {e['similarity']})")
        p(f"  {e['detail']}")
        p(f"  OLD p.{e['old_page']}: \"{e['old_text'][:210]}\"")
        p(f"  NEW p.{e['new_page']}: \"{e['new_text'][:210]}\"")


def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--company", required=True)
    ap.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args(argv)

    res = analyse(args.company, args.threshold)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.company}_langdrift.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    report(res, args.show)
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
