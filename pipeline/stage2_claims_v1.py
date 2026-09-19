"""Stage 2 v1: fast, deterministic claim extraction. No LLM, no embeddings.

Layout-aware block extraction -> heading/section tracking -> sentence split ->
keyword+regex candidate gate -> regex/POS slot filling. Runs in seconds.

Every `sentence` written to the CSV is verbatim from the PDF: the only
transformation is whitespace normalization, and a self-check asserts each
sentence is a substring of its page's normalized text before it is emitted.
Slots that are not found are left blank, never guessed.

Usage:
    python3 pipeline/stage2_claims_v1.py --pdf data/raw/hm/2023_sustainability_report.pdf
    python3 pipeline/stage2_claims_v1.py --pdf ... --max-pages 50 --out out/claims_v1/
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import yaml

ROOT = Path(__file__).resolve().parent.parent
LEXICONS = ROOT / "config" / "lexicons.yaml"

CSV_FIELDS = [
    "sentence", "page", "section_path", "claim_type", "strength", "quantity",
    "baseline", "deadline", "scope", "slot_fill_score", "source_file",
]

# A line appearing on more than this share of pages is running header/footer.
REPEAT_LINE_PAGE_FRAC = 0.30
HEADING_SIZE_RATIO = 1.15   # font size vs page median to count as a heading
HEADING_MAX_WORDS = 12      # headings are short
MIN_SENTENCE_WORDS = 5
MAX_SENTENCE_CHARS = 400

YEAR = r"(?:19|20)\d{2}"
RE_YEAR = re.compile(YEAR)
RE_NUMBER = re.compile(r"\d")
RE_PCT = re.compile(r"%|\bper ?cent\b", re.I)
# quantity: a number with a unit or percent attached
RE_QUANTITY = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*"
    r"(%|per ?cent|tCO2e|tCO₂e|tonnes?|metric tons?|MMT|MtCO2e|kg|GWh|MWh|kWh|"
    r"million|billion|litres?|liters?|gallons?|m³|acres?|hectares?)",
    re.I,
)
RE_BASELINE = re.compile(
    r"(?:versus|vs\.?|compared (?:with|to)|against|from(?: a| our)?|relative to|"
    r"on)\s+(?:an?\s+|our\s+|the\s+)?(?:\w+\s+){0,2}?(" + YEAR + r")"
    r"(?:\s+(?:base ?line|base ?year))?", re.I,
)
RE_BASELINE_LABELLED = re.compile(r"(" + YEAR + r")\s+base\s?(?:line|year)", re.I)
RE_DEADLINE = re.compile(
    r"\b(?:by|before|no later than|ahead of|through(?:out)?|until)\s+(?:the\s+end\s+of\s+)?"
    r"(" + YEAR + r")", re.I,
)
RE_WS = re.compile(r"\s+")


RE_HYPHEN_BREAK = re.compile(r"(\w)[-‐­]\s+(\w)")


def norm(s: str) -> str:
    return RE_WS.sub(" ", s).strip()


def join_lines(lines: List[str]) -> str:
    """Join wrapped lines into prose, repairing words split across a line break
    ('pack-\\naging' -> 'packaging'). Applied identically to block text and to
    the page text used for the verbatim check, so the two stay comparable."""
    joined = " ".join(ln.strip() for ln in lines if ln.strip())
    return RE_HYPHEN_BREAK.sub(r"\1\2", joined)


def looks_like_sentence(s: str) -> bool:
    """Cheap completeness guard: real sentences start with a capital or digit
    and close with terminal punctuation. Catches the fragments that multi-column
    layouts produce when a block is clipped mid-thought."""
    return bool(s) and (s[0].isupper() or s[0].isdigit()) and s.rstrip()[-1] in ".!?"


# Navigational prose. These carry environmental keywords AND numbers, so they
# sail through the candidate gate, but "read more on pages 80-81" asserts
# nothing -- the page number just becomes a bogus quantity.
RE_CROSSREF = re.compile(
    r"\b(?:read|find|see|learn|review|more)\b[^.]{0,40}\b(?:on |at |in )?"
    r"(?:pages?|section|chapter|appendix|table|figure|report)\b"
    r"|\bpages?\s+\d+"
    r"|\b(?:www\.|https?://|[\w.-]+\.com/)"
    r"|\bsee\s+(?:the\s+)?\w+\s+(?:report|statement|policy)\b",
    re.I,
)
# Table-of-contents rows use dot leaders to bridge title and page number
# ("Renewable energy . . . . . . 30"). They survive sentence splitting and then
# match across years as if they were recurring claims.
RE_DOT_LEADER = re.compile(r"(?:\.\s*){4,}|\.{4,}|(?:·\s*){4,}|(?:…){2,}")


# Leading footnote enumerator: "6) This includes..." -- the 6 is a marker, not data.
RE_FOOTNOTE_LEAD = re.compile(r"^\s*(?:\(?\d{1,2}[\).\]]|\*+|[†‡¶])\s+")


def is_navigational(s: str) -> bool:
    return bool(RE_CROSSREF.search(s)) or bool(RE_DOT_LEADER.search(s))


def strip_footnote_marker(s: str) -> str:
    return RE_FOOTNOTE_LEAD.sub("", s, count=1)


def is_heading_text(t: str) -> bool:
    """Reject table cells and page numbers that happen to be large or bold.
    A heading is words, not a row of figures: 'TOTAL 236,035 100.0%' and
    '68 69' were both being promoted into section_path."""
    words = t.split()
    alpha = [w for w in words if any(c.isalpha() for c in w)]
    if len(alpha) < 2:
        return False
    digitish = sum(1 for w in words if any(c.isdigit() for c in w))
    if digitish >= max(2, len(words) / 3):
        return False
    return "%" not in t


@dataclass
class Lex:
    environmental: List[str]
    attribute: List[str]
    strength: Dict[str, List[str]]
    scope_markers: Dict[str, str]

    @staticmethod
    def load(path: Path) -> "Lex":
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
        return Lex(
            environmental=[w.lower() for w in d["environmental"]],
            attribute=[w.lower() for w in d["attribute"]],
            strength={k: [w.lower() for w in v] for k, v in d["strength"].items()},
            scope_markers={k.lower(): v for k, v in d["scope_markers"].items()},
        )


@dataclass
class Block:
    text: str
    page: int
    size: float
    bold: bool


@dataclass
class Stats:
    pages_total: int = 0
    pages_scanned: int = 0
    pages_no_text: List[int] = field(default_factory=list)
    pages_skipped_limit: int = 0
    blocks: int = 0
    headings: int = 0
    dropped_repeated: int = 0
    dropped_small_font: int = 0
    dropped_fragment: int = 0
    dropped_navigational: int = 0
    sentences: int = 0
    candidates: int = 0
    claims: int = 0
    verbatim_failures: int = 0


# --------------------------------------------------------------------------
# 1. blocks, headings, boilerplate
# --------------------------------------------------------------------------

def read_blocks(doc, max_pages: int, st: Stats) -> Tuple[List[Block], Dict[int, str]]:
    """Pull span-level text with font size and weight. Returns blocks plus the
    normalized full text per page, which is used for the verbatim self-check."""
    st.pages_total = doc.page_count
    n = min(max_pages, doc.page_count) if max_pages else doc.page_count
    st.pages_skipped_limit = max(0, doc.page_count - n)

    raw: List[Block] = []
    page_text: Dict[int, str] = {}
    for pno in range(n):
        page = doc[pno]
        d = page.get_text("dict")
        st.pages_scanned += 1
        found = False
        for blk in d.get("blocks", []):
            lines: List[str] = []
            sizes: List[float] = []
            bolds: List[bool] = []
            for line in blk.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                lines.append("".join(s["text"] for s in spans))
                sizes.append(max(float(s.get("size", 0)) for s in spans))
                # PyMuPDF flags: bit 4 (value 16) marks bold
                bolds.append(any(int(s.get("flags", 0)) & 16 for s in spans))
            if not lines:
                continue
            # Join the block's lines into running prose BEFORE sentence
            # splitting -- a sentence routinely spans several visual lines, and
            # splitting per line yields fragments like "Our goal is for 100".
            text = norm(join_lines(lines))
            if not text:
                continue
            found = True
            raw.append(Block(text=text, page=pno + 1,
                             size=max(sizes), bold=all(bolds)))
        page_text[pno + 1] = norm(join_lines(page.get_text("text").split("\n")))
        if not found:
            st.pages_no_text.append(pno + 1)
    st.blocks = len(raw)
    return raw, page_text


def drop_boilerplate(blocks: List[Block], st: Stats) -> List[Block]:
    """Remove running headers/footers (same line on many pages) and the
    smallest font tier (legal footers, GRI index rows, figure captions)."""
    pages = {b.page for b in blocks}
    if not pages:
        return blocks
    seen: Dict[str, set] = {}
    for b in blocks:
        seen.setdefault(b.text.lower(), set()).add(b.page)
    threshold = max(2, int(len(pages) * REPEAT_LINE_PAGE_FRAC))
    repeated = {t for t, ps in seen.items() if len(ps) >= threshold}

    sizes = [round(b.size, 1) for b in blocks]
    tiers = sorted(set(sizes))
    smallest = tiers[0] if tiers else 0.0
    # only treat the bottom tier as noise when there is a real range of sizes
    drop_small = len(tiers) >= 3

    kept: List[Block] = []
    for b in blocks:
        if b.text.lower() in repeated:
            st.dropped_repeated += 1
            continue
        if drop_small and round(b.size, 1) == smallest:
            st.dropped_small_font += 1
            continue
        kept.append(b)
    return kept


def assign_sections(blocks: List[Block], st: Stats) -> List[Tuple[Block, str]]:
    """Walk blocks in reading order, maintaining a section path from headings.
    A heading is markedly larger than the page's median body size, or bold and
    short. Two levels are tracked, which is enough for report structure."""
    by_page: Dict[int, List[float]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b.size)
    median = {p: statistics.median(v) for p, v in by_page.items()}

    out: List[Tuple[Block, str]] = []
    h1 = h2 = ""
    for b in blocks:
        med = median.get(b.page, b.size)
        words = len(b.text.split())
        big = b.size >= med * HEADING_SIZE_RATIO
        boldish = b.bold and words <= HEADING_MAX_WORDS
        if (big or boldish) and words <= HEADING_MAX_WORDS \
                and not b.text.endswith(".") and is_heading_text(b.text):
            st.headings += 1
            if big and b.size >= med * 1.45:
                h1, h2 = b.text, ""
            else:
                h2 = b.text
            continue
        out.append((b, " > ".join(x for x in (h1, h2) if x)))
    return out


# --------------------------------------------------------------------------
# 2. sentences
# --------------------------------------------------------------------------

def make_splitter():
    """spaCy sentenciser if available, else a regex fallback that respects the
    abbreviations these reports actually contain. The fallback exists so the
    stage never hard-blocks on an unavailable model."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer", "textcat"])
        nlp.max_length = 2_000_000

        def split(text: str) -> List[str]:
            return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        return split, "spacy:en_core_web_sm"
    except Exception as exc:  # noqa: BLE001
        print(f"  [sentence splitter] spaCy unavailable ({type(exc).__name__}); "
              f"using regex fallback", file=sys.stderr)
        ABBR = r"(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bNo)(?<!\bInc)(?<!\bLtd)(?<!\bapprox)(?<!\bFig)"
        pat = re.compile(ABBR + r"(?<=[.!?])\s+(?=[A-Z0-9])")

        def split(text: str) -> List[str]:
            return [s.strip() for s in pat.split(text) if s.strip()]
        return split, "regex-fallback"


# --------------------------------------------------------------------------
# 3. candidate gate
# --------------------------------------------------------------------------

def has_term(low: str, terms: List[str]) -> bool:
    """Word-boundary containment. Substring matching is wrong here: 'green'
    matches 'greenhouse gas' and 'clean' matches 'cleaning', both of which
    flooded v1 with false attribute claims."""
    return any(re.search(rf"\b{re.escape(t)}", low) for t in terms)


def is_candidate(s: str, lex: Lex) -> bool:
    low = s.lower()
    if has_term(low, lex.attribute):
        return True
    has_num = bool(RE_NUMBER.search(s)) or bool(RE_PCT.search(s)) or bool(RE_YEAR.search(s))
    return has_num and has_term(low, lex.environmental)


# --------------------------------------------------------------------------
# 4. slots
# --------------------------------------------------------------------------

def find_quantity(s: str) -> str:
    m = RE_QUANTITY.search(s)
    if m:
        return norm(m.group(0))
    m2 = re.search(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])", s)
    return m2.group(0) if m2 else ""


def find_baseline(s: str) -> str:
    m = RE_BASELINE_LABELLED.search(s) or RE_BASELINE.search(s)
    return m.group(1) if m else ""


def find_deadline(s: str) -> str:
    m = RE_DEADLINE.search(s)
    return m.group(1) if m else ""


def find_scope(s: str, lex: Lex) -> str:
    low = s.lower()
    hits = [label for key, label in lex.scope_markers.items() if key in low]
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return "; ".join(out)


def find_strength(s: str, lex: Lex) -> str:
    """firm > hedged > achievement. Firm wins a tie because 'we will aim to'
    still carries a commitment verb the company can be held to."""
    low = f" {s.lower()} "
    for tier in ("firm", "hedged", "achievement"):
        for w in lex.strength[tier]:
            if re.search(rf"\b{re.escape(w)}\b", low):
                return tier
    return ""


def classify(s: str, lex: Lex, deadline: str, strength: str, quantity: str) -> str:
    low = s.lower()
    if deadline or strength in ("firm", "hedged"):
        if deadline or re.search(r"\b(target|goal|ambition|commit|pledge)\b", low):
            return "target"
    if strength == "achievement" and quantity:
        return "achievement"
    if any(a in low for a in lex.attribute):
        return "attribute"
    return "other"


def slot_fill(claim_type: str, slots: Dict[str, str]) -> float:
    """Fraction of slots filled, over the slots that APPLY to this claim type.
    An achievement has no deadline to state, so scoring it against one would
    understate its specificity."""
    applicable = {
        "target": ["quantity", "baseline", "deadline", "scope"],
        "achievement": ["quantity", "baseline", "scope"],
        "attribute": ["quantity", "scope"],
        "other": ["quantity", "scope"],
    }[claim_type]
    filled = sum(1 for k in applicable if slots.get(k))
    return round(filled / len(applicable), 2)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(pdf: Path, max_pages: int, out_dir: Path) -> Tuple[List[dict], Stats, str]:
    lex = Lex.load(LEXICONS)
    st = Stats()
    split, splitter_name = make_splitter()

    doc = fitz.open(pdf)
    try:
        blocks, page_text = read_blocks(doc, max_pages, st)
    finally:
        doc.close()

    blocks = drop_boilerplate(blocks, st)
    with_sections = assign_sections(blocks, st)

    rows: List[dict] = []
    for blk, section in with_sections:
        for sent in split(blk.text):
            sent = norm(sent)
            if len(sent.split()) < MIN_SENTENCE_WORDS or len(sent) > MAX_SENTENCE_CHARS:
                continue
            if not looks_like_sentence(sent):
                st.dropped_fragment += 1
                continue
            st.sentences += 1
            if is_navigational(sent):
                st.dropped_navigational += 1
                continue
            if not is_candidate(sent, lex):
                continue
            st.candidates += 1

            # verbatim guard: the sentence must occur in its page's own text
            if sent not in page_text.get(blk.page, ""):
                st.verbatim_failures += 1
                continue

            body = strip_footnote_marker(sent)
            quantity = find_quantity(body)
            baseline = find_baseline(body)
            deadline = find_deadline(body)
            scope = find_scope(body, lex)
            strength = find_strength(body, lex)
            ctype = classify(body, lex, deadline, strength, quantity)
            slots = {"quantity": quantity, "baseline": baseline,
                     "deadline": deadline, "scope": scope}
            rows.append({
                "sentence": sent, "page": blk.page, "section_path": section,
                "claim_type": ctype, "strength": strength, "quantity": quantity,
                "baseline": baseline, "deadline": deadline, "scope": scope,
                "slot_fill_score": slot_fill(ctype, slots),
                "source_file": str(pdf.relative_to(ROOT)) if pdf.is_relative_to(ROOT) else str(pdf),
            })
    st.claims = len(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{pdf.stem}__{pdf.parent.name}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows, st, str(out_csv)


def report(rows: List[dict], st: Stats, out_csv: str, elapsed: float, seed: int) -> None:
    p = lambda *a: print(*a, file=sys.stderr)  # noqa: E731
    p("\n" + "=" * 72)
    p("PAGES")
    p(f"  in document              {st.pages_total}")
    p(f"  scanned                  {st.pages_scanned}")
    p(f"  skipped (--max-pages)    {st.pages_skipped_limit}")
    p(f"  no extractable text      {len(st.pages_no_text)}"
      + (f"  -> {st.pages_no_text[:12]}" if st.pages_no_text else ""))
    p("\nFUNNEL")
    p(f"  text blocks              {st.blocks}")
    p(f"    dropped, repeated line {st.dropped_repeated}  (header/footer, >30% of pages)")
    p(f"    dropped, smallest font {st.dropped_small_font}")
    p(f"    headings consumed      {st.headings}")
    p(f"  dropped, not a sentence  {st.dropped_fragment}  (clipped mid-thought by column layout)")
    p(f"  dropped, navigational    {st.dropped_navigational}  (cross-refs: 'see pages 80-81', URLs)")
    p(f"  sentences                {st.sentences}")
    pct = lambda a, b: f"{(a / b * 100):5.1f}%" if b else "    -"  # noqa: E731
    p(f"  candidates               {st.candidates}  ({pct(st.candidates, st.sentences)} of sentences)")
    p(f"  claims                   {st.claims}  ({pct(st.claims, st.candidates)} of candidates)")
    if st.verbatim_failures:
        p(f"  dropped, not verbatim    {st.verbatim_failures}")

    if rows:
        p("\nBREAKDOWN")
        for k in ("claim_type", "strength"):
            c = Counter(r[k] or "(none)" for r in rows)
            p(f"  {k:<12} " + "  ".join(f"{v}={n}" for v, n in c.most_common()))
        fills = [r["slot_fill_score"] for r in rows]
        p(f"  slot_fill    mean={statistics.mean(fills):.2f}  median={statistics.median(fills):.2f}")
        for slot in ("quantity", "baseline", "deadline", "scope"):
            n = sum(1 for r in rows if r[slot])
            p(f"    {slot:<9} filled on {n:>4}  ({pct(n, len(rows))})")

        rnd = random.Random(seed)
        sample = rnd.sample(rows, min(20, len(rows)))
        p("\n" + "=" * 72)
        p(f"20 RANDOM CLAIMS FOR MANUAL REVIEW  (seed={seed})")
        p("=" * 72)
        for i, r in enumerate(sample, 1):
            p(f"\n[{i:>2}] p.{r['page']}  {r['claim_type']}/{r['strength'] or '-'}  "
              f"fill={r['slot_fill_score']}")
            if r["section_path"]:
                p(f"     § {r['section_path'][:88]}")
            p(f"     \"{r['sentence'][:300]}\"")
            slots = [f"{k}={r[k]}" for k in ("quantity", "baseline", "deadline", "scope") if r[k]]
            if slots:
                p(f"     {' | '.join(slots)}")

    p("\n" + "=" * 72)
    p(f"wrote {st.claims} rows -> {out_csv}")
    p(f"elapsed {elapsed:.1f}s")


def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--max-pages", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "claims_v1")
    ap.add_argument("--seed", type=int, default=0, help="seed for the 20-claim review sample")
    args = ap.parse_args(argv)

    t0 = time.time()
    rows, st, out_csv = run(args.pdf, args.max_pages, args.out)
    report(rows, st, out_csv, time.time() - t0, args.seed)


if __name__ == "__main__":
    main(sys.argv[1:])
