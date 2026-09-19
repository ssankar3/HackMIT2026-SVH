#!/usr/bin/env python3
"""Draft difference-in-differences (DiD) source tables from raw sustainability-report PDFs.

For each company under data/raw/<company>/*.pdf, scans every page for the target
metrics and writes data/did/<company>_DRAFT_v2.csv with columns:

    year,metric,value,unit,scope,boundary_note,source,page,report_publish_year,
    quote,header_evidence,consistency_check,confidence

Every row is backed by a literal quote and page number pulled straight from the
PDF text layer, and a `header_evidence` field showing exactly what tied the row
to its year (a table's year-header row, or the inline "in <year>" phrase).
Nothing is estimated, interpolated, or backfilled: if a metric's value can't be
tied to a year with real evidence, the candidate is written to
data/did/<company>_REJECTED.txt with a reason instead of being guessed into the
CSV. The same year's figure appearing in multiple reports (a restatement) is
never deduplicated -- each report's value is kept as its own row, tagged with
its own `source`, so a later report's restatement sits right alongside the
originally reported number for comparison.

Extraction method, in order:
  1. Find candidate rows: a metric keyword co-located with a recognizable unit.
  2. Reconstruct the row's true table structure from PDF word positions --
     this project's report pages routinely lay a prose column and a data-table
     column on the very same visual line, and right-align number columns far
     from their row label, both of which defeat plain text extraction (see
     collect_numeric_run / header_years below). PyMuPDF's page.find_tables()
     was evaluated for this step but, on these specific documents, didn't
     locate the year-header row several of the tables actually use (headers
     sitting mid-line, in a wide table column PyMuPDF's detector didn't
     bound) -- verified below on H&M page 41 -- so it's used only for a
     secondary label/unit lookup, not as the primary path.
  3. No LLM fallback: anything the row-reconstruction can't confidently place
     is rejected to the *_REJECTED.txt log for a human to resolve.

confidence values:
  high   - a single year and a single qualifying value are stated together in
           running text/sentence (e.g. "In 2025, ... 95% ...").
  medium - a multi-year table row matched positionally to a year-header row,
           with the value count matching the year count exactly.
  low    - same as medium, but extra trailing numeric columns after the year
           columns were present; kept only when consistency_check passed or
           there weren't enough extra numbers to check, and always worth a
           manual glance. A metric variant inferred by default rather than by
           an explicit boundary phrase in the quote is also capped at "low".

Usage:
    python3 pipeline/draft_did.py [--limit N] [company ...]

--limit N restricts each PDF to its first N pages, for a fast dry run.
With no company arguments, drafts every company found under data/raw/.

This script never reads or writes data/did/<company>.csv (the hand-verified
file). Only *_DRAFT_v2.csv is machine-written; a verified <company>.csv is
something a human (or a separate promotion step) produces by hand from it.
Downstream scoring stages should read ONLY the verified <company>.csv files.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF -- used only for the secondary table lookup described above
import pdfplumber
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "did"

# Words on the same visual row whose x-gap exceeds this are treated as
# belonging to separate side-by-side tables/columns, not one continuous line.
COLUMN_GAP_PT = 18.0
ROW_TOLERANCE_PT = 2.5
LOOKBACK_ROWS = 20  # how far back on the same page to search for a year-header row

CSV_FIELDS = [
    "year", "metric", "value", "unit", "scope", "boundary_note", "source", "page",
    "report_publish_year", "quote", "header_evidence", "consistency_check", "confidence",
]

YEAR_RE = re.compile(r"(?:FY\s?)?((?:19|20)\d{2})(?!,)")  # trailing "," excluded: these tables mark a
# target-year column as e.g. "2030," right before its change-vs-target percentages -- that's not the
# row's reporting year and must not block the table year-header branch (verified on hm 2024 report p.34)
YEAR_CONTEXT_RE = re.compile(r"(?:in|for|during|since)\s+(?:FY\s?)?((?:19|20)\d{2})", re.I)
HEADER_TOKEN_RE = re.compile(r"^\(?(?:FY\s?)?((?:19|20)\d{2})\)?,?$")
NUM_TOKEN_RE = re.compile(r"^[\-–−]?\(?\d[\d,]*\.?\d*\)?%?$")
BARE_YEAR_RE = re.compile(r"^\(?(?:19|20)\d{2}\)?,?$")
YEAR_PLUS_FOOTNOTE_RE = re.compile(r"^(?:19|20)\d{2}\d{1,2}$")  # e.g. "20191" = 2019 + glued footnote marker
FY_TOKEN_RE = re.compile(r"^\(?FY\s?(?:19|20)\d{2}\)?,?$", re.I)
PUBLISH_YEAR_RE = re.compile(r"^(\d{4})_")

COMBINED_SCOPE_RE = re.compile(r"scope\s*1\s*(?:and|&)\s*(?:scope\s*)?2", re.I)
EXCL_USE_PHASE_RE = re.compile(r"excl(?:uding|\.)?\s*(?:the\s*)?use[\s\-]?(?:of\s*sold\s*products|phase)", re.I)

UNIT_PATTERNS = [
    (re.compile(r"MMT\s*CO\s*₂?\s*e", re.I), "MMT CO2e (million metric tons CO2e)"),
    (re.compile(r"MTCO\s*₂?\s*e", re.I), "MTCO2e (thousand metric tons CO2e)"),
    (re.compile(r"metric tons?\s*(?:of\s*)?CO\s*₂?\s*e", re.I), "metric tons CO2e"),
    (re.compile(r"tonnes?\s*CO\s*₂?\s*e", re.I), "tonnes CO2e"),
    (re.compile(r"\btonnes?\s*COe\b", re.I), "tonnes CO2e (subscript '2' dropped by PDF text extraction)"),
    (re.compile(r"tCO\s*₂?\s*e", re.I), "tCO2e"),
    (re.compile(r"\btCOe\b", re.I), "tCO2e (subscript '2' dropped by PDF text extraction)"),
    (re.compile(r"\bgallons?\b", re.I), "gallons"),
    (re.compile(r"\bMWh\b", re.I), "MWh"),
    (re.compile(r"%"), "%"),
]

QUALIFIER_PATTERNS = [
    (re.compile(r"market[\s\-–−]?based", re.I), "market-based"),
    (re.compile(r"location[\s\-–−]?based", re.I), "location-based"),
    (re.compile(r"own operations", re.I), "own operations"),
    (re.compile(r"gross\b", re.I), "gross"),
]

EMISSIONS_METRICS = {
    "Scope 1", "Scope 2 market-based", "Scope 2 location-based",
    "Scope 3 total (excl. use-phase)", "Scope 3 total (incl. use-phase)",
    "Carbon removal contracted",
}
PERCENT_METRICS = {"Renewable electricity share", "Recycled materials share"}

# canonical *base* metric -> label regexes; Scope 2/3 are resolved to their
# final variant name after a match, from the qualifying language in the row.
GENERIC_METRICS: dict[str, list[str]] = {
    "Scope 1": [r"\bscope\s*1\b(?!\s*(?:and|&)\s*2)"],
    "Scope 2": [r"\bscope\s*2\b"],
    "Scope 3": [r"\bscope\s*3\b"],
    "Renewable electricity share": [
        r"share of renewable electricity",
        r"renewable electricity share",
        r"renewable electricity in own operations",
    ],
}

# company-specific metric: (canonical name, [label regexes])
COMPANY_METRICS: dict[str, tuple[str, list[str]]] = {
    "hm": ("Recycled materials share", [r"share of recycled materials?", r"recycled materials?\s*\(%\)"]),
    "microsoft": ("Carbon removal contracted", [r"carbon removal (?:credits?|contracted)", r"tons? of carbon removal"]),
    "delta": ("SAF volume", [r"sustainable aviation fuel", r"\bSAF\b"]),
}


class DidRow(BaseModel):
    year: str
    metric: str
    value: str
    unit: str
    scope: str
    boundary_note: str
    source: str
    page: int
    report_publish_year: Optional[str]
    quote: str
    header_evidence: str
    consistency_check: str
    confidence: str


def metrics_for(company: str) -> dict[str, list[str]]:
    metrics = dict(GENERIC_METRICS)
    extra = COMPANY_METRICS.get(company)
    if extra:
        name, patterns = extra
        metrics[name] = patterns
    return metrics


def unit_kind_ok(metric: str, unit: Optional[str]) -> bool:
    if unit is None:
        return False
    if metric in EMISSIONS_METRICS:
        return "CO" in unit or "ton" in unit.lower()
    if metric in PERCENT_METRICS:
        return unit == "%"
    return unit != "%"


def page_rows(page) -> list[list[dict]]:
    """Group words into visual rows (reading order, top to bottom), each row
    sorted left to right. Report tables commonly right-align number columns
    far from their row label, so rows are kept whole here -- label matching
    and numeric-token collection do their own gap-aware cutoff to avoid
    gluing on an unrelated side-by-side table on the same visual line."""
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    if not words:
        return []
    rows: dict[int, list[dict]] = defaultdict(list)
    for w in words:
        rows[round(w["top"] / ROW_TOLERANCE_PT)].append(w)
    return [sorted(rows[key], key=lambda w: w["x0"]) for key in sorted(rows)]


def row_text(row_words: list[dict]) -> str:
    return " ".join(w["text"] for w in row_words)


def find_unit(text: str) -> Optional[str]:
    for pattern, label in UNIT_PATTERNS:
        if pattern.search(text):
            return label
    return None


def find_qualifiers(text: str) -> list[str]:
    return [label for pattern, label in QUALIFIER_PATTERNS if pattern.search(text)]


def header_years(text: str) -> Optional[list[str]]:
    """Return the first run of >=2 consecutive year-like tokens found in the
    row, wherever it sits. Report pages here often lay a narrative column and
    a data-table column on the same visual line, so a table's year header
    (e.g. "2025 2024 2023 (2019)") can trail behind unrelated prose rather
    than start the row -- a leading-tokens-only check would miss it."""
    tokens = text.split()
    current: list[str] = []
    for tok in tokens:
        m = HEADER_TOKEN_RE.match(tok)
        if m:
            current.append(m.group(1))
        else:
            if len(current) >= 2:
                return current
            current = []
    return current if len(current) >= 2 else None


def clean_value(token: str) -> str:
    negative = token.startswith(("-", "–", "−")) or (token.startswith("(") and token.endswith(")"))
    digits = re.sub(r"[^\d.]", "", token)
    if not digits:
        return token
    return f"-{digits}" if negative and not digits.startswith("-") else digits


def find_label_start(words: list[dict], patterns: list[str]) -> Optional[re.Match]:
    text = row_text(words)
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return m
    return None


def word_index_after(words: list[dict], char_pos: int) -> int:
    offset = 0
    for i, w in enumerate(words):
        word_end = offset + len(w["text"])
        if word_end >= char_pos:
            return i + 1
        offset = word_end + 1  # +1 for the joining space
    return len(words)


def collect_numeric_run(words: list[dict], start_idx: int) -> tuple[list[str], int]:
    """From start_idx (first word after the metric label), walk forward
    collecting value tokens for this row, stopping once an unrelated
    side-by-side table clearly begins: a non-numeric word reached after a
    wide horizontal gap, once at least one value has already been collected."""
    numeric_tokens: list[str] = []
    end_idx = start_idx - 1
    prev_x1 = None
    for i in range(start_idx, len(words)):
        w = words[i]
        token = w["text"]
        gap = (w["x0"] - prev_x1) if prev_x1 is not None else 0
        is_value = bool(NUM_TOKEN_RE.match(token)) and not BARE_YEAR_RE.match(token) and not YEAR_PLUS_FOOTNOTE_RE.match(token)
        if is_value:
            numeric_tokens.append(token)
            end_idx = i
        elif gap > COLUMN_GAP_PT and numeric_tokens:
            break
        else:
            end_idx = i
        prev_x1 = w["x1"]
    return numeric_tokens, end_idx


def resolve_scope23_metric(base: str, quote: str, qualifiers: list[str]) -> tuple[Optional[str], str, bool]:
    """Returns (metric_name, boundary_note, is_default_variant) for a Scope 2
    or Scope 3 match, or (None, reason, False) when the boundary can't be
    determined from the row and the match should be rejected."""
    if base == "Scope 2":
        if "market-based" in qualifiers:
            return "Scope 2 market-based", "", False
        if "location-based" in qualifiers:
            return "Scope 2 location-based", "", False
        return None, "Scope 2 value found but market-based vs location-based basis not stated in the row", False
    if base == "Scope 3":
        if EXCL_USE_PHASE_RE.search(quote):
            return "Scope 3 total (excl. use-phase)", "excludes use-phase emissions per the row's own wording", False
        return ("Scope 3 total (incl. use-phase)",
                "use-phase basis not explicit in this row; defaulted to incl. use-phase -- verify against report footnotes",
                True)
    return base, "", False


def check_consistency(years: list[str], value_tokens: list[str], extra_tokens: list[str]) -> str:
    """Sanity-checks trailing "% change" columns (common right after the
    year columns in these tables) against the extracted absolute values.
    Only ever evaluated for the most recent (first) year's row."""
    if not extra_tokens or len(years) < 2:
        return "not_available"
    try:
        values = [float(clean_value(v)) for v in value_tokens[: len(years)]]
        checks = [float(clean_value(t)) for t in extra_tokens[:2]]
    except ValueError:
        return "not_available"
    if not values or values[1] == 0:
        return "not_available"
    expected_yoy = (values[0] - values[1]) / values[1] * 100
    if abs(expected_yoy - checks[0]) > max(0.5, 0.1 * abs(checks[0])):
        return "fail"
    if len(checks) >= 2 and len(values) >= 2 and values[-1] != 0:
        expected_vs_last = (values[0] - values[-1]) / values[-1] * 100
        if abs(expected_vs_last - checks[1]) > max(0.5, 0.1 * abs(checks[1])):
            return "fail"
    return "pass"


def build_row(**kwargs) -> Optional[dict]:
    try:
        model = DidRow(**kwargs)
    except ValidationError as exc:
        print(f"dropped invalid row: {exc}", file=sys.stderr)
        return None
    return model.model_dump()


def page_is_candidate(rows_of_words: list[list[dict]], metrics: dict[str, list[str]]) -> bool:
    """A page is worth scanning only if it has both a metric keyword and a
    table year-header row (>=3 four-digit years) somewhere on it. This is a
    cheap, literal pre-filter -- it skips narrative-only pages entirely
    rather than trying to parse prose sentences into table data."""
    full_text = " ".join(row_text(r) for r in rows_of_words)
    has_keyword = any(re.search(pat, full_text, re.I) for pats in metrics.values() for pat in pats)
    if not has_keyword:
        return False
    return any((header_years(row_text(r)) or []).__len__() >= 3 for r in rows_of_words)


def process_row(
    rows_of_words: list[list[dict]], row_idx: int, metrics: dict[str, list[str]],
    source: str, page_index: int, publish_year: Optional[str],
    rows: list[dict], rejected: list[str],
) -> None:
    words = rows_of_words[row_idx]
    full_text = row_text(words)
    if not full_text or COMBINED_SCOPE_RE.search(full_text):
        return
    for base_metric, patterns in metrics.items():
        label_match = find_label_start(words, patterns)
        if not label_match:
            continue

        start_idx = word_index_after(words, label_match.end())
        numeric_tokens, end_idx = collect_numeric_run(words, start_idx)
        quote = row_text(words[: end_idx + 1]) if numeric_tokens else full_text

        inline_unit = find_unit(quote)
        qualifiers = find_qualifiers(quote)
        scope = ", ".join(qualifiers) if qualifiers else "unspecified basis"

        metric_name, boundary_note, is_default_variant = (
            resolve_scope23_metric(base_metric, quote, qualifiers)
            if base_metric in ("Scope 2", "Scope 3") else (base_metric, "", False)
        )
        if metric_name is None:
            rejected.append(f"[{source} p.{page_index}] ({base_metric}) {boundary_note}: {full_text}")
            continue

        inline_years = YEAR_RE.findall(quote)
        context_year = YEAR_CONTEXT_RE.search(quote)
        base_conf = "low" if is_default_variant else None

        # Narrative branches (a sentence naming its own year) require the
        # unit stated right there -- a nearby row's unit is too easy to
        # misattribute across unrelated merged rows in these page layouts
        # (verified: doing this for narrative rows produced a false "Scope 1
        # = 0" pulled from an unrelated "% from trading schemes" row). The
        # table branch below is the only one allowed to borrow a unit from a
        # nearby row, since there it's tied to a specific, aligned column.
        if context_year and len(numeric_tokens) >= 1:
            if not unit_kind_ok(metric_name, inline_unit):
                rejected.append(
                    f"[{source} p.{page_index}] ({metric_name}) no recognizable unit stated in this sentence: {full_text}"
                )
                continue
            rows.append(build_row(
                year=context_year.group(1), metric=metric_name, value=clean_value(numeric_tokens[0]),
                unit=inline_unit, scope=scope, boundary_note=boundary_note, source=source, page=page_index,
                report_publish_year=publish_year, quote=quote,
                header_evidence=f"inline phrase: ‘{context_year.group(0)}’",
                consistency_check="not_available", confidence=base_conf or "high",
            ))
        elif len(set(inline_years)) == 1 and len(numeric_tokens) == 1:
            if not unit_kind_ok(metric_name, inline_unit):
                rejected.append(
                    f"[{source} p.{page_index}] ({metric_name}) no recognizable unit stated in this sentence: {full_text}"
                )
                continue
            rows.append(build_row(
                year=inline_years[0], metric=metric_name, value=clean_value(numeric_tokens[0]),
                unit=inline_unit, scope=scope, boundary_note=boundary_note, source=source, page=page_index,
                report_publish_year=publish_year, quote=quote,
                header_evidence=f"single year mentioned in row: {inline_years[0]}",
                consistency_check="not_available", confidence=base_conf or "high",
            ))
        elif not inline_years and numeric_tokens:
            unit = inline_unit
            unit_note = ""
            if unit is None:
                unit, unit_source_row = _lookback_unit(rows_of_words, row_idx)
                if unit is not None:
                    unit_note = f"unit not stated on the data row itself; inferred from a nearby row: '{unit_source_row}'"
            if not unit_kind_ok(metric_name, unit):
                reason = (f"no recognizable {'%' if metric_name in PERCENT_METRICS else 'emissions'} unit found in the row "
                          f"or within {LOOKBACK_ROWS} rows above it")
                rejected.append(f"[{source} p.{page_index}] ({metric_name}) {reason}: {full_text}")
                continue
            row_boundary_note = f"{boundary_note}; {unit_note}" if (boundary_note and unit_note) else (boundary_note or unit_note)
            row_base_conf = "low" if (is_default_variant or unit_note) else None

            years, header_text = _lookback_header(rows_of_words, row_idx)
            if years is None:
                rejected.append(
                    f"[{source} p.{page_index}] ({metric_name}) no year-header row found above "
                    f"within {LOOKBACK_ROWS} rows; year cannot be tied to a column: {quote}"
                )
            elif len(numeric_tokens) < len(years):
                rejected.append(
                    f"[{source} p.{page_index}] ({metric_name}) fewer values ({len(numeric_tokens)}) than "
                    f"year columns ({len(years)}) -- cannot tell which year(s) are missing without "
                    f"guessing: header='{header_text}' row='{quote}'"
                )
            else:
                # The year<->value positional mapping itself (years[i] <-> numeric_tokens[i])
                # is unambiguous once len(numeric_tokens) >= len(years); trailing extra numbers
                # are separate columns (change %, a further baseline/target not itself a literal
                # year, etc.). A failed consistency check flags that those EXTRA columns don't
                # parse as expected -- it does not cast doubt on the year<->value pairs
                # themselves, so per spec ("flag failures") the row is kept, not discarded.
                extra = numeric_tokens[len(years):]
                consistency = check_consistency(years, numeric_tokens, extra) if extra else "not_available"
                exact = len(numeric_tokens) == len(years)
                if row_base_conf:
                    confidence = row_base_conf
                elif consistency == "fail" or not exact:
                    confidence = "low"
                else:
                    confidence = "medium"
                row_note = row_boundary_note
                if consistency == "fail":
                    note = "trailing change-column figures didn't match the extracted values within tolerance"
                    row_note = f"{row_note}; {note}" if row_note else note
                for i, (yr, val) in enumerate(zip(years, numeric_tokens[: len(years)])):
                    rows.append(build_row(
                        year=yr, metric=metric_name, value=clean_value(val), unit=unit, scope=scope,
                        boundary_note=row_note, source=source, page=page_index,
                        report_publish_year=publish_year, quote=quote,
                        header_evidence=f"table year-header row: '{header_text}'",
                        consistency_check=(consistency if i == 0 else "not_available"),
                        confidence=confidence,
                    ))
        else:
            rejected.append(
                f"[{source} p.{page_index}] ({metric_name}) no year could be tied to a value "
                f"(no inline year, no single unambiguous year, and either no values or "
                f"multiple ambiguous years present): {full_text}"
            )


def extract_matches(
    company: str, pdf_path: Path, source: str, publish_year: Optional[str],
    skip_pages: set[int], rows: list[dict], rejected: list[str],
    page_cap: int = 5,
) -> None:
    """Scans up to `page_cap` candidate pages per report (see
    page_is_candidate) -- pages without both a metric keyword and a >=3-year
    table header are skipped outright, not run through row-by-row parsing."""
    metrics = metrics_for(company)
    scanned = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            if page_index in skip_pages or scanned >= page_cap:
                continue
            rows_of_words = page_rows(page)
            if not rows_of_words or not page_is_candidate(rows_of_words, metrics):
                continue
            scanned += 1
            for row_idx in range(len(rows_of_words)):
                process_row(rows_of_words, row_idx, metrics, source, page_index, publish_year, rows, rejected)


def _lookback_header(rows_of_words: list[list[dict]], from_idx: int) -> tuple[Optional[list[str]], str]:
    start = max(0, from_idx - LOOKBACK_ROWS)
    for i in range(from_idx - 1, start - 1, -1):
        years = header_years(row_text(rows_of_words[i]))
        if years:
            return years, row_text(rows_of_words[i])
    return None, ""


def _lookback_unit(rows_of_words: list[list[dict]], from_idx: int) -> tuple[Optional[str], str]:
    """Section-header rows in these tables often carry the unit (e.g. 'Scope
    3 GHG emissions (tCO2e)') a few rows above the data rows themselves. Only
    used when the data row has no unit of its own; the resulting boundary_note
    always says so, so a reviewer can double check it against the page."""
    start = max(0, from_idx - LOOKBACK_ROWS)
    for i in range(from_idx - 1, start - 1, -1):
        text = row_text(rows_of_words[i])
        unit = find_unit(text)
        if unit:
            return unit, text
    return None, ""


def pymupdf_table_probe(pdf_path: Path, page_index_1based: int) -> str:
    """Secondary lookup: what PyMuPDF's own table detector sees on this page.
    Used only to log a cross-check note; never the primary extraction path
    here (see module docstring)."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index_1based - 1]
        tabs = page.find_tables()
        return f"{len(tabs.tables)} table(s) detected by PyMuPDF on this page"
    finally:
        doc.close()


def load_verified_pages(company: str) -> set[tuple[str, int]]:
    verified_path = OUT_DIR / f"{company}.csv"
    if not verified_path.exists():
        return set()
    with verified_path.open(newline="", encoding="utf-8") as f:
        return {(row["source"], int(row["page"])) for row in csv.DictReader(f)}


def draft_company(
    company: str, page_cap: int = 5,
    report_filter: Optional[str] = None, forced_page: Optional[int] = None,
) -> None:
    company_dir = RAW_DIR / company
    pdf_paths = sorted(company_dir.glob("*.pdf"))
    if report_filter:
        pdf_paths = [p for p in pdf_paths if p.name.startswith(report_filter)]
    if not pdf_paths:
        print(f"skip {company}: no matching PDFs found under {company_dir}", file=sys.stderr)
        return

    verified_pages = load_verified_pages(company)
    if verified_pages:
        print(f"{company}: {len(verified_pages)} (source, page) pairs already verified in "
              f"{company}.csv -- skipping re-extraction of those pages", file=sys.stderr)

    rows: list[dict] = []
    rejected: list[str] = []
    for pdf_path in pdf_paths:
        source = f"{company}/{pdf_path.name}"
        m = PUBLISH_YEAR_RE.match(pdf_path.name)
        publish_year = m.group(1) if m else None
        print(f"scanning {source} ...", file=sys.stderr)

        if forced_page is not None:
            print(f"  --page {forced_page}: scanning only this page, bypassing the candidate filter/cap", file=sys.stderr)
            metrics = metrics_for(company)
            with pdfplumber.open(pdf_path) as pdf:
                rows_of_words = page_rows(pdf.pages[forced_page - 1])
            for row_idx in range(len(rows_of_words)):
                process_row(rows_of_words, row_idx, metrics, source, forced_page, publish_year, rows, rejected)
            continue

        skip_pages = {p for s, p in verified_pages if s == source}
        extract_matches(company, pdf_path, source, publish_year, skip_pages, rows, rejected, page_cap=page_cap)

    rows = [r for r in rows if r is not None]
    rows.sort(key=lambda r: (r["metric"], str(r["year"]), r["source"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{company}_DRAFT_v2.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}", file=sys.stderr)

    if rejected:
        rejected_path = OUT_DIR / f"{company}_REJECTED.txt"
        rejected_path.write_text(
            "Candidate metric mentions rejected during drafting, with the reason.\n"
            "Check the page; add a hand-verified row to data/did/<company>.csv yourself\n"
            "if the source actually supports a value -- this script never guesses.\n\n"
            + "\n".join(rejected) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(rejected)} rejected candidates to {rejected_path}", file=sys.stderr)

    print_matrix(company, rows)


def print_matrix(company: str, rows: list[dict]) -> None:
    """Per spec step 4: a metric x year table showing filled / conflicting /
    missing cells, so a reviewer knows exactly what to check by hand."""
    by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in rows:
        by_cell[(r["metric"], str(r["year"]))].add(r["value"])

    metrics = sorted({m for m, _ in by_cell})
    years = sorted({y for _, y in by_cell})
    if not metrics or not years:
        print(f"\n{company}: no rows extracted, nothing to matrix.", file=sys.stderr)
        return

    print(f"\n=== {company}: metric x year coverage ===", file=sys.stderr)
    col_w = max(6, *(len(y) for y in years))
    header = "metric".ljust(38) + "".join(y.rjust(col_w + 1) for y in years)
    print(header, file=sys.stderr)
    for metric in metrics:
        cells = []
        for year in years:
            values = by_cell.get((metric, year), set())
            if not values:
                cells.append("-".rjust(col_w))
            elif len(values) > 1:
                cells.append("CONFLICT".rjust(col_w))
            else:
                cells.append("OK".rjust(col_w))
        print(metric[:37].ljust(38) + " " + " ".join(cells), file=sys.stderr)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("companies", nargs="*", help="company folder names under data/raw/; default: all")
    parser.add_argument("--page-cap", type=int, default=5,
                         help="max candidate pages (keyword + >=3-year table header) to scan per report (default: 5)")
    parser.add_argument("--report", default=None,
                         help="restrict to the one PDF whose filename starts with this (e.g. '2025')")
    parser.add_argument("--page", type=int, default=None,
                         help="debug: scan only this exact 1-based page, bypassing the candidate filter/cap "
                              "(requires --report to pick a single PDF)")
    args = parser.parse_args(argv)

    if args.page is not None and not args.report:
        parser.error("--page requires --report to pick a single PDF")

    companies = args.companies or sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())
    for company in companies:
        draft_company(company, page_cap=args.page_cap, report_filter=args.report, forced_page=args.page)


if __name__ == "__main__":
    main(sys.argv[1:])
