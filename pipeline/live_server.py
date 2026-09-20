"""Local dashboard server with one live endpoint: POST /api/add-company.

Serves web/ exactly like `python3 -m http.server` (drop-in replacement used
by `make demo`), plus a single route that lets the "Try a company" drawer in
the UI submit a company name + a handful of report-year -> PDF-URL pairs and
get a brand-new company dashboard built on demand.

Design constraints (see the "Try a company" plan for the full rationale):
  - No web search/scraping: the caller supplies the PDF URLs directly, or
    uploads the PDF itself (some sites bot-gate direct downloads -- Akamai
    and similar CDNs will silently serve an HTML failover page to a non-
    browser client, which the %PDF-magic-byte check below catches and
    rejects rather than trusting). This process never guesses or discovers
    a URL on its own.
  - No pip dependencies beyond the stdlib -- this project has no
    requirements.txt and no other backend code; adding Flask/requests here
    would be the first dependency in the whole repo. File uploads are parsed
    with the stdlib `cgi` module (deprecated for removal in 3.13+; fine on
    the 3.9 interpreter this project runs on today -- worth revisiting if
    this project ever upgrades its Python version).
  - The numeric drift score (`data/did/{company}.csv`) is treated elsewhere
    in this pipeline as hand-verified ground truth. A company added here
    gets an EMPTY did csv (header only, zero rows) so stage6/stage8 abstain
    honestly on goalpost_drift/say_do_gap instead of either crashing or
    scoring on machine-guessed numbers.

Usage:
    python3 pipeline/live_server.py [port]   # default 8000
"""
from __future__ import annotations

import cgi
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
RAW_DIR = ROOT / "data" / "raw"
DID_DIR = ROOT / "data" / "did"

SEEDED_COMPANIES = {"hm", "amazon", "microsoft"}
MAX_REPORTS = 8
MAX_PDF_BYTES = 50 * 1024 * 1024
DOWNLOAD_TIMEOUT = 30
STAGE_TIMEOUT = 300
CURRENT_YEAR = time.gmtime().tm_year


def slugify(name: str) -> str:
    """hm/amazon/microsoft are single lowercase tokens with no separators --
    match that convention. It also avoids stage6_saydo.py's bare-glob company
    discovery (`if "_" not in p.stem`), which silently skips any slug that
    contains an underscore."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def validate_reports(reports) -> tuple[list[dict], list[str]]:
    """Each valid entry: {"year": int, "url": str|None, "file_bytes": bytes|None}
    -- exactly one of url/file_bytes is set (file_bytes wins if both given)."""
    if not isinstance(reports, list) or not reports:
        return [], ["at least one report (year + url or file) is required"]
    if len(reports) > MAX_REPORTS:
        return [], [f"at most {MAX_REPORTS} reports per request"]
    valid: list[dict] = []
    errors: list[str] = []
    for r in reports:
        if not isinstance(r, dict):
            errors.append("malformed report entry")
            continue
        try:
            year = int(r.get("year"))
        except (TypeError, ValueError):
            errors.append(f"invalid year: {r.get('year')!r}")
            continue
        if not (2000 <= year <= CURRENT_YEAR + 1):
            errors.append(f"year out of range: {year}")
            continue
        file_bytes = r.get("file_bytes")
        if file_bytes:
            valid.append({"year": year, "url": None, "file_bytes": file_bytes})
            continue
        url = str(r.get("url") or "").strip()
        if not url:
            errors.append(f"{year}: no URL or file provided")
            continue
        # http(s)-only, enforced before this ever reaches urlopen: urllib
        # happily follows file:// and would otherwise let a crafted request
        # read an arbitrary local file back into "data/raw/".
        if urlparse(url).scheme.lower() not in ("http", "https"):
            errors.append(f"{year}: URL must be http(s), rejected {url!r}")
            continue
        valid.append({"year": year, "url": url, "file_bytes": None})
    return valid, errors


def save_pdf_bytes(data: bytes, dest: Path) -> Optional[str]:
    """Validate and persist PDF bytes already in hand. Returns an error
    string, or None on success."""
    if len(data) > MAX_PDF_BYTES:
        return "file too large (over 50MB)"
    if not data.startswith(b"%PDF"):
        return "not a PDF (missing %PDF header)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return None


def download_pdf(url: str, dest: Path) -> Optional[str]:
    """Download url to dest. Returns an error string, or None on success."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; RestatedBot/1.0)"})
    try:
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
            data = resp.read(MAX_PDF_BYTES + 1)
    except Exception as exc:  # noqa: BLE001 -- one bad URL must not abort the batch
        return f"download failed: {exc}"
    return save_pdf_bytes(data, dest)


def ensure_empty_did_csv(slug: str) -> None:
    """A MISSING data/did/{slug}.csv makes stage6_saydo.py raise an uncaught
    FileNotFoundError; an EMPTY (header-only) one makes it abstain cleanly
    (goalpost_drift=None, 'no comparable pairs'). We only ever write the
    header -- no row is fabricated."""
    path = DID_DIR / f"{slug}.csv"
    if not path.exists():
        DID_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text("year,metric,value,unit,scope,source,page\n", encoding="utf-8")


def run_stage(args: list[str], tolerant: bool = False) -> Optional[str]:
    """Run a pipeline stage as a subprocess. Returns an error string, or None
    on success. `tolerant=True` mirrors the Makefile's own `lang` loop, which
    lets stage3b/stage4 fail on a claims-free company without aborting the
    rest of the run."""
    try:
        proc = subprocess.run(
            [sys.executable, *args], cwd=str(ROOT),
            capture_output=True, text=True, timeout=STAGE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"{args[0]} timed out after {STAGE_TIMEOUT}s"
    if proc.returncode != 0 and not tolerant:
        return (proc.stderr or proc.stdout or f"exit code {proc.returncode}")[-2000:]
    return None


def add_company(payload: dict) -> dict:
    name = str(payload.get("company") or "").strip()
    slug = slugify(name)
    if len(slug) < 2:
        return {"ok": False, "error": "company name must have at least 2 letters/digits"}
    if slug in SEEDED_COMPANIES:
        return {"ok": False, "error": f"'{slug}' already has hand-verified data seeded; choose a different name"}

    reports, errors = validate_reports(payload.get("reports"))
    if not reports:
        return {"ok": False, "error": "; ".join(errors) or "no valid reports supplied"}

    raw_dir = RAW_DIR / slug
    downloaded: list[int] = []
    failed: list[dict] = [{"year": None, "error": e} for e in errors]
    for report in reports:
        year, url, file_bytes = report["year"], report["url"], report["file_bytes"]
        dest = raw_dir / f"{year}_sustainability_report.pdf"
        err = save_pdf_bytes(file_bytes, dest) if file_bytes is not None else download_pdf(url, dest)
        if err:
            failed.append({"year": year, "error": err})
        else:
            downloaded.append(year)

    if not downloaded:
        return {"ok": False, "error": "no report could be downloaded", "failed": failed}

    ensure_empty_did_csv(slug)

    err = run_stage(["pipeline/stage3_language.py", "--company", slug, "--max-pages", "60"])
    if err:
        return {"ok": False, "error": f"stage3_language failed: {err}",
                "downloaded": downloaded, "failed": failed}
    run_stage(["pipeline/stage3b_langdrift.py", "--company", slug], tolerant=True)
    run_stage(["pipeline/stage4_sins_rules.py", "--company", slug], tolerant=True)

    err = run_stage(["pipeline/stage6_saydo.py", slug])
    if err:
        return {"ok": False, "error": f"stage6_saydo failed: {err}",
                "downloaded": downloaded, "failed": failed}

    # No company arg: re-discovers every out/*.stage6.json, so this also
    # regenerates index.json/data.js and recomputes peer_percentile across
    # all companies including the new one.
    err = run_stage(["pipeline/stage8_score.py"])
    if err:
        return {"ok": False, "error": f"stage8_score failed: {err}",
                "downloaded": downloaded, "failed": failed}

    warnings = []
    if len(downloaded) < 2:
        warnings.append("only 1 report downloaded; cross-year drift signals are unavailable")

    return {"ok": True, "slug": slug, "downloaded": downloaded,
            "failed": failed, "warnings": warnings}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def end_headers(self) -> None:
        """Never let a browser cache this. The dashboard and its data are
        rebuilt in place by the pipeline, so a cached index.html or data.js
        silently shows a stale scoreboard -- which looks exactly like a bug
        that has already been fixed."""
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _parse_json_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if not (0 < length <= 1_000_000):
            raise ValueError("invalid or oversized request body")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("invalid JSON body") from exc

    def _parse_multipart_body(self) -> dict:
        """Uploaded PDFs arrive here (application/x-www-form-urlencoded
        FormData). Fields are indexed (year_0/url_0/file_0, year_1/... ) so
        each row's year/url/file stay unambiguously paired."""
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        max_body = MAX_PDF_BYTES * MAX_REPORTS + 1_000_000
        if not (0 < length <= max_body):
            raise ValueError("invalid or oversized request body")
        form = cgi.FieldStorage(
            fp=self.rfile, headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]},
        )
        company = form.getvalue("company", "")
        reports = []
        i = 0
        while form.getvalue(f"year_{i}") is not None:
            reports.append({
                "year": form.getvalue(f"year_{i}", ""),
                "url": form.getvalue(f"url_{i}", "") or "",
                "file_bytes": self._multipart_file_bytes(form, f"file_{i}"),
            })
            i += 1
        return {"company": company, "reports": reports}

    @staticmethod
    def _multipart_file_bytes(form: "cgi.FieldStorage", key: str) -> Optional[bytes]:
        if key not in form.keys():
            return None
        item = form[key]
        if isinstance(item, list):
            item = item[0]
        if not getattr(item, "filename", None):
            return None
        data = item.file.read()
        return data or None

    def do_POST(self) -> None:
        if self.path != "/api/add-company":
            self.send_error(404, "not found")
            return
        content_type = self.headers.get("Content-Type", "")
        try:
            if content_type.startswith("multipart/form-data"):
                payload = self._parse_multipart_body()
            else:
                payload = self._parse_json_body()
        except ValueError as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        try:
            result = add_company(payload)
        except Exception as exc:  # noqa: BLE001 -- never let this take the server down
            result = {"ok": False, "error": f"internal error: {exc}"}
        self._json(200 if result.get("ok") else 400, result)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Restated dashboard -> http://127.0.0.1:{port}/  (POST /api/add-company enabled)",
          file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
