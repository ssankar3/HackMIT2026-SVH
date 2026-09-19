PY      := python3
PORT    := 8000
COMPANY ?=
LIMIT   ?=

LIMIT_FLAG := $(if $(LIMIT),--limit $(LIMIT),)

.PHONY: help schema saydo score claims lang all demo eval check-leakage clean-out

COMPANIES := hm microsoft amazon

help:
	@echo "Greenwash Time Machine"
	@echo ""
	@echo "  make all            schema + say-do + language + score  (no API key needed)"
	@echo "  make lang           Stage 2/3/3b only: claims, linguistics, language drift"
	@echo "  make demo           build everything, then serve the dashboard"
	@echo "  make schema         export schema/output.schema.json from Pydantic models"
	@echo "  make saydo          Stage 6: goalpost drift from data/did/*.csv"
	@echo "  make score          Stage 8: scoring -> out/*.json + web/public/data"
	@echo "  make eval           compare pipeline output against the held-out prior"
	@echo "  make check-leakage  assert no company name is hardcoded in /pipeline or /config"
	@echo ""
	@echo "  vars: COMPANY=hm  LIMIT=20  PORT=8000"
	@echo ""
	@echo "Stages 2/4/5/7 (claims, sins, evidence, debate) need ANTHROPIC_API_KEY."

schema:
	$(PY) pipeline/models.py

saydo:
	$(PY) pipeline/stage6_saydo.py $(COMPANY) $(LIMIT_FLAG)

score:
	$(PY) pipeline/stage8_score.py $(COMPANY) $(LIMIT_FLAG)

# Stage 2/3/3b: claim extraction + linguistic features + cross-year language
# drift. No API key needed. Slowest step (~30s per company).
lang:
	@for c in $(if $(COMPANY),$(COMPANY),$(COMPANIES)); do \
		echo "--- $$c ---"; \
		$(PY) pipeline/stage3_language.py --company $$c --max-pages 60 2>&1 | tail -3; \
		$(PY) pipeline/stage3b_langdrift.py --company $$c 2>&1 | grep -E "events$$" || true; \
	done

claims:
	$(PY) pipeline/stage2_claims_v1.py --pdf $(PDF) --max-pages $(or $(MAX_PAGES),50)

all: schema saydo lang score
	@echo ""
	@echo "Built. Open the dashboard with:  make demo"

demo: all
	@echo ""
	@echo "Dashboard -> http://localhost:$(PORT)/"
	@cd web && $(PY) -m http.server $(PORT)

eval:
	@$(PY) - <<'EOF'
	import json, pathlib
	rows = []
	for p in sorted(pathlib.Path('out').glob('*.json')):
	    if p.name.endswith('.stage6.json'): continue
	    d = json.loads(p.read_text())
	    s = d['summary']
	    rows.append((s['display_name'], s['sub_scores']['goalpost_drift'], s['confidence']))
	rows.sort(key=lambda r: -1 if r[1] is None else -r[1])
	print("\npipeline ranking by goalpost drift (higher = riskier):")
	for i,(n,g,c) in enumerate(rows,1):
	    print(f"  {i}. {n:<16} {'unmeasurable' if g is None else format(g,'.1f'):>12}   confidence={c}")
	print("\nheld-out prior: see eval/analyst_prior.md (NOT an input to scoring)")
	EOF

# Fails loudly if a company name leaks into SCORING logic or its config. The
# whole signal is worthless if the ranking is encoded rather than derived.
#
# draft_did.py is excluded by design: it is a data-prep locator that searches
# PDFs for sector vocabulary ("SAF" for an airline, "recycled materials" for
# apparel). That is domain vocabulary for finding tables, not an expectation
# about any company's outcome, and it runs before scoring rather than inside it.
SCORING_SRC := pipeline/models.py pipeline/stage6_saydo.py pipeline/stage8_score.py config/

check-leakage:
	@echo "scanning scoring pipeline + config for hardcoded company names..."
	@! grep -rniE '\b(h&m|hennes|microsoft|amazon|delta)\b' $(SCORING_SRC) \
	   | grep -viE 'DISPLAY *=|display_name' \
	   || (echo "LEAKAGE: company name found above" && exit 1)
	@echo "clean: no company-specific scoring logic."
	@echo "(DISPLAY map holds presentation labels only and touches no score.)"

clean-out:
	rm -f out/*.json web/public/data/*.json web/data.js
