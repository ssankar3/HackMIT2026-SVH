"""Stage 0: the output contract.

Every downstream stage writes into these models, and `/schema/output.schema.json`
is generated from them. Freeze this before building on top.

Design rule that matters for judging: anything the pipeline could not establish
from documents, the DiD CSVs, or retrieved evidence is `None` and carries an
`abstain` reason -- never a filled-in default. A missing sub-score must stay
visibly missing rather than silently scoring 0 (which would read as "clean").
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# enums
# --------------------------------------------------------------------------

class SourceType(str, Enum):
    company = "company"
    external = "external"


class ClaimType(str, Enum):
    target = "target"
    achievement = "achievement"
    product_label = "product_label"
    policy = "policy"
    general = "general"


class Sin(str, Enum):
    hidden_trade_off = "hidden_trade_off"
    no_proof = "no_proof"
    vagueness = "vagueness"
    irrelevance = "irrelevance"
    lesser_of_two_evils = "lesser_of_two_evils"
    fibbing = "fibbing"
    false_labels = "false_labels"


class EvidenceStance(str, Enum):
    supports = "supports"
    contradicts = "contradicts"
    irrelevant = "irrelevant"
    context = "context"


class Verdict(str, Enum):
    likely_greenwashing = "likely_greenwashing"
    mixed = "mixed"
    likely_credible = "likely_credible"


class Confidence(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ProgressStatus(str, Enum):
    on_track = "on_track"
    behind = "behind"
    reversed = "reversed"
    unmeasurable = "unmeasurable"


class DriftType(str, Enum):
    target_value_changed = "target_value_changed"
    target_year_moved = "target_year_moved"
    baseline_moved = "baseline_moved"
    scope_narrowed = "scope_narrowed"
    wording_softened = "wording_softened"
    commitment_dropped = "commitment_dropped"
    metric_restated = "metric_restated"
    metric_redefined = "metric_redefined"


# --------------------------------------------------------------------------
# core records
# --------------------------------------------------------------------------

class Evidence(BaseModel):
    """A retrieved passage. `quote` MUST be a verified substring of the source
    chunk after whitespace normalization; unverified quotes are dropped upstream
    and never reach this model."""
    evidence_id: str
    chunk_id: str
    stance: EvidenceStance
    quote: str
    source_type: SourceType
    doc_name: str
    page: Optional[int] = None
    source_url: Optional[str] = None
    doc_date: Optional[str] = Field(None, description="ISO date; drives as-of vs hindsight")
    year: Optional[int] = None
    quote_verified: bool = True


class SinTag(BaseModel):
    sin: Sin
    rationale: str
    severity: float = Field(ge=0.0, le=1.0)
    evidence_needed: Optional[str] = None


class LinguisticFeatures(BaseModel):
    vague_word_ratio: Optional[float] = None
    hedging_ratio: Optional[float] = None
    future_orientation_ratio: Optional[float] = None
    quantification_present: Optional[bool] = None
    baseline_present: Optional[bool] = None
    date_present: Optional[bool] = None
    scope_present: Optional[bool] = None
    verification_present: Optional[bool] = None
    specificity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    readability: Optional[float] = None
    sentiment: Optional[float] = None


class DebateResult(BaseModel):
    prosecutor: Optional[str] = None
    defense: Optional[str] = None
    verdict: Optional[Verdict] = None
    probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    probability_spread: Optional[float] = Field(
        None, description="max-min across judge runs; the confidence band"
    )
    confidence: Optional[Confidence] = None
    key_reasons: List[str] = Field(default_factory=list)
    cited_chunk_ids: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    confidence_capped_reason: Optional[str] = None


class Claim(BaseModel):
    claim_id: str
    claim_text: str = Field(description="verbatim; substring of its chunk")
    claim_type: ClaimType
    metric: Optional[str] = None
    canonical_metric: Optional[str] = None
    target_value: Optional[float] = None
    unit: Optional[str] = None
    baseline_year: Optional[int] = None
    target_year: Optional[int] = None
    scope: Optional[str] = None
    has_third_party_verification: Optional[bool] = None
    verifier_name: Optional[str] = None
    chunk_id: str
    doc_name: Optional[str] = None
    publish_year: int
    page: Optional[int] = None
    features: Optional[LinguisticFeatures] = None
    sins: List[SinTag] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    progress_status: Optional[ProgressStatus] = None
    debate: Optional[DebateResult] = None


class DriftEvent(BaseModel):
    """A goalpost move. `metric_restated` covers the case where a company
    republishes a prior year's actual at a different value."""
    drift_id: str
    year: int
    type: DriftType
    canonical_metric: Optional[str] = None
    old: Optional[str] = None
    new: Optional[str] = None
    magnitude_pct: Optional[float] = Field(
        None, description="signed % change from old to new, where both are numeric"
    )
    description: str
    evidence_chunk_ids: List[str] = Field(default_factory=list)
    old_source: Optional[str] = None
    new_source: Optional[str] = None
    page: Optional[int] = None


class DidPoint(BaseModel):
    """One row of /data/did/{company}.csv -- what the company actually DID."""
    year: int
    metric: str
    value: float
    unit: str
    scope: Optional[str] = None
    source: str
    page: Optional[int] = None


class SubScores(BaseModel):
    """0-100, higher = riskier. None means the pipeline could not measure it;
    the paired `*_basis` string says why."""
    vagueness: Optional[float] = None
    unsupported_claims: Optional[float] = None
    sins_severity: Optional[float] = None
    say_do_gap: Optional[float] = None
    goalpost_drift: Optional[float] = None


class YearScore(BaseModel):
    year: int
    sub_scores: SubScores
    overall: Optional[float] = Field(None, ge=0.0, le=100.0)
    confidence: Confidence
    abstain: bool = False
    abstain_reasons: List[str] = Field(default_factory=list)
    n_claims: int = 0
    n_verified_quotes: int = 0
    n_drift_events: int = 0


class TimelinePoint(BaseModel):
    """One tick of the time machine: what was SAID and what was DID that year."""
    year: int
    said_claim_ids: List[str] = Field(default_factory=list)
    did_points: List[DidPoint] = Field(default_factory=list)
    drift_ids: List[str] = Field(default_factory=list)
    score_as_of: Optional[float] = None
    score_hindsight: Optional[float] = None


class EvalSummary(BaseModel):
    agreement_rate: Optional[float] = None
    per_sin_precision: Dict[str, float] = Field(default_factory=dict)
    per_sin_recall: Dict[str, float] = Field(default_factory=dict)
    brier_score: Optional[float] = None
    calibration_table: List[Dict[str, float]] = Field(default_factory=list)
    n_gold: int = 0
    notes: Optional[str] = None


class DataCoverage(BaseModel):
    """Front-and-centre honesty: what actually backs this company's score."""
    n_documents: int = 0
    n_pages: int = 0
    n_chunks: int = 0
    n_claims: int = 0
    n_evidence_verified: int = 0
    n_did_rows: int = 0
    did_years: List[int] = Field(default_factory=list)
    did_metrics: List[str] = Field(default_factory=list)
    stages_completed: List[str] = Field(default_factory=list)
    stages_blocked: Dict[str, str] = Field(
        default_factory=dict, description="stage -> why it could not run"
    )


class CompanySummary(BaseModel):
    company: str
    display_name: str
    overall_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    confidence: Confidence
    confidence_rationale: str
    peer_percentile: Optional[float] = None
    abstain: bool = False
    abstain_reasons: List[str] = Field(default_factory=list)
    headline: Optional[str] = None
    sub_scores: SubScores = Field(default_factory=SubScores)


class CompanyOutput(BaseModel):
    """The frozen contract: /out/{company}.json and web/public/data/{company}.json."""
    schema_version: str = "1.0.0"
    generated_at: str
    summary: CompanySummary
    coverage: DataCoverage
    timeline: List[TimelinePoint] = Field(default_factory=list)
    year_scores: List[YearScore] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    drift_events: List[DriftEvent] = Field(default_factory=list)
    did_points: List[DidPoint] = Field(default_factory=list)
    top_damaging_claim_ids: List[str] = Field(default_factory=list)
    eval: Optional[EvalSummary] = None


def export_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(CompanyOutput.model_json_schema(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "schema" / "output.schema.json"
    export_schema(out)
    print(f"wrote {out}")
