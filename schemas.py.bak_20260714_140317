from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, Union

MetricValue = Union[float, int, bool, str, None]

class BorrowerProfile(BaseModel):
    name: str = Field(..., description="Borrower legal name")
    industry: str = Field(..., description="Primary industry")
    facility_type: str = Field(..., description="Term loan, revolver, ABL, etc.")
    use_of_proceeds: str = Field(..., description="Working capital, refinance, acquisition, etc.")

class CovenantSet(BaseModel):
    max_total_leverage: Optional[float] = None
    min_fcc: Optional[float] = None
    min_dscr: Optional[float] = None

class ExtractedPeriod(BaseModel):
    period_name: str
    income_statement: Dict[str, Optional[float]] = Field(default_factory=dict)
    balance_sheet: Dict[str, Optional[float]] = Field(default_factory=dict)
    cash_flow: Dict[str, Optional[float]] = Field(default_factory=dict)
    derived_metrics: Dict[str, MetricValue] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)

class ExtractionResult(BaseModel):
    statement_basis: Literal["actual", "thousands", "millions"] = "actual"
    periods: List[ExtractedPeriod] = Field(default_factory=list)
    validation_flags: List[str] = Field(default_factory=list)
    schema_version: str = "1.0"
    extractor_version: str = "2026-01-25"

class MemoRequest(BaseModel):
    borrower: BorrowerProfile
    covenants: CovenantSet
    extracted: ExtractionResult

class MemoResponse(BaseModel):
    memo_markdown: str

# ============================================================
# NEW: Structured Metrics for API Response
# ============================================================

class MetricComparison(BaseModel):
    """YoY comparison for a single metric"""
    metric_name: str
    current_period: str
    current_value: Optional[float] = None
    prior_period: Optional[str] = None
    prior_value: Optional[float] = None
    change_amount: Optional[float] = None
    change_pct: Optional[float] = None
    narrative: Optional[str] = None
    unit: Optional[str] = None  # "MM", "K", "x", "%"

class PeriodMetrics(BaseModel):
    """All metrics for a single period"""
    period_name: str
    revenue: Optional[float] = None
    cost_of_sales: Optional[float] = None
    sga_expense: Optional[float] = None
    operating_income: Optional[float] = None
    ebitda: Optional[float] = None
    ebitda_margin: Optional[float] = None
    net_income: Optional[float] = None

    total_debt: Optional[float] = None
    leverage: Optional[float] = None
    cash: Optional[float] = None

    cfo: Optional[float] = None
    capex: Optional[float] = None
    free_cash_flow: Optional[float] = None
    fcc: Optional[float] = None

    revolver_facility_size: Optional[float] = None
    revolver_borrowings: Optional[float] = None
    revolver_availability: Optional[float] = None

    altman_z_score: Optional[float] = None
    pd_score: Optional[int] = None  # 1-12 scale
    credit_rating: Optional[str] = None  # Investment Grade, Speculative, etc.

class StructuredMetrics(BaseModel):
    """Structured metrics organized by period and with comparisons"""
    statement_basis: str
    periods: List[PeriodMetrics] = Field(default_factory=list)
    yoy_comparisons: List[MetricComparison] = Field(default_factory=list)

# ============================================================
# Updated Response Schemas
# ============================================================

class UnderwriteCreateResponse(BaseModel):
    run_id: str
    build: Optional[str] = None
    status: Optional[Literal["queued", "running", "completed", "failed"]] = None
    completeness: Optional[float] = None
    validation_flags: Optional[List[str]] = None
    memo_markdown: Optional[str] = None
    structured_metrics: Optional[StructuredMetrics] = None  # NEW
    extracted: Optional[ExtractionResult] = None
    mda_summary: Optional[str] = None
    cache_bypass: Optional[bool] = None
    excerpt_preview: Optional[str] = None
    model_raw_preview: Optional[str] = None

class UnderwriteStatusResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    error: Optional[str] = None
    extracted: Optional[ExtractionResult] = None
    memo_markdown: Optional[str] = None
    structured_metrics: Optional[StructuredMetrics] = None  # NEW
    completeness: Optional[float] = None
    validation_flags: Optional[List[str]] = None