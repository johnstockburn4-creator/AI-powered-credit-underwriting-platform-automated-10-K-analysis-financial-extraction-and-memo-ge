"""
profiles_api.py — FastAPI routes for company extraction profiles and rules.

Add to main.py:
    from profiles_api import profiles_router
    app.include_router(profiles_router)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import logging
import profiles as ps

logger = logging.getLogger("credit_ai")

profiles_router = APIRouter(prefix="/v1/profiles", tags=["profiles"])


# ── Models ────────────────────────────────────────────────────────────────────

class CorrectionRequest(BaseModel):
    field_name: str
    value: float
    period_name: Optional[str] = None
    note: Optional[str] = None
    source_hint: Optional[str] = None


class BulkCorrectionRequest(BaseModel):
    corrections: list[CorrectionRequest]


class RuleRequest(BaseModel):
    field_name: str
    rule_type: str                        # "label" | "section" | "exclusion"
    label: Optional[str] = None          # exact line label to match
    section_hint: Optional[str] = None   # section/note header to search within
    exclude_labels: Optional[list[str]] = None  # labels to ignore
    note: Optional[str] = None


class DeleteFieldRequest(BaseModel):
    field_name: str
    period_name: Optional[str] = None


# ── Profile routes ────────────────────────────────────────────────────────────

@profiles_router.get("/")
def list_companies():
    return {"companies": ps.list_companies()}


@profiles_router.get("/{company_name}")
def get_company_profile(company_name: str):
    profile = ps.get_profile(company_name)
    return {
        "company": company_name,
        "profile": profile,
        "exists": bool(profile),
        "rules": profile.get("_rules", {}),
    }


@profiles_router.delete("/{company_name}")
def delete_company_profile(company_name: str):
    ok = ps.delete_profile(company_name)
    if not ok:
        raise HTTPException(status_code=404, detail="Company profile not found")
    return {"status": "deleted", "company": company_name}


# ── Correction routes ─────────────────────────────────────────────────────────

@profiles_router.post("/{company_name}/corrections")
def save_correction(company_name: str, body: CorrectionRequest):
    ok = ps.save_correction(
        company_name=company_name,
        field_name=body.field_name,
        value=body.value,
        period_name=body.period_name,
        note=body.note,
        source_hint=body.source_hint,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save correction")
    return {"status": "saved", "company": company_name, "field": body.field_name}


@profiles_router.post("/{company_name}/corrections/bulk")
def save_bulk_corrections(company_name: str, body: BulkCorrectionRequest):
    corrections = [c.model_dump() for c in body.corrections]
    ok = ps.save_corrections_bulk(company_name=company_name, corrections=corrections)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save corrections")
    return {"status": "saved", "company": company_name, "count": len(corrections)}


@profiles_router.delete("/{company_name}/field")
def delete_field(company_name: str, body: DeleteFieldRequest):
    ok = ps.delete_field(
        company_name=company_name,
        field_name=body.field_name,
        period_name=body.period_name,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Field not found in profile")
    return {"status": "deleted", "company": company_name, "field": body.field_name}


# ── Rule routes ───────────────────────────────────────────────────────────────

@profiles_router.post("/{company_name}/rules")
def save_rule(company_name: str, body: RuleRequest):
    """
    Save an extraction rule for a field.
    This tells the extractor HOW to find the correct value in future filings.
    """
    rule = {
        "rule_type": body.rule_type,
    }
    if body.label:          rule["label"]          = body.label
    if body.section_hint:   rule["section_hint"]   = body.section_hint
    if body.exclude_labels: rule["exclude_labels"]  = body.exclude_labels
    if body.note:           rule["note"]            = body.note

    ok = ps.save_rule(
        company_name=company_name,
        field_name=body.field_name,
        rule=rule,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save rule")
    return {
        "status": "saved",
        "company": company_name,
        "field": body.field_name,
        "rule": rule,
    }


@profiles_router.get("/{company_name}/rules")
def get_rules(company_name: str):
    rules = ps.get_rules(company_name)
    return {"company": company_name, "rules": rules}


@profiles_router.delete("/{company_name}/rules/{field_name}")
def delete_rule(company_name: str, field_name: str):
    ok = ps.delete_rule(company_name=company_name, field_name=field_name)
    if not ok:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"status": "deleted", "company": company_name, "field": field_name}


# ── Context route — what did the extractor find and where ────────────────────

@profiles_router.post("/{company_name}/context")
async def get_field_context(
    company_name: str,
    run_id: str,
    field_name: str,
    extracted_value: Optional[float] = None,
):
    """
    Returns the document context around where a field value was extracted.
    Used by the Review & Correct UI to show "what the extractor found".

    NOTE: RunStore does not persist raw filing text (and shouldn't — filings can
    be tens of MB and contain sensitive borrower data). Context is instead read
    from the per-field notes that extract_financials_from_text() already writes
    onto each period at extraction time, in the form:
        "CONTEXT:{field_name}:{json_blob}"
    This is the same source the ReviewCorrect.jsx frontend reads directly off
    `extracted.periods[].notes` — this endpoint exists for callers that only
    have a run_id and don't already hold the extracted payload client-side.
    """
    try:
        from run_store import RunStore
        store = RunStore()
        run = store.get_run(run_id)
        if not run or not run.extracted_json:
            return {"context": None, "message": "Run not found or not completed"}

        periods = (run.extracted_json or {}).get("periods") or []
        prefix = f"CONTEXT:{field_name}:"

        for period in periods:
            for note in period.get("notes") or []:
                if isinstance(note, str) and note.startswith(prefix):
                    raw = note[len(prefix):]
                    try:
                        context = json.loads(raw)
                    except json.JSONDecodeError:
                        context = {"raw": raw, "formatted": None, "matched_line": None, "score": None}
                    return {
                        "company": company_name,
                        "field": field_name,
                        "period": period.get("period_name"),
                        "extracted_value": extracted_value,
                        "context": context,
                    }

        return {
            "context": None,
            "message": f"No stored extraction context found for '{field_name}' on run {run_id}.",
        }
    except Exception as e:
        logger.error(f"get_field_context failed for run={run_id} field={field_name}: {e}")
        return {"context": None, "message": str(e)}
