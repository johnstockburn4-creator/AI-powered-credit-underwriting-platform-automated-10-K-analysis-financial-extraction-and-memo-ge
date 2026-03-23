import io
import time
import uuid
import logging
import re
from json import JSONDecodeError
from typing import Optional

from dotenv import load_dotenv
from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from bs4 import BeautifulSoup

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from openai import RateLimitError, APIConnectionError, APIStatusError

from schemas import (
    ExtractionResult,
    BorrowerProfile,
    CovenantSet,
)

from agents import (
    extract_financials_from_text,
    validate_and_compute,
    generate_underwriting_memo,
    generate_financial_summary_memo,
)

from run_store import RunStore, compute_completeness

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit_ai")

app = FastAPI(title="Credit AI MVP Backend")

APP_BUILD = "analyze_memo_v7_2026-03-22_HTML_FIX"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # increased to 30MB for large HTM filings
QUALITY_MIN_COMPLETENESS = 0.55

store = RunStore()


@app.middleware("http")
async def add_request_id_and_timing(request, call_next):
    request_id = str(uuid.uuid4())
    start = time.time()
    response = await call_next(request)
    elapsed_ms = int((time.time() - start) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = str(elapsed_ms)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "build": APP_BUILD}


@app.get("/debug/routes")
def debug_routes():
    out = []
    for r in app.routes:
        try:
            methods = sorted(list(getattr(r, "methods", []) or []))
            out.append({"path": getattr(r, "path", None), "name": getattr(r, "name", None), "methods": methods})
        except Exception:
            pass
    return {"build": APP_BUILD, "routes": out}


# ============================================================
# FILE PARSING UTILITIES
# ============================================================

def _pdf_to_text_basic(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _pdf_to_text_with_tables_if_available(file_bytes: bytes) -> str:
    text = _pdf_to_text_basic(file_bytes)
    try:
        import pdfplumber
    except Exception:
        return text

    table_lines = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for t in tables:
                    for row in t:
                        if not row:
                            continue
                        cells = []
                        for c in row:
                            if c is None:
                                continue
                            s = " ".join(str(c).split())
                            if s:
                                cells.append(s)
                        if cells:
                            table_lines.append(" | ".join(cells))
                    table_lines.append("")
    except Exception:
        return text

    if table_lines:
        return text + "\n\nTABLES:\n" + "\n".join(table_lines)
    return text


def _html_to_text_preserve_tables(file_bytes: bytes) -> str:
    """
    Convert HTML/XBRL filing to plain text + pipe-delimited tables.

    KEY FIX: SEC inline XBRL filings (iXBRL) often have the ENTIRE document
    as 7 lines of HTML because every element is on one giant line.
    BeautifulSoup handles this fine but the resulting text needs
    deduplication — table-of-contents entries appear as plain text AND
    as table rows, causing anchors like "Item 8. Financial Statements"
    to be found at page 1 (TOC) instead of page 89 (actual section).

    We solve this by:
    1. Extracting body text normally
    2. Appending tables separately tagged as TABLES:
    3. agents.py _select_relevant_statement_text then skips the TOC
       by finding the LAST occurrence of key anchors, not the first.
    """
    raw_html = file_bytes.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "noscript", "head"]):
        tag.decompose()

    # Get body text
    body_text = soup.get_text(separator="\n")
    body_text = "\n".join(line.strip() for line in body_text.splitlines() if line.strip())

    # Extract tables separately
    table_lines = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = []
            for cell in tr.find_all(["th", "td"]):
                txt = cell.get_text(" ", strip=True)
                if txt:
                    txt = " ".join(txt.split())
                    cells.append(txt)
            if cells:
                table_lines.append(" | ".join(cells))
        table_lines.append("")

    combined = body_text
    if table_lines:
        combined += "\n\nTABLES:\n" + "\n".join(table_lines)

    logger.info(
        "HTML→text: body=%d chars, table_lines=%d, total=%d chars",
        len(body_text), len(table_lines), len(combined)
    )
    return combined


def file_to_text(file_bytes: bytes, filename: str) -> str:
    head = (file_bytes[:2048] or b"").lstrip()
    head_lower = head.lower()

    is_pdf = head.startswith(b"%PDF-")
    looks_like_html = (
        head_lower.startswith(b"<!doctype")
        or head_lower.startswith(b"<html")
        or b"<html" in head_lower
        or b"<head" in head_lower
        or b"<table" in head_lower
        or b"<?xml" in head_lower          # catches iXBRL files that start with <?xml
    )

    if is_pdf:
        try:
            return _pdf_to_text_with_tables_if_available(file_bytes)
        except PdfReadError:
            raise HTTPException(status_code=400, detail="Uploaded PDF could not be read (corrupt or incomplete).")
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file looks like a PDF but could not be parsed.")

    if looks_like_html or (filename or "").lower().endswith((".htm", ".html")):
        return _html_to_text_preserve_tables(file_bytes)

    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF or HTML.")


def _has_financial_amounts(text: str) -> bool:
    if not text:
        return False
    amount_re = re.compile(
        r"(\(\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*\))|(\d{1,3}(?:,\d{3})+(?:\.\d+)?)|(\d+\.\d+)"
    )
    return amount_re.search(text) is not None


def _soft_gate_header(completeness: float, min_required: float) -> str:
    return (
        "## EXTRACTION QUALITY WARNING (SOFT GATE)\n"
        f"- Extraction completeness score: {completeness:.2f}\n"
        f"- Minimum target score: {min_required:.2f}\n"
        "- Interpretation: The memo below is generated from partially complete extracted financials.\n"
        "- Action: Treat quantitative conclusions as provisional; review source statements and address data gaps.\n\n"
    )


def _cache_is_usable(extracted: ExtractionResult) -> bool:
    if not extracted.periods:
        return False
    if len(extracted.periods) >= 2:
        return True

    p = extracted.periods[0]
    is_ = p.income_statement or {}
    bs = p.balance_sheet or {}
    cf = p.cash_flow or {}

    must_have_any = [
        is_.get("revenue"),
        is_.get("operating_income"),
        is_.get("depreciation_amortization"),
        bs.get("total_debt"),
        cf.get("cfo"),
        cf.get("capex"),
    ]
    missing = sum(1 for v in must_have_any if v is None)
    return missing <= 2


def _generate_memo(
    extracted: ExtractionResult,
    borrower_name: Optional[str],
    industry: Optional[str],
    facility_type: Optional[str],
    use_of_proceeds: Optional[str],
    max_total_leverage: Optional[float],
    min_fcc: Optional[float],
    mda_summary: Optional[str] = "",
) -> str:
    """Generate memo — uses full underwriting memo if borrower data provided."""
    has_borrower_data = bool(borrower_name or industry or facility_type or use_of_proceeds)

    if has_borrower_data:
        borrower = BorrowerProfile(
            name=borrower_name or "[Company Name - To Be Determined]",
            industry=industry or "[Industry - To Be Determined]",
            facility_type=facility_type or "[Facility Type - To Be Determined]",
            use_of_proceeds=use_of_proceeds or "[Use of Proceeds - To Be Determined]",
        )

        covenants = CovenantSet(
            max_total_leverage=max_total_leverage,
            min_fcc=min_fcc,
            min_dscr=None,
        )

        logger.info("Generating full underwriting memo with borrower context and MD&A insights")
        return generate_underwriting_memo(borrower, covenants, extracted, mda_summary or "")
    else:
        logger.info("Generating baseline financial summary (no borrower context)")
        return generate_financial_summary_memo(extracted)


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/v1/analyze")
async def analyze_single(
    file: UploadFile = File(...),
    include_extracted: bool = Form(False),
    include_previews: bool = Form(False),
    cache_bypass: bool = Form(False),
    # Optional borrower context
    borrower_name: Optional[str] = Form(None),
    industry: Optional[str] = Form(None),
    facility_type: Optional[str] = Form(None),
    use_of_proceeds: Optional[str] = Form(None),
    max_total_leverage: Optional[float] = Form(None),
    min_fcc: Optional[float] = Form(None),
):
    """
    Upload file -> strip HTML -> extract financials -> compute metrics -> generate memo.
    Always returns: memo_markdown + extracted (full JSON) + mda_summary.
    """
    run_id: Optional[str] = None

    try:
        file_bytes = await file.read()
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large. Max 30MB.")

        run_id = str(uuid.uuid4())
        file_sha = store.create_run(run_id, file.filename, file_bytes)

        cached = None if cache_bypass else store.find_completed_by_file_sha(file_sha)

        # ── Cache path ───────────────────────────────────────────────────
        if cached and cached.extracted_json:
            extracted = ExtractionResult.model_validate(cached.extracted_json)
            extracted = validate_and_compute(extracted)

            if _cache_is_usable(extracted):
                memo_markdown = _generate_memo(
                    extracted, borrower_name, industry, facility_type,
                    use_of_proceeds, max_total_leverage, min_fcc,
                    "",  # No MD&A for cached results — re-run with cache_bypass=true to get MD&A
                )

                extracted_json = extracted.model_dump(mode="json", exclude_none=False)
                completeness = compute_completeness(extracted_json)
                flags = list(extracted.validation_flags or [])

                if completeness < QUALITY_MIN_COMPLETENESS:
                    memo_markdown = _soft_gate_header(float(completeness), QUALITY_MIN_COMPLETENESS) + memo_markdown

                store.set_completed(
                    run_id=run_id,
                    extracted_json=extracted_json,
                    validation_flags=flags,
                    completeness=float(completeness),
                    excerpt_preview=cached.excerpt_preview,
                    model_raw_preview=cached.model_raw_preview,
                    memo_markdown=memo_markdown,
                    model_name=cached.model_name,
                    prompt_version=cached.prompt_version,
                )

                resp = {
                    "run_id": run_id,
                    "status": "completed",
                    "build": APP_BUILD,
                    "completeness": float(completeness),
                    "validation_flags": flags,
                    "memo_markdown": memo_markdown,
                    "extracted": extracted_json,
                    "mda_summary": "(cached run — resubmit with cache_bypass=true to get fresh MD&A)",
                    "cache_bypass": bool(cache_bypass),
                }
                if include_previews:
                    resp["excerpt_preview"] = cached.excerpt_preview
                    resp["model_raw_preview"] = cached.model_raw_preview

                return JSONResponse(resp)

            logger.info("Cache unusable (periods=%d). Re-extracting run=%s", len(extracted.periods or []), run_id)

        # ── Fresh extraction path ────────────────────────────────────────
        store.set_running(run_id)

        text = file_to_text(file_bytes, file.filename)

        logger.info(
            "analyze(fresh) run=%s filename=%s text_len=%d tables_present=%s cache_bypass=%s",
            run_id,
            file.filename,
            len(text),
            ("TABLES:" in text),
            cache_bypass,
        )

        if not text.strip():
            store.set_failed(run_id, "No text extracted. If scanned PDF, OCR is required.")
            raise HTTPException(status_code=400, detail="No text extracted. If scanned PDF, OCR is required.")

        if not _has_financial_amounts(text):
            store.set_failed(run_id, "No financial-amount patterns detected. OCR/table extraction likely required.")
            raise HTTPException(
                status_code=400,
                detail="No financial-amount patterns detected. OCR/table extraction likely required.",
            )

        # agents.py handles HTML stripping internally as a second safety net,
        # but file_to_text above already converts HTM to plain text + tables.
        extracted, excerpt, raw_model, mda_summary = extract_financials_from_text(text)
        extracted = validate_and_compute(extracted)

        memo_markdown = _generate_memo(
            extracted, borrower_name, industry, facility_type,
            use_of_proceeds, max_total_leverage, min_fcc, mda_summary,
        )

        extracted_json = extracted.model_dump(mode="json", exclude_none=False)
        completeness = compute_completeness(extracted_json)
        flags = list(extracted.validation_flags or [])

        excerpt_preview = (excerpt or "")[:50000]
        raw_preview = (raw_model or "")[:50000]

        if completeness < QUALITY_MIN_COMPLETENESS:
            memo_markdown = _soft_gate_header(float(completeness), QUALITY_MIN_COMPLETENESS) + memo_markdown

        store.set_completed(
            run_id=run_id,
            extracted_json=extracted_json,
            validation_flags=flags,
            completeness=float(completeness),
            excerpt_preview=excerpt_preview,
            model_raw_preview=raw_preview,
            memo_markdown=memo_markdown,
            model_name="gpt-4o-mini",
            prompt_version="extractor_v2",
        )

        resp = {
            "run_id": run_id,
            "status": "completed",
            "build": APP_BUILD,
            "completeness": float(completeness),
            "validation_flags": flags,
            "memo_markdown": memo_markdown,
            "extracted": extracted_json,
            # mda_summary is now ALWAYS returned (not gated behind include_previews)
            "mda_summary": mda_summary,
            "cache_bypass": bool(cache_bypass),
        }
        if include_previews:
            resp["excerpt_preview"] = excerpt_preview
            resp["model_raw_preview"] = raw_preview

        return JSONResponse(resp)

    except RateLimitError:
        if run_id:
            store.set_failed(run_id, "OpenAI rate limit/quota exceeded.")
        raise HTTPException(status_code=429, detail="OpenAI quota exceeded or rate limit hit.")
    except APIConnectionError:
        if run_id:
            store.set_failed(run_id, "OpenAI API connection error.")
        raise HTTPException(status_code=503, detail="Could not connect to OpenAI API.")
    except APIStatusError as e:
        if run_id:
            store.set_failed(run_id, f"OpenAI API error: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {e}")
    except ValidationError as e:
        if run_id:
            store.set_failed(run_id, f"Schema validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Schema validation failed: {e}")
    except Exception as e:
        logger.exception("Unexpected error in analyze_single")
        if run_id:
            store.set_failed(run_id, f"Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ============================================================
# EXCEL EXPORT ENDPOINT
# ============================================================

@app.post("/v1/export/excel")
async def export_to_excel(
    file: UploadFile = File(...),
    borrower_name: Optional[str] = Form(None),
    industry: Optional[str] = Form(None),
    facility_type: Optional[str] = Form(None),
    use_of_proceeds: Optional[str] = Form(None),
    max_total_leverage: Optional[float] = Form(None),
    min_fcc: Optional[float] = Form(None),
):
    """Analyze financials and export to Excel file."""
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    import tempfile
    import os

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max 30MB.")

    run_id = str(uuid.uuid4())
    file_sha = store.create_run(run_id, file.filename, file_bytes)

    store.set_running(run_id)

    text = file_to_text(file_bytes, file.filename)

    if not text.strip():
        store.set_failed(run_id, "No text extracted.")
        raise HTTPException(status_code=400, detail="No text extracted.")

    extracted, excerpt, raw_model, mda_summary = extract_financials_from_text(text)
    extracted = validate_and_compute(extracted)

    # Create Excel workbook
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    summary_sheet = wb.create_sheet("Summary")
    income_stmt_sheet = wb.create_sheet("Income Statement")
    balance_sheet_sheet = wb.create_sheet("Balance Sheet")
    cash_flow_sheet = wb.create_sheet("Cash Flow")
    metrics_sheet = wb.create_sheet("Key Metrics")

    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    subheader_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    subheader_font = Font(bold=True, size=10)

    # Summary Sheet
    summary_sheet['A1'] = 'Credit Analysis Summary'
    summary_sheet['A1'].font = Font(bold=True, size=14)

    row = 3
    summary_sheet[f'A{row}'] = 'Borrower Information'
    summary_sheet[f'A{row}'].font = subheader_font
    summary_sheet[f'A{row}'].fill = subheader_fill
    row += 1
    summary_sheet[f'A{row}'] = 'Company Name:'
    summary_sheet[f'B{row}'] = borrower_name or "N/A"
    row += 1
    summary_sheet[f'A{row}'] = 'Industry:'
    summary_sheet[f'B{row}'] = industry or "N/A"
    row += 1
    summary_sheet[f'A{row}'] = 'Facility Type:'
    summary_sheet[f'B{row}'] = facility_type or "N/A"
    row += 2

    summary_sheet[f'A{row}'] = 'Analysis Quality'
    summary_sheet[f'A{row}'].font = subheader_font
    summary_sheet[f'A{row}'].fill = subheader_fill
    row += 1

    completeness = compute_completeness(extracted.model_dump(mode="json", exclude_none=False))
    summary_sheet[f'A{row}'] = 'Completeness Score:'
    summary_sheet[f'B{row}'] = f"{completeness*100:.0f}%"
    row += 1
    summary_sheet[f'A{row}'] = 'Periods Analyzed:'
    summary_sheet[f'B{row}'] = len(extracted.periods)

    periods = extracted.periods
    period_names = [p.period_name for p in periods]

    # Income Statement Sheet
    income_stmt_sheet['A1'] = 'Income Statement'
    income_stmt_sheet['A1'].font = Font(bold=True, size=14)
    income_stmt_sheet['A3'] = 'Line Item'
    income_stmt_sheet['A3'].font = header_font
    income_stmt_sheet['A3'].fill = header_fill
    for i, pname in enumerate(period_names):
        cell = income_stmt_sheet.cell(row=3, column=i+2)
        cell.value = pname
        cell.font = header_font
        cell.fill = header_fill

    is_items = [
        ('Revenue', 'revenue'),
        ('Cost of Sales', 'cost_of_sales'),
        ('Gross Profit', None),
        ('SG&A Expense', 'sga_expense'),
        ('Operating Income', 'operating_income'),
        ('EBITDA', 'ebitda'),
        ('Depreciation & Amortization', 'depreciation_amortization'),
        ('Interest Expense', 'interest_expense'),
        ('Income Tax Expense', 'income_tax_expense'),
        ('Net Income', 'net_income'),
    ]

    row = 4
    for label, field in is_items:
        income_stmt_sheet[f'A{row}'] = label
        for i, period in enumerate(periods):
            is_ = period.income_statement or {}
            if field:
                value = is_.get(field)
            else:
                rev = is_.get('revenue')
                cogs = is_.get('cost_of_sales')
                value = rev - cogs if (rev is not None and cogs is not None) else None
            cell = income_stmt_sheet.cell(row=row, column=i+2)
            if value is not None:
                cell.value = value
                cell.number_format = '#,##0'
        row += 1

    # Balance Sheet
    balance_sheet_sheet['A1'] = 'Balance Sheet'
    balance_sheet_sheet['A1'].font = Font(bold=True, size=14)
    balance_sheet_sheet['A3'] = 'Line Item'
    balance_sheet_sheet['A3'].font = header_font
    balance_sheet_sheet['A3'].fill = header_fill
    for i, pname in enumerate(period_names):
        cell = balance_sheet_sheet.cell(row=3, column=i+2)
        cell.value = pname
        cell.font = header_font
        cell.fill = header_fill

    bs_items = [
        ('Cash & Equivalents', 'cash'),
        ('Total Assets', 'total_assets'),
        ('Total Liabilities', 'total_liabilities'),
        ('Total Equity', 'total_equity'),
        ('Total Debt', 'total_debt'),
        ('Long-term Debt', 'long_term_debt'),
        ('Current Portion LT Debt', 'current_portion_long_term_debt'),
    ]

    row = 4
    for label, field in bs_items:
        balance_sheet_sheet[f'A{row}'] = label
        for i, period in enumerate(periods):
            bs = period.balance_sheet or {}
            value = bs.get(field)
            cell = balance_sheet_sheet.cell(row=row, column=i+2)
            if value is not None:
                cell.value = value
                cell.number_format = '#,##0'
        row += 1

    # Cash Flow
    cash_flow_sheet['A1'] = 'Cash Flow Statement'
    cash_flow_sheet['A1'].font = Font(bold=True, size=14)
    cash_flow_sheet['A3'] = 'Line Item'
    cash_flow_sheet['A3'].font = header_font
    cash_flow_sheet['A3'].fill = header_fill
    for i, pname in enumerate(period_names):
        cell = cash_flow_sheet.cell(row=3, column=i+2)
        cell.value = pname
        cell.font = header_font
        cell.fill = header_fill

    cf_items = [
        ('Operating Cash Flow', 'cfo'),
        ('Investing Cash Flow', 'cfi'),
        ('Financing Cash Flow', 'cff'),
        ('Capex', 'capex'),
        ('Cash Interest Paid', 'cash_paid_for_interest'),
        ('Cash Taxes Paid', 'cash_paid_for_income_taxes'),
        ('Dividends Paid', 'dividends_distributions_paid'),
    ]

    row = 4
    for label, field in cf_items:
        cash_flow_sheet[f'A{row}'] = label
        for i, period in enumerate(periods):
            cf = period.cash_flow or {}
            value = cf.get(field)
            cell = cash_flow_sheet.cell(row=row, column=i+2)
            if value is not None:
                cell.value = value
                cell.number_format = '#,##0'
        row += 1

    # Key Metrics
    metrics_sheet['A1'] = 'Key Credit Metrics'
    metrics_sheet['A1'].font = Font(bold=True, size=14)
    metrics_sheet['A3'] = 'Metric'
    metrics_sheet['A3'].font = header_font
    metrics_sheet['A3'].fill = header_fill
    for i, pname in enumerate(period_names):
        cell = metrics_sheet.cell(row=3, column=i+2)
        cell.value = pname
        cell.font = header_font
        cell.fill = header_fill

    metric_items = [
        ('EBITDA', 'ebitda_computed', '#,##0'),
        ('EBITDA Margin', 'ebitda_margin', '0.0%'),
        ('Gross Margin', None, '0.0%'),
        ('Free Cash Flow', 'free_cash_flow', '#,##0'),
        ('Leverage (Debt/EBITDA)', 'leverage_total_debt_to_ebitda', '0.00"x"'),
        ('FCC', 'fcc', '0.00"x"'),
        ('FCC Numerator', 'fcc_numerator', '#,##0'),
        ('FCC Denominator', 'fcc_denominator', '#,##0'),
    ]

    row = 4
    for label, field, fmt in metric_items:
        metrics_sheet[f'A{row}'] = label
        for i, period in enumerate(periods):
            dm = period.derived_metrics or {}
            is_ = period.income_statement or {}
            if field:
                value = dm.get(field)
            else:
                rev = is_.get('revenue')
                cogs = is_.get('cost_of_sales')
                value = (rev - cogs) / rev if (rev and cogs and rev > 0) else None
            cell = metrics_sheet.cell(row=row, column=i+2)
            if value is not None:
                cell.value = value
                cell.number_format = fmt
        row += 1

    if max_total_leverage or min_fcc:
        row += 2
        metrics_sheet[f'A{row}'] = 'Covenant Compliance'
        metrics_sheet[f'A{row}'].font = subheader_font
        metrics_sheet[f'A{row}'].fill = subheader_fill
        row += 1
        if max_total_leverage:
            metrics_sheet[f'A{row}'] = 'Max Leverage Covenant:'
            metrics_sheet[f'B{row}'] = f"{max_total_leverage}x"
            row += 1
        if min_fcc:
            metrics_sheet[f'A{row}'] = 'Min FCC Covenant:'
            metrics_sheet[f'B{row}'] = f"{min_fcc}x"

    for sheet in [summary_sheet, income_stmt_sheet, balance_sheet_sheet, cash_flow_sheet, metrics_sheet]:
        for column in sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except Exception:
                    pass
            sheet.column_dimensions[column_letter].width = min(max_length + 2, 50)

    temp_dir = tempfile.gettempdir()
    filename = f"{borrower_name or 'company'}_{run_id[:8]}_financials.xlsx".replace(" ", "_")
    filepath = os.path.join(temp_dir, filename)
    wb.save(filepath)

    store.set_completed(
        run_id=run_id,
        extracted_json=extracted.model_dump(mode="json", exclude_none=False),
        validation_flags=list(extracted.validation_flags or []),
        completeness=float(completeness),
        excerpt_preview=excerpt[:50000],
        model_raw_preview=raw_model[:50000],
        memo_markdown="Excel export",
        model_name="gpt-4o-mini",
        prompt_version="extractor_v2",
    )

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )