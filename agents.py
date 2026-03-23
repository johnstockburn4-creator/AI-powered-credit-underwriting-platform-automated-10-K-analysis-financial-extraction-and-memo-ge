from __future__ import annotations

import os
import re
import json
import html
import logging
from typing import Tuple, Optional

from openai import OpenAI
import anthropic

from schemas import ExtractionResult, BorrowerProfile, CovenantSet

from metrics import (
    compute_ebitda,
    compute_free_cash_flow,
    compute_fcc,
    compute_leverage,
    compute_ebitda_margin,
    abs_outflow,
)

from credit_scoring import (
    compute_altman_z_score,
    compute_pd_score,
)

logger = logging.getLogger("credit_ai")

_client = None
_anthropic_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing.")
        _client = OpenAI()
    return _client


def get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is missing.")
        _anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _anthropic_client


def _strip_html_to_text(raw: str) -> str:
    if not raw or not raw.strip():
        return raw
    if "<html" not in raw.lower() and "<span" not in raw.lower():
        return raw
    logger.info(f"HTML stripping: input {len(raw):,} characters")
    text = raw
    text = re.sub(r'<head[\s\S]*?</head>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<ix:header[\s\S]*?</ix:header>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<script[\s\S]*?</script>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<style[\s\S]*?</style>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'</tr\s*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</td\s*>|</th\s*>', '\t', text, flags=re.IGNORECASE)
    block_tags = r'</?(p|div|section|article|header|footer|h[1-6]|li|ul|ol|blockquote|pre)'
    text = re.sub(block_tags + r'[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    entity_map = {
        '&#8217;': "'", '&#8216;': "'", '&#8220;': '"', '&#8221;': '"',
        '&#8211;': '\u2013', '&#8212;': '\u2014', '&#8203;': '', '&#160;': ' ',
        '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&quot;': '"', '&#x2019;': "'", '&#x2018;': "'",
        '&#x201C;': '"', '&#x201D;': '"', '&#x2013;': '\u2013', '&#x2014;': '\u2014',
    }
    for entity, replacement in entity_map.items():
        text = text.replace(entity, replacement)
    text = html.unescape(text)
    text = text.replace('\t', ' ')
    text = re.sub(r'[ ]{2,}', ' ', text)
    lines = [line.rstrip() for line in text.splitlines()]
    clean_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                clean_lines.append('')
        else:
            blank_count = 0
            clean_lines.append(line)
    text = '\n'.join(clean_lines).strip()
    logger.info(f"HTML stripping: output {len(text):,} characters")
    return text


EXPECTED_INCOME_KEYS = [
    "revenue", "cost_of_sales", "sga_expense", "operating_income", "ebitda",
    "net_income", "interest_expense", "income_tax_expense",
    "depreciation_amortization", "rent_expense",
]

EXPECTED_BS_KEYS = [
    "cash", "total_assets", "total_liabilities", "total_equity", "total_debt",
    "long_term_debt", "current_portion_long_term_debt",
    "revolver_facility_size", "revolver_borrowings", "revolver_availability",
]

EXPECTED_CF_KEYS = [
    "cfo", "cfi", "cff", "capex", "cash_paid_for_interest",
    "cash_paid_for_income_taxes", "dividends_distributions_paid",
]

EXTRACTOR_SYSTEM_PROMPT = """
You are a financial statement extraction assistant for a commercial credit analyst.
Return ONLY valid JSON. No extra commentary.
Hard rules:
- The output MUST match the provided schema exactly.
- For each period object, each line item MUST be a single scalar number or null.
- NEVER put multiple years, multiple numbers, ranges, or commentary into a numeric field.
- If you cannot confidently map a number to a field for a given period, set it to null.
- Do not compute derived metrics; derived_metrics must be {}.
- Do not compute EBITDA; leave ebitda null unless explicitly labeled as EBITDA.
- Prioritize: operating_income, depreciation_amortization, total_debt,
  current_portion_long_term_debt, cash_paid_for_interest, cash_paid_for_income_taxes, capex, cfo.
"""

EXTRACTOR_SCHEMA_PROMPT = """
Return JSON with this exact structure:
{
  "statement_basis": "actual" | "thousands" | "millions",
  "periods": [
    {
      "period_name": "FY2024",
      "income_statement": {
        "revenue": number|null, "operating_income": number|null, "ebitda": number|null,
        "cost_of_sales": number|null, "sga_expense": number|null, "net_income": number|null,
        "interest_expense": number|null, "income_tax_expense": number|null,
        "depreciation_amortization": number|null, "rent_expense": number|null
      },
      "balance_sheet": {
        "cash": number|null, "total_assets": number|null, "total_liabilities": number|null,
        "total_equity": number|null, "total_debt": number|null, "long_term_debt": number|null,
        "current_portion_long_term_debt": number|null, "revolver_facility_size": number|null,
        "revolver_borrowings": number|null, "revolver_availability": number|null
      },
      "cash_flow": {
        "cfo": number|null, "cfi": number|null, "cff": number|null, "capex": number|null,
        "cash_paid_for_interest": number|null, "cash_paid_for_income_taxes": number|null,
        "dividends_distributions_paid": number|null
      },
      "derived_metrics": {},
      "notes": []
    }
  ],
  "validation_flags": []
}
"""


def _coerce_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.lstrip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip("\n").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if "|" in s or "\n" in s:
        return None
    years = re.findall(r"\b20\d{2}\b", s)
    if len(set(years)) >= 2:
        return None
    s = s.replace("$", "").replace("€", "").replace("£", "")
    s = re.sub(r"\[\d+\]$", "", s).strip()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    s = s.replace(",", "")
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        return None
    x = float(s)
    return -x if neg else x


def _clean_statement_dict(d: dict, expected_keys: list) -> dict:
    out = {}
    for k in expected_keys:
        out[k] = _to_float(d.get(k)) if isinstance(d, dict) else None
    return out


def _normalize_payload(data: dict) -> dict:
    if not isinstance(data, dict):
        return {"statement_basis": "actual", "periods": [], "validation_flags": ["Top-level JSON was not an object."]}
    sb = data.get("statement_basis")
    if sb not in ("actual", "thousands", "millions"):
        sb = "actual"
    periods = data.get("periods")
    if not isinstance(periods, list):
        periods = []
    validation_flags = data.get("validation_flags")
    if not isinstance(validation_flags, list):
        validation_flags = []
    cleaned_periods = []
    for i, p in enumerate(periods[:5]):
        if not isinstance(p, dict):
            continue
        period_name = p.get("period_name") or f"PERIOD_{i+1}"
        is_raw = p.get("income_statement") if isinstance(p.get("income_statement"), dict) else {}
        bs_raw = p.get("balance_sheet") if isinstance(p.get("balance_sheet"), dict) else {}
        cf_raw = p.get("cash_flow") if isinstance(p.get("cash_flow"), dict) else {}
        cleaned = {
            "period_name": str(period_name),
            "income_statement": _clean_statement_dict(is_raw, EXPECTED_INCOME_KEYS),
            "balance_sheet": _clean_statement_dict(bs_raw, EXPECTED_BS_KEYS),
            "cash_flow": _clean_statement_dict(cf_raw, EXPECTED_CF_KEYS),
            "derived_metrics": {},
            "notes": p.get("notes") if isinstance(p.get("notes"), list) else [],
        }
        cleaned_periods.append(cleaned)
    return {
        "statement_basis": sb,
        "periods": cleaned_periods,
        "validation_flags": validation_flags,
        "schema_version": "1.0",
        "extractor_version": "2026-01-27",
    }


# ============================================================
# MD&A SECTION EXTRACTION  (TOC-aware)
# ============================================================

def _extract_mda_section(raw_text: str, max_chars: int = 350000) -> str:
    """
    Locate the real MD&A section, skipping Table of Contents hits.

    ROOT CAUSE OF PREVIOUS FAILURE:
    The filing's TOC near the top contains lines like:
        "Results of Operations  88"
        "Item 7. Management Discussion and Analysis  88"
    A naive find() matched these TOC entries (at char ~34,000) instead of
    the actual MD&A section (at char ~200,000+).  Claude then received the
    TOC + business description instead of the financial results narrative.

    FIX — _is_toc_hit():
    Before accepting any marker match, scan the surrounding 2,000 characters.
    If 5+ distinct ITEM references appear packed together, or 3+ end-section
    signals (ITEM 7A, ITEM 8, ITEM 9...) appear, it is a TOC entry — skip it
    and keep searching for the next occurrence.

    Four-pass strategy:
      Pass 1: Item 7 / MD&A markers, TOC-skipping, 3+ signal requirement
      Pass 2: Results of Operations markers, same quality check
      Pass 3: rfind() — last occurrence is always past the TOC
      Pass 4: Last 40% of document fallback
    """
    if not raw_text:
        return ""

    upper = raw_text.upper()
    doc_len = len(raw_text)

    MDA_CONTENT_SIGNALS = [
        "NET SALES", "NET REVENUES", "TOTAL REVENUES", "REVENUE",
        "GROSS MARGIN", "OPERATING INCOME", "RESULTS OF OPERATIONS",
        "COMPARED TO", "YEAR ENDED", "INCREASED", "DECREASED",
        "COST OF GOODS", "COST OF SALES", "SELLING GENERAL",
        "SEGMENT", "FAVORABLE", "UNFAVORABLE", "PARTIALLY OFFSET",
        "MILLION", "BILLION",
    ]

    TOC_SIGNALS = [
        "ITEM 7A.", "ITEM 8.", "ITEM 9.", "ITEM 10.",
        "QUANTITATIVE AND QUALITATIVE",
        "FINANCIAL STATEMENTS AND SUPPLEMENTARY",
        "CHANGES IN AND DISAGREEMENTS",
        "CONTROLS AND PROCEDURES",
    ]

    pass1_markers = [
        "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
        "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "ITEM 7 - MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "ITEM 7\u2014MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
        "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS",
        "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
        "MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "MANAGEMENT DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
        "MANAGEMENT DISCUSSION AND ANALYSIS",
        "MD&A",
    ]

    pass2_markers = [
        "RESULTS OF OPERATIONS",
        "CONSOLIDATED RESULTS OF OPERATIONS",
        "RESULTS OF OPERATIONS AND FINANCIAL CONDITION",
        "OPERATING RESULTS",
        "DISCUSSION OF OPERATIONS",
        "DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
        "FINANCIAL REVIEW",
        "FINANCIAL AND OPERATING REVIEW",
        "OPERATING AND FINANCIAL REVIEW",
        "OPERATING AND FINANCIAL REVIEW AND PROSPECTS",
        "BUSINESS PERFORMANCE REVIEW",
        "PERFORMANCE REVIEW",
        "BUSINESS REVIEW",
        "ANNUAL BUSINESS REVIEW",
        "STRATEGIC AND FINANCIAL REVIEW",
        "FINANCIAL DISCUSSION",
        "FINANCIAL OVERVIEW",
        "EXECUTIVE OVERVIEW",
        "REVIEW OF FINANCIAL RESULTS",
        "ANALYSIS OF FINANCIAL RESULTS",
        "REVIEW OF OPERATIONS",
        "OPERATIONAL REVIEW",
        "FINANCIAL AND BUSINESS REVIEW",
    ]

    def _is_toc_hit(idx: int, scan_window: int = 2000) -> bool:
        window = upper[idx: idx + scan_window]
        item_refs = len(re.findall(r'\bITEM\s+\d+[A-Z]?[\.\s]', window))
        if item_refs >= 5:
            logger.debug(f"TOC detected at {idx:,}: {item_refs} ITEM refs in {scan_window} chars")
            return True
        toc_signal_count = sum(1 for sig in TOC_SIGNALS if sig in window)
        if toc_signal_count >= 3:
            logger.debug(f"TOC detected at {idx:,}: {toc_signal_count} TOC signals")
            return True
        return False

    def _find_with_quality_check(markers: list, scan_window: int = 15000) -> tuple:
        for marker in markers:
            search_from = 0
            while True:
                idx = upper.find(marker, search_from)
                if idx == -1:
                    break
                if _is_toc_hit(idx, scan_window=2000):
                    logger.info(f"Skipping TOC hit: '{marker[:50]}' at {idx:,}")
                    search_from = idx + 1
                    continue
                window = upper[idx: idx + scan_window]
                signal_count = sum(1 for sig in MDA_CONTENT_SIGNALS if sig in window)
                non_space = len(window.replace(" ", "").replace("\n", ""))
                if signal_count >= 3 and non_space > 500:
                    logger.info(f"MD&A hit: '{marker[:50]}' at {idx:,} (signals={signal_count})")
                    return idx, marker
                logger.debug(f"Weak hit skipped: '{marker[:50]}' at {idx:,} (signals={signal_count})")
                search_from = idx + 1
        return -1, None

    start_idx = -1
    start_marker_used = None

    start_idx, start_marker_used = _find_with_quality_check(pass1_markers, scan_window=15000)

    if start_idx == -1:
        logger.warning("Pass 1 found nothing — trying Results of Operations markers.")
        start_idx, start_marker_used = _find_with_quality_check(pass2_markers, scan_window=15000)

    if start_idx == -1:
        logger.warning("Pass 2 found nothing — trying rfind fallback.")
        all_markers = pass1_markers + pass2_markers
        for marker in all_markers:
            idx = upper.rfind(marker)
            if idx != -1 and not _is_toc_hit(idx) and len(raw_text[idx: idx + 500].strip()) > 100:
                start_idx = idx
                start_marker_used = f"{marker} (rfind)"
                logger.info(f"rfind hit: '{marker[:50]}' at {idx:,}")
                break

    if start_idx == -1:
        start_idx = max(0, int(doc_len * 0.60))
        start_marker_used = "last-40%-fallback"
        logger.warning(f"No MD&A marker found — using last-40% fallback at char {start_idx:,}")

    logger.info(f"MD&A start: '{start_marker_used[:60]}' at index {start_idx:,}")

    mda_end_markers = [
        "ITEM 7A.", "ITEM 7A ", "ITEM 7A\u2014", "ITEM 7A-",
        "ITEM 8.", "ITEM 8 ", "ITEM 8\u2014", "ITEM 8-",
    ]

    end_idx = doc_len
    for marker in mda_end_markers:
        idx = upper.find(marker, start_idx + 5000)
        if idx != -1 and idx < end_idx:
            end_idx = idx
            logger.info(f"MD&A end: '{marker}' at index {idx:,}")
            break

    mda_text = raw_text[start_idx:end_idx]
    logger.info(f"MD&A extracted: {len(mda_text):,} characters")

    segment_keywords = ["SEGMENT", "BUSINESS UNIT", "DIVISION", "OPERATING SEGMENT"]
    if any(kw in mda_text.upper() for kw in segment_keywords):
        logger.info("Segment discussions confirmed in MD&A.")
    else:
        logger.warning("No segment keywords found — company may not report segments.")

    if len(mda_text) > max_chars:
        logger.warning(f"MD&A truncated from {len(mda_text):,} to {max_chars:,} chars.")
    
    # Trim leading non-financial content — skip forward to where
# actual year-over-year financial comparisons begin.
# These phrases mark the start of real Results of Operations content.
    RESULTS_START_SIGNALS = [
        "NET SALES", "NET REVENUES", "TOTAL REVENUES",
        "REVENUES WERE", "REVENUES DECREASED", "REVENUES INCREASED",
        "THE FOLLOWING TABLE", "YEAR ENDED DECEMBER",
        "COMPARED TO THE PRIOR YEAR", "COMPARED TO FISCAL",
        "FISCAL 2024", "FISCAL 2023",
    ]

    trim_search = mda_text.upper()
    earliest_signal = len(mda_text)
    for signal in RESULTS_START_SIGNALS:
        idx = trim_search.find(signal)
        if idx != -1 and idx < earliest_signal:
            earliest_signal = idx

    # Only trim if we found a signal and it's more than 2000 chars in
    # (don't trim if the content starts immediately)
    if earliest_signal > 2000:
        logger.info(f"MD&A trimmed: skipping first {earliest_signal:,} chars of preamble")
        mda_text = mda_text[earliest_signal:]

    if len(mda_text) > max_chars:
        logger.warning(f"MD&A truncated from {len(mda_text):,} to {max_chars:,} chars.")

    return mda_text[:max_chars]

    


# ============================================================
# MD&A SUMMARISATION — uses Claude to avoid OpenAI content refusals
# ============================================================

def _summarize_mda_for_drivers(mda_text: str, extracted: ExtractionResult) -> str:
    if not mda_text or not mda_text.strip():
        logger.warning("No MD&A text available for driver summarisation.")
        return "No MD&A section found."

    if not extracted.periods:
        logger.warning("No periods extracted — cannot label MD&A analysis.")
        return "No periods extracted."

    p0 = extracted.periods[0]
    p1 = extracted.periods[1] if len(extracted.periods) >= 2 else None
    current_label = p0.period_name
    prior_label = p1.period_name if p1 else "the prior year"

    logger.info(f"MD&A summarisation (Claude): current={current_label} prior={prior_label} mda_len={len(mda_text):,} chars")

    MDA_CHAR_LIMIT = int(os.environ.get("MDA_CHAR_LIMIT", "180000"))
    mda_for_prompt = mda_text[:MDA_CHAR_LIMIT]
    if len(mda_text) > MDA_CHAR_LIMIT:
        logger.warning(f"MD&A truncated from {len(mda_text):,} to {MDA_CHAR_LIMIT:,} chars (set MDA_CHAR_LIMIT to increase).")

    system_prompt = (
        "You are a senior financial analyst. Read SEC 10-K MD&A sections and produce "
        "structured commentary on why each business segment performed the way it did. "
        "Always cite exact dollar amounts from the filing. Never use placeholder text. "
        "If a figure is not disclosed, write 'Not disclosed in filing'."
    )

    user_prompt = f"""Read the MD&A section below and produce a structured performance analysis
comparing {prior_label} to {current_label}.

MD&A TEXT ({len(mda_for_prompt):,} characters):
{mda_for_prompt}

INSTRUCTIONS:

Step 1 — List every business segment or division discussed. Write:
SEGMENTS IDENTIFIED: [name 1], [name 2], ...
If no segments write: SEGMENTS IDENTIFIED: Consolidated (no segments)

Step 2 — CONSOLIDATED PERFORMANCE SUMMARY

CONSOLIDATED SUMMARY:

REVENUE:
Total: [actual figure] vs [prior figure] (% change YoY)
Key themes: [2-3 sentences using actual figures]

COST OF GOODS SOLD:
Total: [actual figure] vs [prior figure] (% change YoY)
Key themes: [2-3 sentences using actual figures]

GROSS MARGIN:
Total: [actual figure] vs [prior figure]
Margin %: [actual]% vs [prior]%
Key themes: [2-3 sentences using actual figures]

Step 3 — INDIVIDUAL SEGMENT ANALYSIS

For each segment in Step 1:

SEGMENT: [Exact name from filing]

REVENUE:
Amount: [actual] vs [prior] (change of [delta])
Drivers:
- [Driver]: [positive/negative] impact of approximately [dollar amount]
[List ALL drivers — volume, pricing, mix, FX, weather, acquisitions, etc.]
Offsetting factors (if any):
- [Factor]: approximately [dollar amount]

Volume and price (if disclosed):
- Volumes: [actual] vs [prior]
- Avg realized price: [actual]/unit vs [prior]/unit

COST OF GOODS SOLD:
Amount: [actual] vs [prior]
If not separately disclosed: Not separately disclosed.
Drivers:
- [Input cost]: approximately [amount] [positive/negative]
Offsetting factors (if any):
- [Factor]: approximately [amount]

GROSS MARGIN:
Amount: [actual] vs [prior]
Margin %: [actual]% vs [prior]%
If not separately disclosed: Not separately disclosed.
Drivers:
- [Pricing/cost/mix/FX/weather factor]: approximately [amount] [positive/negative]
Offsetting factors (if any):
- [Factor]: approximately [amount]

Repeat for every segment.

Step 4 — KEY CROSS-SEGMENT THEMES
3-5 sentences on which segments drove consolidated results, shared headwinds/tailwinds,
and divergences. Use actual figures.

Rules:
- Use EXACT dollar amounts from the filing.
- Read full paragraphs — offsetting factors appear at the end.
- If a metric is not disclosed for a segment: Not separately disclosed.
- No placeholder text like [figure] or [direction].
"""

    try:
        client = get_anthropic_client()
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        result = resp.content[0].text if resp.content else "No driver insights extracted."

        refusal_signals = ["i'm sorry", "i cannot", "i can't assist", "as an ai", "unable to help"]
        if any(sig in result.lower()[:200] for sig in refusal_signals):
            logger.error(f"Claude returned a refusal: {result[:300]}")
            return f"MD&A extraction failed — model returned: {result[:300]}"

        placeholder_signals = ["$X.X", "$Y.Y", "[Direction]", "[actual figure]", "[driver here]"]
        found_ph = [s for s in placeholder_signals if s in result]
        if found_ph:
            logger.warning(f"Placeholder text found in MD&A summary: {found_ph}")

        logger.info(f"MD&A summary produced: {len(result):,} characters")
        return result

    except Exception as e:
        logger.error(f"Error in _summarize_mda_for_drivers: {e}")
        return f"Error extracting MD&A insights: {str(e)}"


# ============================================================
# LEASE DATA EXTRACTION
# ============================================================

def extract_lease_data(raw_text: str, period_names: list[str]) -> dict:
    if not raw_text or not period_names:
        return {}
    upper = raw_text.upper()
    notes_idx = upper.find("NOTE")
    if notes_idx == -1:
        return {}
    lease_keywords = ["LEASES", "LEASE EXPENSE", "OPERATING LEASE COST", "FINANCE LEASE COST",
                      "RIGHT-OF-USE ASSETS", "RENTAL EXPENSE", "LEASE LIABILITIES"]
    lease_section_start = -1
    for keyword in lease_keywords:
        idx = upper.find(keyword, notes_idx)
        if idx != -1 and idx < notes_idx + 500000:
            lease_section_start = idx
            logger.info(f"Lease section found at index {idx} with keyword '{keyword}'")
            break
    if lease_section_start == -1:
        return {}
    lease_section = raw_text[lease_section_start:lease_section_start + 50000]
    prompt = f"""
Extract lease-related expense data for periods: {', '.join(period_names)}.
TEXT:
{lease_section[:30000]}
Look for: Rental expense, Operating lease cost, Finance lease cost (Amortization of ROU assets,
Interest on lease liabilities), Short-term lease cost, Variable lease cost, Total lease cost.
Return ONLY valid JSON:
{{{{
  "FY2024": {{{{
    "rental_expense": 269.4, "operating_lease_cost": 87.2,
    "amortization_of_rou_assets": 45.5, "interest_on_lease_liabilities": 6.1,
    "short_term_lease_cost": 0.2, "variable_lease_cost": 19.5, "total_lease_cost": 158.5
  }}}}
}}}}
Use null if not found. Return ONLY the JSON.
"""
    try:
        resp = get_client().chat.completions.create(
            model="gpt-4o-mini", temperature=0.0, max_tokens=2000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Extract lease expense data and return valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        result_text = resp.choices[0].message.content or "{}"
        lease_data = json.loads(result_text)
        logger.info(f"Lease data extracted: {json.dumps(lease_data, indent=2)}")
        return lease_data
    except Exception as e:
        logger.error(f"Error extracting lease data: {e}")
        return {}


def calculate_adjusted_rent_expense(
    rental_expense, operating_lease_cost, amortization_of_rou,
    interest_on_lease_liab, short_term_lease_cost, total_lease_cost,
) -> Optional[float]:
    if rental_expense is not None and (
        amortization_of_rou is not None or interest_on_lease_liab is not None
        or short_term_lease_cost is not None
    ):
        adjusted = rental_expense
        if amortization_of_rou is not None:
            adjusted -= amortization_of_rou
        if interest_on_lease_liab is not None:
            adjusted -= interest_on_lease_liab
        if short_term_lease_cost is not None:
            adjusted -= short_term_lease_cost
        return adjusted
    if operating_lease_cost is not None:
        return operating_lease_cost
    if rental_expense is not None:
        return rental_expense
    if total_lease_cost is not None:
        return total_lease_cost
    return None


# ============================================================
# REVOLVER EXTRACTION
# ============================================================

def extract_revolver_data(raw_text: str, period_names: list[str]) -> dict:
    if not raw_text or not period_names:
        return {}
    upper = raw_text.upper()
    debt_section_start = -1
    debt_section_end = -1
    note_header = None
    note_patterns = [
        r'NOTE\s+\d+[\.\:\-\s]+(DEBT|LONG[\-\s]?TERM DEBT|BORROWINGS)',
        r'NOTE\s+\d+[\.\:\-\s]+(CREDIT FACILITIES|CREDIT AGREEMENTS?)',
        r'NOTE\s+\d+[\.\:\-\s]+(FINANCING|FINANCIAL OBLIGATIONS)',
    ]
    for pattern in note_patterns:
        matches = list(re.finditer(pattern, upper))
        if matches:
            note_match = matches[0]
            note_header = note_match.group(0)
            debt_section_start = note_match.start()
            logger.info(f"Debt note found: '{note_header}' at index {debt_section_start:,}")
            break
    if debt_section_start != -1:
        remaining_text = upper[debt_section_start + 100:]
        next_note = re.search(r'NOTE\s+\d+[\.\:\-]', remaining_text)
        if next_note:
            debt_section_end = debt_section_start + 100 + next_note.start()
        else:
            debt_section_end = min(len(raw_text), debt_section_start + 250000)
    else:
        logger.warning("No debt NOTE header found; using keyword search fallback.")
        for keyword in ["CREDIT FACILIT", "REVOLVING CREDIT", "BORROWINGS", "DEBT"]:
            idx = upper.find(keyword)
            if idx != -1:
                note_header = f"Keyword: {keyword}"
                debt_section_start = max(0, idx - 30000)
                debt_section_end = min(len(raw_text), idx + 250000)
                break
    if debt_section_start == -1:
        logger.warning("Could not find debt/financing section.")
        return {}
    debt_section = raw_text[debt_section_start:debt_section_end]
    prompt = f"""
Extract ALL revolving credit facilities for periods: {', '.join(period_names)}.
DEBT NOTE ({len(debt_section):,} chars):
{debt_section[:120000]}
Find every revolving credit facility (not term loans, not LC sub-limits).
For each: name, facility_size, borrowings, availability.
If availability not stated, calculate: facility_size - borrowings - letters_of_credit.
Return ONLY valid JSON:
{{
  "FY2024": {{"facilities": [{{"name": "...", "facility_size": 1500.0, "borrowings": 0.0, "availability": 1500.0}}]}},
  "FY2023": {{"facilities": [...]}}
}}
Use null if not found. Return ONLY the JSON.
"""
    try:
        client = get_anthropic_client()
        resp = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=4000, temperature=0.0,
            system="Extract revolving credit facility data from 10-K debt notes. Return valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        result_text = resp.content[0].text if resp.content else "{}"
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        revolver_data = json.loads(result_text)
        logger.info("=" * 60)
        logger.info("REVOLVER EXTRACTION SUMMARY")
        for period, data in revolver_data.items():
            facilities = data.get("facilities", [])
            logger.info(f"Period: {period} — {len(facilities)} facility/ies found")
            for f in facilities:
                logger.info(f"  {f.get('name')}: size=${f.get('facility_size')}MM borrowed=${f.get('borrowings')}MM avail=${f.get('availability')}MM")
        logger.info("=" * 60)
        return revolver_data
    except Exception as e:
        logger.error(f"Error extracting revolver data: {e}")
        return {}


# ============================================================
# TEXT SELECTION FOR FINANCIAL STATEMENTS
# ============================================================

def _select_relevant_statement_text(raw_text: str, max_chars: int = 260000) -> str:
    if not raw_text:
        return ""
    upper = raw_text.upper()
    FS_CONTENT_SIGNALS = [
        "NET SALES", "NET REVENUES", "TOTAL REVENUES", "COST OF GOODS SOLD",
        "COST OF SALES", "COST OF PRODUCTS", "GROSS PROFIT", "OPERATING INCOME",
        "NET INCOME", "TOTAL ASSETS", "TOTAL LIABILITIES", "STOCKHOLDERS",
        "CASH AND CASH EQUIVALENTS", "DEPRECIATION",
    ]
    tier1_anchors = [
        "CONSOLIDATED STATEMENTS OF OPERATIONS", "CONSOLIDATED STATEMENTS OF INCOME",
        "CONSOLIDATED STATEMENTS OF EARNINGS", "CONSOLIDATED BALANCE SHEETS",
        "CONSOLIDATED STATEMENTS OF CASH FLOWS", "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS",
    ]
    tier2_anchors = [
        "FINANCIAL STATEMENTS", "ITEM 8. FINANCIAL STATEMENTS",
        "STATEMENTS OF CASH FLOWS", "SUPPLEMENTAL CASH FLOW",
    ]
    start = -1
    for anchor in tier1_anchors:
        search_from = 0
        while True:
            idx = upper.find(anchor, search_from)
            if idx == -1:
                break
            window = raw_text[idx: idx + 3000].upper()
            if any(sig in window for sig in FS_CONTENT_SIGNALS):
                start = max(0, idx - 5000)
                logger.info(f"FS tier-1 anchor: '{anchor[:40]}' at {idx:,}")
                break
            search_from = idx + 1
        if start != -1:
            break
    if start == -1:
        for anchor in tier2_anchors:
            search_from = 0
            while True:
                idx = upper.find(anchor, search_from)
                if idx == -1:
                    break
                window = raw_text[idx: idx + 3000].upper()
                if any(sig in window for sig in FS_CONTENT_SIGNALS):
                    start = max(0, idx - 5000)
                    logger.info(f"FS tier-2 anchor: '{anchor[:40]}' at {idx:,}")
                    break
                search_from = idx + 1
            if start != -1:
                break
    if start == -1:
        for anchor in tier1_anchors:
            idx = upper.rfind(anchor)
            if idx != -1:
                start = max(0, idx - 5000)
                logger.info(f"FS rfind anchor: '{anchor[:40]}' at {idx:,}")
                break
    if start == -1:
        start = max(0, int(len(raw_text) * 0.65))
        logger.warning(f"No FS anchor found — using last-35% fallback start={start:,}")
    end = min(len(raw_text), start + max_chars)
    slice_text = raw_text[start:end]
    tables_idx = upper.rfind("TABLES:")
    if tables_idx == -1:
        tables_idx = upper.find("TABLES:")
    if tables_idx != -1 and "TABLES:" not in slice_text:
        slice_text += "\n\n" + raw_text[tables_idx: min(len(raw_text), tables_idx + 220000)]
    logger.info("_select_relevant_statement_text: start=%d end=%d slice_len=%d", start, end, len(slice_text))
    return slice_text


def _extract_tables_block(text: str, max_chars: int = 160000) -> str:
    if not text:
        return ""
    idx = text.upper().find("TABLES:")
    if idx == -1:
        return ""
    return text[idx: idx + max_chars]


def _numeric_excerpt(text: str, max_chars: int = 130000, context_lines: int = 1) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    keep = [False] * len(lines)
    amount_re = re.compile(r"(\(\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*\))|(\d{1,3}(?:,\d{3})+(?:\.\d+)?)|(\d+\.\d+)")
    for i, line in enumerate(lines):
        if amount_re.search(line):
            for j in range(max(0, i - context_lines), min(len(lines), i + context_lines + 1)):
                keep[j] = True
    filtered = [lines[i].strip() for i in range(len(lines)) if keep[i] and lines[i].strip()]
    out = "\n".join(filtered) if filtered else text
    return out[:max_chars]


# ============================================================
# MAIN EXTRACTION ENTRY POINT
# ============================================================

def extract_financials_from_text(raw_text: str) -> Tuple[ExtractionResult, str, str, str]:
    raw_text = _strip_html_to_text(raw_text)

    selected = _select_relevant_statement_text(raw_text, max_chars=260000)
    tables_block = _extract_tables_block(selected, max_chars=160000)
    numeric_block = _numeric_excerpt(selected, max_chars=130000, context_lines=1)

    combined_excerpt = ""
    if tables_block:
        combined_excerpt += tables_block + "\n\n"
    combined_excerpt += "NUMERIC LINES + CONTEXT:\n" + numeric_block

    user_prompt = f"""
Extract key financial statement line items from the following statement text.

MULTI-PERIOD REQUIREMENTS:
- If statements show multiple columns/years, return ONE period object per year.
- Return up to 5 most recent periods, most recent first.
- period_name: if column is "2024" use "FY2024".
- Do NOT put multiple years into a single numeric field.

GENERAL RULES:
- Prefer TABLES rows (pipe-delimited) when present.
- Each numeric field: single scalar or null.
- derived_metrics must be {{}}.
- Leave ebitda null unless explicitly labeled EBITDA.

ALIAS / LABEL RULES:
Income statement:
- revenue: Net revenue, Net sales, Total revenues, Revenues
- cost_of_sales: Cost of sales, Cost of products sold, Cost of goods sold
- sga_expense: Selling general and administrative, SGA
- operating_income: Operating income, Operating profit, Income from operations, EBIT
- net_income: Net income, Net earnings
- interest_expense: Interest expense
- income_tax_expense: Provision for income taxes, Income tax expense
- depreciation_amortization: Depreciation and amortization, D&A
- rent_expense: Operating lease cost, Lease cost, Rent expense, Rental expense

Balance sheet:
- cash: Cash, Cash and cash equivalents
- total_debt: Total debt, Total borrowings
- long_term_debt: Long-term debt
- current_portion_long_term_debt: Current portion of long-term debt

Cash flow:
- cfo: Net cash provided by operating activities
- cfi: Net cash used in investing activities
- cff: Net cash used in financing activities
- capex: Capital expenditures, Additions to PP&E
- cash_paid_for_interest: Cash paid for interest
- cash_paid_for_income_taxes: Cash paid for income taxes
- dividends_distributions_paid: Dividends paid, Distributions

SIGN CONVENTION: Capture values as shown in the document.

STATEMENT TEXT:
{combined_excerpt}

{EXTRACTOR_SCHEMA_PROMPT}
"""

    resp = get_client().chat.completions.create(
        model="gpt-4o-mini", temperature=0.0,
        messages=[
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_model_text = resp.choices[0].message.content or ""
    data = _coerce_json(raw_model_text)
    data = _normalize_payload(data)
    extracted = ExtractionResult.model_validate(data)

    period_names = [p.period_name for p in extracted.periods] if extracted.periods else []

    # Lease data
    lease_data = extract_lease_data(raw_text, period_names) if period_names else {}
    if extracted.periods:
        for p in extracted.periods:
            is_ = p.income_statement or {}
            period_lease_data = lease_data.get(p.period_name, {})
            if period_lease_data:
                if not p.notes:
                    p.notes = []
                p.notes.append(f"Lease data: {json.dumps(period_lease_data)}")
                adjusted_rent = calculate_adjusted_rent_expense(
                    rental_expense=period_lease_data.get("rental_expense"),
                    operating_lease_cost=period_lease_data.get("operating_lease_cost"),
                    amortization_of_rou=period_lease_data.get("amortization_of_rou_assets"),
                    interest_on_lease_liab=period_lease_data.get("interest_on_lease_liabilities"),
                    short_term_lease_cost=period_lease_data.get("short_term_lease_cost"),
                    total_lease_cost=period_lease_data.get("total_lease_cost"),
                )
                if adjusted_rent is not None:
                    is_["rent_expense"] = adjusted_rent
            if is_.get("rent_expense") is None:
                rent = _extract_rent_from_mda(raw_text, p.period_name)
                if rent is not None:
                    is_["rent_expense"] = rent

    # Revolver data
    revolver_data = extract_revolver_data(raw_text, period_names) if period_names else {}
    if extracted.periods:
        for p in extracted.periods:
            bs_ = p.balance_sheet or {}
            period_revolver_data = revolver_data.get(p.period_name, {})
            if period_revolver_data:
                facilities = period_revolver_data.get("facilities", [])
                if facilities:
                    total_size = sum(f.get("facility_size", 0) or 0 for f in facilities)
                    total_borrowings = sum(f.get("borrowings", 0) or 0 for f in facilities)
                    total_availability = sum(f.get("availability", 0) or 0 for f in facilities)
                    bs_["revolver_facility_size"] = total_size if total_size > 0 else None
                    bs_["revolver_borrowings"] = total_borrowings if total_borrowings > 0 else 0.0
                    bs_["revolver_availability"] = total_availability if total_availability > 0 else None
                    bs_["revolver_facilities_detail"] = facilities
                else:
                    bs_["revolver_facility_size"] = period_revolver_data.get("revolver_facility_size")
                    bs_["revolver_borrowings"] = period_revolver_data.get("revolver_borrowings")
                    bs_["revolver_availability"] = period_revolver_data.get("revolver_availability")

    # MD&A
    mda_text = _extract_mda_section(raw_text, max_chars=350000)
    if not mda_text:
        logger.error("CRITICAL: _extract_mda_section returned empty string.")
    else:
        seg_kws = ["segment", "business unit", "division"]
        if not any(kw in mda_text.lower() for kw in seg_kws):
            logger.warning("No segment keywords in extracted MD&A.")
        logger.info(f"MD&A first 500 chars: {mda_text[:500].replace(chr(10), ' ')}")

    mda_summary = _summarize_mda_for_drivers(mda_text, extracted)

    return extracted, combined_excerpt, raw_model_text, mda_summary


def _extract_rent_from_mda(raw_text: str, period_name: str) -> Optional[float]:
    upper = raw_text.upper()
    lease_markers = ["OPERATING LEASE COST", "LEASE EXPENSE", "OPERATING LEASE EXPENSE", "RENT EXPENSE"]
    year_match = re.search(r'20\d{2}', period_name)
    if not year_match:
        return None
    year = year_match.group()
    for marker in lease_markers:
        idx = upper.find(marker)
        if idx != -1:
            context = raw_text[max(0, idx - 500):idx + 1500]
            amount_pattern = (
                r'(?:' + year + r'[^\d]*?[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?))'
                r'|(?:[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)[^\d]*?' + year + r')'
            )
            matches = re.findall(amount_pattern, context, re.IGNORECASE)
            if matches:
                for match_tuple in matches:
                    amount_str = match_tuple[0] or match_tuple[1]
                    if amount_str:
                        try:
                            amount = float(amount_str.replace(',', ''))
                            if 1 < amount < 10000:
                                return amount
                        except Exception:
                            continue
    return None


# ============================================================
# COMPUTATIONS
# ============================================================

def validate_and_compute(result: ExtractionResult) -> ExtractionResult:
    flags: list[str] = []
    for p in result.periods:
        p.income_statement = dict(p.income_statement or {})
        p.balance_sheet = dict(p.balance_sheet or {})
        p.cash_flow = dict(p.cash_flow or {})
        p.derived_metrics = {}
        is_ = p.income_statement
        bs = p.balance_sheet
        cf = p.cash_flow
        op_inc = is_.get("operating_income")
        da = is_.get("depreciation_amortization")
        rent = is_.get("rent_expense")
        ebitda = compute_ebitda(op_inc, da, rent_expense=rent, include_rent=True)
        if ebitda is None:
            if op_inc is None:
                flags.append(f"{p.period_name}: operating_income missing; EBITDA not computed.")
            if da is None:
                flags.append(f"{p.period_name}: depreciation_amortization missing; EBITDA not computed.")
        else:
            if rent is None:
                p.derived_metrics["ebitda_includes_rent"] = False
                flags.append(f"{p.period_name}: rent_expense not found; EBITDA excludes rent.")
            else:
                p.derived_metrics["ebitda_includes_rent"] = True
            is_["ebitda"] = float(ebitda)
            p.derived_metrics["ebitda_computed"] = float(ebitda)
        fcf = compute_free_cash_flow(cf.get("cfo"), cf.get("capex"))
        if fcf is not None:
            p.derived_metrics["free_cash_flow"] = float(fcf)
        ebitda_margin = compute_ebitda_margin(is_.get("ebitda"), is_.get("revenue"))
        if ebitda_margin is not None:
            p.derived_metrics["ebitda_margin"] = float(ebitda_margin)
        capex = cf.get("capex")
        cash_taxes = cf.get("cash_paid_for_income_taxes")
        cpltd = bs.get("current_portion_long_term_debt")
        cash_interest = cf.get("cash_paid_for_interest")
        fcc = compute_fcc(ebitda=is_.get("ebitda"), capex=capex, cash_taxes=cash_taxes, cpltd=cpltd, cash_interest=cash_interest)
        if fcc is None:
            if is_.get("ebitda") is None:
                flags.append(f"{p.period_name}: EBITDA missing; FCC not computed.")
            if capex is None:
                flags.append(f"{p.period_name}: capex missing; FCC not computed.")
            if cash_taxes is None:
                flags.append(f"{p.period_name}: cash_paid_for_income_taxes missing; FCC not computed.")
            if cpltd is None:
                flags.append(f"{p.period_name}: CPLTD missing; FCC not computed.")
            if cash_interest is None:
                flags.append(f"{p.period_name}: cash_paid_for_interest missing; FCC not computed.")
            if cpltd is not None and cash_interest is not None:
                denom = float(cpltd) + float(abs_outflow(cash_interest) or 0.0)
                if denom == 0:
                    flags.append(f"{p.period_name}: FCC denominator is 0.")
        else:
            p.derived_metrics["fcc"] = float(fcc)
            capex_mag = abs_outflow(capex)
            taxes_mag = abs_outflow(cash_taxes)
            int_mag = abs_outflow(cash_interest)
            if isinstance(is_.get("ebitda"), (int, float)) and capex_mag is not None and taxes_mag is not None:
                p.derived_metrics["fcc_numerator"] = float(is_.get("ebitda")) - float(capex_mag) - float(taxes_mag)
            if cpltd is not None and int_mag is not None:
                p.derived_metrics["fcc_denominator"] = float(cpltd) + float(int_mag)
        lev = compute_leverage(bs.get("total_debt"), is_.get("ebitda"))
        if lev is None:
            if bs.get("total_debt") is None:
                flags.append(f"{p.period_name}: total_debt missing; leverage not computed.")
            ebitda_val = is_.get("ebitda")
            if ebitda_val is None:
                flags.append(f"{p.period_name}: EBITDA missing; leverage not computed.")
            elif isinstance(ebitda_val, (int, float)) and ebitda_val <= 0:
                flags.append(f"{p.period_name}: EBITDA <= 0; leverage not computed.")
        else:
            p.derived_metrics["leverage_total_debt_to_ebitda"] = float(lev)
        z_score = compute_altman_z_score(
            revenue=is_.get("revenue"), operating_income=is_.get("operating_income"),
            total_assets=bs.get("total_assets"), total_liabilities=bs.get("total_liabilities"),
            total_equity=bs.get("total_equity"), retained_earnings=None,
        )
        if z_score is not None:
            p.derived_metrics["altman_z_score"] = float(z_score)
        else:
            flags.append(f"{p.period_name}: Insufficient data for Altman Z-Score.")
        revenue_growth = None
        if len(result.periods) >= 2:
            curr_idx = result.periods.index(p)
            if curr_idx < len(result.periods) - 1:
                prior_p = result.periods[curr_idx + 1]
                curr_revenue = is_.get("revenue")
                prior_revenue = (prior_p.income_statement or {}).get("revenue")
                if curr_revenue is not None and prior_revenue is not None and prior_revenue != 0:
                    revenue_growth = (curr_revenue - prior_revenue) / prior_revenue
        pd_score, credit_rating = compute_pd_score(
            leverage=lev, fcc=fcc, ebitda_margin=p.derived_metrics.get("ebitda_margin"),
            altman_z=z_score, revenue_growth=revenue_growth,
            free_cash_flow=p.derived_metrics.get("free_cash_flow"), ebitda=is_.get("ebitda"),
        )
        if pd_score is not None:
            p.derived_metrics["pd_score"] = int(pd_score)
            p.derived_metrics["credit_rating"] = credit_rating
    seen: set[str] = set()
    deduped: list[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    result.validation_flags = deduped
    return result


# ============================================================
# FINANCIAL SUMMARY MEMO (no borrower context)
# ============================================================

def generate_financial_summary_memo(extracted: ExtractionResult) -> str:
    if not extracted.periods:
        return "## Financial Summary\n- No periods extracted.\n"
    extracted = validate_and_compute(extracted)
    def m(x): return "N/A" if x is None else f"${float(x):,.0f}"
    def xfmt(x): return "N/A" if x is None else f"{float(x):.2f}x"
    def pct(x): return "N/A" if x is None else f"{float(x) * 100:.1f}%"
    def yoy(new, old):
        if new is None or old is None or float(old) == 0: return None
        return (float(new) - float(old)) / float(old)
    p0 = extracted.periods[0]
    basis = extracted.statement_basis or "actual"
    is0 = p0.income_statement or {}
    bs0 = p0.balance_sheet or {}
    cf0 = p0.cash_flow or {}
    dm0 = p0.derived_metrics or {}
    memo = []
    memo.append("## Financial Summary (Extracted + Computed)")
    memo.append(f"*Statement basis:* `{basis}`  ")
    memo.append(f"*Most recent period:* `{p0.period_name}`\n")
    memo.append("### Executive Takeaways")
    memo.append(f"- Revenue: {m(is0.get('revenue'))}.")
    memo.append(f"- EBITDA: {m(is0.get('ebitda'))} ({pct(dm0.get('ebitda_margin'))} margin).")
    memo.append(f"- Total debt: {m(bs0.get('total_debt'))}; leverage: {xfmt(dm0.get('leverage_total_debt_to_ebitda'))}.")
    memo.append(f"- FCC: {xfmt(dm0.get('fcc'))}.")
    memo.append(f"- CFO: {m(cf0.get('cfo'))}; Capex: {m(abs(cf0.get('capex')) if isinstance(cf0.get('capex'), (int, float)) else None)}; FCF: {m(dm0.get('free_cash_flow'))}.")
    memo.append(f"- Liquidity (cash): {m(bs0.get('cash'))}.\n")
    if len(extracted.periods) >= 2:
        p1 = extracted.periods[1]
        is1 = p1.income_statement or {}
        dm1 = p1.derived_metrics or {}
        memo.append(f"### YoY Trend ({p1.period_name} -> {p0.period_name})")
        memo.append(f"- Revenue: {m(is1.get('revenue'))} -> {m(is0.get('revenue'))} ({pct(yoy(is0.get('revenue'), is1.get('revenue')))})")
        memo.append(f"- EBITDA: {m(is1.get('ebitda'))} -> {m(is0.get('ebitda'))} ({pct(yoy(is0.get('ebitda'), is1.get('ebitda')))})")
        memo.append(f"- EBITDA margin: {pct(dm1.get('ebitda_margin'))} -> {pct(dm0.get('ebitda_margin'))}")
        memo.append("")
    flags = list(extracted.validation_flags or [])
    if flags:
        memo.append("### Data Gaps / Extraction Notes")
        memo.extend([f"- {f}" for f in flags])
        memo.append("")
    return "\n".join(memo).strip() + "\n"


# ============================================================
# FORMATTING HELPERS
# ============================================================

def _fmt_money(x: Optional[float], basis: str) -> str:
    if x is None: return "N/A"
    suffix = "MM" if basis == "millions" else ("K" if basis == "thousands" else "")
    return f"${x:,.0f}{suffix}"

def _fmt_pct(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x * 100:.1f}%"

def _fmt_x(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x:.2f}x"

def _safe_pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0: return None
    return (new - old) / old


def _format_extracted_data(extracted: ExtractionResult, basis: str) -> str:
    suffix = "MM" if basis == "millions" else ("K" if basis == "thousands" else "")
    lines = ["## Extracted Financial Data", "", f"**Statement Basis:** {basis}", ""]
    for period in extracted.periods:
        lines.append(f"### {period.period_name}")
        lines.append("")
        is_ = period.income_statement or {}
        if any(v is not None for v in is_.values()):
            lines.append("#### Income Statement")
            for label, key in [
                ("Revenue", "revenue"), ("Cost of Sales", "cost_of_sales"),
                ("SG&A Expense", "sga_expense"), ("Operating Income", "operating_income"),
                ("EBITDA", "ebitda"), ("Net Income", "net_income"),
                ("Interest Expense", "interest_expense"), ("Income Tax Expense", "income_tax_expense"),
                ("Depreciation & Amortization", "depreciation_amortization"), ("Rent Expense", "rent_expense"),
            ]:
                if is_.get(key) is not None:
                    lines.append(f"- {label}: ${is_[key]:,.0f}{suffix}")
            lines.append("")
        bs = period.balance_sheet or {}
        if any(v is not None for v in bs.values()):
            lines.append("#### Balance Sheet")
            for label, key in [
                ("Cash", "cash"), ("Total Assets", "total_assets"),
                ("Total Liabilities", "total_liabilities"), ("Total Equity", "total_equity"),
                ("Total Debt", "total_debt"), ("Long-term Debt", "long_term_debt"),
                ("Current Portion of Long-term Debt", "current_portion_long_term_debt"),
            ]:
                if bs.get(key) is not None:
                    lines.append(f"- {label}: ${bs[key]:,.0f}{suffix}")
            lines.append("")
        cf = period.cash_flow or {}
        if any(v is not None for v in cf.values()):
            lines.append("#### Cash Flow Statement")
            for label, key in [
                ("Operating Cash Flow", "cfo"), ("Investing Cash Flow", "cfi"),
                ("Financing Cash Flow", "cff"), ("Capex", "capex"),
                ("Cash Paid for Interest", "cash_paid_for_interest"),
                ("Cash Paid for Income Taxes", "cash_paid_for_income_taxes"),
                ("Dividends/Distributions Paid", "dividends_distributions_paid"),
            ]:
                if cf.get(key) is not None:
                    lines.append(f"- {label}: ${cf[key]:,.0f}{suffix}")
            lines.append("")
        dm = period.derived_metrics or {}
        if any(v is not None for v in dm.values()):
            lines.append("#### Computed Metrics")
            if dm.get("ebitda_computed") is not None:
                lines.append(f"- EBITDA (Computed): ${dm['ebitda_computed']:,.0f}{suffix}")
            if dm.get("ebitda_margin") is not None:
                lines.append(f"- EBITDA Margin: {dm['ebitda_margin'] * 100:.1f}%")
            if dm.get("leverage_total_debt_to_ebitda") is not None:
                lines.append(f"- Leverage (Debt/EBITDA): {dm['leverage_total_debt_to_ebitda']:.2f}x")
            if dm.get("fcc") is not None:
                lines.append(f"- FCC: {dm['fcc']:.2f}x")
            if dm.get("free_cash_flow") is not None:
                lines.append(f"- Free Cash Flow: ${dm['free_cash_flow']:,.0f}{suffix}")
            if dm.get("fcc_numerator") is not None:
                lines.append(f"- FCC Numerator: ${dm['fcc_numerator']:,.0f}{suffix}")
            if dm.get("fcc_denominator") is not None:
                lines.append(f"- FCC Denominator: ${dm['fcc_denominator']:,.0f}{suffix}")
            lines.append("")
    return "\n".join(lines)


def _build_metric_summary(extracted: ExtractionResult, basis: str) -> str:
    if not extracted.periods:
        return "No periods available"
    lines = []
    p0 = extracted.periods[0]
    is0 = p0.income_statement or {}
    bs0 = p0.balance_sheet or {}
    cf0 = p0.cash_flow or {}
    dm0 = p0.derived_metrics or {}
    lines += [
        f"MOST RECENT PERIOD: {p0.period_name}",
        f"Revenue: {_fmt_money(is0.get('revenue'), basis)}",
        f"Cost of Sales: {_fmt_money(is0.get('cost_of_sales'), basis)}",
        f"SG&A: {_fmt_money(is0.get('sga_expense'), basis)}",
        f"Operating Income: {_fmt_money(is0.get('operating_income'), basis)}",
        f"EBITDA: {_fmt_money(is0.get('ebitda'), basis)}",
        f"EBITDA Margin: {_fmt_pct(dm0.get('ebitda_margin'))}",
        f"Net Income: {_fmt_money(is0.get('net_income'), basis)}",
        f"Total Debt: {_fmt_money(bs0.get('total_debt'), basis)}",
        f"Leverage (Debt/EBITDA): {_fmt_x(dm0.get('leverage_total_debt_to_ebitda'))}",
        f"Operating Cash Flow: {_fmt_money(cf0.get('cfo'), basis)}",
        f"Capex: {_fmt_money(abs_outflow(cf0.get('capex')), basis)}",
        f"Free Cash Flow: {_fmt_money(dm0.get('free_cash_flow'), basis)}",
        f"FCC: {_fmt_x(dm0.get('fcc'))}",
        f"Cash: {_fmt_money(bs0.get('cash'), basis)}",
        f"Revolver Facility Size: {_fmt_money(bs0.get('revolver_facility_size'), basis)}",
        f"Revolver Borrowings: {_fmt_money(bs0.get('revolver_borrowings'), basis)}",
        f"Revolver Availability: {_fmt_money(bs0.get('revolver_availability'), basis)}",
    ]
    if len(extracted.periods) >= 2:
        p1 = extracted.periods[1]
        is1 = p1.income_statement or {}
        cf1 = p1.cash_flow or {}
        dm1 = p1.derived_metrics or {}
        lines += [
            f"\nPRIOR PERIOD: {p1.period_name}",
            f"Revenue: {_fmt_money(is1.get('revenue'), basis)}",
            f"Cost of Sales: {_fmt_money(is1.get('cost_of_sales'), basis)}",
            f"SG&A: {_fmt_money(is1.get('sga_expense'), basis)}",
            f"Operating Income: {_fmt_money(is1.get('operating_income'), basis)}",
            f"EBITDA: {_fmt_money(is1.get('ebitda'), basis)}",
            f"EBITDA Margin: {_fmt_pct(dm1.get('ebitda_margin'))}",
            f"Net Income: {_fmt_money(is1.get('net_income'), basis)}",
            f"Total Debt: {_fmt_money((p1.balance_sheet or {}).get('total_debt'), basis)}",
            f"Leverage (Debt/EBITDA): {_fmt_x(dm1.get('leverage_total_debt_to_ebitda'))}",
            f"Operating Cash Flow: {_fmt_money(cf1.get('cfo'), basis)}",
            f"Capex: {_fmt_money(abs_outflow(cf1.get('capex')), basis)}",
            f"Free Cash Flow: {_fmt_money(dm1.get('free_cash_flow'), basis)}",
            f"FCC: {_fmt_x(dm1.get('fcc'))}",
            f"Cash: {_fmt_money((p1.balance_sheet or {}).get('cash'), basis)}",
            f"\nYoY CHANGES ({p1.period_name} -> {p0.period_name}):",
        ]
        for label, new_val, old_val in [
            ("Revenue", is0.get('revenue'), is1.get('revenue')),
            ("Cost of Sales", is0.get('cost_of_sales'), is1.get('cost_of_sales')),
            ("EBITDA", is0.get('ebitda'), is1.get('ebitda')),
        ]:
            chg = _safe_pct_change(new_val, old_val)
            if chg is not None:
                lines.append(f"{label}: {_fmt_pct(chg)}")
        if dm0.get('ebitda_margin') is not None and dm1.get('ebitda_margin') is not None:
            delta = dm0['ebitda_margin'] - dm1['ebitda_margin']
            lines.append(f"EBITDA Margin: {delta * 100:+.1f}pp")
    return "\n".join(lines)


# ============================================================
# MEMO SYSTEM PROMPT
# ============================================================

MEMO_SYSTEM_PROMPT = """
You are a senior commercial credit analyst writing a comprehensive, metric-focused credit memo.

CRITICAL RULE: Never write placeholder text. If data is not available write: Not disclosed in MD&A.

HOW TO USE THE MD&A INSIGHTS:
- SEGMENTS IDENTIFIED line gives exact segment names — use them as headers throughout.
- CONSOLIDATED SUMMARY block gives consolidated Revenue, COGS, Gross Margin drivers.
- SEGMENT blocks give per-segment drivers, amounts, and offsetting factors — use them exactly.
"""

MEMO_MODEL = os.getenv("MEMO_MODEL", "gpt-4o")
MEMO_MAX_TOKENS = int(os.getenv("MEMO_MAX_TOKENS", "10000"))
MEMO_TEMPERATURE = float(os.getenv("MEMO_TEMPERATURE", "0.1"))


# ============================================================
# UNDERWRITING MEMO GENERATOR
# ============================================================

def generate_underwriting_memo(
    borrower: BorrowerProfile,
    covenants: CovenantSet,
    extracted: ExtractionResult,
    mda_summary: str = "",
) -> str:
    extracted = validate_and_compute(extracted)
    if not extracted.periods:
        return "## Credit Memo\n\nNo financial periods available for analysis."
    basis = extracted.statement_basis or "actual"
    extracted_section = _format_extracted_data(extracted, basis)
    metric_data = _build_metric_summary(extracted, basis)

    user_prompt = f"""
Write a comprehensive credit memo for the borrower below.

BORROWER
Name:            {borrower.name}
Industry:        {borrower.industry}
Facility Type:   {borrower.facility_type}
Use of Proceeds: {borrower.use_of_proceeds}

COVENANTS
Max Total Leverage: {covenants.max_total_leverage if covenants.max_total_leverage else "Not specified"}
Min FCC:            {covenants.min_fcc if covenants.min_fcc else "Not specified"}

FINANCIAL DATA (basis: {basis})
{metric_data}

MD&A DRIVER INSIGHTS
{mda_summary}

VALIDATION FLAGS
{json.dumps(extracted.validation_flags, indent=2) if extracted.validation_flags else "None"}

Write the memo using this structure:

## Executive Summary
2-3 bullets with actual figures on overall credit assessment.

## Credit Risk Assessment

### PD Score: [score]/12 — [rating]
- Scale: 1-3 Investment Grade | 4-6 Non-Investment Grade | 7-9 Speculative | 10-11 High Risk | 12 Default
- Interpretation: 1 sentence.

### Altman Z-Score: [score]
- Zones: Z > 2.9 Safe | 1.23-2.9 Grey | Z < 1.23 Distress
- Assessment: current zone and implication.

### Key Credit Metrics
- Leverage: [X.XX]x (vs covenant if specified)
- FCC: [X.XX]x (vs minimum if specified)
- EBITDA Margin: [XX.X]%
- Liquidity: Cash $[X]MM + $[Y]MM revolver availability = $[Z]MM total

## Financial Performance Analysis

### Revenue
One sentence with actual YoY % and $ change.
Consolidated Drivers: from CONSOLIDATED SUMMARY in MD&A insights.
Segment Analysis: one block per segment using exact names from SEGMENTS IDENTIFIED.
Each segment block: actual revenue figures, all drivers with dollar amounts, offsetting factors.

### Cost of Goods Sold
Same structure as Revenue.

### Gross Margin
Same structure as Revenue.

### EBITDA and Margins
EBITDA $ and % change YoY. EBITDA margin pp change.

### Cash Flow
CFO, Capex, FCF each with $ and % change YoY.

### Leverage and Debt Service
Total Debt change, Leverage old->new, FCC old->new.

### Liquidity
Cash change. Revolver: facility size, borrowings, availability.

## Credit Risks
3+ specific risks with actual data points.

## Mitigants
2+ specific mitigants with actual data points.

## Covenant Compliance
Leverage vs covenant with headroom. FCC vs minimum with headroom.

## Recommendation
Approve / Decline / Approve with conditions — with rationale.

## Data Gaps and Limitations
List any missing or unextracted data points.
"""

    resp = get_client().chat.completions.create(
        model=MEMO_MODEL,
        temperature=MEMO_TEMPERATURE,
        max_tokens=MEMO_MAX_TOKENS,
        messages=[
            {"role": "system", "content": MEMO_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    memo = resp.choices[0].message.content or ""
    full_memo = memo.strip() + "\n\n" + extracted_section
    return full_memo.strip()