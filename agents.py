from __future__ import annotations

import os
import json
import re
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

_client = None
_anthropic_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Ensure .env is set and loaded.")
        _client = OpenAI()
    return _client


def get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is missing. Ensure .env is set and loaded.")
        _anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _anthropic_client


# ============================================================
# EXPECTED KEYS (ENFORCE SHAPE)
# ============================================================

EXPECTED_INCOME_KEYS = [
    "revenue",
    "cost_of_sales",
    "sga_expense",
    "operating_income",
    "ebitda",
    "net_income",
    "interest_expense",
    "income_tax_expense",
    "depreciation_amortization",
    "rent_expense",
]

EXPECTED_BS_KEYS = [
    "cash",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "total_debt",
    "long_term_debt",
    "current_portion_long_term_debt",
    "revolver_facility_size",
    "revolver_borrowings",
    "revolver_availability",
]

EXPECTED_CF_KEYS = [
    "cfo",
    "cfi",
    "cff",
    "capex",
    "cash_paid_for_interest",
    "cash_paid_for_income_taxes",
    "dividends_distributions_paid",
]


# ============================================================
# EXTRACTION PROMPTS
# ============================================================

EXTRACTOR_SYSTEM_PROMPT = """
You are a financial statement extraction assistant for a commercial credit analyst.

Return ONLY valid JSON. No extra commentary.

Hard rules:
- The output MUST match the provided schema exactly.
- For each period object, each line item MUST be a single scalar number or null.
- NEVER put multiple years, multiple numbers, ranges, or commentary into a numeric field.
  If you see multiple years on a row (e.g., 2024 | 2023 | 2022), create multiple period objects instead.
- If you cannot confidently map a number to a field for a given period, set it to null.
- Do not compute derived metrics; derived_metrics must be {}.
- Do not compute EBITDA; leave ebitda null unless explicitly labeled as "EBITDA".
- Prioritize extracting fields required for EBITDA, FCC, and leverage: operating_income, depreciation_amortization,
  total_debt, current_portion_long_term_debt, cash_paid_for_interest, cash_paid_for_income_taxes, capex, and cfo.
"""

EXTRACTOR_SCHEMA_PROMPT = """
Return JSON with this exact structure:

{
  "statement_basis": "actual" | "thousands" | "millions",
  "periods": [
    {
      "period_name": "FY2024",
      "income_statement": {
        "revenue": number|null,
        "operating_income": number|null,
        "ebitda": number|null,
        "cost_of_sales": number|null,
        "sga_expense": number|null,
        "net_income": number|null,
        "interest_expense": number|null,
        "income_tax_expense": number|null,
        "depreciation_amortization": number|null,
        "rent_expense": number|null
      },
      "balance_sheet": {
        "cash": number|null,
        "total_assets": number|null,
        "total_liabilities": number|null,
        "total_equity": number|null,
        "total_debt": number|null,
        "long_term_debt": number|null,
        "current_portion_long_term_debt": number|null,
        "revolver_facility_size": number|null,
        "revolver_borrowings": number|null,
        "revolver_availability": number|null
      },
      "cash_flow": {
        "cfo": number|null,
        "cfi": number|null,
        "cff": number|null,
        "capex": number|null,
        "cash_paid_for_interest": number|null,
        "cash_paid_for_income_taxes": number|null,
        "dividends_distributions_paid": number|null
      },
      "derived_metrics": {},
      "notes": []
    }
  ],
  "validation_flags": []
}
"""


# ============================================================
# PARSING / NORMALIZATION HELPERS
# ============================================================

def _coerce_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip()
        text = text.lstrip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip("\n").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


def _to_float(value):
    """
    Convert common financial formats to float.
    Returns None if not a clean scalar number.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    s = value.strip()

    # Reject multi-column or multi-line blobs
    if "|" in s or "\n" in s:
        return None

    # Reject "two years in one field" patterns
    years = re.findall(r"\b20\d{2}\b", s)
    if len(set(years)) >= 2:
        return None

    # Remove currency symbols and common junk
    s = s.replace("$", "").replace("€", "").replace("£", "")
    s = re.sub(r"\[\d+\]$", "", s).strip()

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # Remove commas
    s = s.replace(",", "")

    # Must be a simple numeric token
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
    """
    Enforce strict schema shape and numeric scalar fields.
    """
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
    for i, p in enumerate(periods[:5]):  # keep max 5
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
# MD&A EXTRACTION (NEW)
# ============================================================

def _extract_mda_section(raw_text: str, max_chars: int = 350000) -> str:
    """
    Extract Management's Discussion and Analysis section from 10-K/10-Q

    IMPORTANT: This function now extracts MORE generously to ensure all segment
    discussions are captured. Segment results often appear throughout MD&A,
    not just in "Results of Operations".

    Enhanced to capture:
    - Results of Operations sections
    - Segment Results / Segment Performance sections
    - Business unit discussions anywhere in MD&A
    - All subsections within Item 7 up to Item 7A or Item 8
    """
    import logging
    logger = logging.getLogger("credit_ai")

    if not raw_text:
        return ""

    upper = raw_text.upper()

    # STRATEGY: Extract the ENTIRE Item 7 MD&A section (not just Results of Operations)
    # This ensures we capture all segment discussions regardless of where they appear

    # Find start of Item 7 MD&A
    mda_start_markers = [
        "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "ITEM 7 - MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "ITEM 7—MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "ITEM 7.MANAGEMENT'S DISCUSSION",
        "ITEM 7. MANAGEMENT",
        "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION",
        "MANAGEMENT'S DISCUSSION AND ANALYSIS",
    ]

    start_idx = -1
    start_marker_used = None
    for marker in mda_start_markers:
        idx = upper.find(marker)
        if idx != -1:
            # Verify this isn't just table of contents
            text_after = raw_text[idx:idx+500]
            if len(text_after.strip()) > 100:
                start_idx = idx
                start_marker_used = marker
                break

    if start_idx == -1:
        # Fallback: Look for Results of Operations
        results_markers = [
            "RESULTS OF OPERATIONS",
            "OPERATING RESULTS",
            "CONSOLIDATED RESULTS OF OPERATIONS",
        ]
        for marker in results_markers:
            idx = upper.find(marker)
            if idx != -1:
                text_after = raw_text[idx:idx+500]
                if len(text_after.strip()) > 100:
                    start_idx = idx
                    start_marker_used = marker
                    break

    if start_idx == -1:
        logger.warning("✗ Could not find MD&A section start")
        return ""

    logger.info(f"✓ Found MD&A start: '{start_marker_used}' at index {start_idx:,}")

    # Find end of MD&A - ONLY stop at Item 7A or Item 8, NOT at subsection headers
    # This ensures we capture ALL segment discussions
    mda_end_markers = [
        "ITEM 7A.",
        "ITEM 7A ",
        "ITEM 7A—",
        "ITEM 7A-",
        "ITEM 8.",
        "ITEM 8 ",
        "ITEM 8—",
        "ITEM 8-",
    ]

    end_idx = len(raw_text)
    for marker in mda_end_markers:
        idx = upper.find(marker, start_idx + 5000)  # Skip ahead to avoid false positives
        if idx != -1 and idx < end_idx:
            end_idx = idx
            logger.info(f"✓ Found MD&A end: '{marker}' at index {idx:,}")
            break

    mda_text = raw_text[start_idx:end_idx]

    # Log what we captured
    logger.info(f"✓ MD&A section extracted: {len(mda_text):,} characters")

    # Check if we have segment-related content
    segment_keywords = ["SEGMENT", "BUSINESS UNIT", "DIVISION", "OPERATING SEGMENT"]
    has_segments = any(kw in mda_text.upper() for kw in segment_keywords)
    if has_segments:
        logger.info("✓ Segment discussions found in MD&A")
    else:
        logger.warning("⚠ No segment keywords found in extracted MD&A")

    # Return full MD&A up to max_chars (increased to 350k)
    if len(mda_text) > max_chars:
        logger.warning(f"⚠ MD&A truncated from {len(mda_text):,} to {max_chars:,} characters")

    return mda_text[:max_chars]


def _summarize_mda_for_drivers(mda_text: str, extracted: ExtractionResult) -> str:
    """
    Read the ENTIRE MD&A and extract segment commentary including offsetting factors.

    IMPORTANT: This function sends the full MD&A to the model in a single call
    to ensure all segments are captured together. No chunking is used.
    """
    import logging
    import re
    logger = logging.getLogger("credit_ai")

    if not mda_text or not mda_text.strip():
        logger.warning("✗ No MD&A section found in document!")
        return "No MD&A section found."

    if not extracted.periods or len(extracted.periods) < 2:
        return "Insufficient period data for YoY driver analysis."

    p0 = extracted.periods[0]
    p1 = extracted.periods[1]

    # Log MD&A size before processing
    logger.info(f"✓ MD&A text received for summarization: {len(mda_text):,} characters")

    # PRE-SCAN: Detect potential segment names in the MD&A
    # This helps verify we're capturing all segments
    mda_upper = mda_text.upper()
    potential_segments = []

    # Look for common segment header patterns
    segment_patterns = [
        r"(?:THE\s+)?(\w+(?:\s+\w+)?)\s+SEGMENT['\"]?S?\s+(?:NET\s+SALES|REVENUE|RESULTS)",
        r"SEGMENT:\s*(\w+(?:\s+\w+)?)",
        r"(\w+(?:\s+\w+)?)\s+BUSINESS\s+UNIT",
        r"(\w+(?:\s+\w+)?)\s+OPERATING\s+SEGMENT",
    ]

    for pattern in segment_patterns:
        matches = re.findall(pattern, mda_upper)
        for match in matches:
            seg_name = match.strip()
            if seg_name and len(seg_name) > 2 and seg_name not in ["THE", "OUR", "THIS", "EACH"]:
                if seg_name not in potential_segments:
                    potential_segments.append(seg_name)

    if potential_segments:
        logger.info(f"✓ Pre-scan detected potential segments: {', '.join(potential_segments)}")
    else:
        logger.warning("⚠ Pre-scan found no segment headers - model will need to identify them")

    # Use full MD&A text (up to 300k chars to fit in context window)
    # GPT-4o has 128k token context, ~300k chars is roughly 75k tokens
    mda_for_prompt = mda_text[:300000]

    if len(mda_text) > 300000:
        logger.warning(f"⚠ MD&A truncated from {len(mda_text):,} to 300,000 characters for model context")

    # First, identify all business units/segments in the MD&A
    prompt = f"""
Read the ENTIRE MD&A section below from beginning to end and extract business driver commentary for ALL business units/segments.

MD&A TEXT LENGTH: {len(mda_for_prompt):,} characters
MD&A TEXT (READ COMPLETELY - DO NOT SKIP ANY SECTION):
{mda_for_prompt}

CRITICAL INSTRUCTIONS:

STEP 1: IDENTIFY ALL BUSINESS UNITS/SEGMENTS
- Scan the ENTIRE MD&A section
- Identify EVERY business unit, segment, or division discussed
- Common segment structures: by product line, geography, business division
- Look for headers like "Segment Results", "Business Unit Performance", or individual segment names
- List ALL segments before extracting details

STEP 2: EXTRACT COMMENTARY FOR EACH SEGMENT
- For EACH segment identified in Step 1, extract complete driver commentary
- Do NOT skip any segments - if you found 5 segments in Step 1, you must provide commentary for all 5
- Extract period comparison commentary (e.g., "{p1.period_name} vs {p0.period_name}")

STEP 3: READ COMPLETE PARAGRAPHS
- For each segment, read the ENTIRE paragraph discussing that segment
- Do not stop mid-paragraph - offsetting factors typically appear at the end

CRITICAL EXAMPLE (from Phosphates segment):

FULL TEXT IN MD&A:
"The Phosphates segment's net sales were $4.5 billion for the year ended December 31, 2024, compared to $4.7 billion for the same period a year ago. The decrease in net sales was primarily due to lower finished goods sales volumes, which unfavorably impacted net sales by approximately $310 million. In addition, Miski Mayo operations had an unfavorable impact of approximately $40 million compared to the prior year period due to lower selling prices. These impacts were partially offset by approximately $150 million due to higher finished product selling prices in the year current period."

YOU MUST EXTRACT:
Revenue drivers:
- Lower finished goods sales volumes: unfavorably impacted by ~$310M
- Miski Mayo lower selling prices: unfavorable impact of ~$40M
- OFFSETTING: Higher finished product selling prices: ~$150M (partially offset)

ANOTHER EXAMPLE (Gross Margin):

FULL TEXT:
"The current year gross margin was also unfavorably impacted by approximately $40 million related to selling a higher proportion of purchased tonnes compared to the prior year period, higher idle costs of approximately $40 million, primarily due to impacts from Hurricane Milton, and higher freight costs of approximately $10 million. In addition, Miski Mayo gross margin was approximately $30 million lower than the prior year primarily due to a decrease in selling prices. These impacts were partially offset by favorable impacts from higher finished goods selling prices of approximately $150 million and lower raw material costs, primarily sulfur as discussed below, of approximately $150 million."

YOU MUST EXTRACT:
Gross margin drivers:
- Higher proportion of purchased tonnes: unfavorably impacted by ~$40M
- Higher idle costs (Hurricane Milton): ~$40M
- Higher freight costs: ~$10M
- Miski Mayo lower selling prices: ~$30M lower
- OFFSETTING: Higher finished goods selling prices: ~$150M (partially offset)
- OFFSETTING: Lower raw material costs (sulfur): ~$150M (partially offset)

BEFORE YOU START EXTRACTION:
1. List all business units/segments you found in the MD&A
2. This ensures you don't miss any segments
3. Format: "SEGMENTS IDENTIFIED: [Segment 1], [Segment 2], [Segment 3], etc."

FORMAT YOUR RESPONSE:

SEGMENTS IDENTIFIED: [List ALL segments found in MD&A]

---

**SEGMENT: [Segment Name]**

REVENUE:
Net sales: $X.X billion vs $Y.Y billion

ALL Revenue drivers (extract EVERY driver mentioned - not just volume/pricing):
- [Driver 1 with dollar amount]
- [Driver 2 with dollar amount]
- [Driver 3 with dollar amount]
- [Include: volume, pricing, mix, FX, acquisitions, weather, etc.]
- OFFSETTING: [Factor 1 with dollar amount]
- OFFSETTING: [Factor 2 with dollar amount]

Volume/Price detail (if provided):
- Volumes: X.X million [units] vs Y.Y million [units]
- Price: $XXX/[unit] vs $YYY/[unit]

COGS (if discussed):
- [ALL cost drivers: raw materials, labor, overhead, production, etc.]

GROSS MARGIN:
Gross margin: $XXX million vs $YYY million

ALL Gross margin drivers (extract EVERY driver mentioned):
- [Driver 1: e.g., "Higher proportion of purchased tonnes: ~$40M unfavorable"]
- [Driver 2: e.g., "Higher idle costs (Hurricane Milton): ~$40M"]
- [Driver 3: e.g., "Higher freight costs: ~$10M"]
- [Driver 4: e.g., "Lower selling prices: ~$30M"]
- [Include: raw materials, labor, overhead, efficiency, mix, weather, etc.]
- OFFSETTING: [Factor 1 with dollar amount]
- OFFSETTING: [Factor 2 with dollar amount]

Input costs (if discussed):
- [Material 1]: $XXX vs $YYY per [unit]
- [Material 2]: $XXX vs $YYY per [unit]

---

[Repeat for EVERY segment in "SEGMENTS IDENTIFIED"]

CRITICAL FORMAT RULES:
1. Extract ALL drivers mentioned by management - not just volume and pricing
2. Include: raw materials, labor, mix, efficiency, weather, FX, acquisitions, idle costs, freight, etc.
3. List ALL offsetting factors with "OFFSETTING:" prefix
4. Include dollar amounts for each driver where provided
5. If MD&A says "partially offset by X and Y", you must have TWO offsetting bullets

MANDATORY RULES - THIS IS CRITICAL:

0. SCAN THE ENTIRE MD&A FOR ALL SEGMENTS FIRST:
   - Before extracting details, read through the ENTIRE MD&A section
   - Identify ALL business units, segments, or divisions discussed
   - List them at the beginning: "SEGMENTS IDENTIFIED: [all segments]"
   - Common mistake: extracting only the first 2-3 segments and stopping
   - You must extract commentary for EVERY segment you identify

1. Read the COMPLETE paragraph for each segment FROM START TO END:
   - Do NOT stop reading in the middle of a paragraph
   - Offsetting factors typically appear at the END of paragraphs
   - Read until you reach the next segment or section header

2. Extract ALL factors with dollar amounts - don't skip any:
   - Include negative impacts AND positive offsets
   - Include all segments, not just the first few mentioned

3. OFFSETTING FACTORS ARE MANDATORY:
   - Look for phrases: "These impacts were partially offset by", "partially offset by", "offset by", "These were partially offset by"
   - When you find these phrases, the factors that follow are OFFSETTING FACTORS
   - Mark them as "OFFSETTING:" with dollar amounts
   - There are often MULTIPLE offsetting factors - extract ALL of them

4. "Partially offset by" language may be:
   - In the same sentence
   - In a separate sentence immediately after the negative impacts
   - At the end of a long paragraph describing negative factors
   - STILL extract them as OFFSETTING even if separated

5. Include EVERY dollar amount mentioned:
   - Even small ones like $10M
   - Both negative impacts ($310M unfavorable) AND offsetting factors ($150M favorable)

6. Use management's exact language and numbers - don't paraphrase or summarize

7. SPECIFIC SEARCH STRATEGY:
   - For each segment paragraph, read to the very last sentence
   - If you see "These impacts were partially offset by", what follows is CRITICAL
   - Common pattern: [negative factors listed]... "These impacts were partially offset by [positive factors]"
   - Extract BOTH the negative factors AND the offsetting factors

The offsetting factors are ALWAYS present in MD&A when companies discuss YoY changes.
They appear after phrases like:
- "These impacts were partially offset by"
- "partially offset by"
- "offset by"
- "These were partially offset by"
- "were offset by"

YOUR JOB:
1. Find and list ALL business units/segments in the MD&A (don't miss any!)
2. Extract complete commentary for EVERY segment identified
3. Find and extract EVERY offsetting factor with its dollar amount for each segment

FINAL CHECK BEFORE SUBMITTING:
- Did you list all segments at the beginning?
- Did you provide commentary for EVERY segment you listed?
- Did you read complete paragraphs to the end?
- Did you extract offsetting factors for each segment?
"""
    
    try:
        resp = get_client().chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            max_tokens=12000,
            messages=[
                {
                    "role": "system",
                    "content": """You are a financial analyst extracting detailed segment-level MD&A commentary.

CRITICAL: Extract ALL drivers for Revenue, COGS, AND Gross Margin for EACH segment.

OUTPUT FORMAT:

First line: SEGMENTS IDENTIFIED: [Segment1], [Segment2], [Segment3], etc.

Then for EACH segment identified:

---

**SEGMENT: [Segment Name]**

REVENUE:
Net sales: $X.X billion vs $Y.Y billion (increase/decrease of $XXX million)

ALL Revenue drivers (list EVERY driver mentioned):
- [Driver 1]: [impact] by approximately $XXM
- [Driver 2]: [impact] by approximately $XXM
- [Driver 3]: [impact] by approximately $XXM
- [Continue for ALL drivers - volume, pricing, mix, FX, acquisitions, weather, etc.]
- OFFSETTING: [Factor 1]: ~$XXM (partially offset)
- OFFSETTING: [Factor 2]: ~$XXM (partially offset)

COST OF GOODS SOLD / COST OF SALES:
COGS: $X.X billion vs $Y.Y billion (increase/decrease of $XXX million)

ALL COGS drivers (list EVERY driver mentioned):
- [Driver 1]: [impact] by approximately $XXM
- [Driver 2]: [impact] by approximately $XXM
- [Raw material costs, labor, overhead, production volumes, efficiency, etc.]
- OFFSETTING: [Factor]: ~$XXM (partially offset)

Input costs (if discussed):
- [Material 1]: $XXX per [unit] vs $YYY per [unit]
- [Material 2]: $XXX per [unit] vs $YYY per [unit]

GROSS MARGIN:
Gross margin: $XXX million vs $YYY million (increase/decrease of $XXX million)
Gross margin %: XX.X% vs YY.Y%

ALL Gross margin drivers (list EVERY driver mentioned):
- [Driver 1]: [impact] by approximately $XXM
- [Driver 2]: [impact] by approximately $XXM
- [Pricing, costs, mix, volume leverage, efficiency, weather, etc.]
- OFFSETTING: [Factor 1]: ~$XXM (partially offset)
- OFFSETTING: [Factor 2]: ~$XXM (partially offset)

---

[REPEAT THE ABOVE FOR EVERY SEGMENT IN "SEGMENTS IDENTIFIED"]

EXTRACTION RULES:
1. Extract ALL segments mentioned in the MD&A - do not skip any
2. For EACH segment, extract Revenue, COGS, AND Gross Margin drivers
3. Include ALL drivers mentioned - not just volume and pricing
4. Include: raw materials, labor, overhead, mix, efficiency, weather, FX, acquisitions, etc.
5. ALWAYS extract offsetting factors with dollar amounts
6. If COGS is not discussed separately, note "COGS drivers not separately discussed"
7. Use exact numbers from the MD&A"""
                },
                {"role": "user", "content": prompt}
            ]
        )

        result = resp.choices[0].message.content or "No driver insights extracted."
        return result

    except Exception as e:
        return f"Error extracting MD&A insights: {str(e)}"


# ============================================================
# LEASE DATA EXTRACTION
# ============================================================

def extract_lease_data(raw_text: str, period_names: list[str]) -> dict:
    """
    Extract lease-related data from the Lease/Leases note for rent expense adjustment.

    Returns dict mapping period_name to dict of lease components:
    {
        "FY2024": {
            "rental_expense": 269.4,
            "operating_lease_cost": 87.2,
            "amortization_of_rou_assets": 45.5,
            "interest_on_lease_liabilities": 6.1,
            "short_term_lease_cost": 0.2,
            "variable_lease_cost": 19.5,
            "total_lease_cost": 158.5
        }
    }
    """
    import logging
    import datetime
    logger = logging.getLogger("credit_ai")

    if not raw_text or not period_names:
        return {}

    upper = raw_text.upper()

    # Find lease note section with more flexible matching
    lease_anchors = [
        "NOTE",  # Look for notes section first
    ]

    # Find notes section
    notes_idx = -1
    for anchor in lease_anchors:
        idx = upper.find(anchor)
        if idx != -1:
            notes_idx = idx
            break

    if notes_idx == -1:
        logger.info("DEBUG: Could not find notes section")
        return {}

    # Now search for lease-related content after notes section
    lease_keywords = [
        "LEASES",
        "LEASE EXPENSE",
        "OPERATING LEASE COST",
        "FINANCE LEASE COST",
        "RIGHT-OF-USE ASSETS",
        "RENTAL EXPENSE",
        "LEASE LIABILITIES"
    ]

    lease_section_start = -1
    for keyword in lease_keywords:
        idx = upper.find(keyword, notes_idx)  # Search after notes section
        if idx != -1 and idx < notes_idx + 500000:  # Within reasonable distance
            lease_section_start = idx
            logger.info(f"DEBUG: Found lease section at index {idx} with keyword '{keyword}'")
            break

    if lease_section_start == -1:
        logger.info("DEBUG: Could not find lease section in notes")
        return {}

    # Extract section around lease note (take ~50k chars)
    lease_section = raw_text[lease_section_start:lease_section_start + 50000]

    prompt = f"""
Extract lease-related expense data from the following text for the periods: {', '.join(period_names)}.

TEXT:
{lease_section[:30000]}

INSTRUCTIONS:
Look for a table showing lease expenses with the following line items:
- "Rental expense" or "Rent expense"
- "Operating lease cost"
- "Finance lease cost" (which may be broken down into):
  - "Amortization of right-of-use assets"
  - "Interest on lease liabilities"
- "Short-term lease cost"
- "Variable lease cost"
- "Total lease cost"

For EACH period ({', '.join(period_names)}), extract ALL available values.

Return ONLY valid JSON in this exact format:
{{{{
  "FY2024": {{{{
    "rental_expense": 269.4,
    "operating_lease_cost": 87.2,
    "amortization_of_rou_assets": 45.5,
    "interest_on_lease_liabilities": 6.1,
    "short_term_lease_cost": 0.2,
    "variable_lease_cost": 19.5,
    "total_lease_cost": 158.5
  }}}},
  "FY2023": {{{{
    "rental_expense": 252.1,
    ...
  }}}}
}}}}

If a value is not found, use null. Return ONLY the JSON, no other text.
"""

    try:
        resp = get_client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=2000,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You extract lease expense data from financial disclosures and return valid JSON."
                },
                {"role": "user", "content": prompt}
            ]
        )

        result_text = resp.choices[0].message.content or "{}"
        lease_data = json.loads(result_text)

        # Debug logging
        logger.info(f"DEBUG: Lease data extraction result: {json.dumps(lease_data, indent=2)}")

        return lease_data

    except Exception as e:
        print(f"Error extracting lease data: {e}")
        import traceback
        traceback.print_exc()
        return {}


def calculate_adjusted_rent_expense(
    rental_expense: Optional[float],
    operating_lease_cost: Optional[float],
    amortization_of_rou: Optional[float],
    interest_on_lease_liab: Optional[float],
    short_term_lease_cost: Optional[float],
    total_lease_cost: Optional[float]
) -> Optional[float]:
    """
    Calculate adjusted rent expense using the following logic:

    1. IF rental_expense exists with finance lease breakdown:
       Adjusted = Rental Expense - Amortization of ROU - Interest - Short-term

    2. ELSE IF operating_lease_cost is shown separately:
       Adjusted = Operating lease cost

    3. ELSE IF rental_expense exists without breakdown:
       Adjusted = Rental expense (as-is)

    4. ELSE IF only total_lease_cost:
       Adjusted = Total lease cost

    5. ELSE: None
    """

    # Scenario 1: Rental expense with finance lease breakdown
    if rental_expense is not None and (amortization_of_rou is not None or interest_on_lease_liab is not None or short_term_lease_cost is not None):
        adjusted = rental_expense
        if amortization_of_rou is not None:
            adjusted -= amortization_of_rou
        if interest_on_lease_liab is not None:
            adjusted -= interest_on_lease_liab
        if short_term_lease_cost is not None:
            adjusted -= short_term_lease_cost
        return adjusted

    # Scenario 2: Operating lease cost shown separately
    if operating_lease_cost is not None:
        return operating_lease_cost

    # Scenario 3: Rental expense without breakdown
    if rental_expense is not None:
        return rental_expense

    # Scenario 4: Only total lease cost
    if total_lease_cost is not None:
        return total_lease_cost

    # Scenario 5: No data
    return None


# ============================================================
# REVOLVER / CREDIT FACILITY EXTRACTION
# ============================================================

def extract_revolver_data(raw_text: str, period_names: list[str]) -> dict:
    """
    Extract revolving credit facility data from the Debt/Credit Facility notes.

    Returns dict mapping period_name to dict of revolver components:
    {
        "FY2024": {
            "revolver_facility_size": 1500.0,
            "revolver_borrowings": 350.0,
            "revolver_availability": 1150.0
        }
    }
    """
    import logging
    import re
    logger = logging.getLogger("credit_ai")

    if not raw_text or not period_names:
        return {}

    upper = raw_text.upper()

    # Strategy: Find the ENTIRE debt note section and read it completely
    # This ensures we capture ALL facilities, not just the first one mentioned

    debt_section_start = -1
    debt_section_end = -1
    note_header = None

    # Step 1: Look for the debt/financing NOTE section header
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
            logger.info(f"✓ Found debt note: '{note_header}' at index {debt_section_start:,}")
            break

    if debt_section_start != -1:
        # Found a debt note - extract from note start to next note (= entire debt note)
        remaining_text = upper[debt_section_start + 100:]
        next_note_pattern = r'NOTE\s+\d+[\.\:\-]'
        next_note = re.search(next_note_pattern, remaining_text)

        if next_note:
            # End at the next note
            debt_section_end = debt_section_start + 100 + next_note.start()
            logger.info(f"  Note ends at next NOTE at index {debt_section_end:,}")
        else:
            # No next note found - take a very large chunk (250k chars) to ensure completeness
            debt_section_end = min(len(raw_text), debt_section_start + 250000)
            logger.info(f"  No next NOTE found, extracting up to {debt_section_end:,}")

        logger.info(f"  Extracting ENTIRE debt note: {debt_section_end - debt_section_start:,} characters")
    else:
        # Fallback: search for debt keywords and take large surrounding context
        logger.warning("⚠ No debt NOTE header found, using keyword search fallback")
        keywords = ["CREDIT FACILIT", "REVOLVING CREDIT", "BORROWINGS", "DEBT"]
        for keyword in keywords:
            idx = upper.find(keyword)
            if idx != -1:
                note_header = f"Keyword: {keyword}"
                debt_section_start = max(0, idx - 30000)
                debt_section_end = min(len(raw_text), idx + 250000)
                logger.info(f"  Found '{keyword}' at {idx:,}, extracting {debt_section_end - debt_section_start:,} chars")
                break

    if debt_section_start == -1:
        logger.warning("✗ Could not find debt/financing section in document")
        return {}

    # Extract the section with surrounding context
    debt_section = raw_text[debt_section_start:debt_section_end]

    # Log preview of what we're extracting
    preview_start = debt_section[:800].replace('\n', ' ')[:400]
    preview_end = debt_section[-800:].replace('\n', ' ')[-400:]
    logger.info(f"  Preview (start): {preview_start}...")
    logger.info(f"  Preview (end): ...{preview_end}")

    prompt = f"""
You are reading the COMPLETE debt note from a 10-K filing. This note contains ALL information about the company's debt and credit facilities.

Your task: Extract ALL revolving credit facilities found anywhere in this note for periods: {', '.join(period_names)}.

COMPLETE DEBT NOTE SECTION ({len(debt_section):,} characters):
{debt_section[:120000]}

CRITICAL INSTRUCTIONS:

1. READ THE ENTIRE NOTE ABOVE - FIND ALL REVOLVING CREDIT FACILITIES:
   - **CRITICAL: Extract ALL revolving credit facilities separately - do not combine them**
   - Look for EACH distinct facility with its own name, size, and terms
   - Examples of separate facilities:
     * "Mosaic Credit Facility" - $2.5 billion
     * "Bilateral Revolving Facility" - $500 million
     * "ABL Facility" - $1 billion
   - For EACH facility found, extract:
     * Facility name (e.g., "Mosaic Credit Facility", "Senior Secured Revolving Credit Facility")
     * Facility size/commitment
     * Current borrowings
     * Availability
   - DO NOT aggregate or sum facilities together
   - Ignore letter of credit sub-limits, term loans, and non-revolving facilities
   - The facilities are typically described in the debt note, often at the beginning

2. EXTRACT THREE KEY METRICS:

   A. TOTAL FACILITY SIZE (commitment amount):
      - Look for phrases like:
        * "$1.5 billion revolving credit facility"
        * "aggregate commitments of $1,500 million"
        * "revolving credit facility with a maximum of $X"
      - This is the total capacity of the revolver

   B. CURRENT BORROWINGS/OUTSTANDING:
      - Look for the amount currently drawn/borrowed under the facility
      - Search in tables showing outstanding debt by period
      - Common phrases:
        * "borrowings under the revolving credit facility of $X million"
        * "outstanding at December 31, 2024: $X"
        * Look for the revolver row in debt maturity tables
      - IMPORTANT: If you see "net available borrowings" or "net availability", this is NOT the borrowings
      - If you find Letters of Credit amounts, borrowings can be calculated as:
        Borrowings = Facility Size - Net Availability - Letters of Credit
      - Some facilities have $0 borrowings but still show availability

   C. AVAILABILITY (Net Available Borrowings):
      - Look for explicit statements like:
        * "net available borrowings for revolving loans under the [Facility] were approximately $X billion"
        * "availability under the facility was $X million"
        * "remaining capacity", "available capacity", "undrawn availability"
      - This represents how much can still be borrowed
      - If facility has no borrowings and no letters of credit, availability = facility size

3. EXTRACT FOR EACH PERIOD:
   - {', '.join(period_names)}
   - Look for period-specific information (usually shown in tables or by date)
   - The facility size typically stays constant across periods
   - Borrowings and availability change by period

4. RETURN FORMAT:
Return ONLY valid JSON with an array of facilities for each period:
{{
  "FY2024": {{
    "facilities": [
      {{
        "name": "Mosaic Credit Facility",
        "facility_size": 2500.0,
        "borrowings": 0.0,
        "availability": 2500.0
      }},
      {{
        "name": "Bilateral Revolving Facility",
        "facility_size": 500.0,
        "borrowings": 0.0,
        "availability": 500.0
      }}
    ]
  }},
  "FY2023": {{
    "facilities": [
      {{
        "name": "Mosaic Credit Facility",
        "facility_size": 2500.0,
        "borrowings": 0.0,
        "availability": 2500.0
      }},
      {{
        "name": "Bilateral Revolving Facility",
        "facility_size": 500.0,
        "borrowings": 0.0,
        "availability": 500.0
      }}
    ]
  }}
}}

If only one facility exists, still return it as an array with one item.
If a value is not found, use null. Return ONLY the JSON, no other text.
"""

    try:
        # Use Claude Sonnet for high-quality extraction (behaves like Claude.ai)
        system_prompt = """You are a financial analyst specializing in debt structure analysis.

Your task is to read debt/financing note sections from 10-K filings and extract revolving credit facility information.

KEY CONCEPTS:
- Revolving Credit Facility: A credit line that can be drawn, repaid, and drawn again (like a corporate credit card)
- Facility Size: Maximum commitment amount (e.g., "$2.5 billion revolving credit facility")
- Borrowings/Outstanding: Current amount borrowed under the facility
- Net Availability: Amount that can still be borrowed = Facility Size - Borrowings - Letters of Credit
- Letters of Credit: May reduce availability even if there are no borrowings

CRITICAL NOTES:
- **EXTRACT ALL FACILITIES SEPARATELY**: If multiple revolving facilities exist, extract EACH one as a separate item
- DO NOT combine or aggregate facilities - list each one individually with its own name and metrics
- Examples:   * "Mosaic Credit Facility" - $2.5B + "Bilateral Facility" - $500MM = TWO separate facilities in the array
  * "Senior Credit Facility" - $1.5B only = ONE facility in the array
- For EACH facility, include its specific name (e.g., "Mosaic Credit Facility", not just "Credit Facility")
- "Net available borrowings of $2.5 billion" means AVAILABILITY, not borrowings
- If availability equals facility size, there are likely ZERO borrowings
- Look for explicit statements about "outstanding letters of credit" and actual "borrowings"
- Ignore letter of credit sub-limits, term loans, and old/terminated facilities

COMMON LOCATIONS:
- Opening paragraph describing the credit agreement
- Tables showing "Long-term Debt" or "Debt Outstanding by Type"
- Narrative sections about "Liquidity" or "Financing Arrangements"
- Statements like "As of December 31, 2024... we had outstanding letters of credit... net available borrowings were..."

Return valid JSON with the extracted data. Use the EXACT values stated in the document."""

        # Call Claude Sonnet for extraction
        client = get_anthropic_client()
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            temperature=0.0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract JSON from Claude's response
        result_text = resp.content[0].text if resp.content else "{}"

        # Claude might wrap JSON in markdown code blocks, so extract it
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        revolver_data = json.loads(result_text)

        # Detailed summary logging
        logger.info("=" * 80)
        logger.info("REVOLVER EXTRACTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Note header: {note_header}")
        logger.info(f"Text analyzed: {len(debt_section):,} characters (entire debt note)")
        logger.info(f"Periods requested: {', '.join(period_names)}")
        logger.info("")

        if revolver_data:
            for period, data in revolver_data.items():
                logger.info(f"Period: {period}")
                facility_size = data.get('revolver_facility_size')
                borrowings = data.get('revolver_borrowings')
                availability = data.get('revolver_availability')

                logger.info(f"  - Facility Size: ${facility_size:,.0f}M" if facility_size else "  - Facility Size: Not found")
                logger.info(f"  - Current Borrowings: ${borrowings:,.0f}M" if borrowings else "  - Current Borrowings: Not found")
                logger.info(f"  - Availability: ${availability:,.0f}M" if availability else "  - Availability: Not found")

                # Check for issues
                if availability == 0:
                    logger.warning(f"  ⚠ WARNING: Availability is 0 for {period} - this may be incorrect!")
                if facility_size and borrowings and not availability:
                    calculated = facility_size - borrowings
                    logger.warning(f"  ⚠ NOTE: Availability not found but could be calculated as ${calculated:,.0f}M")
                logger.info("")
        else:
            logger.warning("✗ No revolver data extracted from the section")

        logger.info("=" * 80)

        return revolver_data

    except Exception as e:
        logger.error(f"✗ Error extracting revolver data: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================
# TEXT SELECTION
# ============================================================

def _select_relevant_statement_text(raw_text: str, max_chars: int = 260000) -> str:
    if not raw_text:
        return ""
    upper = raw_text.upper()
    anchors = [
        "ITEM 8. FINANCIAL STATEMENTS",
        "FINANCIAL STATEMENTS",
        "CONSOLIDATED STATEMENTS OF OPERATIONS",
        "CONSOLIDATED STATEMENTS OF INCOME",
        "CONSOLIDATED BALANCE SHEETS",
        "CONSOLIDATED STATEMENTS OF CASH FLOWS",
        "STATEMENTS OF CASH FLOWS",
        "SUPPLEMENTAL CASH FLOW",
        "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS",
    ]
    idxs = [upper.find(a) for a in anchors]
    idxs = [i for i in idxs if i != -1]
    start = max(0, min(idxs) - 15000) if idxs else 0
    end = min(len(raw_text), start + max_chars)
    slice_text = raw_text[start:end]

    tables_idx = upper.find("TABLES:")
    if tables_idx != -1 and "TABLES:" not in slice_text:
        slice_text += "\n\n" + raw_text[tables_idx: min(len(raw_text), tables_idx + 220000)]

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
    amount_re = re.compile(
        r"(\(\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*\))|(\d{1,3}(?:,\d{3})+(?:\.\d+)?)|(\d+\.\d+)"
    )
    for i, line in enumerate(lines):
        if amount_re.search(line):
            for j in range(max(0, i - context_lines), min(len(lines), i + context_lines + 1)):
                keep[j] = True
    filtered = [lines[i].strip() for i in range(len(lines)) if keep[i] and lines[i].strip()]
    out = "\n".join(filtered) if filtered else text
    return out[:max_chars]


# ============================================================
# EXTRACTION (ENHANCED WITH MD&A)
# ============================================================

def extract_financials_from_text(raw_text: str) -> Tuple[ExtractionResult, str, str, str]:
    """
    Returns:
      (extracted: ExtractionResult, excerpt_sent_to_model: str, raw_model_text: str, mda_summary: str)
    """
    # Extract financial statements
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
- If the statements show multiple columns/years (e.g., 2024 | 2023 | 2022), return ONE period object per column year.
- Return up to the 5 most recent periods shown, most recent first.
- period_name should match the column label; if the column is "2024" use "FY2024".
- Do NOT put multiple years into a single numeric field.

GENERAL RULES:
- Prefer extracting values from TABLES rows when present (pipe-delimited).
- For each numeric field, return a SINGLE scalar number or null.
- Do NOT compute derived metrics; derived_metrics must be {{}}.
- Do NOT compute EBITDA; leave ebitda null unless explicitly labeled as EBITDA.

ALIAS / LABEL RULES (pick the clearest GAAP line items):

Income statement:
- revenue: Net revenue, Net sales, Total revenue(s), Revenues
- cost_of_sales: Cost of sales, Cost of products sold, Cost of goods sold (COGS), Cost of revenue
- sga_expense: Selling, general and administrative (SG&A), Selling and administrative, Operating expenses (if clearly SG&A),
  Selling and marketing + G&A (if broken out, prefer total)
- operating_income: Operating income, Operating profit, Income from operations, EBIT
- net_income: Net income, Net earnings, Net income attributable to
- interest_expense: Interest expense, Interest expense net, Interest and other debt expense
- income_tax_expense: Provision for income taxes, Income tax expense, Income taxes
- depreciation_amortization: Depreciation and amortization, D&A (may be in CFO reconciliation)
- rent_expense: Operating lease cost, Lease cost, Total lease cost, Rent expense, Rental expense, Lease expense, Operating lease expense
  Check MD&A and lease footnotes if not in statements. May be disclosed separately.
  Exclude: interest on lease liabilities / finance lease interest; ROU amortization alone.

Balance sheet:
- cash: Cash, Cash and cash equivalents
- total_debt: Total debt, Total borrowings, Debt
- long_term_debt: Long-term debt
- current_portion_long_term_debt: Current portion/maturities of long-term debt

Cash flow / supplemental:
- cfo: Net cash provided by operating activities
- cfi: Net cash used in investing activities
- cff: Net cash used in financing activities
- capex: Capital expenditures, Additions to PP&E, Payments for PP&E, Purchases of property and equipment
- cash_paid_for_interest: Cash paid for interest, Interest paid, Cash interest paid
- cash_paid_for_income_taxes: Cash paid for income taxes, Income taxes paid, Cash taxes paid
- dividends_distributions_paid: Dividends paid, Cash dividends paid, Distributions

SIGN CONVENTION:
- Capture values as shown. If outflows are negative in the table, keep them negative.

STATEMENT TEXT:
{combined_excerpt}

{EXTRACTOR_SCHEMA_PROMPT}
"""

    resp = get_client().chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_model_text = resp.choices[0].message.content or ""
    data = _coerce_json(raw_model_text)
    data = _normalize_payload(data)
    extracted = ExtractionResult.model_validate(data)

    # Extract lease data for rent expense adjustment
    import logging
    logger = logging.getLogger("credit_ai")

    period_names = [p.period_name for p in extracted.periods] if extracted.periods else []
    lease_data = extract_lease_data(raw_text, period_names) if period_names else {}

    # Store lease data in each period and calculate adjusted rent expense
    if extracted.periods:
        for p in extracted.periods:
            is_ = p.income_statement or {}

            # Get lease data for this period
            period_lease_data = lease_data.get(p.period_name, {})

            if period_lease_data:
                # Store raw lease components in notes for reference
                if not p.notes:
                    p.notes = []
                p.notes.append(f"Lease data: {json.dumps(period_lease_data)}")

                # Calculate adjusted rent expense using the lease data
                adjusted_rent = calculate_adjusted_rent_expense(
                    rental_expense=period_lease_data.get("rental_expense"),
                    operating_lease_cost=period_lease_data.get("operating_lease_cost"),
                    amortization_of_rou=period_lease_data.get("amortization_of_rou_assets"),
                    interest_on_lease_liab=period_lease_data.get("interest_on_lease_liabilities"),
                    short_term_lease_cost=period_lease_data.get("short_term_lease_cost"),
                    total_lease_cost=period_lease_data.get("total_lease_cost")
                )

                if adjusted_rent is not None:
                    is_["rent_expense"] = adjusted_rent

            # Fallback: Try to extract from MD&A if still missing
            if is_.get("rent_expense") is None:
                rent = _extract_rent_from_mda(raw_text, p.period_name)
                if rent is not None:
                    is_["rent_expense"] = rent

    # Extract revolver/credit facility data
    revolver_data = extract_revolver_data(raw_text, period_names) if period_names else {}

    # Store revolver data in each period's balance sheet
    if extracted.periods:
        for p in extracted.periods:
            bs_ = p.balance_sheet or {}

            # Get revolver data for this period
            period_revolver_data = revolver_data.get(p.period_name, {})

            if period_revolver_data:
                # Check if we have the new array format with multiple facilities
                facilities = period_revolver_data.get("facilities", [])

                if facilities:
                    # Aggregate totals from all facilities
                    total_size = sum(f.get("facility_size", 0) or 0 for f in facilities)
                    total_borrowings = sum(f.get("borrowings", 0) or 0 for f in facilities)
                    total_availability = sum(f.get("availability", 0) or 0 for f in facilities)

                    # Store aggregated values in balance sheet
                    bs_["revolver_facility_size"] = total_size if total_size > 0 else None
                    bs_["revolver_borrowings"] = total_borrowings if total_borrowings > 0 else 0.0
                    bs_["revolver_availability"] = total_availability if total_availability > 0 else None

                    # Store individual facilities for memo generation
                    bs_["revolver_facilities_detail"] = facilities

                    logger.info(f"DEBUG: Period {p.period_name} - Found {len(facilities)} facilities:")
                    for fac in facilities:
                        logger.info(f"  - {fac.get('name')}: Size=${fac.get('facility_size')}MM, " +
                                  f"Borrowings=${fac.get('borrowings')}MM, " +
                                  f"Availability=${fac.get('availability')}MM")
                    logger.info(f"  TOTAL: Size=${total_size}MM, Borrowings=${total_borrowings}MM, Availability=${total_availability}MM")
                else:
                    # Fallback to old format (single facility)
                    bs_["revolver_facility_size"] = period_revolver_data.get("revolver_facility_size")
                    bs_["revolver_borrowings"] = period_revolver_data.get("revolver_borrowings")
                    bs_["revolver_availability"] = period_revolver_data.get("revolver_availability")

                    logger.info(f"DEBUG: Period {p.period_name} - Added revolver data (old format): " +
                              f"Size={bs_.get('revolver_facility_size')}, " +
                              f"Borrowings={bs_.get('revolver_borrowings')}, " +
                              f"Availability={bs_.get('revolver_availability')}")

    # Extract and summarize MD&A
    mda_text = _extract_mda_section(raw_text, max_chars=250000)
    mda_summary = _summarize_mda_for_drivers(mda_text, extracted)

    return extracted, combined_excerpt, raw_model_text, mda_summary


def _extract_rent_from_mda(raw_text: str, period_name: str) -> Optional[float]:
    """
    Try to extract rent/lease expense from MD&A or footnotes when not in statements
    """
    upper = raw_text.upper()
    
    # Look for lease expense disclosures
    lease_markers = [
        "OPERATING LEASE COST",
        "LEASE EXPENSE",
        "OPERATING LEASE EXPENSE",
        "RENT EXPENSE",
    ]
    
    # Extract year from period name (e.g., "FY2024" -> "2024")
    import re
    year_match = re.search(r'20\d{2}', period_name)
    if not year_match:
        return None
    
    year = year_match.group()
    
    # Search for lease expense sections
    for marker in lease_markers:
        idx = upper.find(marker)
        if idx != -1:
            # Extract surrounding text (500 chars before and after)
            context = raw_text[max(0, idx-500):idx+1500]
            
            # Look for dollar amounts near the year
            # Pattern: year followed by amount, or amount followed by year
            amount_pattern = r'(?:' + year + r'[^\d]*?[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?))|(?:[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)[^\d]*?' + year + r')'
            
            matches = re.findall(amount_pattern, context, re.IGNORECASE)
            if matches:
                # Get first non-empty match
                for match_tuple in matches:
                    amount_str = match_tuple[0] or match_tuple[1]
                    if amount_str:
                        try:
                            amount = float(amount_str.replace(',', ''))
                            # Sanity check: rent should be reasonable (between 1M and 10B typically)
                            if 1 < amount < 10000:
                                return amount
                        except:
                            continue
    
    return None


# ============================================================
# COMPUTATIONS (DETERMINISTIC VIA metrics.py)
# ============================================================

def validate_and_compute(result: ExtractionResult) -> ExtractionResult:
    """
    Deterministic + idempotent computation of derived metrics
    """
    flags: list[str] = []

    for p in result.periods:
        p.income_statement = dict(p.income_statement or {})
        p.balance_sheet = dict(p.balance_sheet or {})
        p.cash_flow = dict(p.cash_flow or {})
        p.derived_metrics = {}

        is_ = p.income_statement
        bs = p.balance_sheet
        cf = p.cash_flow

        # EBITDA
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

        # FCF
        fcf = compute_free_cash_flow(cf.get("cfo"), cf.get("capex"))
        if fcf is not None:
            p.derived_metrics["free_cash_flow"] = float(fcf)

        # EBITDA margin
        ebitda_margin = compute_ebitda_margin(is_.get("ebitda"), is_.get("revenue"))
        if ebitda_margin is not None:
            p.derived_metrics["ebitda_margin"] = float(ebitda_margin)

        # FCC
        capex = cf.get("capex")
        cash_taxes = cf.get("cash_paid_for_income_taxes")
        cpltd = bs.get("current_portion_long_term_debt")
        cash_interest = cf.get("cash_paid_for_interest")

        fcc = compute_fcc(
            ebitda=is_.get("ebitda"),
            capex=capex,
            cash_taxes=cash_taxes,
            cpltd=cpltd,
            cash_interest=cash_interest,
        )

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
                    flags.append(f"{p.period_name}: FCC denominator is 0; FCC not computed.")
        else:
            p.derived_metrics["fcc"] = float(fcc)

            capex_mag = abs_outflow(capex)
            taxes_mag = abs_outflow(cash_taxes)
            int_mag = abs_outflow(cash_interest)

            if isinstance(is_.get("ebitda"), (int, float)) and capex_mag is not None and taxes_mag is not None:
                p.derived_metrics["fcc_numerator"] = float(is_.get("ebitda")) - float(capex_mag) - float(taxes_mag)
            if cpltd is not None and int_mag is not None:
                p.derived_metrics["fcc_denominator"] = float(cpltd) + float(int_mag)

        # Leverage
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

        # Altman Z-Score
        z_score = compute_altman_z_score(
            revenue=is_.get("revenue"),
            operating_income=is_.get("operating_income"),
            total_assets=bs.get("total_assets"),
            total_liabilities=bs.get("total_liabilities"),
            total_equity=bs.get("total_equity"),
            retained_earnings=None  # Not typically extracted, will use 0
        )
        if z_score is not None:
            p.derived_metrics["altman_z_score"] = float(z_score)
        else:
            flags.append(f"{p.period_name}: Insufficient data for Altman Z-Score calculation.")

        # PD Score (requires previously calculated metrics)
        # Calculate revenue growth if we have a prior period
        revenue_growth = None
        if len(result.periods) >= 2:
            # Find current period index
            curr_idx = result.periods.index(p)
            if curr_idx < len(result.periods) - 1:
                prior_p = result.periods[curr_idx + 1]
                curr_revenue = is_.get("revenue")
                prior_revenue = (prior_p.income_statement or {}).get("revenue")
                if curr_revenue is not None and prior_revenue is not None and prior_revenue != 0:
                    revenue_growth = (curr_revenue - prior_revenue) / prior_revenue

        pd_score, credit_rating = compute_pd_score(
            leverage=lev,
            fcc=fcc,
            ebitda_margin=p.derived_metrics.get("ebitda_margin"),
            altman_z=z_score,
            revenue_growth=revenue_growth,
            free_cash_flow=p.derived_metrics.get("free_cash_flow"),
            ebitda=is_.get("ebitda")
        )
        if pd_score is not None:
            p.derived_metrics["pd_score"] = int(pd_score)
            p.derived_metrics["credit_rating"] = credit_rating

    # De-dupe flags while preserving order
    seen = set()
    deduped = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            deduped.append(f)

    result.validation_flags = deduped
    return result


# ============================================================
# DETERMINISTIC FINANCIAL SUMMARY MEMO
# ============================================================

def generate_financial_summary_memo(extracted: ExtractionResult) -> str:
    """
    Deterministic memo built ONLY from extracted financials + derived_metrics.
    No borrower/covenant context. No invented facts.
    """
    if not extracted.periods:
        return "## Financial Summary\n- No periods extracted.\n"

    extracted = validate_and_compute(extracted)

    def m(x):
        if x is None:
            return "N/A"
        return f"${float(x):,.0f}"

    def xfmt(x):
        if x is None:
            return "N/A"
        return f"{float(x):.2f}x"

    def pct(x):
        if x is None:
            return "N/A"
        return f"{float(x) * 100:.1f}%"

    def yoy(new, old):
        if new is None or old is None:
            return None
        old = float(old)
        if old == 0:
            return None
        return (float(new) - old) / old

    p0 = extracted.periods[0]
    basis = extracted.statement_basis or "actual"
    is0 = p0.income_statement or {}
    bs0 = p0.balance_sheet or {}
    cf0 = p0.cash_flow or {}
    dm0 = p0.derived_metrics or {}

    revenue = is0.get("revenue")
    ebitda = is0.get("ebitda")
    ebitda_margin = dm0.get("ebitda_margin")
    debt = bs0.get("total_debt")
    leverage = dm0.get("leverage_total_debt_to_ebitda")
    fcc = dm0.get("fcc")
    cfo = cf0.get("cfo")
    capex = cf0.get("capex")
    fcf = dm0.get("free_cash_flow")
    cash = bs0.get("cash")

    memo = []
    memo.append("## Financial Summary (Extracted + Computed)")
    memo.append(f"*Statement basis:* `{basis}`  ")
    memo.append(f"*Most recent period:* `{p0.period_name}`\n")

    memo.append("### Executive Takeaways")
    memo.append(f"- Revenue: {m(revenue)}.")
    memo.append(f"- EBITDA: {m(ebitda)} ({pct(ebitda_margin)} margin).")
    memo.append(f"- Total debt: {m(debt)}; leverage (Debt/EBITDA): {xfmt(leverage)}.")
    memo.append(f"- FCC: {xfmt(fcc)}.")
    memo.append(f"- CFO: {m(cfo)}; Capex: {m(abs(capex) if isinstance(capex, (int, float)) else None)}; FCF: {m(fcf)}.")
    memo.append(f"- Liquidity (cash): {m(cash)}.\n")

    if len(extracted.periods) >= 2:
        p1 = extracted.periods[1]
        is1 = p1.income_statement or {}
        dm1 = p1.derived_metrics or {}

        memo.append(f"### YoY Trend ({p1.period_name} → {p0.period_name})")
        memo.append(f"- Revenue: {m(is1.get('revenue'))} → {m(revenue)} ({pct(yoy(revenue, is1.get('revenue')))})")
        memo.append(f"- EBITDA: {m(is1.get('ebitda'))} → {m(ebitda)} ({pct(yoy(ebitda, is1.get('ebitda')))})")
        memo.append(f"- EBITDA margin: {pct(dm1.get('ebitda_margin'))} → {pct(ebitda_margin)}")
        memo.append("")

    flags = list(extracted.validation_flags or [])
    if flags:
        memo.append("### Data Gaps / Extraction Notes")
        memo.extend([f"- {f}" for f in flags])
        memo.append("")

    return "\n".join(memo).strip() + "\n"


# ============================================================
# ENHANCED UNDERWRITING MEMO WITH MD&A INSIGHTS
# ============================================================

def _fmt_money(x: Optional[float], basis: str) -> str:
    if x is None:
        return "N/A"
    suffix = ""
    if basis == "millions":
        suffix = "MM"
    elif basis == "thousands":
        suffix = "K"
    return f"${x:,.0f}{suffix}"


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x*100:.1f}%"


def _fmt_x(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x:.2f}x"


def _safe_pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / old


def _format_extracted_data(extracted: ExtractionResult, basis: str) -> str:
    """
    Format extracted financial data as nested bullets for readability
    """
    suffix = "MM" if basis == "millions" else ("K" if basis == "thousands" else "")
    
    lines = ["## Extracted Financial Data", ""]
    lines.append(f"**Statement Basis:** {basis}")
    lines.append("")
    
    for period in extracted.periods:
        lines.append(f"### {period.period_name}")
        lines.append("")
        
        # Income Statement
        is_ = period.income_statement or {}
        if any(v is not None for v in is_.values()):
            lines.append("#### Income Statement")
            if is_.get("revenue") is not None:
                lines.append(f"- Revenue: ${is_['revenue']:,.0f}{suffix}")
            if is_.get("cost_of_sales") is not None:
                lines.append(f"- Cost of Sales: ${is_['cost_of_sales']:,.0f}{suffix}")
            if is_.get("sga_expense") is not None:
                lines.append(f"- SG&A Expense: ${is_['sga_expense']:,.0f}{suffix}")
            if is_.get("operating_income") is not None:
                lines.append(f"- Operating Income: ${is_['operating_income']:,.0f}{suffix}")
            if is_.get("ebitda") is not None:
                lines.append(f"- EBITDA: ${is_['ebitda']:,.0f}{suffix}")
            if is_.get("net_income") is not None:
                lines.append(f"- Net Income: ${is_['net_income']:,.0f}{suffix}")
            if is_.get("interest_expense") is not None:
                lines.append(f"- Interest Expense: ${is_['interest_expense']:,.0f}{suffix}")
            if is_.get("income_tax_expense") is not None:
                lines.append(f"- Income Tax Expense: ${is_['income_tax_expense']:,.0f}{suffix}")
            if is_.get("depreciation_amortization") is not None:
                lines.append(f"- Depreciation & Amortization: ${is_['depreciation_amortization']:,.0f}{suffix}")
            if is_.get("rent_expense") is not None:
                lines.append(f"- Rent Expense: ${is_['rent_expense']:,.0f}{suffix}")
            lines.append("")
        
        # Balance Sheet
        bs = period.balance_sheet or {}
        if any(v is not None for v in bs.values()):
            lines.append("#### Balance Sheet")
            if bs.get("cash") is not None:
                lines.append(f"- Cash: ${bs['cash']:,.0f}{suffix}")
            if bs.get("total_assets") is not None:
                lines.append(f"- Total Assets: ${bs['total_assets']:,.0f}{suffix}")
            if bs.get("total_liabilities") is not None:
                lines.append(f"- Total Liabilities: ${bs['total_liabilities']:,.0f}{suffix}")
            if bs.get("total_equity") is not None:
                lines.append(f"- Total Equity: ${bs['total_equity']:,.0f}{suffix}")
            if bs.get("total_debt") is not None:
                lines.append(f"- Total Debt: ${bs['total_debt']:,.0f}{suffix}")
            if bs.get("long_term_debt") is not None:
                lines.append(f"- Long-term Debt: ${bs['long_term_debt']:,.0f}{suffix}")
            if bs.get("current_portion_long_term_debt") is not None:
                lines.append(f"- Current Portion of Long-term Debt: ${bs['current_portion_long_term_debt']:,.0f}{suffix}")
            lines.append("")
        
        # Cash Flow
        cf = period.cash_flow or {}
        if any(v is not None for v in cf.values()):
            lines.append("#### Cash Flow Statement")
            if cf.get("cfo") is not None:
                lines.append(f"- Operating Cash Flow: ${cf['cfo']:,.0f}{suffix}")
            if cf.get("cfi") is not None:
                lines.append(f"- Investing Cash Flow: ${cf['cfi']:,.0f}{suffix}")
            if cf.get("cff") is not None:
                lines.append(f"- Financing Cash Flow: ${cf['cff']:,.0f}{suffix}")
            if cf.get("capex") is not None:
                lines.append(f"- Capex: ${cf['capex']:,.0f}{suffix}")
            if cf.get("cash_paid_for_interest") is not None:
                lines.append(f"- Cash Paid for Interest: ${cf['cash_paid_for_interest']:,.0f}{suffix}")
            if cf.get("cash_paid_for_income_taxes") is not None:
                lines.append(f"- Cash Paid for Income Taxes: ${cf['cash_paid_for_income_taxes']:,.0f}{suffix}")
            if cf.get("dividends_distributions_paid") is not None:
                lines.append(f"- Dividends/Distributions Paid: ${cf['dividends_distributions_paid']:,.0f}{suffix}")
            lines.append("")
        
        # Derived Metrics
        dm = period.derived_metrics or {}
        if any(v is not None for v in dm.values()):
            lines.append("#### Computed Metrics")
            if dm.get("ebitda_computed") is not None:
                lines.append(f"- EBITDA (Computed): ${dm['ebitda_computed']:,.0f}{suffix}")
            if dm.get("ebitda_margin") is not None:
                lines.append(f"- EBITDA Margin: {dm['ebitda_margin']*100:.1f}%")
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
    """Build a structured summary of all metrics for the LLM to reference"""
    if not extracted.periods:
        return "No periods available"
    
    lines = []
    
    # Most recent period
    p0 = extracted.periods[0]
    is0 = p0.income_statement or {}
    bs0 = p0.balance_sheet or {}
    cf0 = p0.cash_flow or {}
    dm0 = p0.derived_metrics or {}
    
    lines.append(f"MOST RECENT PERIOD: {p0.period_name}")
    lines.append(f"Revenue: {_fmt_money(is0.get('revenue'), basis)}")
    lines.append(f"Cost of Sales: {_fmt_money(is0.get('cost_of_sales'), basis)}")
    lines.append(f"SG&A: {_fmt_money(is0.get('sga_expense'), basis)}")
    lines.append(f"Operating Income: {_fmt_money(is0.get('operating_income'), basis)}")
    lines.append(f"EBITDA: {_fmt_money(is0.get('ebitda'), basis)}")
    lines.append(f"EBITDA Margin: {_fmt_pct(dm0.get('ebitda_margin'))}")
    lines.append(f"Net Income: {_fmt_money(is0.get('net_income'), basis)}")
    lines.append(f"")
    lines.append(f"Total Debt: {_fmt_money(bs0.get('total_debt'), basis)}")
    lines.append(f"Leverage (Debt/EBITDA): {_fmt_x(dm0.get('leverage_total_debt_to_ebitda'))}")
    lines.append(f"")
    lines.append(f"Operating Cash Flow: {_fmt_money(cf0.get('cfo'), basis)}")
    lines.append(f"Capex: {_fmt_money(abs_outflow(cf0.get('capex')), basis)}")
    lines.append(f"Free Cash Flow: {_fmt_money(dm0.get('free_cash_flow'), basis)}")
    lines.append(f"FCC: {_fmt_x(dm0.get('fcc'))}")
    lines.append(f"")
    lines.append(f"Cash: {_fmt_money(bs0.get('cash'), basis)}")
    
    # Prior period comparison if available
    if len(extracted.periods) >= 2:
        p1 = extracted.periods[1]
        is1 = p1.income_statement or {}
        bs1 = p1.balance_sheet or {}
        cf1 = p1.cash_flow or {}
        dm1 = p1.derived_metrics or {}
        
        lines.append(f"\nPRIOR PERIOD: {p1.period_name}")
        lines.append(f"Revenue: {_fmt_money(is1.get('revenue'), basis)}")
        lines.append(f"Cost of Sales: {_fmt_money(is1.get('cost_of_sales'), basis)}")
        lines.append(f"SG&A: {_fmt_money(is1.get('sga_expense'), basis)}")
        lines.append(f"Operating Income: {_fmt_money(is1.get('operating_income'), basis)}")
        lines.append(f"EBITDA: {_fmt_money(is1.get('ebitda'), basis)}")
        lines.append(f"EBITDA Margin: {_fmt_pct(dm1.get('ebitda_margin'))}")
        lines.append(f"Net Income: {_fmt_money(is1.get('net_income'), basis)}")
        lines.append(f"")
        lines.append(f"Total Debt: {_fmt_money(bs1.get('total_debt'), basis)}")
        lines.append(f"Leverage (Debt/EBITDA): {_fmt_x(dm1.get('leverage_total_debt_to_ebitda'))}")
        lines.append(f"")
        lines.append(f"Operating Cash Flow: {_fmt_money(cf1.get('cfo'), basis)}")
        lines.append(f"Capex: {_fmt_money(abs_outflow(cf1.get('capex')), basis)}")
        lines.append(f"Free Cash Flow: {_fmt_money(dm1.get('free_cash_flow'), basis)}")
        lines.append(f"FCC: {_fmt_x(dm1.get('fcc'))}")
        lines.append(f"")
        lines.append(f"Cash: {_fmt_money(bs1.get('cash'), basis)}")
        
        # Calculate YoY changes
        lines.append(f"\nYoY CHANGES ({p1.period_name} → {p0.period_name}):")
        
        rev_change = _safe_pct_change(is0.get('revenue'), is1.get('revenue'))
        if rev_change is not None:
            lines.append(f"Revenue: {_fmt_pct(rev_change)}")
        
        cogs_change = _safe_pct_change(is0.get('cost_of_sales'), is1.get('cost_of_sales'))
        if cogs_change is not None:
            lines.append(f"Cost of Sales: {_fmt_pct(cogs_change)}")
        
        ebitda_change = _safe_pct_change(is0.get('ebitda'), is1.get('ebitda'))
        if ebitda_change is not None:
            lines.append(f"EBITDA: {_fmt_pct(ebitda_change)}")
        
        margin_change = None
        if dm0.get('ebitda_margin') is not None and dm1.get('ebitda_margin') is not None:
            margin_change = dm0.get('ebitda_margin') - dm1.get('ebitda_margin')
            lines.append(f"EBITDA Margin: {margin_change*100:+.1f}pp")
        
        lev_change = _safe_pct_change(dm0.get('leverage_total_debt_to_ebitda'), 
                                      dm1.get('leverage_total_debt_to_ebitda'))
        if lev_change is not None:
            lines.append(f"Leverage: {_fmt_pct(lev_change)}")
        
        fcc_change = _safe_pct_change(dm0.get('fcc'), dm1.get('fcc'))
        if fcc_change is not None:
            lines.append(f"FCC: {_fmt_pct(fcc_change)}")
    
    return "\n".join(lines)


MEMO_SYSTEM_PROMPT = """
You are a senior commercial credit analyst writing comprehensive, metric-focused credit memos.

CRITICAL INSTRUCTION: YOU MUST INCLUDE DETAILED SEGMENT-LEVEL ANALYSIS WITH ACTUAL SEGMENT NAMES.

=== STEP 1: EXTRACT ACTUAL SEGMENT NAMES FROM MD&A ===

BEFORE WRITING ANYTHING:
1. Find "SEGMENTS IDENTIFIED:" in the MD&A insights - this lists the ACTUAL segment names
   Example: "SEGMENTS IDENTIFIED: Phosphates, Potash, Mosaic Fertilizantes"
2. Copy these EXACT names - they become your segment headers
3. For each segment, find its "**SEGMENT: [Name]**" block in the MD&A insights

CRITICAL WARNING:
- NEVER use generic names like "Segment 1", "Segment 2", "Segment 3"
- NEVER use placeholder text - always use the ACTUAL segment names from MD&A
- If MD&A says "SEGMENTS IDENTIFIED: Phosphates, Potash, Mosaic Fertilizantes" then your headers MUST be:
  - **Phosphates**:
  - **Potash**:
  - **Mosaic Fertilizantes**:

=== EXACT OUTPUT FORMAT FOR FINANCIAL PERFORMANCE ANALYSIS ===

### Revenue
Revenue [increased/decreased] by [%]% YoY from $[amount]MM in [prior] to $[amount]MM in [current].

**Segment Analysis:**

  - **Phosphates**: Revenue was $4.5B in FY2024 vs $4.7B in FY2023. Decrease of $200M driven by:
    - Lower finished goods sales volumes: unfavorably impacted by approximately $310M
    - Miski Mayo lower selling prices: unfavorable impact of approximately $40M
    - Partially offset by higher finished product selling prices of approximately $150M

  - **Potash**: Revenue was $2.8B in FY2024 vs $3.1B in FY2023. Decrease of $300M driven by:
    - Lower average selling prices: approximately $250M
    - Lower sales volumes: approximately $80M
    - Partially offset by favorable product mix of approximately $30M

  - **Mosaic Fertilizantes**: Revenue was $X.XB in FY2024 vs $Y.YB in FY2023. [Direction] of $XXXM driven by:
    - [ALL drivers from that segment's MD&A block]

(NOTE: The above uses example segment names. YOU MUST use the ACTUAL names from "SEGMENTS IDENTIFIED:" in your MD&A)

### Cost of Goods Sold
COGS [increased/decreased] by [%]% YoY from $[amount]MM in [prior] to $[amount]MM in [current].

**Segment Analysis:**

  - **[ACTUAL Segment Name 1 from MD&A]**: COGS was $X.XB vs $Y.YB. [Direction] of $XXXM driven by:
    - [Raw material cost changes - name specific materials]: approximately $XXM
    - [Labor/overhead changes]: approximately $XXM
    - [Production volume impacts]: approximately $XXM
    - [ALL other COGS drivers from that segment's MD&A block]
    - Partially offset by [offsetting factors] of approximately $XXM

  - **[ACTUAL Segment Name 2 from MD&A]**: COGS was $X.XB vs $Y.YB. [Direction] of $XXXM driven by:
    - [ALL COGS drivers from that segment's MD&A block]

  - **[ACTUAL Segment Name 3 from MD&A]**: [Continue for EVERY segment in SEGMENTS IDENTIFIED]

### Gross Margin
Gross margin [increased/decreased] from [%]% in [prior] to [%]% in [current] (change of [X]pp).

**Segment Analysis:**

  - **[ACTUAL Segment Name 1 from MD&A]**: Gross margin was $XXXM vs $YYYM. [Direction] of $XXXM driven by:
    - [Pricing impacts]: approximately $XXM
    - [Cost impacts - raw materials, labor, etc.]: approximately $XXM
    - [Mix impacts]: approximately $XXM
    - [Weather/one-time items]: approximately $XXM
    - [ALL other drivers from that segment's MD&A block]
    - Partially offset by [offsetting factor 1] of approximately $XXM
    - Partially offset by [offsetting factor 2] of approximately $XXM

  - **[ACTUAL Segment Name 2 from MD&A]**: Gross margin was $XXXM vs $YYYM. [Direction] driven by:
    - [ALL gross margin drivers from that segment's MD&A block]

  - **[ACTUAL Segment Name 3 from MD&A]**: [Continue for EVERY segment]

=== CRITICAL RULES ===

1. USE ACTUAL SEGMENT NAMES - NEVER USE "SEGMENT 1", "SEGMENT 2", ETC.:
   - Read "SEGMENTS IDENTIFIED:" line in MD&A to get the real segment names
   - Use those EXACT names as your bold headers (e.g., **Phosphates**, **Potash**)
   - This is MANDATORY - generic segment numbers are NEVER acceptable

2. SEGMENT ANALYSIS IS MANDATORY:
   - You MUST include "**Segment Analysis:**" header under Revenue, COGS, and Gross Margin
   - You MUST have a sub-section for EACH segment listed in "SEGMENTS IDENTIFIED:"
   - If SEGMENTS IDENTIFIED shows 3 segments, you must have 3 segment sub-sections

3. EXTRACT ALL DRIVERS - NOT JUST VOLUME AND PRICING:
   - Read each segment's "**SEGMENT:**" block in MD&A completely
   - Include EVERY driver management discusses:
     * Volume changes
     * Pricing changes
     * Mix/product mix changes
     * Raw material costs (name specific materials: sulfur, ammonia, potash, etc.)
     * Labor and overhead costs
     * Currency/FX impacts
     * Weather events (hurricanes, droughts, etc.)
     * Idle costs, production issues
     * Freight and logistics costs
     * Acquisitions/divestitures
     * One-time items
     * Efficiency gains/losses
     * ANY other factor mentioned
   - Each driver should have a dollar impact where provided

4. INCLUDE ALL OFFSETTING FACTORS:
   - Look for "OFFSETTING:" tags in each segment's MD&A block
   - Include ALL offsetting factors with dollar amounts
   - Each offsetting factor gets its own bullet point

5. USE ACTUAL DATA FROM MD&A:
   - Copy exact numbers from MD&A insights
   - If data is not provided for a specific segment, note "Data not separately disclosed for this segment"

TONE: Professional, factual, comprehensive - capture ALL drivers management discusses for EVERY segment
"""

MEMO_MODEL = os.getenv("MEMO_MODEL", "gpt-4o-mini")
MEMO_MAX_TOKENS = int(os.getenv("MEMO_MAX_TOKENS", "6000"))  # Increased for detailed segment breakdowns
MEMO_TEMPERATURE = float(os.getenv("MEMO_TEMPERATURE", "0.1"))  # Low for factual reporting


def generate_underwriting_memo(
    borrower: BorrowerProfile,
    covenants: CovenantSet,
    extracted: ExtractionResult,
    mda_summary: str = "",
) -> str:
    """
    Generate metric-focused underwriting memo with YoY comparisons using MD&A insights
    """
    extracted = validate_and_compute(extracted)
    
    if not extracted.periods:
        return "## Credit Memo\n\nNo financial periods available for analysis."
    
    basis = extracted.statement_basis or "actual"
    suffix = "MM" if basis == "millions" else ("K" if basis == "thousands" else "")
    p0 = extracted.periods[0]
    
    # Build formatted extracted data section
    extracted_section = _format_extracted_data(extracted, basis)
    
    # Build metric summary for the LLM
    metric_data = _build_metric_summary(extracted, basis)
    
    user_prompt = f"""
Write a credit memo for the following borrower.

BORROWER INFORMATION:
- Name: {borrower.name}
- Industry: {borrower.industry}
- Facility Type: {borrower.facility_type}
- Use of Proceeds: {borrower.use_of_proceeds}

COVENANTS:
- Max Total Leverage: {covenants.max_total_leverage if covenants.max_total_leverage else "Not specified"}
- Min FCC: {covenants.min_fcc if covenants.min_fcc else "Not specified"}

FINANCIAL DATA (statement basis: {basis}):
{metric_data}

MD&A BUSINESS DRIVER INSIGHTS:
{mda_summary}

VALIDATION FLAGS / DATA GAPS:
{json.dumps(extracted.validation_flags, indent=2) if extracted.validation_flags else "None"}

REQUIRED STRUCTURE:

## Executive Summary
- [Overall credit assessment in 2-3 bullets]

## Credit Risk Assessment

### PD Score: [score]/12 - [Credit Rating]
- **Scale**: 1-3 = Investment Grade | 4-6 = Non-Investment Grade | 7-9 = Speculative | 10-11 = High Risk | 12 = Default/Distressed
- **Interpretation**: [Brief assessment based on score - e.g., "Strong credit profile with low default risk" or "Elevated credit risk requires close monitoring"]

### Altman Z-Score: [score]
- **Interpretation**: Z > 2.9 = Safe Zone | 1.23-2.9 = Grey Zone | Z < 1.23 = Distress Zone
- **Assessment**: [Current zone and implications]

### Key Credit Drivers:
- Leverage: [X.XX]x ([above/below] target/covenant of [Y.YY]x)
- Fixed Charge Coverage: [X.XX]x ([above/below] minimum of [Y.YY]x)
- EBITDA Margin: [XX.X]%
- Liquidity Position: Cash of $[XXX]MM + $[YYY]MM revolver availability = $[ZZZ]MM total liquidity

## Financial Performance Analysis

CRITICAL: You MUST include detailed SEGMENT ANALYSIS using ACTUAL SEGMENT NAMES from MD&A.
1. Find "SEGMENTS IDENTIFIED:" in MD&A - this lists ALL segment names (e.g., "Phosphates, Potash, Mosaic Fertilizantes")
2. Use those EXACT names as your segment headers - NEVER use "Segment 1", "Segment 2", etc.
3. For each segment, find its "**SEGMENT:**" block and extract ALL drivers

### Revenue
Revenue [increased/decreased] by [%]% YoY from $[amount]MM in [prior] to $[amount]MM in [current].

**Segment Analysis:**

(Use ACTUAL segment names from "SEGMENTS IDENTIFIED:" - example format below)

  - **Phosphates**: Revenue was $4.5B in FY2024 vs $4.7B in FY2023. Decrease of $200M driven by:
    - Lower finished goods sales volumes: unfavorably impacted by approximately $310M
    - Miski Mayo lower selling prices: unfavorable impact of approximately $40M
    - Partially offset by higher finished product selling prices of approximately $150M

  - **Potash**: Revenue was $2.8B in FY2024 vs $3.1B in FY2023. Decrease of $300M driven by:
    - [ALL drivers from Potash section of MD&A with dollar amounts]
    - Partially offset by [offsetting factors from MD&A]

  - **Mosaic Fertilizantes**: Revenue was $X.XB vs $Y.YB. [Direction] driven by:
    - [ALL drivers from Mosaic Fertilizantes section of MD&A]

(Replace above example names with ACTUAL names from your MD&A "SEGMENTS IDENTIFIED:" line)

### Cost of Goods Sold
COGS [increased/decreased] by [%]% YoY from $[amount]MM in [prior] to $[amount]MM in [current].

**Segment Analysis:**

(Use ACTUAL segment names from "SEGMENTS IDENTIFIED:")

  - **[First segment name from MD&A]**: COGS was $X.XB vs $Y.YB. [Direction] of $XXXM driven by:
    - [Raw material costs - name specific materials]: approximately $XXM
    - [Labor/overhead changes]: approximately $XXM
    - [Production volume impacts]: approximately $XXM
    - [ALL other COGS drivers from that segment's MD&A block]
    - Partially offset by [offsetting factors] of approximately $XXM

  - **[Second segment name from MD&A]**: COGS was $X.XB vs $Y.YB. [Direction] driven by:
    - [ALL COGS drivers from that segment's MD&A block]

  - **[Third segment name from MD&A]**: [Continue for EVERY segment]

### Gross Margin
Gross margin [increased/decreased] from [%]% in [prior] to [%]% in [current] (change of [X]pp).

**Segment Analysis:**

(Use ACTUAL segment names from "SEGMENTS IDENTIFIED:")

  - **[First segment name from MD&A]**: Gross margin was $XXXM vs $YYYM. [Direction] of $XXXM driven by:
    - [Pricing impacts]: approximately $XXM
    - [Cost impacts - raw materials, labor, etc.]: approximately $XXM
    - [Mix impacts]: approximately $XXM
    - [Weather/one-time items]: approximately $XXM
    - [ALL other drivers from that segment's MD&A block]
    - Partially offset by [offsetting factor 1] of approximately $XXM
    - Partially offset by [offsetting factor 2] of approximately $XXM

  - **[Second segment name from MD&A]**: Gross margin was $XXXM vs $YYYM. [Direction] driven by:
    - [ALL gross margin drivers from that segment's MD&A block]

  - **[Third segment name from MD&A]**: [Continue for EVERY segment]

### EBITDA & Margins
- EBITDA: [increased/decreased] by [X]% YoY from $[old][MM/K] in [old period] to $[new][MM/K] in [new period] due to [specific operational drivers from MD&A]
- EBITDA Margin: [increased/decreased] by [X]pp YoY from [old]% in [old period] to [new]% in [new period] due to [margin drivers from MD&A]

### Profitability Drivers
- SG&A Expense: [increased/decreased] by [X]% YoY from $[old][MM/K] in [old period] to $[new][MM/K] in [new period] due to [primary drivers from MD&A], partially offset by [offsetting factors from MD&A]

### Cash Flow Metrics
- Operating Cash Flow: [increased/decreased] by [X]% YoY from $[old][MM/K] in [old period] to $[new][MM/K] in [new period] due to [working capital changes, profitability impacts from MD&A]
- Capex: [increased/decreased] by [X]% YoY from $[old][MM/K] in [old period] to $[new][MM/K] in [new period] due to [specific capital projects or efficiency initiatives from MD&A]
- Free Cash Flow: [increased/decreased] by [X]% YoY from $[old][MM/K] in [old period] to $[new][MM/K] in [new period]

### Leverage & Debt Service
- Total Debt: [increased/decreased] by [X]% YoY from $[old][MM/K] in [old period] to $[new][MM/K] in [new period]
- Leverage (Debt/EBITDA): [increased/decreased] from [old]x in [old period] to [new]x in [new period], [well below/below/near/above] the covenant maximum of [covenant level]x
- FCC (Fixed Charge Coverage): [increased/decreased] from [old]x in [old period] to [new]x in [new period], [comfortably above/above/near/below] the covenant minimum of [covenant level]x

### Liquidity
- Cash: [increased/decreased] by [X]% YoY from $[old][MM/K] in [old period] to $[new][MM/K] in [new period]
- Revolving Credit Facility:
  - Total Facility Size: $[size][MM/K]
  - Current Borrowings: $[borrowings][MM/K] ([X]% utilized)
  - Available Capacity: $[availability][MM/K]

## Credit Risks
- [Specific risk based on MD&A and financials]
- [Specific risk based on MD&A and financials]
- [Specific risk based on MD&A and financials]

## Mitigants
- [Specific mitigant based on MD&A and financials]
- [Specific mitigant based on MD&A and financials]

## Covenant Compliance
- [Assessment vs stated covenants with specific numbers]

## Recommendation
- [Approve/Decline/Approve with conditions]

## Data Gaps & Limitations
- [List any missing data points]

CRITICAL INSTRUCTIONS:

0. CREDIT RISK ASSESSMENT SECTION (MANDATORY):
   - ALWAYS include the Credit Risk Assessment section immediately after Executive Summary
   - Extract PD Score, Credit Rating, and Altman Z-Score from the METRIC DATA provided

1. USE ACTUAL SEGMENT NAMES - THIS IS CRITICAL:
   - NEVER use generic names like "Segment 1", "Segment 2", "Segment 3" - these are NEVER acceptable
   - Find "SEGMENTS IDENTIFIED:" in MD&A insights to get the ACTUAL segment names
   - Example: If MD&A says "SEGMENTS IDENTIFIED: Phosphates, Potash, Mosaic Fertilizantes"
     Your headers MUST be: **Phosphates**, **Potash**, **Mosaic Fertilizantes**
   - Use these EXACT names as bold headers in your segment analysis

2. SEGMENT ANALYSIS IS MANDATORY - THIS IS THE MOST IMPORTANT REQUIREMENT:
   - For Revenue, COGS, AND Gross Margin, you MUST include "**Segment Analysis:**" header
   - Under each "**Segment Analysis:**" header, include a sub-section for EVERY segment in "SEGMENTS IDENTIFIED:"
   - If SEGMENTS IDENTIFIED lists 3 segments, you must have 3 segment sub-sections with ACTUAL names
   - Each segment sub-section MUST include ALL drivers from that segment's MD&A block
   - DO NOT skip any segments - every segment needs its own detailed analysis

3. DYNAMIC EXTRACTION FROM MD&A (MANDATORY):
   - First, find "SEGMENTS IDENTIFIED:" line in MD&A insights - this lists ALL segments to analyze
   - These could be ANY names: product lines, geographies, business units, etc.
   - For EACH segment listed, find its "**SEGMENT:**" section and extract ALL data
   - Use the EXACT segment names from MD&A - do NOT use hardcoded names or generic placeholders

3. REVENUE, COGS, GROSS MARGIN FORMAT:
   - One-line summary with actual % change and dollar amounts
   - Then "**Segment Analysis:**" header
   - Then indented segment breakdown with ALL drivers for EACH segment from "SEGMENTS IDENTIFIED"
   - ALL values must come from the MD&A insights provided

4. EXTRACT ALL DRIVERS - NOT JUST VOLUME AND PRICING:
   - Include EVERY driver mentioned in MD&A for each segment, such as:
     * Volume changes
     * Pricing/selling price changes
     * Product mix shifts
     * Raw material costs (sulfur, ammonia, energy, etc.)
     * Labor and overhead costs
     * Currency/FX impacts
     * Weather events (hurricanes, droughts)
     * Acquisitions or divestitures
     * Operating efficiency gains/losses
     * Idle costs or production issues
     * Freight and logistics costs
     * One-time items
     * ANY other factor management discusses
   - Each driver should have a dollar impact where provided

5. OFFSETTING FACTORS (MANDATORY):
   - Every segment MUST include offsetting factors from "OFFSETTING:" tags in MD&A
   - Format: "Partially offset by [factor] of approximately $XXX million"
   - Include ALL offsetting factors mentioned, not just one

6. LIQUIDITY AND REVOLVER: Include revolving credit facility details when available
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
    
    # Append formatted extracted data
    full_memo = memo.strip() + "\n\n" + extracted_section
    
    return full_memo.strip()