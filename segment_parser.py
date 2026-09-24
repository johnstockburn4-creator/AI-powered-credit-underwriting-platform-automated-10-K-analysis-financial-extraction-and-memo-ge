"""
segment_parser.py
-----------------
Calls Claude to parse a free-text MD&A summary into structured segment data
suitable for rendering in the narrative Word document.

Designed to be used by doc_builder.build_word_narrative() before invoking
build_narrative_doc_v2.js.

Usage:
    from segment_parser import prepare_segment_data
    seg_data = prepare_segment_data(mda_summary, period_names=["FY2024","FY2023"])
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("credit_ai")

SEGMENT_PARSER_SYSTEM = """You are a financial data extraction assistant.
Your job is to parse MD&A narrative text and return ONLY valid JSON — no preamble, no markdown fences, no explanation.

Extract:
1. Consolidated-level drivers (revenue, COGS, gross margin) — brief, specific bullets
2. Per-segment data: revenue rows with numbers, COGS rows, driver bullets

Rules:
- Extract numbers as plain floats (e.g. 3772.9 not "$3,772.9M")  
- If a number is not explicitly stated, use null — never guess or fabricate
- Keep each bullet concise: one specific point, max 20 words
- "negative_drivers" = factors that reduced revenue; "positive_drivers" = factors that increased/offset
- "cost_increasing" = factors that increased COGS; "cost_reducing" = factors that reduced COGS
- Revenue rows: list sub-items first (e.g. North America, International), Total last
- If a segment has no COGS data in the text, return an empty rows array
- Return ONLY the JSON object, nothing else"""

SEGMENT_PARSER_TEMPLATE = """Parse the following MD&A summary (current period: {curr}, prior period: {prior}).

Return this exact JSON structure:
{{
  "consolidated_drivers": {{
    "revenue": {{
      "key_drivers": ["<bullet>", ...],
      "offsetting_factors": ["<bullet>", ...]
    }},
    "cogs": {{
      "cost_increasing": ["<bullet>", ...],
      "cost_reducing": ["<bullet>", ...]
    }},
    "gross_margin": {{
      "commentary": ["<bullet>", ...]
    }}
  }},
  "segments": [
    {{
      "name": "<Segment Name>",
      "revenue": {{
        "rows": [
          {{"label": "<row label>", "curr": <float or null>, "prior": <float or null>}},
          {{"label": "Total Net Sales", "curr": <float or null>, "prior": <float or null>}}
        ],
        "negative_drivers": ["<bullet>", ...],
        "positive_drivers": ["<bullet>", ...]
      }},
      "cogs": {{
        "rows": [
          {{"label": "Total COGS", "curr": <float or null>, "prior": <float or null>}}
        ],
        "cost_increasing": ["<bullet>", ...],
        "cost_reducing": ["<bullet>", ...]
      }}
    }}
  ]
}}

MD&A SUMMARY:
{mda_summary}"""


def prepare_segment_data(
    mda_summary: str,
    period_names: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> Optional[Dict]:
    """
    Call Claude to parse the MD&A summary into structured segment data.

    Args:
        mda_summary:  The full MD&A summary string from the extraction pipeline
        period_names: [current_period, prior_period] e.g. ["FY2024", "FY2023"]
        api_key:      Anthropic API key (defaults to ANTHROPIC_API_KEY env var)

    Returns:
        dict with keys: consolidated_drivers, segments
        Returns None on failure (caller should degrade gracefully)
    """
    if not mda_summary or not mda_summary.strip():
        logger.warning("prepare_segment_data: empty mda_summary, skipping")
        return None

    try:
        import anthropic
    except ImportError:
        logger.error("prepare_segment_data: anthropic package not installed")
        return None

    curr  = period_names[0] if period_names else "Current Period"
    prior = period_names[1] if period_names and len(period_names) > 1 else "Prior Period"

    prompt = SEGMENT_PARSER_TEMPLATE.format(
        curr=curr,
        prior=prior,
        mda_summary=mda_summary[:80_000],   # guard against very long summaries
    )

    try:
        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SEGMENT_PARSER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()

        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```").strip()

        parsed = json.loads(raw)
        logger.info(
            f"prepare_segment_data: parsed {len(parsed.get('segments', []))} segments, "
            f"{len(parsed.get('consolidated_drivers', {}).get('revenue', {}).get('key_drivers', []))} revenue drivers"
        )
        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"prepare_segment_data: JSON parse failed — {e}. Raw response: {raw[:300]}")
        return None
    except Exception as e:
        logger.error(f"prepare_segment_data: API call failed — {e}")
        return None


def build_word_narrative_v2(
    run_record,
    borrower_name: str,
    output_path: str,
    memo_markdown: Optional[str] = None,
    mda_summary: Optional[str] = None,
    covenants: Optional[Dict] = None,
    segment_data: Optional[Dict] = None,   # pass pre-computed, or let this function compute it
) -> str:
    """
    Produces the v2 narrative analyst Word doc.

    If segment_data is None and mda_summary is provided, calls prepare_segment_data()
    automatically to parse segment structure from the MD&A text.

    Requires:
      - node installed
      - build_narrative_doc_v2.js in the same directory as this file
    """
    import subprocess, tempfile, json as _json, os as _os

    extracted_json = getattr(run_record, "extracted_json", None) or {}
    period_names = [p.get("period_name", "") for p in (extracted_json.get("periods") or [])]

    # Parse MD&A into structured segment data if not already provided
    if segment_data is None and mda_summary:
        logger.info("build_word_narrative_v2: parsing segment data from MD&A summary...")
        segment_data = prepare_segment_data(mda_summary, period_names=period_names)
        if segment_data:
            logger.info("build_word_narrative_v2: segment data ready")
        else:
            logger.warning("build_word_narrative_v2: segment parsing failed, doc will omit segment section")

    payload = {
        "borrower_name": borrower_name,
        "statement_basis": extracted_json.get("statement_basis", "actual"),
        "periods": extracted_json.get("periods", []),
        "memo_markdown": memo_markdown,
        "segment_data": segment_data,
        "covenants": covenants or {},
    }

    script = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "build_narrative_doc_v2.js")
    if not _os.path.exists(script):
        raise FileNotFoundError(f"build_narrative_doc_v2.js not found at {script}")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        _json.dump(payload, tf, ensure_ascii=False)
        tf_path = tf.name

    try:
        result = subprocess.run(
            ["node", script, tf_path, output_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"build_narrative_doc_v2.js failed:\n{result.stderr}")
    finally:
        _os.unlink(tf_path)

    return output_path
