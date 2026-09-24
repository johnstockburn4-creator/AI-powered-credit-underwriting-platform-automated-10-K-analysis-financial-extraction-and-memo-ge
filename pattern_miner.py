"""
pattern_miner.py
================
Analyses correction history across ALL companies to find systematic extraction
mistakes and generate general guidance that improves future extractions.

Run this manually after accumulating corrections across several companies:
    python pattern_miner.py

What it does:
  1. Reads company_profiles.json — all corrections analysts have saved
  2. Groups corrections by field name across all companies
  3. When a field has 3+ corrections, calls Claude to identify the pattern:
     "The extractor keeps making THIS mistake on THIS field"
  4. Saves the patterns to extraction_hints.json
  5. agents.py reads extraction_hints.json and injects these hints into
     the GPT-4o extraction prompt — so ALL future filings benefit

Example: If 5 different companies all needed cash_paid_for_income_taxes
corrected because the extractor was reading the accrual tax line instead,
pattern_miner generates: "For cash_paid_for_income_taxes, always look in
the Supplemental Cash Flow footnote, not the income statement."

That hint gets injected into the prompt for every future company,
even ones you've never analysed before.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("credit_ai")

# File paths (configurable via environment variables)
PROFILES_PATH = os.environ.get("PROFILES_PATH", "company_profiles.json")
HINTS_PATH    = os.environ.get("EXTRACTION_HINTS_PATH", "extraction_hints.json")

# Minimum number of corrections on a field before we try to mine a pattern.
# We need enough examples to identify a real systematic problem vs. a one-off.
MIN_EXAMPLES_TO_LEARN = 3

# System prompt for Claude when generating patterns
PATTERN_MINER_SYSTEM = """You are a financial extraction expert analysing patterns in AI extraction mistakes.

You will be given a list of corrections analysts made, grouped by field.
Identify the systematic mistake and write clear, actionable guidance to prevent it.

Return ONLY valid JSON:
{
  "field_name": "<field>",
  "pattern_summary": "<1-2 sentences: what systematic mistake the AI makes>",
  "extraction_guidance": [
    "<specific actionable instruction>",
    "<specific actionable instruction>"
  ],
  "common_locations": ["<where this field is usually found in 10-Ks>"],
  "common_mistakes": ["<what the AI tends to pick up incorrectly>"],
  "confidence": "high" | "medium"
}

Be specific — generic guidance like "look in the right place" is useless.
"Search the Supplemental Cash Flow footnote for a line labeled
Cash paid for income taxes, net of refunds" is useful."""


def load_all_corrections() -> Dict[str, List[dict]]:
    """
    Load all analyst corrections from company_profiles.json.

    Returns a dictionary grouped by field name, where each entry contains
    all corrections analysts have made for that field across all companies.

    Example return value:
    {
      "rent_expense": [
        {"company": "CHD", "period": "_default", "value": 40.2, ...},
        {"company": "MOS", "period": "FY2024",   "value": 38.1, ...},
      ],
      "cash_paid_for_income_taxes": [...]
    }
    """
    if not os.path.exists(PROFILES_PATH):
        logger.warning(f"No profiles file found at {PROFILES_PATH}")
        return {}

    with open(PROFILES_PATH, encoding="utf-8") as f:
        profiles = json.load(f)

    # Group all user corrections by field name across all companies
    by_field: Dict[str, List[dict]] = defaultdict(list)

    for company_name, company_data in profiles.items():
        if not isinstance(company_data, dict):
            continue

        for period_key, period_data in company_data.items():
            # Skip the rules section — we only want value corrections
            if period_key.startswith("_rules"):
                continue
            if not isinstance(period_data, dict):
                continue

            for field_name, entry in period_data.items():
                if not isinstance(entry, dict) or "value" not in entry:
                    continue
                # Only include corrections the analyst explicitly made
                # (not system defaults)
                if entry.get("corrected_by") != "user":
                    continue

                by_field[field_name].append({
                    "company":      company_name,
                    "period":       period_key,
                    "value":        entry["value"],
                    "corrected_at": entry.get("corrected_at"),
                })

    return dict(by_field)


def mine_patterns(
    api_key:      Optional[str] = None,
    min_examples: int           = MIN_EXAMPLES_TO_LEARN,
) -> Dict[str, dict]:
    """
    Analyse all corrections to find systematic extraction patterns.

    For each field with enough corrections (min_examples or more),
    calls Claude to identify the systematic mistake and generate guidance.

    Args:
        api_key:      Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        min_examples: Minimum corrections needed before mining a pattern

    Returns:
        Dictionary of field_name -> pattern dict.
        Also saves patterns to extraction_hints.json automatically.
    """
    import anthropic

    corrections_by_field = load_all_corrections()

    if not corrections_by_field:
        logger.info("pattern_miner: no corrections found in profiles")
        return {}

    # Only mine fields with enough examples to learn from
    fields_to_mine = {
        field: examples
        for field, examples in corrections_by_field.items()
        if len(examples) >= min_examples
    }

    if not fields_to_mine:
        logger.info(
            f"pattern_miner: no fields have {min_examples}+ corrections yet. "
            f"Fields found: {list(corrections_by_field.keys())}"
        )
        return {}

    logger.info(f"pattern_miner: mining {len(fields_to_mine)} fields")

    client  = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    results = {}

    for field_name, examples in fields_to_mine.items():
        try:
            user_prompt = (
                f"Field: {field_name}\n"
                f"Number of analyst corrections: {len(examples)}\n\n"
                f"Corrections:\n{json.dumps(examples, indent=2)}\n\n"
                f"What systematic mistake is the AI making? Generate extraction guidance to fix it."
            )

            response = client.messages.create(
                model      = "claude-sonnet-4-6",
                max_tokens = 800,
                system     = PATTERN_MINER_SYSTEM,
                messages   = [{"role": "user", "content": user_prompt}],
            )

            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip().rstrip("```").strip()

            pattern                    = json.loads(raw)
            pattern["examples_count"]  = len(examples)
            pattern["mined_at"]        = datetime.utcnow().isoformat()
            results[field_name]        = pattern

            logger.info(
                f"pattern_miner: mined {field_name} -> "
                f"{pattern.get('confidence')} confidence, "
                f"{len(pattern.get('extraction_guidance', []))} guidance items"
            )

        except Exception as e:
            logger.error(f"pattern_miner: failed on {field_name}: {e}")
            continue

    # Save the patterns to disk so agents.py can inject them into prompts
    if results:
        with open(HINTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"pattern_miner: saved {len(results)} patterns to {HINTS_PATH}")

    return results


def load_hints_for_prompt() -> str:
    """
    Load mined patterns and format them for injection into the extraction prompt.

    Called by agents.py before building the GPT-4o extraction system prompt.
    Returns a formatted string block like:

        LEARNED EXTRACTION GUIDANCE (from analyst corrections):

        rent_expense:
          Summary: Extractor reads total lease cost instead of operating lease cost
          - Look in the LEASES footnote, not the income statement
          - Use the line labeled "Operating lease cost", not "Total lease cost"

    Returns empty string if no hints have been mined yet.
    """
    if not os.path.exists(HINTS_PATH):
        return ""

    try:
        with open(HINTS_PATH, encoding="utf-8") as f:
            hints = json.load(f)
    except Exception:
        return ""

    if not hints:
        return ""

    lines = ["LEARNED EXTRACTION GUIDANCE (from analyst corrections across prior filings):"]

    for field_name, pattern in hints.items():
        # Only include high or medium confidence patterns
        if pattern.get("confidence") not in ("high", "medium"):
            continue

        guidance  = pattern.get("extraction_guidance") or []
        locations = pattern.get("common_locations") or []
        mistakes  = pattern.get("common_mistakes") or []

        lines.append(f"\n{field_name}:")
        lines.append(f"  Summary: {pattern.get('pattern_summary', '')}")

        # Cap at 3 guidance items to keep the prompt concise
        for g in guidance[:3]:
            lines.append(f"  - {g}")

        if locations:
            lines.append(f"  Usually found in: {'; '.join(locations[:2])}")
        if mistakes:
            lines.append(f"  Common wrong picks: {'; '.join(mistakes[:2])}")

    return "\n".join(lines)


# ── Run directly from command line ────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 60)
    print("  Credit AI — Extraction Pattern Miner")
    print("=" * 60)

    # Show what corrections exist across all companies
    corrections = load_all_corrections()
    if not corrections:
        print("\nNo corrections found in company_profiles.json.")
        print("Save some corrections in the Review & Correct interface first.")
        sys.exit(0)

    print(f"\nCorrections found:")
    for field, examples in sorted(corrections.items(), key=lambda x: -len(x[1])):
        companies = list({e["company"] for e in examples})
        print(f"  {field}: {len(examples)} corrections from {len(companies)} companies")

    fields_ready = [f for f, ex in corrections.items() if len(ex) >= MIN_EXAMPLES_TO_LEARN]
    if not fields_ready:
        print(f"\nNo field has {MIN_EXAMPLES_TO_LEARN}+ corrections yet.")
        print("Keep correcting extractions — patterns will be mined once enough data accumulates.")
        sys.exit(0)

    print(f"\nMining patterns for {len(fields_ready)} field(s)...")
    patterns = mine_patterns()

    if patterns:
        print(f"\nPatterns saved to {HINTS_PATH}:")
        for field, p in patterns.items():
            print(f"\n  {field} ({p.get('confidence')} confidence, {p.get('examples_count')} examples):")
            print(f"    {p.get('pattern_summary')}")
            for g in (p.get("extraction_guidance") or [])[:2]:
                print(f"    - {g}")
    else:
        print("No patterns mined.")
