"""
auto_rule.py
============
Automatically suggests an extraction rule when an analyst saves a correction
in the Review & Correct interface.

How it works:
  1. The analyst corrects a field value (e.g. rent_expense = 40.2 instead of 9.3)
  2. ReviewCorrect.jsx calls POST /v1/profiles/{company}/suggest_rule
  3. This module looks at where the extractor originally found the value
     (stored in period.notes as CONTEXT:{field}:{json})
  4. It calls Claude to figure out WHY the extractor got it wrong and
     suggests a specific rule for where to look next time
  5. The suggestion is returned to the analyst for one-click approval

The rule is NOT saved automatically — the analyst sees it and clicks "Apply"
which calls POST /v1/profiles/{company}/rules to save it.

High-confidence rules (where Claude is certain of the fix) are flagged with
auto_approved=True so the frontend can apply them with less friction.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("credit_ai")

# Human-readable descriptions of each field so Claude understands
# what it is looking for in the filing document
FIELD_DESCRIPTIONS = {
    "cash_paid_for_income_taxes":      "actual cash paid for income taxes (in supplemental cash flow disclosure)",
    "cash_paid_for_interest":          "actual cash paid for interest (in supplemental cash flow disclosure)",
    "rent_expense":                    "rent or operating lease cost (usually in the lease footnote)",
    "depreciation_amortization":       "depreciation and amortization (from income statement or cash flow)",
    "capex":                           "capital expenditures (from investing activities section)",
    "revolver_facility_size":          "total revolving credit facility commitment size",
    "revolver_borrowings":             "current drawn amount on the revolving credit facility",
    "revolver_availability":           "available undrawn capacity on the revolving credit facility",
    "current_portion_long_term_debt":  "current portion of long-term debt (balance sheet)",
    "total_debt":                      "total debt including current and long-term portions",
    "operating_income":                "operating income / EBIT from the income statement",
    "revenue":                         "total net revenues or net sales",
    "cost_of_sales":                   "cost of goods sold or cost of revenues",
    "sga_expense":                     "selling, general and administrative expense",
    "net_income":                      "net income attributable to common shareholders",
    "cash":                            "cash and cash equivalents from the balance sheet",
    "total_assets":                    "total assets from the balance sheet",
    "total_liabilities":               "total liabilities from the balance sheet",
    "total_equity":                    "total stockholders equity from the balance sheet",
}

# System prompt sent to Claude when generating a rule suggestion
SUGGEST_RULE_SYSTEM = """You are a financial document analyst specialising in SEC 10-K filings.

An AI extractor got a financial field value wrong. Your job is to:
1. Look at WHERE the extractor found the wrong value (the extraction context)
2. Generate a precise rule so it finds the RIGHT value next time

Return ONLY valid JSON — no explanation, no markdown:
{
  "rule_type": "label" | "section" | "exclusion",
  "label": "<exact line label to match, if rule_type=label>",
  "section_hint": "<section or footnote header to search within>",
  "exclude_labels": ["<labels the extractor keeps picking up incorrectly>"],
  "explanation": "<one sentence: what was wrong and what this rule fixes>",
  "confidence": "high" | "medium" | "low"
}

Rule type guidance:
  label     — the extractor found a line with a similar name but wrong meaning.
               Specify the exact correct label to look for.
  section   — the extractor looked in the wrong part of the document.
               Specify which section/footnote header to search within.
  exclusion — the extractor is too broad and picks up multiple lines.
               Specify which labels to skip.

Be specific — vague rules are useless to the extractor."""


@dataclass
class RuleSuggestion:
    """
    A suggested extraction rule returned to the analyst for approval.

    Attributes:
        field_name:     Which field this rule applies to (e.g. "rent_expense")
        rule_type:      "label", "section", or "exclusion"
        label:          The exact line label to match (for label/section types)
        section_hint:   Which document section to search within
        exclude_labels: Labels the extractor should skip
        explanation:    Plain-English explanation of what the rule fixes
        confidence:     How confident Claude is: "high", "medium", or "low"
        auto_approved:  True if confidence=high and rule is specific enough
                        to apply without analyst review
    """
    field_name:     str
    rule_type:      str
    label:          Optional[str]
    section_hint:   Optional[str]
    exclude_labels: list
    explanation:    str
    confidence:     str
    auto_approved:  bool


def suggest_rule_for_correction(
    company_name:    str,
    field_name:      str,
    correct_value:   float,
    extracted_value: Optional[float],
    period_name:     str,
    run_id:          str,
    api_key:         Optional[str] = None,
) -> Optional[RuleSuggestion]:
    """
    Generate an extraction rule suggestion based on an analyst correction.

    Looks up the stored extraction context (where the extractor originally
    found the value) and asks Claude to generate a rule that would fix it.

    Args:
        company_name:    Company the correction is for (e.g. "CHD")
        field_name:      Field that was wrong (e.g. "rent_expense")
        correct_value:   What the value should be (analyst-provided)
        extracted_value: What the extractor returned (the wrong value)
        period_name:     Which fiscal year (e.g. "FY2024")
        run_id:          The run ID so we can look up the extraction context
        api_key:         Anthropic API key (defaults to ANTHROPIC_API_KEY env var)

    Returns:
        RuleSuggestion if successful, None if suggestion could not be generated.
    """
    try:
        import anthropic
        from run_store import RunStore

        # ── Look up where the extractor originally found the value ──────────
        # The extraction context is stored in period.notes as:
        # "CONTEXT:{field_name}:{json_data}"
        # This tells us exactly what line in the filing the extractor matched
        store   = RunStore()
        run     = store.get_run(run_id)
        context_json = None

        if run and run.extracted_json:
            for period in (run.extracted_json.get("periods") or []):
                if period.get("period_name") != period_name:
                    continue
                # Search through notes for the context entry for this field
                for note in (period.get("notes") or []):
                    prefix = f"CONTEXT:{field_name}:"
                    if isinstance(note, str) and note.startswith(prefix):
                        try:
                            context_json = json.loads(note[len(prefix):])
                        except json.JSONDecodeError:
                            context_json = {"raw": note[len(prefix):]}
                        break

        # Build the prompt for Claude
        field_desc = FIELD_DESCRIPTIONS.get(field_name, field_name.replace("_", " "))

        context_section = ""
        if context_json:
            context_section = f"""
WHERE THE EXTRACTOR LOOKED (this is where it found the wrong value):
{json.dumps(context_json, indent=2)}
"""
        else:
            context_section = "\nExtraction context not available for this run.\n"

        user_prompt = f"""Company: {company_name}
Field: {field_name} ({field_desc})
Fiscal year: {period_name}

What the extractor returned: {extracted_value if extracted_value is not None else "null (not found)"}
What the correct value is:   {correct_value}
{context_section}
Generate an extraction rule so the extractor finds {correct_value} instead of {extracted_value} next time."""

        # ── Call Claude to generate the rule ─────────────────────────────────
        client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=SUGGEST_RULE_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Parse the JSON response from Claude
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```").strip()

        parsed = json.loads(raw)

        # A rule is auto-approved if Claude is highly confident AND the rule
        # has a specific label to match (not just a vague section hint)
        auto_approved = (
            parsed.get("confidence") == "high"
            and parsed.get("rule_type") == "label"
            and bool(parsed.get("label"))
        )

        suggestion = RuleSuggestion(
            field_name     = field_name,
            rule_type      = parsed.get("rule_type", "section"),
            label          = parsed.get("label"),
            section_hint   = parsed.get("section_hint"),
            exclude_labels = parsed.get("exclude_labels") or [],
            explanation    = parsed.get("explanation", ""),
            confidence     = parsed.get("confidence", "medium"),
            auto_approved  = auto_approved,
        )

        logger.info(
            f"suggest_rule: {company_name}/{field_name} -> "
            f"type={suggestion.rule_type} confidence={suggestion.confidence} "
            f"auto_approved={suggestion.auto_approved}"
        )
        return suggestion

    except Exception as e:
        logger.error(f"suggest_rule failed for {company_name}/{field_name}: {e}")
        return None


def apply_suggestion(
    company_name: str,
    field_name:   str,
    suggestion:   RuleSuggestion,
) -> bool:
    """
    Save an approved rule suggestion to the company profile.

    Called when the analyst clicks "Apply Rule" in the frontend,
    or automatically when auto_approved=True.

    Args:
        company_name: Company to save the rule for
        field_name:   Field the rule applies to
        suggestion:   The RuleSuggestion to save

    Returns:
        True if saved successfully, False on error.
    """
    try:
        import profiles as ps

        # Build the rule dict from the suggestion
        rule = {"rule_type": suggestion.rule_type}
        if suggestion.label:          rule["label"]          = suggestion.label
        if suggestion.section_hint:   rule["section_hint"]   = suggestion.section_hint
        if suggestion.exclude_labels: rule["exclude_labels"]  = suggestion.exclude_labels
        if suggestion.explanation:    rule["note"]            = suggestion.explanation

        return ps.save_rule(
            company_name=company_name,
            field_name=field_name,
            rule=rule,
        )
    except Exception as e:
        logger.error(f"apply_suggestion failed: {e}")
        return False
