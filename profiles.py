"""
profiles.py
-----------
Company extraction profile store.

Two things are stored per company:

1. VALUE CORRECTIONS — the correct numeric value for a specific field/period.
   Used to override extraction errors after the fact.

2. EXTRACTION RULES — instructions telling the extractor HOW to find the right
   number in future filings. These are injected into the GPT-4o prompt so the
   extractor learns where to look rather than just having its output corrected.

Rule types:
  "label"     — find the line with this exact label, ignore others
  "section"   — look in this specific section/note of the document
  "exclusion" — ignore lines whose labels contain these strings

Storage: company_profiles.json in the project root.

Structure:
{
  "Church & Dwight Co.": {
    "_rules": {
      "rent_expense": {
        "rule_type": "label",
        "label": "Operating lease cost",
        "section_hint": "LEASES",
        "exclude_labels": ["Total lease cost", "Finance lease"],
        "note": "Use operating lease cost only, not total",
        "created_at": "2026-06-17T..."
      },
      "cash_paid_for_income_taxes": {
        "rule_type": "section",
        "section_hint": "SUPPLEMENTAL CASH FLOW INFORMATION",
        "label": "Income taxes paid",
        "note": "In supplemental disclosures, not tax footnote"
      }
    },
    "_default": {
      "rent_expense": {"value": 40.2, "corrected_at": "..."}
    },
    "FY2024": {
      "rent_expense": {"value": 40.2, "corrected_at": "..."}
    }
  }
}
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("credit_ai")

PROFILES_PATH = os.environ.get(
    "PROFILES_PATH",
    os.path.join(os.path.dirname(__file__), "company_profiles.json")
)

FIELD_SECTION_MAP = {
    "revenue":                      "income_statement",
    "cost_of_sales":                "income_statement",
    "sga_expense":                  "income_statement",
    "operating_income":             "income_statement",
    "ebitda":                       "income_statement",
    "net_income":                   "income_statement",
    "interest_expense":             "income_statement",
    "income_tax_expense":           "income_statement",
    "depreciation_amortization":    "income_statement",
    "rent_expense":                 "income_statement",
    "cash":                             "balance_sheet",
    "total_assets":                     "balance_sheet",
    "total_liabilities":                "balance_sheet",
    "total_equity":                     "balance_sheet",
    "total_debt":                       "balance_sheet",
    "long_term_debt":                   "balance_sheet",
    "current_portion_long_term_debt":   "balance_sheet",
    "revolver_facility_size":           "balance_sheet",
    "revolver_borrowings":              "balance_sheet",
    "revolver_availability":            "balance_sheet",
    "cfo":                          "cash_flow",
    "cfi":                          "cash_flow",
    "cff":                          "cash_flow",
    "capex":                        "cash_flow",
    "cash_paid_for_interest":       "cash_flow",
    "cash_paid_for_income_taxes":   "cash_flow",
    "dividends_distributions_paid": "cash_flow",
}

FIELD_LABELS = {
    "revenue": "Revenue",
    "cost_of_sales": "Cost of Sales",
    "sga_expense": "SG&A",
    "operating_income": "Operating Income",
    "ebitda": "EBITDA",
    "net_income": "Net Income",
    "interest_expense": "Interest Expense",
    "income_tax_expense": "Income Tax Expense",
    "depreciation_amortization": "D&A",
    "rent_expense": "Rent / Lease Expense",
    "cash": "Cash",
    "total_assets": "Total Assets",
    "total_liabilities": "Total Liabilities",
    "total_equity": "Total Equity",
    "total_debt": "Total Debt",
    "long_term_debt": "Long-term Debt",
    "current_portion_long_term_debt": "Current Portion LTD",
    "revolver_facility_size": "Revolver Facility Size",
    "revolver_borrowings": "Revolver Borrowings",
    "revolver_availability": "Revolver Availability",
    "cfo": "Operating Cash Flow",
    "cfi": "Investing Cash Flow",
    "cff": "Financing Cash Flow",
    "capex": "Capex",
    "cash_paid_for_interest": "Cash Paid for Interest",
    "cash_paid_for_income_taxes": "Cash Paid for Taxes",
    "dividends_distributions_paid": "Dividends Paid",
}


# ── Storage helpers ───────────────────────────────────────────────────────────

def _load_profiles() -> dict:
    if not os.path.exists(PROFILES_PATH):
        return {}
    try:
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load profiles: {e}")
        return {}


def _save_profiles(profiles: dict) -> bool:
    try:
        with open(PROFILES_PATH, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save profiles: {e}")
        return False


# ── Profile accessors ─────────────────────────────────────────────────────────

def get_profile(company_name: str) -> dict:
    if not company_name:
        return {}
    return _load_profiles().get(company_name, {})


def get_field_value(company_name: str, field_name: str,
                    period_name: Optional[str] = None) -> Optional[dict]:
    profile = get_profile(company_name)
    if not profile:
        return None
    if period_name and period_name in profile:
        if field_name in profile[period_name]:
            return profile[period_name][field_name]
    default = profile.get("_default", {})
    return default.get(field_name)


def get_rules(company_name: str) -> dict:
    """Return all extraction rules for a company. Keys are field names."""
    if not company_name:
        return {}
    return get_profile(company_name).get("_rules", {})


def get_rule(company_name: str, field_name: str) -> Optional[dict]:
    """Return the extraction rule for a specific field, or None."""
    return get_rules(company_name).get(field_name)


# ── Correction saving ─────────────────────────────────────────────────────────

def save_correction(company_name: str, field_name: str, value: float,
                    period_name: Optional[str] = None,
                    note: Optional[str] = None,
                    source_hint: Optional[str] = None) -> bool:
    """
    Save a value correction.

    Value corrections are ONE-TIME FIXES — they apply to the current run to
    fix calculations in the memo. They do NOT apply to future runs if an
    extraction rule exists for the same field.

    To fix the cause permanently, save an extraction rule via save_rule().
    The rule teaches the extractor where to look; the value just patches
    the current output.
    """
    if not company_name or not field_name:
        return False
    profiles = _load_profiles()
    if company_name not in profiles:
        profiles[company_name] = {"_default": {}, "_rules": {}}

    entry = {
        "value": value,
        "corrected_by": "user",
        "corrected_at": datetime.now(timezone.utc).isoformat(),
        "one_time_fix": True,  # Flag: this value won't override if a rule exists
    }
    if note:        entry["note"] = note
    if source_hint: entry["source_hint"] = source_hint

    if period_name:
        if period_name not in profiles[company_name]:
            profiles[company_name][period_name] = {}
        profiles[company_name][period_name][field_name] = entry
    else:
        if "_default" not in profiles[company_name]:
            profiles[company_name]["_default"] = {}
        profiles[company_name]["_default"][field_name] = entry

    logger.info(f"Correction saved (one-time): {company_name}/{period_name or '_default'}/{field_name}={value}")
    return _save_profiles(profiles)


def save_corrections_bulk(company_name: str, corrections: list[dict]) -> bool:
    if not company_name or not corrections:
        return False
    profiles = _load_profiles()
    if company_name not in profiles:
        profiles[company_name] = {"_default": {}, "_rules": {}}

    now = datetime.now(timezone.utc).isoformat()
    for c in corrections:
        field_name  = c.get("field_name")
        value       = c.get("value")
        period_name = c.get("period_name")
        note        = c.get("note")
        source_hint = c.get("source_hint")
        if not field_name or value is None:
            continue
        entry = {"value": float(value), "corrected_by": "user", "corrected_at": now}
        if note:        entry["note"] = note
        if source_hint: entry["source_hint"] = source_hint
        if period_name:
            if period_name not in profiles[company_name]:
                profiles[company_name][period_name] = {}
            profiles[company_name][period_name][field_name] = entry
        else:
            if "_default" not in profiles[company_name]:
                profiles[company_name]["_default"] = {}
            profiles[company_name]["_default"][field_name] = entry

    logger.info(f"Bulk corrections: {company_name} — {len(corrections)} saved")
    return _save_profiles(profiles)


# ── Rule saving ───────────────────────────────────────────────────────────────

def save_rule(company_name: str, field_name: str, rule: dict) -> bool:
    """
    Save an extraction rule for a field.

    rule dict keys:
      rule_type:      "label" | "section" | "exclusion"
      label:          exact line label to match (for label/section types)
      section_hint:   section/note header to search within
      exclude_labels: list of label strings to ignore (for exclusion type)
      note:           human-readable explanation
    """
    if not company_name or not field_name or not rule:
        return False
    profiles = _load_profiles()
    if company_name not in profiles:
        profiles[company_name] = {"_default": {}, "_rules": {}}
    if "_rules" not in profiles[company_name]:
        profiles[company_name]["_rules"] = {}

    rule["created_at"] = datetime.now(timezone.utc).isoformat()
    rule["created_by"] = "user"
    profiles[company_name]["_rules"][field_name] = rule

    logger.info(f"Rule saved: {company_name}/{field_name} — type={rule.get('rule_type')}")
    return _save_profiles(profiles)


def delete_rule(company_name: str, field_name: str) -> bool:
    profiles = _load_profiles()
    rules = profiles.get(company_name, {}).get("_rules", {})
    if field_name in rules:
        del rules[field_name]
        return _save_profiles(profiles)
    return False


# ── Rule injection into extraction prompt ────────────────────────────────────

def build_rules_prompt_block(company_name: str) -> str:
    """
    Build a prompt block that tells GPT-4o exactly how to find each field
    for this specific company based on saved rules.

    This gets injected into the extraction prompt BEFORE the alias list,
    so the model treats company-specific rules as higher priority than
    the general aliases.

    Returns empty string if no rules exist.
    """
    rules = get_rules(company_name)
    if not rules:
        return ""

    lines = [
        f"\nCOMPANY-SPECIFIC EXTRACTION RULES FOR {company_name.upper()}:",
        "These rules override the general aliases below. Follow them exactly.\n",
    ]

    for field_name, rule in rules.items():
        label = FIELD_LABELS.get(field_name, field_name)
        rule_type = rule.get("rule_type", "label")
        note = rule.get("note", "")

        if rule_type == "label":
            line = (
                f"- {field_name} ({label}): "
                f"Extract ONLY from the line labeled '{rule.get('label')}'"
            )
            if rule.get("section_hint"):
                line += f" in the '{rule['section_hint']}' section"
            excl = rule.get("exclude_labels", [])
            if excl:
                line += f". IGNORE lines labeled: {', '.join(repr(e) for e in excl)}"
            if note:
                line += f". Note: {note}"
            lines.append(line)

        elif rule_type == "section":
            line = (
                f"- {field_name} ({label}): "
                f"Look in the '{rule.get('section_hint', '')}' section"
            )
            if rule.get("label"):
                line += f" for a line labeled '{rule['label']}'"
            if note:
                line += f". Note: {note}"
            lines.append(line)

        elif rule_type == "exclusion":
            excl = rule.get("exclude_labels", [])
            line = (
                f"- {field_name} ({label}): "
                f"Do NOT use lines labeled: {', '.join(repr(e) for e in excl)}"
            )
            if rule.get("label"):
                line += f". Use line labeled '{rule['label']}' instead"
            if note:
                line += f". Note: {note}"
            lines.append(line)

    lines.append("")
    return "\n".join(lines)


# ── Context extraction (what the extractor actually found) ───────────────────

def extract_field_context(raw_text: str, field_name: str,
                          extracted_value: Optional[float],
                          context_lines: int = 3) -> Optional[str]:
    """
    Find where in the raw document the extracted value was likely extracted from.

    Returns a dict with two keys:
      "formatted" — clean label/value table (expanded by default in UI)
      "raw"       — raw document lines surrounding the match (collapsed in UI)

    Serialized as JSON string in the note for storage.

    Improvements over v1:
    - Two-pass matching: first try label+value on same line, then value alone
    - Stronger scoring: requires label hint within 2 lines of value
    - Section anchoring: boosts score if section header is nearby (within 10 lines)
    - Penalises common false-positive sections (TOC, page headers)
    - Returns None only if truly cannot locate — never returns random context
    """
    import json as _json
    if extracted_value is None or not raw_text:
        return None

    val = abs(extracted_value)

    # Build number format variants the value might appear as in the document
    val_str_variants = set()
    val_str_variants.add(f"{val:.1f}")
    val_str_variants.add(f"{val:.0f}")
    val_str_variants.add(f"{val:,.1f}")
    val_str_variants.add(f"{val:,.0f}")
    val_str_variants.add(f"{int(val):,}")
    # For values in millions, also check the raw number without MM suffix
    if val >= 1000:
        val_str_variants.add(f"{val/1000:.1f}")
    # Parenthetical negative form
    val_str_variants.add(f"({val:.1f})")
    val_str_variants.add(f"({val:,.1f})")

    # Field-specific configuration
    # label_hints: strong signals this line is the right one
    # section_hints: section headers that should be nearby
    # avoid_sections: section headers that indicate a false positive
    FIELD_CONFIG = {
        "rent_expense": {
            "label_hints": ["operating lease cost", "operating leases", "lease cost"],
            "section_hints": ["leases", "lease obligations", "operating lease"],
            "avoid_sections": ["table of contents", "item 7", "item 2"],
            "avoid_labels": ["total lease cost", "finance lease", "variable lease",
                            "short-term lease", "sublease"],
        },
        "cash_paid_for_income_taxes": {
            "label_hints": ["income taxes paid", "cash paid for income taxes",
                           "cash income tax payments", "taxes paid"],
            "section_hints": ["supplemental cash flow", "supplemental disclosures",
                             "cash flow information"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": ["federal", "state", "international", "deferred"],
        },
        "cash_paid_for_interest": {
            "label_hints": ["interest paid", "cash paid for interest",
                           "cash interest paid"],
            "section_hints": ["supplemental cash flow", "supplemental disclosures"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": ["interest expense", "interest income"],
        },
        "capex": {
            "label_hints": ["capital expenditures", "additions to property",
                           "purchases of property", "capital spending",
                           "payments for property"],
            "section_hints": ["investing activities", "cash flow"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "depreciation_amortization": {
            "label_hints": ["depreciation and amortization", "depreciation",
                           "amortization"],
            "section_hints": ["operating activities", "cash flow", "income statement"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "cfo": {
            "label_hints": ["net cash provided by operating", "cash provided by operating",
                           "net cash from operating"],
            "section_hints": ["operating activities", "cash flow"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "total_debt": {
            "label_hints": ["total debt", "total long-term debt", "total borrowings"],
            "section_hints": ["long-term debt", "debt", "borrowings", "balance sheet"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "cost_of_sales": {
            "label_hints": ["cost of sales", "cost of goods sold", "cost of products sold",
                           "cost of revenue"],
            "section_hints": ["income statement", "statements of income", "statements of earnings"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "sga_expense": {
            "label_hints": ["selling, general", "sg&a", "selling and administrative",
                           "general and administrative"],
            "section_hints": ["income statement", "statements of income"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "net_income": {
            "label_hints": ["net income", "net earnings", "net income attributable"],
            "section_hints": ["income statement", "statements of income", "statements of earnings"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "interest_expense": {
            "label_hints": ["interest expense", "interest expense, net", "interest costs"],
            "section_hints": ["income statement", "statements of income"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": ["cash paid", "interest income"],
        },
        "income_tax_expense": {
            "label_hints": ["provision for income taxes", "income tax expense",
                           "income taxes", "tax expense"],
            "section_hints": ["income statement", "statements of income"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": ["deferred", "cash paid"],
        },
        "cash": {
            "label_hints": ["cash and cash equivalents", "cash and equivalents"],
            "section_hints": ["balance sheet", "balance sheets"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "total_assets": {
            "label_hints": ["total assets"],
            "section_hints": ["balance sheet", "balance sheets"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "total_liabilities": {
            "label_hints": ["total liabilities"],
            "section_hints": ["balance sheet", "balance sheets"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "total_equity": {
            "label_hints": ["total stockholders", "total equity", "total shareholders",
                           "stockholders equity"],
            "section_hints": ["balance sheet", "balance sheets"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "long_term_debt": {
            "label_hints": ["long-term debt", "long term debt"],
            "section_hints": ["balance sheet", "balance sheets", "debt"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": ["current portion"],
        },
        "current_portion_long_term_debt": {
            "label_hints": ["current portion", "current maturities", "current portion of long-term"],
            "section_hints": ["balance sheet", "balance sheets"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "revolver_borrowings": {
            "label_hints": ["revolving credit", "revolver", "borrowings outstanding",
                           "amounts outstanding"],
            "section_hints": ["credit facilities", "long-term debt", "debt"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "revolver_availability": {
            "label_hints": ["available", "availability", "unused capacity"],
            "section_hints": ["credit facilities", "revolving credit"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "cfi": {
            "label_hints": ["net cash used in investing", "cash used in investing",
                           "investing activities"],
            "section_hints": ["cash flow", "investing activities"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "cff": {
            "label_hints": ["net cash used in financing", "cash used in financing",
                           "financing activities"],
            "section_hints": ["cash flow", "financing activities"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "dividends_distributions_paid": {
            "label_hints": ["dividends paid", "cash dividends", "distributions paid"],
            "section_hints": ["cash flow", "financing activities"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "revolver_facility_size": {
            "label_hints": ["revolving credit facility", "revolver", "credit facility"],
            "section_hints": ["credit facilities", "long-term debt", "debt"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "revenue": {
            "label_hints": ["net sales", "net revenues", "revenues", "total revenues"],
            "section_hints": ["income statement", "statements of income",
                             "statements of earnings"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
        "operating_income": {
            "label_hints": ["operating income", "income from operations",
                           "operating profit"],
            "section_hints": ["income statement", "statements of income"],
            "avoid_sections": ["table of contents"],
            "avoid_labels": [],
        },
    }

    config = FIELD_CONFIG.get(field_name, {
        "label_hints": [], "section_hints": [],
        "avoid_sections": [], "avoid_labels": [],
    })

    lines = raw_text.splitlines()
    best_match_idx = None
    best_score = -1

    for i, line in enumerate(lines):
        line_lower = line.lower()
        line_clean = line.replace(",", "").replace("$", "").strip()

        # Must contain the numeric value
        has_value = any(v in line_clean for v in val_str_variants)
        if not has_value:
            continue

        # Skip clearly wrong contexts
        skip = False
        for avoid in config.get("avoid_sections", []):
            if avoid in line_lower:
                skip = True
                break
        if skip:
            continue

        # Skip if line label is one we want to avoid
        for avoid_label in config.get("avoid_labels", []):
            if avoid_label in line_lower:
                skip = True
                break
        if skip:
            continue

        score = 1  # Base score: value found

        # Strong boost: label hint on the SAME line as the value
        for hint in config.get("label_hints", []):
            if hint in line_lower:
                score += 10
                break

        # Medium boost: label hint within 2 lines
        if score < 5:
            window = lines[max(0, i-2):min(len(lines), i+3)]
            for nearby in window:
                for hint in config.get("label_hints", []):
                    if hint in nearby.lower():
                        score += 5
                        break

        # Boost: section header within 10 lines above
        section_window = lines[max(0, i-10):i]
        for nearby in section_window:
            for hint in config.get("section_hints", []):
                if hint in nearby.lower():
                    score += 3
                    break

        # Penalise: looks like a TOC entry (short line, page number pattern)
        import re as _re
        if _re.search(r'\d{1,3}\s*$', line.strip()) and len(line.strip()) < 80:
            score -= 5

        if score > best_score:
            best_score = score
            best_match_idx = i

    # Require minimum score to avoid returning random context
    if best_match_idx is None or best_score < 1:
        return None

    # Extract context window
    start = max(0, best_match_idx - context_lines)
    end   = min(len(lines), best_match_idx + context_lines + 1)

    raw_lines = []
    for i in range(start, end):
        line = lines[i].strip()
        if not line:
            continue
        marker = ">>>" if i == best_match_idx else "   "
        raw_lines.append(f"{marker} {line}")

    raw_context = "\n".join(raw_lines)

    # Build formatted version: parse pipe-delimited columns into clean rows
    matched_line = lines[best_match_idx].strip()
    formatted_rows = []

    # Try to parse surrounding lines as a table
    table_lines = []
    for i in range(start, end):
        line = lines[i].strip()
        if not line:
            continue
        is_match = (i == best_match_idx)
        # Split on pipe or multiple spaces
        if '|' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
        else:
            parts = _re.split(r'\s{2,}', line)
            parts = [p.strip() for p in parts if p.strip()]

        if parts:
            table_lines.append({"parts": parts, "is_match": is_match})

    # Format as aligned rows
    if table_lines:
        for row in table_lines:
            parts = row["parts"]
            is_match = row["is_match"]
            if len(parts) == 1:
                formatted_rows.append({
                    "label": parts[0], "values": [],
                    "is_match": is_match
                })
            else:
                formatted_rows.append({
                    "label": parts[0],
                    "values": parts[1:],
                    "is_match": is_match,
                })

    result = {
        "raw": raw_context,
        "formatted": formatted_rows,
        "matched_line": matched_line,
        "score": best_score,
    }
    return _json.dumps(result)


def apply_profile_to_extracted(company_name: str, extracted) -> list[str]:
    """
    Compare extracted values against saved corrections.
    Returns conflict flags — does NOT override values.
    Overriding is done separately via apply_profile_overrides.
    """
    if not company_name:
        return []
    profile = get_profile(company_name)
    if not profile:
        return []

    conflicts = []
    for period in (extracted.periods or []):
        period_name = period.period_name
        all_fields = {}
        for f, e in profile.get("_default", {}).items():
            all_fields[f] = e
        for f, e in profile.get(period_name, {}).items():
            all_fields[f] = e

        for field_name, entry in all_fields.items():
            profile_value = entry.get("value")
            if profile_value is None:
                continue
            section_name = FIELD_SECTION_MAP.get(field_name)
            if not section_name:
                continue
            section = getattr(period, section_name, None) or {}
            extracted_value = section.get(field_name) if isinstance(section, dict) else None

            if extracted_value is None:
                conflicts.append(
                    f"PROFILE: {period_name} {field_name} — "
                    f"extractor returned null, profile has {profile_value}. "
                    f"Review and apply if correct."
                )
            elif abs(float(extracted_value) - float(profile_value)) > 0.5:
                conflicts.append(
                    f"PROFILE CONFLICT: {period_name} {field_name} — "
                    f"extractor={extracted_value:.1f}, profile={profile_value:.1f}. "
                    f"Review which is correct."
                )
    return conflicts


def apply_profile_overrides(company_name: str, extracted) -> list[str]:
    """
    Apply saved value corrections to extracted data for the CURRENT RUN ONLY.

    IMPORTANT DESIGN DECISION:
    If an extraction rule exists for a field, the value override is NOT applied.
    The rule is injected into the extraction prompt so the model finds the correct
    value dynamically — the saved numeric value is only a one-time fix for runs
    where no rule was available yet.

    This means:
    - Rule saved + value saved → rule is used to find value dynamically, saved value ignored
    - Value saved, no rule    → value override applied (one-time fix until rule is added)
    - Rule saved, no value    → rule guides extraction, no override needed

    Returns list of applied override messages for validation_flags.
    """
    if not company_name:
        return []
    profile = get_profile(company_name)
    if not profile:
        return []

    # Get all rules for this company — fields with rules skip value overrides
    rules = get_rules(company_name)

    applied = []
    for period in (extracted.periods or []):
        period_name = period.period_name
        all_fields = {}
        for f, e in profile.get("_default", {}).items():
            all_fields[f] = e
        for f, e in profile.get(period_name, {}).items():
            all_fields[f] = e

        for field_name, entry in all_fields.items():
            if field_name.startswith("_"):
                continue
            profile_value = entry.get("value")
            if profile_value is None:
                continue

            # SKIP override if a rule exists for this field.
            # The rule will guide the extractor to find the right value dynamically.
            # Applying a saved value on top of a rule-guided extraction would
            # defeat the purpose of the rule system.
            if field_name in rules:
                logger.debug(
                    f"Profile override skipped: {company_name}/{field_name} "
                    f"has a rule — rule will find value dynamically"
                )
                continue

            section_name = FIELD_SECTION_MAP.get(field_name)
            if not section_name:
                continue
            section = getattr(period, section_name, None)
            if not isinstance(section, dict):
                continue
            old_value = section.get(field_name)
            section[field_name] = float(profile_value)
            if old_value is None or abs(float(old_value) - float(profile_value)) > 0.5:
                applied.append(
                    f"PROFILE APPLIED (one-time): {period_name} {field_name} "
                    f"{'(was null)' if old_value is None else f'({old_value:.1f} -> {profile_value:.1f})'} "
                    f"— add an extraction rule to fix this permanently"
                )
                logger.info(
                    f"Profile value override (no rule): {company_name}/{period_name}/{field_name} "
                    f"{old_value} -> {profile_value}"
                )
    return applied


# ── Utility ───────────────────────────────────────────────────────────────────

def list_companies() -> list[str]:
    return sorted(_load_profiles().keys())


def delete_profile(company_name: str) -> bool:
    profiles = _load_profiles()
    if company_name in profiles:
        del profiles[company_name]
        return _save_profiles(profiles)
    return False


def delete_field(company_name: str, field_name: str,
                 period_name: Optional[str] = None) -> bool:
    profiles = _load_profiles()
    if company_name not in profiles:
        return False
    if period_name:
        period_data = profiles[company_name].get(period_name, {})
        if field_name in period_data:
            del period_data[field_name]
    else:
        default_data = profiles[company_name].get("_default", {})
        if field_name in default_data:
            del default_data[field_name]
    return _save_profiles(profiles)
