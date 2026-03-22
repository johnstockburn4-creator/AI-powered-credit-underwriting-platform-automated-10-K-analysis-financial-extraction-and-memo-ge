from __future__ import annotations

from typing import Optional, List, Dict, Tuple
from schemas import ExtractionResult
from metrics import (
    compute_ebitda,
    compute_fcc,
    compute_leverage,
    compute_ebitda_margin,
    compute_free_cash_flow,
    abs_outflow,
)

FINANCIAL_SUMMARY_BUILD = "fin_summary_narrative_v1_2026-01-26"


# =========================
# Formatting helpers
# =========================

def _fmt_money(x: Optional[float], basis: str) -> str:
    if x is None:
        return "N/A"
    return f"${x:,.0f}"


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x * 100:.1f}%"


def _fmt_x(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x:.2f}x"


def _safe_pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / old


def _period_dict(p) -> Tuple[Dict, Dict, Dict, Dict]:
    is_ = p.income_statement or {}
    bs = p.balance_sheet or {}
    cf = p.cash_flow or {}
    dm = p.derived_metrics or {}
    return is_, bs, cf, dm


def _get_ebitda(is_: Dict, dm: Dict) -> Optional[float]:
    e = is_.get("ebitda")
    if isinstance(e, (int, float)):
        return float(e)

    e2 = dm.get("ebitda_computed")
    if isinstance(e2, (int, float)):
        return float(e2)

    return compute_ebitda(
        is_.get("operating_income"),
        is_.get("depreciation_amortization"),
        rent_expense=is_.get("rent_expense"),
        include_rent=True,
    )


def _get_margin(is_: Dict, dm: Dict) -> Optional[float]:
    m = dm.get("ebitda_margin")
    if isinstance(m, (int, float)):
        return float(m)
    ebitda = _get_ebitda(is_, dm)
    return compute_ebitda_margin(ebitda, is_.get("revenue"))


def _get_leverage(is_: Dict, bs: Dict, dm: Dict) -> Optional[float]:
    lev = dm.get("leverage_total_debt_to_ebitda")
    if isinstance(lev, (int, float)):
        return float(lev)
    ebitda = _get_ebitda(is_, dm)
    return compute_leverage(bs.get("total_debt"), ebitda)


def _get_fcc(is_: Dict, bs: Dict, cf: Dict, dm: Dict) -> Optional[float]:
    f = dm.get("fcc")
    if isinstance(f, (int, float)):
        return float(f)
    ebitda = _get_ebitda(is_, dm)
    return compute_fcc(
        ebitda=ebitda,
        capex=cf.get("capex"),
        cash_taxes=cf.get("cash_paid_for_income_taxes"),
        cpltd=bs.get("current_portion_long_term_debt"),
        cash_interest=cf.get("cash_paid_for_interest"),
    )


def _get_fcf(cf: Dict, dm: Dict) -> Optional[float]:
    fcf = dm.get("free_cash_flow")
    if isinstance(fcf, (int, float)):
        return float(fcf)
    return compute_free_cash_flow(cf.get("cfo"), cf.get("capex"))


def _missing_fields_for_credit_metrics(p) -> List[str]:
    is_, bs, cf, _ = _period_dict(p)
    missing = []

    # EBITDA
    if is_.get("operating_income") is None:
        missing.append("income_statement.operating_income")
    if is_.get("depreciation_amortization") is None:
        missing.append("income_statement.depreciation_amortization")

    # Leverage
    if bs.get("total_debt") is None:
        missing.append("balance_sheet.total_debt")

    # FCC inputs
    if cf.get("capex") is None:
        missing.append("cash_flow.capex")
    if cf.get("cash_paid_for_interest") is None:
        missing.append("cash_flow.cash_paid_for_interest")
    if cf.get("cash_paid_for_income_taxes") is None:
        missing.append("cash_flow.cash_paid_for_income_taxes")
    if bs.get("current_portion_long_term_debt") is None:
        missing.append("balance_sheet.current_portion_long_term_debt")

    # Helpful drivers
    if is_.get("cost_of_sales") is None:
        missing.append("income_statement.cost_of_sales")
    if is_.get("sga_expense") is None:
        missing.append("income_statement.sga_expense")

    # Liquidity/cash flow
    if cf.get("cfo") is None:
        missing.append("cash_flow.cfo")
    if bs.get("cash") is None:
        missing.append("balance_sheet.cash")

    # de-dupe preserve order
    seen = set()
    out = []
    for m in missing:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def build_financial_memo(extracted: ExtractionResult) -> str:
    """
    Deterministic baseline memo with sentences + bullets + audit table.
    """
    periods = extracted.periods or []
    basis = extracted.statement_basis or "actual"

    if not periods:
        return "## Financial Summary (Baseline)\n- No financial periods were extracted."

    p0 = periods[0]  # most recent (per your extractor requirements)
    is0, bs0, cf0, dm0 = _period_dict(p0)

    rev0 = is0.get("revenue")
    ebitda0 = _get_ebitda(is0, dm0)
    margin0 = _get_margin(is0, dm0)
    lev0 = _get_leverage(is0, bs0, dm0)
    fcc0 = _get_fcc(is0, bs0, cf0, dm0)
    fcf0 = _get_fcf(cf0, dm0)

    cfo0 = cf0.get("cfo")
    capex0 = cf0.get("capex")
    capex_mag0 = abs_outflow(capex0)
    cash0 = bs0.get("cash")
    debt0 = bs0.get("total_debt")

    lines: List[str] = []
    lines.append(f"## Financial Summary (Baseline)\n*build:* `{FINANCIAL_SUMMARY_BUILD}`")
    lines.append(f"*Statement basis:* `{basis}`  \n*Most recent period:* `{p0.period_name}`\n")

    # Executive Takeaways
    lines.append("### Executive Takeaways")
    lines.append(f"- Revenue: {_fmt_money(rev0, basis)}.")
    lines.append(f"- EBITDA: {_fmt_money(ebitda0, basis)} ({_fmt_pct(margin0)} margin).")
    lines.append(f"- Total debt: {_fmt_money(debt0, basis)}; leverage (Debt/EBITDA): {_fmt_x(lev0)}.")
    lines.append(f"- FCC: {_fmt_x(fcc0)}.")
    lines.append(f"- CFO: {_fmt_money(cfo0, basis)}; Capex: {_fmt_money(capex_mag0, basis)}; FCF: {_fmt_money(fcf0, basis)}.")
    lines.append(f"- Liquidity (cash): {_fmt_money(cash0, basis)}.\n")

    # YoY trend (if available)
    if len(periods) >= 2:
        p1 = periods[1]
        is1, bs1, cf1, dm1 = _period_dict(p1)

        rev1 = is1.get("revenue")
        ebitda1 = _get_ebitda(is1, dm1)
        margin1 = _get_margin(is1, dm1)

        lines.append(f"### YoY Trend ({p1.period_name} → {p0.period_name})")
        lines.append(f"- Revenue: {_fmt_money(rev1, basis)} → {_fmt_money(rev0, basis)} ({_fmt_pct(_safe_pct_change(rev0, rev1))}).")
        lines.append(f"- EBITDA: {_fmt_money(ebitda1, basis)} → {_fmt_money(ebitda0, basis)} ({_fmt_pct(_safe_pct_change(ebitda0, ebitda1))}).")

        if margin0 is not None and margin1 is not None:
            lines.append(f"- EBITDA margin: {_fmt_pct(margin1)} → {_fmt_pct(margin0)} (Δ {(margin0 - margin1)*100:.1f}pp).")
        else:
            lines.append("- EBITDA margin: N/A for YoY delta (requires revenue + EBITDA in both periods).")

        # Optional drivers (only if present)
        cos0, cos1 = is0.get("cost_of_sales"), is1.get("cost_of_sales")
        sga0, sga1 = is0.get("sga_expense"), is1.get("sga_expense")
        if rev0 is not None and cos0 is not None and rev1 is not None and cos1 is not None and rev0 != 0 and rev1 != 0:
            gm0 = (rev0 - cos0) / rev0
            gm1 = (rev1 - cos1) / rev1
            lines.append(f"- Gross margin: {_fmt_pct(gm1)} → {_fmt_pct(gm0)} (Δ {(gm0 - gm1)*100:.1f}pp).")
        else:
            lines.append("- Gross margin: N/A (requires revenue + cost of sales in both periods).")

        if sga0 is not None and rev0 is not None and rev0 != 0 and sga1 is not None and rev1 is not None and rev1 != 0:
            sga_rate0 = sga0 / rev0
            sga_rate1 = sga1 / rev1
            lines.append(f"- SG&A as % of revenue: {_fmt_pct(sga_rate1)} → {_fmt_pct(sga_rate0)} (Δ {(sga_rate0 - sga_rate1)*100:.1f}pp).")
        else:
            lines.append("- SG&A as % of revenue: N/A (requires SG&A + revenue in both periods).")
        lines.append("")

    # Data gaps + flags
    lines.append("### Data Gaps / Extraction Notes")
    gaps0 = _missing_fields_for_credit_metrics(p0)
    if gaps0:
        lines.append("- Missing fields (most recent period) that limit leverage/FCC analysis:")
        for g in gaps0:
            lines.append(f"  - {g}")
    else:
        lines.append("- No critical data gaps detected for leverage/FCC inputs in the most recent period.")

    if extracted.validation_flags:
        lines.append("- Validation flags:")
        for f in extracted.validation_flags:
            lines.append(f"  - {f}")
    else:
        lines.append("- No validation flags.")
    lines.append("")

    # Audit table
    lines.append("### Metrics Table (Audit)")
    lines.append("| Metric | " + " | ".join(p.period_name for p in periods) + " |")
    lines.append("|---" + "|---" * len(periods) + "|")

    def pm(p, key: str) -> str:
        is_, bs, cf, dm = _period_dict(p)
        if key == "revenue":
            return _fmt_money(is_.get("revenue"), basis)
        if key == "operating_income":
            return _fmt_money(is_.get("operating_income"), basis)
        if key == "da":
            return _fmt_money(is_.get("depreciation_amortization"), basis)
        if key == "ebitda":
            return _fmt_money(_get_ebitda(is_, dm), basis)
        if key == "margin":
            return _fmt_pct(_get_margin(is_, dm))
        if key == "debt":
            return _fmt_money(bs.get("total_debt"), basis)
        if key == "lev":
            return _fmt_x(_get_leverage(is_, bs, dm))
        if key == "fcc":
            return _fmt_x(_get_fcc(is_, bs, cf, dm))
        if key == "cfo":
            return _fmt_money(cf.get("cfo"), basis)
        if key == "capex":
            return _fmt_money(abs_outflow(cf.get("capex")), basis)
        if key == "fcf":
            return _fmt_money(_get_fcf(cf, dm), basis)
        if key == "cash":
            return _fmt_money(bs.get("cash"), basis)
        return "N/A"

    rows = [
        ("Revenue", "revenue"),
        ("Operating income", "operating_income"),
        ("D&A", "da"),
        ("EBITDA", "ebitda"),
        ("EBITDA margin", "margin"),
        ("Total debt", "debt"),
        ("Leverage (Debt/EBITDA)", "lev"),
        ("FCC", "fcc"),
        ("CFO", "cfo"),
        ("Capex (abs outflow)", "capex"),
        ("Free cash flow", "fcf"),
        ("Cash", "cash"),
    ]

    for label, key in rows:
        lines.append("| " + label + " | " + " | ".join(pm(p, key) for p in periods) + " |")

    return "\n".join(lines)
