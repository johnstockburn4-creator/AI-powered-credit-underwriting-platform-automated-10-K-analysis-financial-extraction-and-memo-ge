"""
Helper functions to build structured metrics from extracted financials
"""
from typing import Optional, List
from schemas import ExtractionResult, StructuredMetrics, PeriodMetrics, MetricComparison
from metrics import abs_outflow


def _safe_pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    """Calculate percentage change, returns None if inputs invalid"""
    if new is None or old is None or old == 0:
        return None
    return ((new - old) / old) * 100


def build_structured_metrics(extracted: ExtractionResult) -> StructuredMetrics:
    """
    Build structured metrics response with:
    1. Metrics organized by period
    2. YoY comparison bullets
    """
    if not extracted.periods:
        return StructuredMetrics(
            statement_basis=extracted.statement_basis or "actual",
            periods=[],
            yoy_comparisons=[]
        )
    
    basis = extracted.statement_basis or "actual"
    unit_suffix = "MM" if basis == "millions" else ("K" if basis == "thousands" else "")
    
    # Build period metrics
    period_metrics: List[PeriodMetrics] = []
    for p in extracted.periods:
        is_ = p.income_statement or {}
        bs = p.balance_sheet or {}
        cf = p.cash_flow or {}
        dm = p.derived_metrics or {}
        
        pm = PeriodMetrics(
            period_name=p.period_name,
            revenue=is_.get("revenue"),
            cost_of_sales=is_.get("cost_of_sales"),
            sga_expense=is_.get("sga_expense"),
            operating_income=is_.get("operating_income"),
            ebitda=is_.get("ebitda"),
            ebitda_margin=dm.get("ebitda_margin"),
            net_income=is_.get("net_income"),
            total_debt=bs.get("total_debt"),
            leverage=dm.get("leverage_total_debt_to_ebitda"),
            cash=bs.get("cash"),
            cfo=cf.get("cfo"),
            capex=abs_outflow(cf.get("capex")),
            free_cash_flow=dm.get("free_cash_flow"),
            fcc=dm.get("fcc"),
            revolver_facility_size=bs.get("revolver_facility_size"),
            revolver_borrowings=bs.get("revolver_borrowings"),
            revolver_availability=bs.get("revolver_availability"),
            altman_z_score=dm.get("altman_z_score"),
            pd_score=dm.get("pd_score"),
            credit_rating=dm.get("credit_rating")
        )
        period_metrics.append(pm)
    
    # Build YoY comparisons if we have at least 2 periods
    comparisons: List[MetricComparison] = []
    if len(extracted.periods) >= 2:
        p0 = extracted.periods[0]  # Current
        p1 = extracted.periods[1]  # Prior
        
        is0 = p0.income_statement or {}
        is1 = p1.income_statement or {}
        bs0 = p0.balance_sheet or {}
        bs1 = p1.balance_sheet or {}
        cf0 = p0.cash_flow or {}
        cf1 = p1.cash_flow or {}
        dm0 = p0.derived_metrics or {}
        dm1 = p1.derived_metrics or {}
        
        # Helper to create comparison
        def add_comparison(name: str, curr_val: Optional[float], prior_val: Optional[float], 
                          unit: str = unit_suffix, narrative_template: Optional[str] = None):
            change_amt = None
            change_pct = None
            narrative = None
            
            if curr_val is not None and prior_val is not None:
                change_amt = curr_val - prior_val
                change_pct = _safe_pct_change(curr_val, prior_val)
                
                if narrative_template and change_pct is not None:
                    direction = "increased" if change_amt >= 0 else "decreased"
                    narrative = narrative_template.format(
                        direction=direction,
                        change_pct=abs(change_pct),
                        prior_val=prior_val,
                        curr_val=curr_val,
                        prior_period=p1.period_name,
                        curr_period=p0.period_name,
                        unit=unit
                    )
            
            comparisons.append(MetricComparison(
                metric_name=name,
                current_period=p0.period_name,
                current_value=curr_val,
                prior_period=p1.period_name,
                prior_value=prior_val,
                change_amount=change_amt,
                change_pct=change_pct,
                narrative=narrative,
                unit=unit
            ))
        
        # Add all key metrics
        add_comparison(
            "Revenue",
            is0.get("revenue"),
            is1.get("revenue"),
            unit_suffix,
            "Revenue {direction} by {change_pct:.1f}% YoY from ${prior_val:,.0f}{unit} in {prior_period} to ${curr_val:,.0f}{unit} in {curr_period}"
        )
        
        add_comparison(
            "EBITDA",
            is0.get("ebitda"),
            is1.get("ebitda"),
            unit_suffix,
            "EBITDA {direction} by {change_pct:.1f}% YoY from ${prior_val:,.0f}{unit} in {prior_period} to ${curr_val:,.0f}{unit} in {curr_period}"
        )
        
        # EBITDA Margin (special handling for percentage points)
        margin0 = dm0.get("ebitda_margin")
        margin1 = dm1.get("ebitda_margin")
        if margin0 is not None and margin1 is not None:
            margin_change_pp = (margin0 - margin1) * 100  # Convert to percentage points
            direction = "increased" if margin_change_pp >= 0 else "decreased"
            comparisons.append(MetricComparison(
                metric_name="EBITDA Margin",
                current_period=p0.period_name,
                current_value=margin0 * 100,  # Store as percentage
                prior_period=p1.period_name,
                prior_value=margin1 * 100,
                change_amount=margin_change_pp,
                change_pct=None,  # Not meaningful for margins
                narrative=f"EBITDA Margin {direction} by {abs(margin_change_pp):.1f}pp YoY from {margin1*100:.1f}% in {p1.period_name} to {margin0*100:.1f}% in {p0.period_name}",
                unit="%"
            ))
        
        add_comparison(
            "Cost of Sales",
            is0.get("cost_of_sales"),
            is1.get("cost_of_sales"),
            unit_suffix,
            "Cost of Sales {direction} by {change_pct:.1f}% YoY from ${prior_val:,.0f}{unit} in {prior_period} to ${curr_val:,.0f}{unit} in {curr_period}"
        )
        
        add_comparison(
            "SG&A Expense",
            is0.get("sga_expense"),
            is1.get("sga_expense"),
            unit_suffix,
            "SG&A {direction} by {change_pct:.1f}% YoY from ${prior_val:,.0f}{unit} in {prior_period} to ${curr_val:,.0f}{unit} in {curr_period}"
        )
        
        add_comparison(
            "Operating Cash Flow",
            cf0.get("cfo"),
            cf1.get("cfo"),
            unit_suffix,
            "Operating Cash Flow {direction} by {change_pct:.1f}% YoY from ${prior_val:,.0f}{unit} in {prior_period} to ${curr_val:,.0f}{unit} in {curr_period}"
        )
        
        add_comparison(
            "Capex",
            abs_outflow(cf0.get("capex")),
            abs_outflow(cf1.get("capex")),
            unit_suffix,
            "Capex {direction} by {change_pct:.1f}% YoY from ${prior_val:,.0f}{unit} in {prior_period} to ${curr_val:,.0f}{unit} in {curr_period}"
        )
        
        add_comparison(
            "Free Cash Flow",
            dm0.get("free_cash_flow"),
            dm1.get("free_cash_flow"),
            unit_suffix,
            "Free Cash Flow {direction} by {change_pct:.1f}% YoY from ${prior_val:,.0f}{unit} in {prior_period} to ${curr_val:,.0f}{unit} in {curr_period}"
        )
        
        add_comparison(
            "Total Debt",
            bs0.get("total_debt"),
            bs1.get("total_debt"),
            unit_suffix,
            "Total Debt {direction} by {change_pct:.1f}% YoY from ${prior_val:,.0f}{unit} in {prior_period} to ${curr_val:,.0f}{unit} in {curr_period}"
        )
        
        add_comparison(
            "Leverage (Debt/EBITDA)",
            dm0.get("leverage_total_debt_to_ebitda"),
            dm1.get("leverage_total_debt_to_ebitda"),
            "x",
            "Leverage {direction} by {change_pct:.1f}% YoY from {prior_val:.2f}x in {prior_period} to {curr_val:.2f}x in {curr_period}"
        )
        
        add_comparison(
            "FCC",
            dm0.get("fcc"),
            dm1.get("fcc"),
            "x",
            "FCC {direction} by {change_pct:.1f}% YoY from {prior_val:.2f}x in {prior_period} to {curr_val:.2f}x in {curr_period}"
        )
        
        add_comparison(
            "Cash",
            bs0.get("cash"),
            bs1.get("cash"),
            unit_suffix,
            "Cash {direction} by {change_pct:.1f}% YoY from ${prior_val:,.0f}{unit} in {prior_period} to ${curr_val:,.0f}{unit} in {curr_period}"
        )
    
    return StructuredMetrics(
        statement_basis=basis,
        periods=period_metrics,
        yoy_comparisons=comparisons
    )