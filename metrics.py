from __future__ import annotations

from typing import Optional


def safe_divide(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None:
        return None
    if d == 0:
        return None
    return n / d


def abs_outflow(x: Optional[float]) -> Optional[float]:
    """
    Treat cash outflows as positive magnitudes, regardless of sign convention in statements.
    """
    if x is None:
        return None
    return abs(float(x))


def compute_ebitda(
    operating_income: Optional[float],
    depreciation_amortization: Optional[float],
    rent_expense: Optional[float] = None,
    include_rent: bool = True,
) -> Optional[float]:
    """
    Your definition:
      EBITDA = EBIT (operating income) + D&A + rent (optional)
    """
    if operating_income is None or depreciation_amortization is None:
        return None

    ebitda = float(operating_income) + float(depreciation_amortization)

    if include_rent and rent_expense is not None:
        ebitda += float(rent_expense)

    return ebitda


def compute_free_cash_flow(cfo: Optional[float], capex: Optional[float]) -> Optional[float]:
    """
    FCF = CFO - |capex|
    """
    capex_mag = abs_outflow(capex)
    if cfo is None or capex_mag is None:
        return None
    return float(cfo) - capex_mag


def compute_fcc(
    ebitda: Optional[float],
    capex: Optional[float],
    cash_taxes: Optional[float],
    cpltd: Optional[float],
    cash_interest: Optional[float],
) -> Optional[float]:
    """
    Your definition:
      FCC = (EBITDA - |capex| - |cash taxes|) / (CPLTD + |cash interest|)
    """
    capex_mag = abs_outflow(capex)
    taxes_mag = abs_outflow(cash_taxes)
    int_mag = abs_outflow(cash_interest)

    if ebitda is None or capex_mag is None or taxes_mag is None or cpltd is None or int_mag is None:
        return None

    numerator = float(ebitda) - capex_mag - taxes_mag
    denominator = float(cpltd) + int_mag
    return safe_divide(numerator, denominator)


def compute_leverage(total_debt: Optional[float], ebitda: Optional[float]) -> Optional[float]:
    """
    Your definition:
      leverage = total debt / EBITDA
    """
    if total_debt is None or ebitda is None:
        return None
    if ebitda <= 0:
        return None
    return float(total_debt) / float(ebitda)


def compute_ebitda_margin(ebitda: Optional[float], revenue: Optional[float]) -> Optional[float]:
    if ebitda is None or revenue is None:
        return None
    if revenue == 0:
        return None
    return float(ebitda) / float(revenue)
