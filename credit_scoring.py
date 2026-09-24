"""
Credit Risk Scoring Module

This module contains functions for calculating credit risk metrics:
- Altman Z-Score (bankruptcy prediction model)
- PD Score (Probability of Default on 1-12 scale)
"""

from typing import Optional, Tuple


def compute_altman_z_score(
    revenue: Optional[float],
    operating_income: Optional[float],
    total_assets: Optional[float],
    total_liabilities: Optional[float],
    total_equity: Optional[float],
    retained_earnings: Optional[float] = None,
) -> Optional[float]:
    """
    Modified Altman Z-Score for private companies:
    Z' = 0.717*A + 0.847*B + 3.107*C + 0.420*D + 0.998*E

    Where:
    A = Working Capital / Total Assets (approximated as equity/assets for simplicity)
    B = Retained Earnings / Total Assets (if not available, use 0)
    C = EBIT / Total Assets
    D = Book Value of Equity / Total Liabilities
    E = Sales / Total Assets

    Interpretation:
    - Z > 2.9: Safe zone (low default risk)
    - 1.23 < Z < 2.9: Grey zone (moderate risk)
    - Z < 1.23: Distress zone (high default risk)
    """
    if (total_assets is None or total_assets <= 0 or
        total_liabilities is None or total_liabilities <= 0 or
        total_equity is None or
        operating_income is None or
        revenue is None):
        return None

    # A = Working Capital / Total Assets
    # Simplified as equity/assets since we don't always have current assets/liabilities
    a_ratio = float(total_equity) / float(total_assets)

    # B = Retained Earnings / Total Assets
    # If not available, use 0 (conservative approach)
    b_ratio = 0.0
    if retained_earnings is not None:
        b_ratio = float(retained_earnings) / float(total_assets)

    # C = EBIT / Total Assets
    c_ratio = float(operating_income) / float(total_assets)

    # D = Book Value of Equity / Total Liabilities
    d_ratio = float(total_equity) / float(total_liabilities)

    # E = Sales / Total Assets
    e_ratio = float(revenue) / float(total_assets)

    # Modified Altman Z-Score formula for private companies
    z_score = (0.717 * a_ratio +
               0.847 * b_ratio +
               3.107 * c_ratio +
               0.420 * d_ratio +
               0.998 * e_ratio)

    return z_score


def compute_pd_score(
    leverage: Optional[float],
    fcc: Optional[float],
    ebitda_margin: Optional[float],
    altman_z: Optional[float],
    revenue_growth: Optional[float] = None,
    free_cash_flow: Optional[float] = None,
    ebitda: Optional[float] = None,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Calculate Probability of Default (PD) score on a 1-12 scale.

    Returns: (pd_score, credit_rating)

    Scale:
    1-3: Investment Grade (AAA to BBB)
    4-6: Non-Investment Grade / Crossover (BB to B)
    7-9: Speculative (B- to CCC)
    10-11: High Risk (CC to C)
    12: Default / Distressed (D)

    Methodology:
    - Start with base score from Altman Z-Score
    - Adjust based on leverage, coverage ratios, profitability, growth
    - Each metric contributes to risk assessment

    Factor Weights:
    - Altman Z-Score: 30%
    - Leverage: 25%
    - Fixed Charge Coverage: 25%
    - Profitability (EBITDA Margin): 15%
    - Revenue Growth: 5%
    """
    # Start with base score of 6 (middle of range)
    score = 6.0

    # Weight factors
    altman_weight = 0.30
    leverage_weight = 0.25
    fcc_weight = 0.25
    profitability_weight = 0.15
    growth_weight = 0.05

    # Factor 1: Altman Z-Score (30% weight)
    if altman_z is not None:
        if altman_z > 2.9:
            # Safe zone: reduces score (better credit)
            score -= 3.0 * altman_weight / 0.30
        elif altman_z > 1.23:
            # Grey zone: slight increase in score
            score += 1.0 * altman_weight / 0.30
        else:
            # Distress zone: increases score significantly
            score += 5.0 * altman_weight / 0.30

    # Factor 2: Leverage (25% weight)
    if leverage is not None:
        if leverage < 2.0:
            # Low leverage: reduces score
            score -= 2.5 * leverage_weight / 0.25
        elif leverage < 4.0:
            # Moderate leverage: slight increase
            score += 0.5 * leverage_weight / 0.25
        elif leverage < 6.0:
            # High leverage: moderate increase
            score += 2.5 * leverage_weight / 0.25
        else:
            # Very high leverage: significant increase
            score += 4.5 * leverage_weight / 0.25

    # Factor 3: Fixed Charge Coverage (25% weight)
    if fcc is not None:
        if fcc > 2.0:
            # Strong coverage: reduces score
            score -= 2.5 * fcc_weight / 0.25
        elif fcc > 1.25:
            # Adequate coverage: slight reduction
            score -= 0.5 * fcc_weight / 0.25
        elif fcc > 1.0:
            # Tight coverage: slight increase
            score += 1.5 * fcc_weight / 0.25
        else:
            # Weak coverage: significant increase
            score += 4.0 * fcc_weight / 0.25

    # Factor 4: Profitability (15% weight)
    if ebitda_margin is not None:
        if ebitda_margin > 0.20:
            # Strong margins: reduces score
            score -= 1.5 * profitability_weight / 0.15
        elif ebitda_margin > 0.10:
            # Moderate margins: neutral
            score += 0.0
        elif ebitda_margin > 0.05:
            # Weak margins: increases score
            score += 1.5 * profitability_weight / 0.15
        else:
            # Very weak/negative margins: significant increase
            score += 3.0 * profitability_weight / 0.15

    # Factor 5: Revenue Growth (5% weight)
    if revenue_growth is not None:
        if revenue_growth > 0.10:
            # Strong growth: reduces score
            score -= 0.5 * growth_weight / 0.05
        elif revenue_growth < -0.10:
            # Declining revenue: increases score
            score += 1.5 * growth_weight / 0.05

    # Factor 6: Cash flow quality
    if free_cash_flow is not None and ebitda is not None and ebitda > 0:
        fcf_to_ebitda = free_cash_flow / ebitda
        if fcf_to_ebitda > 0.5:
            # Strong cash conversion: reduces score
            score -= 0.5
        elif fcf_to_ebitda < 0:
            # Negative FCF: increases score
            score += 1.0

    # Bound the score to 1-12 range
    pd_score = max(1, min(12, round(score)))

    # Determine credit rating based on PD score
    if pd_score <= 3:
        credit_rating = "Investment Grade"
    elif pd_score <= 6:
        credit_rating = "Non-Investment Grade"
    elif pd_score <= 9:
        credit_rating = "Speculative"
    elif pd_score <= 11:
        credit_rating = "High Risk"
    else:
        credit_rating = "Default/Distressed"

    return pd_score, credit_rating
