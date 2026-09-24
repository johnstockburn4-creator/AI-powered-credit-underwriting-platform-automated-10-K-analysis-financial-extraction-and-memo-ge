"""
filing_fetcher.py
=================
Orchestrates the multi-source filing lookup chain for the Credit AI platform.

The platform supports three sources for financial data, attempted in order:

  1. SEC EDGAR (US public companies)
     ─────────────────────────────────────────────────────────────────────────
     Best source. Fetches the actual 10-K HTML filing and runs it through the
     full extraction pipeline including MD&A analysis, segment drivers, and
     memo generation. Free, no API key required.

     Coverage: All US public companies (~10,000 companies on major exchanges
     plus smaller reporting companies, REITs, and Rule 144A debt filers).

  2. Yahoo Finance via yfinance library (international public companies)
     ─────────────────────────────────────────────────────────────────────────
     Fallback for companies not on SEC EDGAR. Returns pre-structured financial
     data (income statement, balance sheet, cash flow) for 3-5 years. Faster
     and cheaper than document parsing but provides no MD&A narrative.

     Coverage: Most globally-listed public companies (~50,000 tickers across
     NYSE, NASDAQ, LSE, TSX, ASX, Euronext, etc.)

     Limitations:
       - No MD&A analysis (no segment drivers, no management commentary)
       - Numbers may differ from audited filings due to restatements
       - Requires yfinance to be installed: pip install yfinance

  3. Manual URL / file upload
     ─────────────────────────────────────────────────────────────────────────
     If both EDGAR and yfinance fail, the system prompts the user to provide
     a URL to an annual report or upload the file directly. This covers:
       - Private companies with publicly available financial statements
       - Companies whose filings are not indexed by EDGAR or Yahoo Finance
       - Historical filings older than what yfinance provides

The result always indicates which source was used and what limitations apply,
so the frontend can show appropriate caveats to the analyst.

Usage:
    from filing_fetcher import FilingFetcher, FetchResult

    fetcher = FilingFetcher()

    # Try to find and fetch the most recent 10-K for a company
    result = fetcher.find_and_fetch("Rolls-Royce Holdings")

    if result.source == "edgar":
        # Full analysis possible — text is the 10-K HTML
        text = result.filing_text
        run_full_extraction(text)

    elif result.source == "yfinance":
        # Metrics only — structured data, no MD&A
        metrics = result.structured_data
        display_metrics_only(metrics)

    elif result.source == "not_found":
        # Prompt user for manual input
        prompt_user_for_url_or_upload()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from edgar_client import EdgarClient, CompanyResult, FilingResult

logger = logging.getLogger("credit_ai")


# ── Result types ──────────────────────────────────────────────────────────────

# The source that successfully provided the filing data
SOURCE_EDGAR     = "edgar"      # Full 10-K document from SEC EDGAR
SOURCE_YFINANCE  = "yfinance"   # Structured data from Yahoo Finance
SOURCE_URL       = "url"        # User-provided URL or uploaded file
SOURCE_NOT_FOUND = "not_found"  # No source found — user must provide manually


@dataclass
class FetchResult:
    """
    The result of a filing fetch attempt.

    Attributes:
        source:          Which data source provided the result. One of:
                         "edgar", "yfinance", "url", "not_found"

        company_name:    The company name as identified by the data source
                         (e.g. "CHURCH & DWIGHT CO INC"). May differ slightly
                         from what the user typed.

        cik:             SEC CIK number if the company was found on EDGAR,
                         otherwise None.

        ticker:          Stock ticker symbol if available (e.g. "CHD")

        fiscal_year:     The fiscal year this data covers (e.g. "FY2024")

        filing_text:     The full text of the annual report document, ready to
                         pass to extract_financials_from_text() in agents.py.
                         Only set when source="edgar" or source="url".
                         None when source="yfinance" (no document available).

        structured_data: Pre-structured financial data from Yahoo Finance.
                         Only set when source="yfinance".
                         Format matches the extraction schema as closely as
                         possible so it can be used to populate the analysis
                         without running the LLM extraction.

        supports_mda:    True if MD&A analysis is possible (requires filing_text).
                         False for yfinance results — no narrative available.

        limitations:     Human-readable list of limitations for this result.
                         Displayed as caveats in the frontend.
                         e.g. ["MD&A analysis not available for non-EDGAR filings",
                               "Numbers sourced from Yahoo Finance — verify against filing"]

        available_filings: For EDGAR results, the list of available historical
                           filings so the frontend can offer year selection.

        error:           Error message if source="not_found", describing why
                         the search failed.
    """
    source:             str
    company_name:       str                    = ""
    cik:                Optional[str]          = None
    ticker:             Optional[str]          = None
    fiscal_year:        str                    = ""
    filing_text:        Optional[str]          = None
    structured_data:    Optional[Dict]         = None
    supports_mda:       bool                   = True
    limitations:        List[str]              = field(default_factory=list)
    available_filings:  List[FilingResult]     = field(default_factory=list)
    error:              Optional[str]          = None


# ── Main fetcher ──────────────────────────────────────────────────────────────

class FilingFetcher:
    """
    Orchestrates multi-source filing lookup with automatic fallback.

    This class is the single entry point for "find me the annual report for
    company X." It tries each source in order and returns the best available
    result, always indicating what source was used and what limitations apply.

    Design philosophy:
        The system should always give the analyst something useful, even if it
        can't provide a full analysis. A metrics-only result from Yahoo Finance
        is better than an error. The limitations list tells the analyst exactly
        what they're getting and what to verify manually.

    Thread safety:
        The EdgarClient is not thread-safe. Create one FilingFetcher per request.
    """

    def __init__(self):
        """Initialises the fetcher with an EDGAR client."""
        self._edgar = EdgarClient()

    def search_companies(self, query: str) -> List[CompanyResult]:
        """
        Searches for companies by name across all data sources.

        Currently searches EDGAR (US public companies). Results include the
        CIK, ticker, and official name that can be used with get_filing().

        Args:
            query: Company name to search for (partial names work well)

        Returns:
            List of CompanyResult objects ranked by match quality.
            Empty list if no matches found.

        Example:
            >>> fetcher.search_companies("mosaic")
            [CompanyResult(name='MOSAIC CO', cik='1285785', ticker='MOS'), ...]
        """
        logger.info(f"Searching companies for query: {query!r}")
        return self._edgar.search_companies(query, max_results=10)

    def get_available_filings(
        self,
        cik:   str,
        count: int = 5,
    ) -> List[FilingResult]:
        """
        Returns the list of available 10-K filings for a company.

        Used by the frontend to show a year selector after the analyst has
        chosen a company from the search results.

        Args:
            cik:   The company's SEC CIK number
            count: Maximum number of filings to return

        Returns:
            List of FilingResult objects, most recent first.
        """
        logger.info(f"Getting available 10-K filings for CIK: {cik}")
        return self._edgar.get_10k_filings(cik, count=count)

    def fetch_by_edgar(
        self,
        cik:              str,
        accession_number: Optional[str] = None,
    ) -> FetchResult:
        """
        Fetches a filing directly from SEC EDGAR.

        This is the primary path for US public companies. It returns the full
        10-K HTML document which supports complete analysis including MD&A,
        segment drivers, revolver data, and memo generation.

        Args:
            cik:              The company's SEC CIK number
            accession_number: Specific filing to fetch (e.g. "0000313927-25-000009").
                              If None, fetches the most recent 10-K.

        Returns:
            FetchResult with source="edgar" and filing_text set to the full
            10-K HTML, ready for extract_financials_from_text().

            If the filing cannot be fetched, returns source="not_found" with
            an error message.
        """
        try:
            # Get the list of filings to find the right document URL
            filings = self._edgar.get_10k_filings(cik, count=10)

            if not filings:
                return FetchResult(
                    source="not_found",
                    error=f"No 10-K filings found for CIK {cik} on EDGAR."
                )

            # Select the target filing
            if accession_number:
                # Find the specific filing by accession number
                target = next(
                    (f for f in filings if f.accession_number == accession_number),
                    None
                )
                if not target:
                    return FetchResult(
                        source="not_found",
                        error=f"Filing {accession_number} not found for CIK {cik}."
                    )
            else:
                # Default to the most recent filing
                target = filings[0]

            # Fetch the actual document text
            filing_text = self._edgar.fetch_filing_text(target.document_url)
            if not filing_text:
                return FetchResult(
                    source="not_found",
                    error=f"Could not download filing document from {target.document_url}"
                )

            # Get the company's official name for the borrower field
            company_name = self._edgar.get_company_name(cik) or f"CIK {cik}"

            logger.info(
                f"EDGAR fetch successful: {company_name} {target.fiscal_year} "
                f"({len(filing_text):,} chars)"
            )

            return FetchResult(
                source            = SOURCE_EDGAR,
                company_name      = company_name,
                cik               = cik,
                ticker            = None,  # Could add from tickers list if needed
                fiscal_year       = target.fiscal_year,
                filing_text       = filing_text,
                structured_data   = None,
                supports_mda      = True,   # Full document — MD&A analysis supported
                limitations       = [],     # No limitations for EDGAR filings
                available_filings = filings,
            )

        except Exception as e:
            logger.error(f"EDGAR fetch failed for CIK {cik}: {e}")
            return FetchResult(
                source="not_found",
                error=f"EDGAR fetch failed: {str(e)}"
            )

    def fetch_by_yfinance(
        self,
        ticker:      str,
        company_name: str = "",
    ) -> FetchResult:
        """
        Fetches structured financial data from Yahoo Finance for international
        and non-EDGAR companies.

        Yahoo Finance provides pre-structured income statement, balance sheet,
        and cash flow data for most globally-listed companies. The data is
        returned in a format compatible with the extraction schema so it can
        be used to populate the analysis without running LLM extraction.

        Limitations:
          - No MD&A analysis (no segment drivers or management commentary)
          - Numbers from Yahoo Finance may differ from audited filings due to
            restatements, rounding, or accounting adjustments
          - Typically provides 3-4 years of annual data

        Args:
            ticker:       Yahoo Finance ticker symbol (e.g. "RR.L" for
                          Rolls-Royce on the London Stock Exchange)
            company_name: Display name for the company

        Returns:
            FetchResult with source="yfinance" and structured_data containing
            the financial data formatted to match the extraction schema.
            If yfinance is not installed or the ticker is not found, returns
            source="not_found".

        Requires:
            pip install yfinance
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed — skipping Yahoo Finance fallback")
            return FetchResult(
                source="not_found",
                error="Yahoo Finance integration not available. Install with: pip install yfinance"
            )

        try:
            logger.info(f"Fetching Yahoo Finance data for ticker: {ticker}")
            stock = yf.Ticker(ticker)
            info  = stock.info or {}

            # Validate that we got real data back
            # Yahoo Finance returns an empty or minimal info dict for unknown tickers
            if not info.get("longName") and not info.get("shortName"):
                return FetchResult(
                    source="not_found",
                    error=f"Ticker {ticker!r} not found on Yahoo Finance."
                )

            display_name = company_name or info.get("longName") or info.get("shortName", ticker)

            # Fetch financial statements
            # These return pandas DataFrames with years as columns
            income_stmt = stock.financials         # Annual income statement
            balance_sht = stock.balance_sheet      # Annual balance sheet
            cash_flow   = stock.cashflow           # Annual cash flow statement

            # Convert to the extraction schema format
            # Each period becomes an ExtractedPeriod-compatible dict
            periods = self._yfinance_to_periods(
                income_stmt = income_stmt,
                balance_sht = balance_sht,
                cash_flow   = cash_flow,
            )

            if not periods:
                return FetchResult(
                    source="not_found",
                    error=f"No financial data available for {ticker} on Yahoo Finance."
                )

            # Determine currency and scale
            currency = info.get("currency", "USD")
            scale    = "millions"  # yfinance returns raw values, usually in actual dollars
            # Most large companies report in millions — we'll note this as a caveat

            structured_data = {
                "statement_basis":  scale,
                "periods":          periods,
                "validation_flags": [
                    f"Data sourced from Yahoo Finance ({ticker}) — verify key figures against the audited annual report",
                    f"Currency: {currency}. Values may be in actual dollars, not millions — verify scale.",
                    "MD&A analysis not available for Yahoo Finance data — segment drivers and management commentary are not included",
                ],
                "source":           "yfinance",
                "ticker":           ticker,
            }

            fiscal_year = periods[0].get("period_name", "Recent") if periods else "Recent"

            logger.info(
                f"Yahoo Finance fetch successful: {display_name} ({ticker}), "
                f"{len(periods)} periods"
            )

            return FetchResult(
                source          = SOURCE_YFINANCE,
                company_name    = display_name,
                cik             = None,
                ticker          = ticker,
                fiscal_year     = fiscal_year,
                filing_text     = None,    # No document — structured data only
                structured_data = structured_data,
                supports_mda    = False,   # No MD&A available from Yahoo Finance
                limitations     = [
                    "MD&A analysis not available — no filing document was obtained",
                    "Financial data sourced from Yahoo Finance — verify against audited statements",
                    f"Currency: {currency}",
                ],
            )

        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed for {ticker}: {e}")
            return FetchResult(
                source="not_found",
                error=f"Yahoo Finance fetch failed: {str(e)}"
            )

    def fetch_by_url(self, url: str) -> FetchResult:
        """
        Fetches a filing document from a user-provided URL.

        This is the fallback of last resort. The analyst pastes a URL to an
        annual report PDF or HTML page (e.g. from a company's investor relations
        site) and the system fetches and analyses it exactly like an EDGAR filing.

        Supports:
          - Direct links to PDF annual reports
          - Direct links to HTML filing pages
          - Most investor relations page formats

        Args:
            url: Direct URL to an annual report document

        Returns:
            FetchResult with source="url" and filing_text set to the document
            content. The actual parsing (PDF vs HTML) is handled by agents.py's
            existing pipeline.
        """
        import httpx

        try:
            logger.info(f"Fetching filing from user-provided URL: {url}")

            with httpx.Client(
                timeout=60.0,
                follow_redirects=True,
                headers={"User-Agent": "CreditAI/1.0 credit-ai@example.com"}
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                content = response.text

            logger.info(f"URL fetch successful: {len(content):,} chars from {url}")

            return FetchResult(
                source       = SOURCE_URL,
                company_name = "",          # Will be filled from borrower form
                filing_text  = content,
                supports_mda = True,        # Full document — MD&A supported
                limitations  = [
                    "Document sourced from user-provided URL — verify it is the correct annual report",
                ],
            )

        except Exception as e:
            logger.error(f"URL fetch failed for {url}: {e}")
            return FetchResult(
                source="not_found",
                error=f"Could not fetch document from URL: {str(e)}"
            )

    # ── yfinance data conversion ──────────────────────────────────────────────

    def _yfinance_to_periods(
        self,
        income_stmt: Any,
        balance_sht: Any,
        cash_flow:   Any,
    ) -> List[Dict]:
        """
        Converts Yahoo Finance DataFrame financials to the extraction schema format.

        Yahoo Finance returns pandas DataFrames with financial line items as rows
        and fiscal year end dates as columns. This method converts that structure
        into the list-of-periods format used throughout the extraction schema.

        The field mapping attempts to align Yahoo Finance's line item names with
        the extraction schema field names. Not all fields will be available for
        all companies — missing fields are left as None.

        Args:
            income_stmt: yfinance Ticker.financials DataFrame
            balance_sht: yfinance Ticker.balance_sheet DataFrame
            cash_flow:   yfinance Ticker.cashflow DataFrame

        Returns:
            List of period dicts compatible with ExtractedPeriod schema,
            ordered most recent first.
        """
        try:
            import pandas as pd
        except ImportError:
            return []

        if income_stmt is None or income_stmt.empty:
            return []

        periods = []

        # Each column in the DataFrame is a year-end date
        for col in income_stmt.columns:
            try:
                period_name = f"FY{col.year}" if hasattr(col, 'year') else str(col)[:4]

                def safe_get(df, *keys):
                    """
                    Safely extracts a value from a DataFrame trying multiple
                    possible key names. Yahoo Finance uses inconsistent naming
                    across companies and regions.

                    Returns the value in millions (divides by 1,000,000) or None.
                    """
                    if df is None or df.empty:
                        return None
                    for key in keys:
                        if key in df.index:
                            val = df.loc[key, col]
                            if pd.notna(val):
                                # Convert from raw dollars to millions
                                return float(val) / 1_000_000
                    return None

                # Income statement fields
                # Yahoo Finance uses various names for the same line item
                revenue = safe_get(income_stmt,
                    "Total Revenue", "Revenue", "Net Revenue", "TotalRevenue")
                cogs    = safe_get(income_stmt,
                    "Cost Of Revenue", "CostOfRevenue", "Cost of Goods Sold")
                op_inc  = safe_get(income_stmt,
                    "Operating Income", "OperatingIncome", "EBIT")
                net_inc = safe_get(income_stmt,
                    "Net Income", "NetIncome", "Net Income Common Stockholders")
                ebitda  = safe_get(income_stmt,
                    "EBITDA", "Ebitda", "Normalized EBITDA")
                da      = safe_get(income_stmt,
                    "Reconciled Depreciation", "Depreciation And Amortization",
                    "Depreciation", "D&A")
                int_exp = safe_get(income_stmt,
                    "Interest Expense", "InterestExpense", "Total Interest Expense")

                # Balance sheet fields
                cash       = safe_get(balance_sht, "Cash And Cash Equivalents",
                                      "Cash", "CashAndCashEquivalents")
                total_debt = safe_get(balance_sht, "Total Debt", "Long Term Debt And Capital Lease Obligation",
                                      "LongTermDebt", "Total Long Term Debt")
                lt_debt    = safe_get(balance_sht, "Long Term Debt", "LongTermDebt")
                tot_assets = safe_get(balance_sht, "Total Assets", "TotalAssets")
                tot_eq     = safe_get(balance_sht, "Total Equity Gross Minority Interest",
                                      "Stockholders Equity", "TotalStockholdersEquity")
                tot_liab   = safe_get(balance_sht, "Total Liabilities Net Minority Interest",
                                      "TotalLiabilities")

                # Cash flow fields
                cfo    = safe_get(cash_flow, "Operating Cash Flow", "OperatingCashFlow",
                                  "Cash From Operations")
                capex  = safe_get(cash_flow, "Capital Expenditure", "CapitalExpenditure",
                                  "Purchases Of Property Plant And Equipment")
                # Capex is typically negative in cash flow statements — normalise to negative
                if capex is not None and capex > 0:
                    capex = -capex

                period = {
                    "period_name": period_name,
                    "income_statement": {
                        "revenue":                   revenue,
                        "cost_of_sales":             cogs,
                        "operating_income":          op_inc,
                        "net_income":                net_inc,
                        "ebitda":                    ebitda,
                        "depreciation_amortization": da,
                        "interest_expense":          int_exp,
                        "sga_expense":               None,  # Not directly available from yfinance
                        "rent_expense":              None,
                    },
                    "balance_sheet": {
                        "cash":        cash,
                        "total_debt":  total_debt,
                        "long_term_debt": lt_debt,
                        "total_assets": tot_assets,
                        "total_equity": tot_eq,
                        "total_liabilities": tot_liab,
                        "current_portion_long_term_debt": None,
                        "revolver_facility_size":     None,
                        "revolver_borrowings":        None,
                        "revolver_availability":      None,
                    },
                    "cash_flow": {
                        "cfo":                       cfo,
                        "capex":                     capex,
                        "cash_paid_for_interest":    None,  # Not available from yfinance
                        "cash_paid_for_income_taxes": None,
                    },
                    "derived_metrics": {},
                    "notes": [
                        "DATA SOURCE: Yahoo Finance — verify against audited annual report",
                        "MD&A analysis not available for Yahoo Finance data",
                    ],
                }

                periods.append(period)

            except Exception as e:
                logger.warning(f"Failed to parse yfinance period {col}: {e}")
                continue

        return periods

    def close(self):
        """Releases resources. Call when done with the fetcher."""
        self._edgar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
