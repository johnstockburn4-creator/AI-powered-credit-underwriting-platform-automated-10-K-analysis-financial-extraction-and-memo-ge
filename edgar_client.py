"""
edgar_client.py
===============
Client for the SEC EDGAR public API.

The SEC provides several free, unauthenticated REST APIs for accessing company
filings. This module wraps those APIs into a clean interface used by
filing_fetcher.py to locate and download 10-K documents for analysis.

Key EDGAR APIs used:
  1. Company Tickers JSON — full list of ~10,000 public companies with CIK numbers
     https://www.sec.gov/files/company_tickers.json

  2. Submissions API — all filings for a company, indexed by CIK
     https://data.sec.gov/submissions/CIK{cik:010d}.json

  3. EDGAR Archives — the actual filing documents (HTML/PDF)
     https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no}/

Rate limiting:
  The SEC requests a maximum of 10 requests/second and requires a descriptive
  User-Agent header. This client enforces a 0.15-second delay between requests
  and uses a configurable User-Agent.

Caching:
  The company tickers list (~3MB) is cached locally for 24 hours since it
  changes infrequently. Per-company submission data is not cached (small files,
  always want latest filings).

Usage:
    from edgar_client import EdgarClient

    client = EdgarClient()

    # Search for companies matching a name
    companies = client.search_companies("Church & Dwight")
    # [{"name": "CHURCH & DWIGHT CO INC", "cik": "313927", "ticker": "CHD"}, ...]

    # Get recent 10-K filings for a company
    filings = client.get_10k_filings(cik="313927", count=5)
    # [{"fiscal_year": "FY2024", "filing_date": "2025-02-14", "url": "https://..."}, ...]

    # Fetch the actual filing HTML text
    text = client.fetch_filing_text(filings[0]["url"])
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger("credit_ai")

# ── Configuration ─────────────────────────────────────────────────────────────

# The SEC requires a descriptive User-Agent identifying your application and
# contact email. Failure to provide this may result in IP blocks.
# Override via environment variable EDGAR_USER_AGENT.
DEFAULT_USER_AGENT = os.environ.get(
    "EDGAR_USER_AGENT",
    "CreditAI/1.0 credit-ai@example.com"
)

# Minimum seconds between EDGAR API requests to stay within the 10 req/s limit
EDGAR_REQUEST_DELAY = 0.15

# How long to cache the company tickers list locally (seconds)
TICKERS_CACHE_TTL = 60 * 60 * 24  # 24 hours

# Local path for the cached tickers file
TICKERS_CACHE_PATH = os.environ.get("EDGAR_TICKERS_CACHE", "/tmp/edgar_company_tickers.json")

# EDGAR API base URLs
EDGAR_BASE          = "https://data.sec.gov"
EDGAR_SUBMISSIONS   = f"{EDGAR_BASE}/submissions/CIK{{cik:010d}}.json"
EDGAR_ARCHIVES      = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_doc}"
EDGAR_TICKERS_URL   = "https://www.sec.gov/files/company_tickers.json"


# ── Data classes ──────────────────────────────────────────────────────────────

class CompanyResult:
    """
    A company found in the SEC EDGAR company registry.

    Attributes:
        name:    Official company name as registered with the SEC
                 (e.g. "CHURCH & DWIGHT CO INC")
        cik:     Central Index Key — the SEC's unique identifier for the company
                 (e.g. "313927"). Always a string to preserve leading zeros.
        ticker:  Stock ticker symbol (e.g. "CHD"). May be empty for private issuers.
    """
    def __init__(self, name: str, cik: str, ticker: str = ""):
        self.name   = name
        self.cik    = cik
        self.ticker = ticker

    def to_dict(self) -> Dict:
        return {"name": self.name, "cik": self.cik, "ticker": self.ticker}

    def __repr__(self):
        return f"CompanyResult(name={self.name!r}, cik={self.cik!r}, ticker={self.ticker!r})"


class FilingResult:
    """
    A single 10-K (or 10-K/A amendment) filing for a company.

    Attributes:
        accession_number: SEC accession number in standard format
                          (e.g. "0000313927-25-000009")
        filing_date:      Date the filing was submitted to the SEC
        period_of_report: The last day of the fiscal year covered by the filing
                          (e.g. "2024-12-31" for a FY2024 annual report)
        fiscal_year:      Human-readable fiscal year label (e.g. "FY2024")
        document_url:     Direct URL to the primary HTML filing document.
                          This is what gets passed to extract_financials_from_text().
        form_type:        "10-K" for annual report, "10-K/A" for amendment
    """
    def __init__(
        self,
        accession_number: str,
        filing_date:      str,
        period_of_report: str,
        fiscal_year:      str,
        document_url:     str,
        form_type:        str = "10-K",
    ):
        self.accession_number = accession_number
        self.filing_date      = filing_date
        self.period_of_report = period_of_report
        self.fiscal_year      = fiscal_year
        self.document_url     = document_url
        self.form_type        = form_type

    def to_dict(self) -> Dict:
        return {
            "accession_number": self.accession_number,
            "filing_date":      self.filing_date,
            "period_of_report": self.period_of_report,
            "fiscal_year":      self.fiscal_year,
            "document_url":     self.document_url,
            "form_type":        self.form_type,
        }

    def __repr__(self):
        return (
            f"FilingResult(fiscal_year={self.fiscal_year!r}, "
            f"filing_date={self.filing_date!r}, form_type={self.form_type!r})"
        )


# ── Main client ───────────────────────────────────────────────────────────────

class EdgarClient:
    """
    Client for the SEC EDGAR public filing API.

    This class handles all communication with EDGAR — searching for companies,
    retrieving their filing history, and fetching the actual filing documents.

    Thread safety:
        This client is not thread-safe. Create one instance per request
        or use a lock if sharing across threads.

    Example:
        client = EdgarClient()
        companies = client.search_companies("Mosaic Company")
        filings   = client.get_10k_filings(companies[0].cik, count=3)
        text      = client.fetch_filing_text(filings[0].document_url)
    """

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        """
        Args:
            user_agent: String identifying your application, sent with every
                        request as the HTTP User-Agent header.
                        Format: "AppName/version contact@email.com"
        """
        self.user_agent   = user_agent
        self._last_request = 0.0  # Unix timestamp of the last request made

        # httpx client with persistent connection pooling for efficiency
        self._http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            timeout=30.0,
            follow_redirects=True,
        )

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def _rate_limited_get(self, url: str, **kwargs) -> httpx.Response:
        """
        Makes a GET request while respecting EDGAR's rate limit.

        Sleeps if the minimum inter-request delay has not elapsed since the
        last request. This ensures we never exceed 10 requests/second.

        Args:
            url:    The URL to fetch
            kwargs: Additional arguments passed to httpx.Client.get()

        Returns:
            httpx.Response

        Raises:
            httpx.HTTPStatusError: If the server returns a 4xx or 5xx status
        """
        # Calculate how long we need to wait before making the next request
        elapsed = time.time() - self._last_request
        if elapsed < EDGAR_REQUEST_DELAY:
            time.sleep(EDGAR_REQUEST_DELAY - elapsed)

        logger.debug(f"EDGAR GET: {url}")
        response = self._http.get(url, **kwargs)
        self._last_request = time.time()
        response.raise_for_status()
        return response

    # ── Company search ────────────────────────────────────────────────────────

    def _load_company_tickers(self) -> Dict[str, Dict]:
        """
        Loads the SEC's full list of registered companies with their CIK numbers.

        The SEC publishes a JSON file (~3MB) containing every company registered
        with EDGAR, including their CIK number and ticker symbol. We download
        this file and cache it locally for 24 hours to avoid hammering the SEC
        server on every search.

        Returns:
            Dict mapping string index → {"cik_str": "313927", "ticker": "CHD",
                                         "title": "CHURCH & DWIGHT CO INC"}

        Example:
            {
              "0": {"cik_str": "1750",   "ticker": "A",   "title": "AGILENT TECHNOLOGIES INC"},
              "1": {"cik_str": "313927", "ticker": "CHD", "title": "CHURCH & DWIGHT CO INC"},
              ...
            }
        """
        # Check if we have a valid cached copy
        if os.path.exists(TICKERS_CACHE_PATH):
            cache_age = time.time() - os.path.getmtime(TICKERS_CACHE_PATH)
            if cache_age < TICKERS_CACHE_TTL:
                logger.debug(f"Loading company tickers from cache ({cache_age:.0f}s old)")
                with open(TICKERS_CACHE_PATH, encoding="utf-8") as f:
                    return json.load(f)

        # Cache is stale or missing — fetch fresh copy from SEC
        logger.info("Fetching company tickers from SEC EDGAR (caching for 24h)...")
        response = self._rate_limited_get(EDGAR_TICKERS_URL)
        tickers  = response.json()

        # Save to cache
        try:
            with open(TICKERS_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(tickers, f)
        except OSError as e:
            # Cache write failure is non-fatal — we still have the data in memory
            logger.warning(f"Could not write tickers cache: {e}")

        return tickers

    def search_companies(
        self,
        query:      str,
        max_results: int = 10,
    ) -> List[CompanyResult]:
        """
        Searches for companies registered with the SEC by name.

        Searches the SEC's complete company registry (~10,000 companies) using
        case-insensitive substring matching on the company name. Returns results
        ranked by how closely the name matches the query.

        Args:
            query:       Company name to search for (e.g. "Church & Dwight",
                         "mosaic", "3M"). Partial names work well.
            max_results: Maximum number of companies to return (default 10)

        Returns:
            List of CompanyResult objects, sorted by name match quality.
            Empty list if no matches found.

        Example:
            >>> client.search_companies("church dwight")
            [CompanyResult(name='CHURCH & DWIGHT CO INC', cik='313927', ticker='CHD')]

        Notes:
            - The search is performed locally against a cached company list
            - Companies with no ticker (private debt issuers) are included
            - The CIK is zero-padded to 10 digits internally but returned as-is
        """
        if not query or not query.strip():
            return []

        # Normalise the query for matching:
        # Remove punctuation, lowercase, split into words
        # "Church & Dwight" → ["church", "dwight"]
        query_clean = re.sub(r"[&,./]", " ", query).lower()
        query_words = [w for w in query_clean.split() if len(w) > 1]

        if not query_words:
            return []

        tickers = self._load_company_tickers()

        results = []
        for entry in tickers.values():
            name_lower = entry.get("title", "").lower()

            # Score: count how many query words appear in the company name
            # Higher score = better match
            score = sum(1 for word in query_words if word in name_lower)
            if score == 0:
                continue

            # Bonus: exact phrase match scores higher than individual word matches
            if query_clean.strip() in name_lower:
                score += 10

            results.append((score, CompanyResult(
                name   = entry.get("title", ""),
                cik    = str(entry.get("cik_str", "")),
                ticker = entry.get("ticker", ""),
            )))

        # Sort by score descending, then alphabetically within same score
        results.sort(key=lambda x: (-x[0], x[1].name))

        return [r for _, r in results[:max_results]]

    # ── Filing retrieval ──────────────────────────────────────────────────────

    def get_10k_filings(
        self,
        cik:   str,
        count: int = 5,
    ) -> List[FilingResult]:
        """
        Retrieves recent 10-K annual report filings for a company.

        Uses EDGAR's Submissions API which returns the complete filing history
        for a company including form type, filing date, and document references.

        Args:
            cik:   The company's SEC Central Index Key (e.g. "313927").
                   Can be zero-padded or not — both are handled.
            count: Maximum number of filings to return (default 5).
                   Returns the most recent filings first.

        Returns:
            List of FilingResult objects, most recent first.
            Includes both 10-K (full) and 10-K/A (amendments).
            Empty list if the company has no 10-K filings on record.

        Raises:
            httpx.HTTPStatusError: If the CIK is invalid or EDGAR is unreachable

        Example:
            >>> filings = client.get_10k_filings("313927", count=3)
            >>> filings[0].fiscal_year
            'FY2024'
            >>> filings[0].document_url
            'https://www.sec.gov/Archives/edgar/data/313927/000031392725000009/chd-20241231.htm'
        """
        # EDGAR Submissions API requires the CIK zero-padded to exactly 10 digits
        cik_int     = int(cik)
        submissions_url = EDGAR_SUBMISSIONS.format(cik=cik_int)

        logger.info(f"Fetching submissions for CIK {cik}")
        response = self._rate_limited_get(submissions_url)
        data     = response.json()

        # The submissions JSON contains a "filings" → "recent" section with
        # parallel arrays: one value per filing, all arrays have the same length.
        # Example structure:
        # {
        #   "name": "CHURCH & DWIGHT CO INC",
        #   "filings": {
        #     "recent": {
        #       "form":             ["10-K", "8-K", "10-K/A", ...],
        #       "filingDate":       ["2025-02-14", "2025-01-08", ...],
        #       "reportDate":       ["2024-12-31", ...],
        #       "accessionNumber":  ["0000313927-25-000009", ...],
        #       "primaryDocument":  ["chd-20241231.htm", ...],
        #     }
        #   }
        # }
        recent = data.get("filings", {}).get("recent", {})

        forms        = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        accessions   = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        results = []
        for i, form_type in enumerate(forms):
            # We only want annual reports (10-K) and their amendments (10-K/A)
            if form_type not in ("10-K", "10-K/A"):
                continue

            if len(results) >= count:
                break

            accession_no         = accessions[i] if i < len(accessions) else ""
            filing_date          = filing_dates[i] if i < len(filing_dates) else ""
            period               = report_dates[i] if i < len(report_dates) else ""
            primary_doc          = primary_docs[i] if i < len(primary_docs) else ""

            if not accession_no or not primary_doc:
                continue

            # Construct the document URL:
            # The accession number "0000313927-25-000009" becomes
            # the directory "000031392725000009" (dashes removed)
            accession_no_dashes = accession_no.replace("-", "")
            document_url = EDGAR_ARCHIVES.format(
                cik              = cik_int,
                accession_no_dashes = accession_no_dashes,
                primary_doc      = primary_doc,
            )

            # Derive the fiscal year from the period end date
            # "2024-12-31" → "FY2024"
            fiscal_year = self._period_to_fiscal_year(period)

            results.append(FilingResult(
                accession_number = accession_no,
                filing_date      = filing_date,
                period_of_report = period,
                fiscal_year      = fiscal_year,
                document_url     = document_url,
                form_type        = form_type,
            ))

        logger.info(f"Found {len(results)} 10-K filings for CIK {cik}")
        return results

    def _period_to_fiscal_year(self, period: str) -> str:
        """
        Converts a period end date to a fiscal year label.

        The SEC stores the period end date as "YYYY-MM-DD". We extract the year
        and format it as "FYYYY".

        Note: For companies with fiscal years ending before June (e.g. a company
        with a March year-end whose period is "2024-03-31"), the fiscal year label
        may differ from the calendar year in their own reporting. We use the
        calendar year of the period end date for simplicity.

        Args:
            period: Date string in "YYYY-MM-DD" format (e.g. "2024-12-31")

        Returns:
            Fiscal year label (e.g. "FY2024"). Returns "Unknown" if parsing fails.

        Examples:
            "2024-12-31" → "FY2024"
            "2024-03-31" → "FY2024"
            ""           → "Unknown"
        """
        if not period or len(period) < 4:
            return "Unknown"
        try:
            year = datetime.strptime(period, "%Y-%m-%d").year
            return f"FY{year}"
        except ValueError:
            return f"FY{period[:4]}"

    # ── Document fetching ─────────────────────────────────────────────────────

    def fetch_filing_text(self, url: str) -> Optional[str]:
        """
        Downloads and returns the text content of a filing document.

        The filing document is typically an HTML file (.htm) that contains the
        full 10-K text including financial statements and MD&A. It may be large
        (1-5MB for complex filings). The content is passed directly to
        extract_financials_from_text() in agents.py for analysis.

        Args:
            url: Direct URL to the filing document
                 (e.g. "https://www.sec.gov/Archives/edgar/data/...")

        Returns:
            The full text content of the document as a string, or None if the
            document could not be retrieved.

        Notes:
            - HTML documents are returned as-is (agents.py handles HTML parsing)
            - Very large documents (>50MB) are truncated to avoid memory issues
            - The SEC sometimes returns inline XBRL documents (.htm files that
              contain embedded financial data in XBRL tags) — these are handled
              by the existing HTML-to-text conversion in agents.py
        """
        try:
            logger.info(f"Fetching filing document: {url}")
            response = self._rate_limited_get(url)
            content  = response.text

            # Guard against extremely large documents
            max_chars = 50_000_000  # 50MB of text
            if len(content) > max_chars:
                logger.warning(
                    f"Filing document is very large ({len(content):,} chars), "
                    f"truncating to {max_chars:,} chars"
                )
                content = content[:max_chars]

            logger.info(f"Fetched filing document: {len(content):,} chars")
            return content

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching filing {url}: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch filing {url}: {e}")
            return None

    def get_company_name(self, cik: str) -> Optional[str]:
        """
        Returns the official company name for a given CIK.

        Fetches the submissions JSON for the company and extracts the name field.
        Used to pre-populate the borrower name field in the analysis request.

        Args:
            cik: The company's SEC CIK number

        Returns:
            Official company name (e.g. "CHURCH & DWIGHT CO INC"), or None on error.
        """
        try:
            url      = EDGAR_SUBMISSIONS.format(cik=int(cik))
            response = self._rate_limited_get(url)
            data     = response.json()
            return data.get("name")
        except Exception as e:
            logger.error(f"Failed to get company name for CIK {cik}: {e}")
            return None

    def close(self):
        """Closes the underlying HTTP connection pool. Call when done."""
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
