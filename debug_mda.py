"""
debug_mda.py
============
Drop this file next to your extraction_agent.py and run:

    python debug_mda.py path/to/your/filing.txt

It will print exactly what each stage of the MD&A pipeline sees,
so you can pinpoint where the content is being lost.
"""

from __future__ import annotations

import sys
import os
import re

# ── Load the raw document ────────────────────────────────────────────────────

def load_doc(path: str) -> str:
    """Load a file as plain text, trying UTF-8 then latin-1 as fallback."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


# ── Stage 1: confirm the raw document loaded ─────────────────────────────────

def stage1_raw(raw_text: str) -> None:
    print("\n" + "=" * 70)
    print("STAGE 1 — RAW DOCUMENT")
    print("=" * 70)
    print(f"Total characters : {len(raw_text):,}")
    print(f"Total lines      : {raw_text.count(chr(10)):,}")

    # Show the first 500 chars so we can confirm it's the right file
    print("\nFirst 500 characters:")
    print("-" * 40)
    print(raw_text[:500])
    print("-" * 40)


# ── Stage 2: search for every MD&A header variant ───────────────────────────

MDA_MARKERS = [
    # Primary SEC headers
    "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
    "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS",
    "ITEM 7 - MANAGEMENT'S DISCUSSION AND ANALYSIS",
    "ITEM 7— MANAGEMENT'S DISCUSSION AND ANALYSIS",
    "ITEM 7.MANAGEMENT'S DISCUSSION",
    # Without apostrophe
    "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
    "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS",
    # Standalone titles
    "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
    "MANAGEMENT'S DISCUSSION AND ANALYSIS",
    "MANAGEMENT DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
    "MANAGEMENT DISCUSSION AND ANALYSIS",
    "MD&A",
    # Results of Operations variants
    "RESULTS OF OPERATIONS",
    "CONSOLIDATED RESULTS OF OPERATIONS",
    "RESULTS OF OPERATIONS AND FINANCIAL CONDITION",
    "OPERATING RESULTS",
    "DISCUSSION OF OPERATIONS",
    "DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
    # Financial Review
    "FINANCIAL REVIEW",
    "FINANCIAL AND OPERATING REVIEW",
    "OPERATING AND FINANCIAL REVIEW",
    "OPERATING AND FINANCIAL REVIEW AND PROSPECTS",
    # Performance Review
    "BUSINESS PERFORMANCE REVIEW",
    "PERFORMANCE REVIEW",
    "BUSINESS REVIEW",
    "ANNUAL BUSINESS REVIEW",
    "STRATEGIC AND FINANCIAL REVIEW",
    # Other
    "FINANCIAL DISCUSSION",
    "FINANCIAL OVERVIEW",
    "EXECUTIVE OVERVIEW",
    "REVIEW OF FINANCIAL RESULTS",
    "ANALYSIS OF FINANCIAL RESULTS",
    "REVIEW OF OPERATIONS",
    "OPERATIONAL REVIEW",
    "FINANCIAL AND BUSINESS REVIEW",
    "CEO/CFO REVIEW",
]


def stage2_header_scan(raw_text: str) -> str | None:
    """
    Scan for every known MD&A header.
    Returns the text starting at the first valid hit, or None.
    """
    print("\n" + "=" * 70)
    print("STAGE 2 — MD&A HEADER SCAN")
    print("=" * 70)

    upper = raw_text.upper()
    hits = []

    for marker in MDA_MARKERS:
        idx = upper.find(marker)
        if idx != -1:
            hits.append((idx, marker))

    if not hits:
        print("❌  NO MD&A HEADERS FOUND IN DOCUMENT.")
        print()
        print("    Possible causes:")
        print("    1. The file is a PDF that was not properly converted to text.")
        print("    2. The MD&A section uses an unusual header not in the search list.")
        print("    3. The text is encoded in a way that breaks string matching.")
        print()
        print("    Showing first 3,000 characters to help identify the format:")
        print("-" * 40)
        print(raw_text[:3000])
        print("-" * 40)
        return None

    # Sort by position
    hits.sort(key=lambda x: x[0])

    print(f"Found {len(hits)} header hit(s):\n")
    for idx, marker in hits:
        text_after = raw_text[idx: idx + 300].replace("\n", " ")
        is_toc = len(raw_text[idx: idx + 500].strip()) < 100
        toc_flag = "  ← likely TABLE OF CONTENTS hit" if is_toc else ""
        print(f"  [{idx:>8,}]  {marker}{toc_flag}")
        print(f"             Preview: {text_after[:120]}")
        print()

    # Pick first non-TOC hit
    chosen_idx = None
    chosen_marker = None
    for idx, marker in hits:
        text_after = raw_text[idx: idx + 500]
        if len(text_after.strip()) > 100:
            chosen_idx = idx
            chosen_marker = marker
            break

    if chosen_idx is None:
        print("❌  All hits look like table-of-contents entries (too little text after header).")
        return None

    print(f"✅  Using: '{chosen_marker}' at index {chosen_idx:,}")
    return chosen_idx, chosen_marker, raw_text


# ── Stage 3: extract the MD&A section ───────────────────────────────────────

MDA_END_MARKERS = [
    "ITEM 7A.",
    "ITEM 7A ",
    "ITEM 7A—",
    "ITEM 7A-",
    "ITEM 8.",
    "ITEM 8 ",
    "ITEM 8—",
    "ITEM 8-",
]


def stage3_extract(raw_text: str, start_idx: int, start_marker: str) -> str:
    print("\n" + "=" * 70)
    print("STAGE 3 — MD&A SECTION EXTRACTION")
    print("=" * 70)

    upper = raw_text.upper()
    end_idx = len(raw_text)
    end_marker_used = "end of document"

    for marker in MDA_END_MARKERS:
        idx = upper.find(marker, start_idx + 5000)
        if idx != -1 and idx < end_idx:
            end_idx = idx
            end_marker_used = marker
            break

    mda_text = raw_text[start_idx:end_idx]

    print(f"Start marker : '{start_marker}' at {start_idx:,}")
    print(f"End marker   : '{end_marker_used}' at {end_idx:,}")
    print(f"MD&A length  : {len(mda_text):,} characters")
    print()

    if len(mda_text) < 500:
        print("❌  MD&A section is suspiciously short (< 500 chars).")
        print("    Full extracted text:")
        print(mda_text)
        return mda_text

    # Segment keyword check
    seg_kws = ["SEGMENT", "BUSINESS UNIT", "DIVISION", "OPERATING SEGMENT",
               "RESULTS OF OPERATIONS", "PRODUCT LINE"]
    found_kws = [kw for kw in seg_kws if kw in mda_text.upper()]
    if found_kws:
        print(f"✅  Segment-related keywords found: {', '.join(found_kws)}")
    else:
        print("⚠️   No segment keywords found in MD&A.")
        print("    Company may not report segments, or wrong section captured.")

    # Show first and last 1,000 chars of the extracted section
    print("\nFirst 1,000 characters of extracted MD&A:")
    print("-" * 40)
    print(mda_text[:1000])
    print("-" * 40)

    print("\nLast 1,000 characters of extracted MD&A:")
    print("-" * 40)
    print(mda_text[-1000:])
    print("-" * 40)

    return mda_text


# ── Stage 4: call the LLM summarizer and show raw output ────────────────────

def stage4_llm(mda_text: str) -> None:
    print("\n" + "=" * 70)
    print("STAGE 4 — LLM SUMMARIZER TEST (minimal prompt)")
    print("=" * 70)

    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        client = OpenAI()
    except ImportError:
        print("⚠️   openai or dotenv not installed — skipping LLM stage.")
        print("    Install with:  pip install openai python-dotenv")
        return

    # Use a very short slice to minimise cost — just enough to test the pipeline
    test_slice = mda_text[:40000]

    print(f"Sending first {len(test_slice):,} chars of MD&A to gpt-4o-mini (test only)...\n")

    system = (
        "You are a financial analyst. Read the MD&A text and answer these three questions:\n"
        "1. What business segments are discussed?\n"
        "2. What is the first revenue driver mentioned for any segment?\n"
        "3. Is there any mention of 'partially offset by'? If yes, quote the exact sentence.\n\n"
        "Use only information present in the text. "
        "If you cannot find the answer, say 'Not found in the provided text.'"
    )

    user = f"MD&A TEXT:\n{test_slice}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    result = resp.choices[0].message.content or ""
    print("LLM RESPONSE:")
    print("-" * 40)
    print(result)
    print("-" * 40)

    if "not found" in result.lower() or "not provided" in result.lower() or "not present" in result.lower():
        print()
        print("⚠️   LLM says content is not present in the text slice.")
        print("    This means the MD&A text is either:")
        print("    a) Correctly extracted but segments are described later in the section.")
        print("       → Solution: the 40k char test slice may be too small; full run should work.")
        print("    b) The extracted text does not actually contain MD&A narrative content.")
        print("       → Solution: check Stage 3 output above — is it real prose or garbled text?")
    else:
        print()
        print("✅  LLM successfully found content in the MD&A slice.")
        print("    If the full pipeline still returns 'Not disclosed', the issue is likely")
        print("    token truncation in the summarizer — increase max_tokens or reduce mda_text slice.")


# ── Stage 5: check the summarizer output for placeholders ───────────────────

def stage5_placeholder_check(mda_summary: str) -> None:
    print("\n" + "=" * 70)
    print("STAGE 5 — PLACEHOLDER CHECK ON MD&A SUMMARY")
    print("=" * 70)

    if not mda_summary:
        print("❌  mda_summary is empty.")
        return

    placeholder_signals = [
        "$X.X", "$Y.Y", "[Direction]", "[ALL drivers",
        "[actual figure]", "[driver here]", "[favorable/unfavorable]",
        "Not disclosed in MD&A",
    ]

    found = [s for s in placeholder_signals if s in mda_summary]

    if found:
        print(f"⚠️   Placeholder signals found in summary: {found}")
    else:
        print("✅  No placeholder signals detected.")

    print(f"\nSummary length: {len(mda_summary):,} characters")
    print("\nFirst 2,000 characters of mda_summary:")
    print("-" * 40)
    print(mda_summary[:2000])
    print("-" * 40)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_mda.py path/to/filing.txt")
        print()
        print("If your filing is a PDF, first convert it to text:")
        print("  pdftotext -layout filing.pdf filing.txt")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"\nLoading: {path}")
    raw_text = load_doc(path)

    # Stage 1
    stage1_raw(raw_text)

    # Stage 2
    result = stage2_header_scan(raw_text)
    if result is None:
        print("\n⛔  Pipeline stopped at Stage 2 — no MD&A header found.")
        sys.exit(1)

    start_idx, start_marker, raw_text = result

    # Stage 3
    mda_text = stage3_extract(raw_text, start_idx, start_marker)

    # Stage 4 — only runs if OPENAI_API_KEY is set
    if os.environ.get("OPENAI_API_KEY"):
        stage4_llm(mda_text)
    else:
        print("\n" + "=" * 70)
        print("STAGE 4 — SKIPPED (OPENAI_API_KEY not set)")
        print("=" * 70)
        print("Set OPENAI_API_KEY in your environment or .env file to run the LLM test.")

    # Stage 5 — only if you already have a cached summary to check
    cached_summary_path = path.replace(".txt", "_mda_summary.txt")
    if os.path.exists(cached_summary_path):
        with open(cached_summary_path, "r", encoding="utf-8") as f:
            cached = f.read()
        stage5_placeholder_check(cached)
    else:
        print("\n" + "=" * 70)
        print("STAGE 5 — SKIPPED (no cached summary file found)")
        print("=" * 70)
        print(f"To test Stage 5, save your mda_summary string to: {cached_summary_path}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    print()
    print("What to look for:")
    print("  Stage 2 ❌ → Header not found. Check file encoding or add missing header variant.")
    print("  Stage 3 short → Wrong section captured or PDF-to-text conversion is garbled.")
    print("  Stage 3 no segments → Segments are present but under unexpected keyword names.")
    print("  Stage 4 'Not found' → MD&A content exists but segment data is beyond the 40k slice.")
    print("  Stage 4 ✅ but pipeline fails → Token truncation in _summarize_mda_for_drivers.")
    print("  Stage 5 placeholders → Summarizer prompt not constraining the model tightly enough.")
    print()


if __name__ == "__main__":
    main()
