"""
doc_builder.py
==============
Produces an Excel workbook and a Word document from the credit AI pipeline output.

Usage (standalone):
    from doc_builder import build_documents
    paths = build_documents(
        run_record=run,           # RunRecord from RunStore
        borrower_name="CHD",
        output_dir="./outputs",
    )
    # paths["excel"] -> "outputs/CHD_credit_summary.xlsx"
    # paths["word"]  -> "outputs/CHD_credit_memo.docx"

Or call directly from main.py after a completed run.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from openpyxl import Workbook
from openpyxl.styles import (
    Alignment, Border, Font, PatternFill, Side
)
from openpyxl.utils import get_column_letter

# ── Docx imports ─────────────────────────────────────────────────────────────
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


# =============================================================================
# Shared helpers
# =============================================================================

def _v(x: Any, fmt: str = "money", basis: str = "actual") -> str:
    """Format a raw numeric value for display."""
    if x is None:
        return "—"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)

    unit = "MM" if basis == "millions" else ("K" if basis == "thousands" else "")

    if fmt == "money":
        return f"${f:,.1f}{unit}"
    if fmt == "pct":
        return f"{f * 100:.1f}%"
    if fmt == "pct_already":         # value already in percent (e.g. ebitda_margin stored as 0.xx)
        return f"{f:.1f}%"
    if fmt == "x":
        return f"{f:.2f}x"
    if fmt == "raw":
        return f"{f:,.1f}"
    return str(x)


def _periods_from_run(run_record) -> List[Dict]:
    """Pull the periods list from a RunRecord's extracted_json."""
    if not run_record or not run_record.extracted_json:
        return []
    return run_record.extracted_json.get("periods") or []


def _basis_from_run(run_record) -> str:
    if not run_record or not run_record.extracted_json:
        return "actual"
    return run_record.extracted_json.get("statement_basis") or "actual"


def _period_sections(p: Dict):
    """Unpack a period dict into its sub-dicts."""
    return (
        p.get("income_statement") or {},
        p.get("balance_sheet") or {},
        p.get("cash_flow") or {},
        p.get("derived_metrics") or {},
    )


# =============================================================================
# Excel builder
# =============================================================================

# Colour palette (openpyxl uses ARGB strings, no '#')
_NAVY   = "001F5B"
_BLUE   = "0000FF"   # hardcoded input convention
_BLACK  = "000000"
_WHITE  = "FFFFFF"
_LGRAY  = "F2F2F2"
_MGRAY  = "D9D9D9"
_GREEN  = "00B050"
_AMBER  = "FFC000"
_RED    = "FF0000"


def _header_font(bold=True, color=_WHITE, size=10):
    return Font(name="Arial", bold=bold, color=color, size=size)


def _body_font(bold=False, color=_BLACK, size=10):
    return Font(name="Arial", bold=bold, color=color, size=size)


def _fill(hex_color: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_color)


def _thin_border() -> Border:
    s = Side(style="thin", color=_MGRAY)
    return Border(left=s, right=s, top=s, bottom=s)


def _apply_header_row(ws, row: int, values: List, col_start: int = 1):
    for i, v in enumerate(values, start=col_start):
        cell = ws.cell(row=row, column=i, value=v)
        cell.font = _header_font()
        cell.fill = _fill(_NAVY)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _thin_border()


def _apply_subheader_row(ws, row: int, values: List, col_start: int = 1):
    for i, v in enumerate(values, start=col_start):
        cell = ws.cell(row=row, column=i, value=v)
        cell.font = Font(name="Arial", bold=True, color=_WHITE, size=9)
        cell.fill = _fill("2E4057")
        cell.alignment = Alignment(horizontal="left", vertical="center")
        cell.border = _thin_border()


def _write_metric_row(ws, row: int, label: str, values: List,
                      fmt: str = "money", is_alt: bool = False,
                      bold: bool = False, indent: bool = False):
    bg = _LGRAY if is_alt else _WHITE
    label_cell = ws.cell(row=row, column=1, value=("  " if indent else "") + label)
    label_cell.font = Font(name="Arial", bold=bold, color=_BLACK, size=10)
    label_cell.fill = _fill(bg)
    label_cell.border = _thin_border()

    for i, v in enumerate(values, start=2):
        raw = None if v == "—" else v
        cell = ws.cell(row=row, column=i)
        cell.fill = _fill(bg)
        cell.border = _thin_border()
        cell.alignment = Alignment(horizontal="right")
        cell.font = Font(name="Arial", bold=bold, color=_BLUE, size=10)

        if raw is None:
            cell.value = "—"
            cell.font = Font(name="Arial", color=_MGRAY, size=10)
        elif fmt == "money":
            cell.value = raw
            cell.number_format = '$#,##0.0;($#,##0.0);"-"'
        elif fmt == "pct":
            cell.value = raw / 100 if abs(raw) > 1 else raw
            cell.number_format = "0.0%;(0.0%);-"
        elif fmt == "x":
            cell.value = raw
            cell.number_format = '0.00"x";(0.00"x");"-"'
        else:
            cell.value = raw


def _write_section_label(ws, row: int, label: str, n_cols: int):
    cell = ws.cell(row=row, column=1, value=label)
    cell.font = Font(name="Arial", bold=True, color=_WHITE, size=10)
    cell.fill = _fill("3A5683")
    cell.border = _thin_border()
    ws.merge_cells(start_row=row, start_column=1,
                   end_row=row, end_column=n_cols)
    cell.alignment = Alignment(horizontal="left", vertical="center")


def _build_excel(run_record, borrower_name: str, output_path: str) -> str:
    periods = _periods_from_run(run_record)
    basis   = _basis_from_run(run_record)
    n_periods = len(periods)

    wb = Workbook()
    wb.remove(wb.active)

    # ── Sheet 1: Financial Summary ────────────────────────────────────────────
    ws = wb.create_sheet("Financial Summary")
    ws.freeze_panes = "B4"

    period_names = [p.get("period_name", f"Period {i+1}") for i, p in enumerate(periods)]
    n_cols = 1 + n_periods

    # Title
    title = ws.cell(row=1, column=1,
                    value=f"{borrower_name} — Credit Financial Summary")
    title.font = Font(name="Arial", bold=True, size=14, color=_NAVY)
    ws.merge_cells(start_row=1, start_column=1,
                   end_row=1, end_column=max(n_cols, 4))

    sub = ws.cell(row=2, column=1,
                  value=f"Source: 10-K Filing  |  Generated: {datetime.now().strftime('%B %d, %Y')}  |  Basis: {basis.title()}")
    sub.font = Font(name="Arial", size=9, color="666666")
    ws.merge_cells(start_row=2, start_column=1,
                   end_row=2, end_column=max(n_cols, 4))

    # Column headers
    _apply_header_row(ws, 3, ["Metric"] + period_names)

    # Column widths
    ws.column_dimensions["A"].width = 34
    for i in range(n_periods):
        ws.column_dimensions[get_column_letter(i + 2)].width = 16

    row = 4

    def vals(field: str, section: str) -> List:
        out = []
        for p in periods:
            s = p.get(section) or {}
            out.append(s.get(field))
        return out

    def dm_vals(field: str) -> List:
        return vals(field, "derived_metrics")

    def cf_vals(field: str) -> List:
        return vals(field, "cash_flow")

    # ── Income Statement ──────────────────────────────────────────────────────
    _write_section_label(ws, row, "INCOME STATEMENT", n_cols); row += 1
    rows_is = [
        ("Revenue",                  vals("revenue",          "income_statement"), "money", False),
        ("Cost of Sales",            vals("cost_of_sales",    "income_statement"), "money", True),
        ("  Gross Profit",           None,                                         "money", False),   # formula
        ("SG&A Expense",             vals("sga_expense",      "income_statement"), "money", True),
        ("Operating Income (EBIT)",  vals("operating_income", "income_statement"), "money", False),
        ("D&A",                      vals("depreciation_amortization", "income_statement"), "money", True),
        ("Rent / Lease Expense",     vals("rent_expense",     "income_statement"), "money", True),
        ("EBITDA",                   vals("ebitda",           "income_statement"), "money", False),
        ("Net Income",               vals("net_income",       "income_statement"), "money", True),
    ]
    for i, (label, data_vals, fmt, alt) in enumerate(rows_is):
        if data_vals is None:
            # Gross profit: =Revenue - CoS formula
            label_cell = ws.cell(row=row, column=1, value=label)
            label_cell.font = Font(name="Arial", bold=True, color=_BLACK, size=10)
            label_cell.fill = _fill(_LGRAY if alt else _WHITE)
            label_cell.border = _thin_border()
            for col in range(2, 2 + n_periods):
                rev_row = row - 2   # revenue is 2 rows up
                cos_row = row - 1
                cell = ws.cell(row=row, column=col)
                cell.value = f"={get_column_letter(col)}{rev_row}-{get_column_letter(col)}{cos_row}"
                cell.number_format = '$#,##0.0;($#,##0.0);"-"'
                cell.font = Font(name="Arial", bold=True, color=_BLACK, size=10)
                cell.fill = _fill(_WHITE)
                cell.border = _thin_border()
                cell.alignment = Alignment(horizontal="right")
        else:
            _write_metric_row(ws, row, label, data_vals, fmt=fmt, is_alt=alt,
                              bold=label in ("EBITDA", "Operating Income (EBIT)"))
        row += 1

    row += 1  # spacer

    # ── Balance Sheet ─────────────────────────────────────────────────────────
    _write_section_label(ws, row, "BALANCE SHEET", n_cols); row += 1
    rows_bs = [
        ("Cash & Equivalents",           vals("cash",                           "balance_sheet"), "money", False),
        ("Total Assets",                 vals("total_assets",                   "balance_sheet"), "money", True),
        ("Total Liabilities",            vals("total_liabilities",              "balance_sheet"), "money", False),
        ("Total Equity",                 vals("total_equity",                   "balance_sheet"), "money", True),
        ("Total Debt",                   vals("total_debt",                     "balance_sheet"), "money", False),
        ("  Long-Term Debt",             vals("long_term_debt",                 "balance_sheet"), "money", True),
        ("  Current Portion LTD (CPLTD)",vals("current_portion_long_term_debt", "balance_sheet"), "money", True),
        ("Revolver — Facility Size",     vals("revolver_facility_size",         "balance_sheet"), "money", False),
        ("Revolver — Borrowings",        vals("revolver_borrowings",            "balance_sheet"), "money", True),
        ("Revolver — Availability",      vals("revolver_availability",          "balance_sheet"), "money", False),
    ]
    for label, data_vals, fmt, alt in rows_bs:
        _write_metric_row(ws, row, label, data_vals, fmt=fmt, is_alt=alt,
                         bold=label in ("Total Debt",))
        row += 1

    row += 1

    # ── Cash Flow ─────────────────────────────────────────────────────────────
    _write_section_label(ws, row, "CASH FLOW", n_cols); row += 1
    rows_cf = [
        ("Cash from Operations (CFO)",   cf_vals("cfo"),                    "money", False),
        ("Capital Expenditures (CapEx)", cf_vals("capex"),                  "money", True),
        ("Free Cash Flow",               dm_vals("free_cash_flow"),         "money", False),
        ("Cash Paid for Interest",       cf_vals("cash_paid_for_interest"), "money", True),
        ("Cash Paid for Taxes",          cf_vals("cash_paid_for_income_taxes"), "money", True),
    ]
    for label, data_vals, fmt, alt in rows_cf:
        _write_metric_row(ws, row, label, data_vals, fmt=fmt, is_alt=alt,
                         bold=label in ("Free Cash Flow",))
        row += 1

    row += 1

    # ── Credit Metrics ────────────────────────────────────────────────────────
    _write_section_label(ws, row, "CREDIT METRICS", n_cols); row += 1
    rows_cr = [
        ("EBITDA Margin",               dm_vals("ebitda_margin"),                    "pct",   False),
        ("Leverage (Total Debt / EBITDA)", dm_vals("leverage_total_debt_to_ebitda"), "x",     True),
        ("FCC",                          dm_vals("fcc"),                             "x",     False),
        ("Altman Z-Score",               dm_vals("altman_z_score"),                  "raw",   True),
        ("PD Score (1–12)",              dm_vals("pd_score"),                        "raw",   False),
        ("Credit Rating",                dm_vals("credit_rating"),                   "label", True),
    ]
    for label, data_vals, fmt, alt in rows_cr:
        if fmt == "label":
            label_cell = ws.cell(row=row, column=1, value=label)
            label_cell.font = _body_font()
            label_cell.fill = _fill(_LGRAY if alt else _WHITE)
            label_cell.border = _thin_border()
            for i, v in enumerate(data_vals, start=2):
                cell = ws.cell(row=row, column=i, value=v or "—")
                cell.font = Font(name="Arial", color=_NAVY if v else _MGRAY, size=10)
                cell.fill = _fill(_LGRAY if alt else _WHITE)
                cell.border = _thin_border()
                cell.alignment = Alignment(horizontal="center")
        else:
            _write_metric_row(ws, row, label, data_vals, fmt=fmt, is_alt=alt,
                             bold=label in ("Leverage (Total Debt / EBITDA)", "FCC"))
        row += 1

    # ── Sheet 2: YoY Bridge ───────────────────────────────────────────────────
    ws2 = wb.create_sheet("YoY Comparisons")
    ws2.column_dimensions["A"].width = 36
    ws2.column_dimensions["B"].width = 18
    ws2.column_dimensions["C"].width = 18
    ws2.column_dimensions["D"].width = 16
    ws2.column_dimensions["E"].width = 16

    t2 = ws2.cell(row=1, column=1,
                  value=f"{borrower_name} — Year-over-Year Comparison")
    t2.font = Font(name="Arial", bold=True, size=14, color=_NAVY)
    ws2.merge_cells("A1:E1")

    if n_periods >= 2:
        curr_name = period_names[0]
        prior_name = period_names[1]
        _apply_header_row(ws2, 3,
                         ["Metric", curr_name, prior_name, "Change ($)", "Change (%)"])

        bridge_rows = []
        for p_curr, p_prior in zip([periods[0]], [periods[1]]):
            is0, bs0, cf0, dm0 = _period_sections(p_curr)
            is1, bs1, cf1, dm1 = _period_sections(p_prior)

            def bridge(label, curr, prior, fmt="money"):
                bridge_rows.append((label, curr, prior, fmt))

            bridge("Revenue",         is0.get("revenue"),         is1.get("revenue"))
            bridge("Cost of Sales",   is0.get("cost_of_sales"),   is1.get("cost_of_sales"))
            bridge("SG&A Expense",    is0.get("sga_expense"),     is1.get("sga_expense"))
            bridge("Operating Income",is0.get("operating_income"),is1.get("operating_income"))
            bridge("EBITDA",          is0.get("ebitda"),          is1.get("ebitda"))
            bridge("EBITDA Margin",   dm0.get("ebitda_margin"),   dm1.get("ebitda_margin"), "pct")
            bridge("Net Income",      is0.get("net_income"),      is1.get("net_income"))
            bridge("CFO",             cf0.get("cfo"),             cf1.get("cfo"))
            bridge("CapEx",           cf0.get("capex"),           cf1.get("capex"))
            bridge("Free Cash Flow",  dm0.get("free_cash_flow"),  dm1.get("free_cash_flow"))
            bridge("Total Debt",      bs0.get("total_debt"),      bs1.get("total_debt"))
            bridge("Cash",            bs0.get("cash"),            bs1.get("cash"))
            bridge("Leverage",        dm0.get("leverage_total_debt_to_ebitda"),
                                      dm1.get("leverage_total_debt_to_ebitda"), "x")
            bridge("FCC",             dm0.get("fcc"),             dm1.get("fcc"), "x")

        for i, (label, curr, prior, fmt) in enumerate(bridge_rows, start=4):
            alt = i % 2 == 0
            bg = _LGRAY if alt else _WHITE

            lc = ws2.cell(row=i, column=1, value=label)
            lc.font = _body_font()
            lc.fill = _fill(bg)
            lc.border = _thin_border()

            for col, val in [(2, curr), (3, prior)]:
                cell = ws2.cell(row=i, column=col, value=val)
                cell.fill = _fill(bg)
                cell.border = _thin_border()
                cell.alignment = Alignment(horizontal="right")
                cell.font = Font(name="Arial", color=_BLUE, size=10)
                if val is None:
                    cell.value = "—"
                    cell.font = Font(name="Arial", color=_MGRAY, size=10)
                elif fmt == "money":
                    cell.number_format = '$#,##0.0;($#,##0.0);"-"'
                elif fmt == "pct":
                    cell.value = val if abs(val) <= 1 else val / 100
                    cell.number_format = "0.0%;(0.0%);-"
                elif fmt == "x":
                    cell.number_format = '0.00"x";(0.00"x");"-"'

            # Change ($) — formula: =B{row}-C{row}
            chg = ws2.cell(row=i, column=4)
            chg.fill = _fill(bg)
            chg.border = _thin_border()
            chg.alignment = Alignment(horizontal="right")
            if curr is not None and prior is not None:
                chg.value = f"=B{i}-C{i}"
                if fmt == "money":
                    chg.number_format = '$#,##0.0;($#,##0.0);"-"'
                elif fmt == "pct":
                    chg.number_format = "0.0%;(0.0%);-"
                elif fmt == "x":
                    chg.number_format = '0.00"x";(0.00"x");"-"'
                chg.font = Font(name="Arial", color=_BLACK, size=10)
            else:
                chg.value = "—"
                chg.font = Font(name="Arial", color=_MGRAY, size=10)

            # Change (%) — formula: =IF(C{row}=0,"—",D{row}/C{row})
            pct_cell = ws2.cell(row=i, column=5)
            pct_cell.fill = _fill(bg)
            pct_cell.border = _thin_border()
            pct_cell.alignment = Alignment(horizontal="right")
            if curr is not None and prior is not None and fmt != "pct":
                pct_cell.value = f'=IF(C{i}=0,"—",D{i}/C{i})'
                pct_cell.number_format = "0.0%;(0.0%);-"
                pct_cell.font = Font(name="Arial", color=_BLACK, size=10)
            else:
                pct_cell.value = "N/A"
                pct_cell.font = Font(name="Arial", color=_MGRAY, size=10)

    else:
        note = ws2.cell(row=3, column=1,
                        value="Only one period extracted — YoY comparison not available.")
        note.font = Font(name="Arial", color="888888", italic=True, size=10)

    # ── Sheet 3: Validation Flags ─────────────────────────────────────────────
    ws3 = wb.create_sheet("Extraction Notes")
    ws3.column_dimensions["A"].width = 14
    ws3.column_dimensions["B"].width = 90
    ws3.row_dimensions[1].height = 22

    t3 = ws3.cell(row=1, column=1, value="Extraction & Validation Notes")
    t3.font = Font(name="Arial", bold=True, size=13, color=_NAVY)
    ws3.merge_cells("A1:B1")

    _apply_header_row(ws3, 3, ["Type", "Note"])
    ws3.column_dimensions["A"].width = 18

    flags = []
    if run_record and run_record.validation_flags:
        flags = run_record.validation_flags or []

    profile_flags = [f for f in flags if f.startswith("PROFILE")]
    other_flags   = [f for f in flags if not f.startswith("PROFILE")]

    row3 = 4
    for flag in other_flags:
        tag = ws3.cell(row=row3, column=1,
                       value="⚠ Extraction" if "missing" in flag.lower() else "ℹ Info")
        tag.font = Font(name="Arial", bold=True, color=_AMBER, size=9)
        tag.fill = _fill("FFFDE7")
        tag.border = _thin_border()
        tag.alignment = Alignment(horizontal="center")
        body = ws3.cell(row=row3, column=2, value=flag)
        body.font = _body_font(size=9)
        body.fill = _fill("FFFDE7")
        body.border = _thin_border()
        body.alignment = Alignment(wrap_text=True, vertical="top")
        ws3.row_dimensions[row3].height = 30
        row3 += 1

    for flag in profile_flags:
        tag = ws3.cell(row=row3, column=1, value="⚡ Profile")
        tag.font = Font(name="Arial", bold=True, color="0050A0", size=9)
        tag.fill = _fill("E8F0FE")
        tag.border = _thin_border()
        tag.alignment = Alignment(horizontal="center")
        body = ws3.cell(row=row3, column=2, value=flag)
        body.font = _body_font(size=9)
        body.fill = _fill("E8F0FE")
        body.border = _thin_border()
        body.alignment = Alignment(wrap_text=True, vertical="top")
        ws3.row_dimensions[row3].height = 30
        row3 += 1

    if not flags:
        note = ws3.cell(row=4, column=2,
                        value="No extraction flags raised — all key metrics extracted cleanly.")
        note.font = Font(name="Arial", color=_GREEN, size=10)
        note.fill = _fill("E8F5E9")
        ws3.cell(row=4, column=1).fill = _fill("E8F5E9")
        ws3.cell(row=4, column=1).border = _thin_border()
        note.border = _thin_border()

    completeness = getattr(run_record, "completeness", None)
    if completeness is not None:
        row3 += 2
        cl = ws3.cell(row=row3, column=1, value="Completeness Score")
        cl.font = Font(name="Arial", bold=True, size=10, color=_NAVY)
        cv = ws3.cell(row=row3, column=2,
                      value=f"{float(completeness):.0%} — "
                            + ("Sufficient for leverage/FCC analysis."
                               if completeness >= 0.7
                               else "Below threshold — review missing fields above."))
        cv.font = Font(name="Arial", size=10,
                       color=_GREEN if completeness >= 0.7 else _AMBER)

    wb.save(output_path)
    return output_path


# =============================================================================
# Word builder
# =============================================================================

def _set_cell_bg(cell, hex_color: str):
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(shading)


def _bold_run(para, text: str, size_pt: int = 10, color: str = None):
    run = para.add_run(text)
    run.bold = True
    run.font.size = Pt(size_pt)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    return run


def _add_heading(doc: Document, text: str, level: int):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0, 31, 91)  # navy
    return h


def _add_metric_table(doc: Document, rows: List[tuple], period_names: List[str]):
    """
    rows: [(label, [val_period0, val_period1, ...]), ...]
    """
    n_cols = 1 + len(period_names)
    table = doc.add_table(rows=1, cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT

    # Header row
    hdr = table.rows[0].cells
    hdr[0].text = "Metric"
    hdr[0].paragraphs[0].runs[0].bold = True
    _set_cell_bg(hdr[0], "001F5B")
    hdr[0].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    for i, name in enumerate(period_names, start=1):
        hdr[i].text = name
        if hdr[i].paragraphs[0].runs:
            hdr[i].paragraphs[0].runs[0].bold = True
            hdr[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        _set_cell_bg(hdr[i], "001F5B")
        hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for r_idx, (label, vals) in enumerate(rows):
        row = table.add_row().cells
        row[0].text = label
        if row[0].paragraphs[0].runs:
            row[0].paragraphs[0].runs[0].font.size = Pt(9)
        bg = "F2F2F2" if r_idx % 2 == 0 else "FFFFFF"
        _set_cell_bg(row[0], bg)
        for i, v in enumerate(vals, start=1):
            row[i].text = str(v) if v is not None else "—"
            row[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
            if row[i].paragraphs[0].runs:
                row[i].paragraphs[0].runs[0].font.size = Pt(9)
            _set_cell_bg(row[i], bg)

    return table


def _build_word(run_record, borrower_name: str, output_path: str,
                memo_markdown: Optional[str] = None,
                mda_summary: Optional[str] = None) -> str:

    periods  = _periods_from_run(run_record)
    basis    = _basis_from_run(run_record)
    n_periods = len(periods)
    period_names = [p.get("period_name", f"Period {i+1}") for i, p in enumerate(periods)]

    doc = Document()

    # Page margins (1 inch all around)
    for section in doc.sections:
        section.top_margin    = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin   = Inches(1.1)
        section.right_margin  = Inches(1.1)

    # ── Cover / Title ─────────────────────────────────────────────────────────
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_para.add_run(f"CREDIT UNDERWRITING SUMMARY")
    run.bold = True
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor(0, 31, 91)

    sub_para = doc.add_paragraph()
    sub_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub_para.add_run(borrower_name)
    sub_run.bold = True
    sub_run.font.size = Pt(14)
    sub_run.font.color.rgb = RGBColor(58, 86, 131)

    meta_para = doc.add_paragraph()
    meta_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_para.add_run(
        f"Source: Annual 10-K Filing  |  Periods: {', '.join(period_names)}  |  "
        f"Generated: {datetime.now().strftime('%B %d, %Y')}"
    ).font.size = Pt(9)

    doc.add_paragraph()  # spacer

    # ── Section 1: Executive Summary ─────────────────────────────────────────
    _add_heading(doc, "1. Executive Summary", 1)

    completeness = getattr(run_record, "completeness", None)
    if completeness is not None:
        ep = doc.add_paragraph()
        ep.add_run("Extraction Completeness: ").bold = True
        color = "00B050" if completeness >= 0.7 else "FFC000"
        cr = ep.add_run(f"{float(completeness):.0%}")
        cr.font.color.rgb = RGBColor.from_string(color)
        cr.bold = True

    if periods:
        p = periods[0]
        is_, bs, cf, dm = _period_sections(p)
        period_name = p.get("period_name", "Most Recent Period")

        summary_para = doc.add_paragraph()
        summary_para.add_run(
            f"For {period_name}, {borrower_name} reported revenue of "
            f"{_v(is_.get('revenue'), 'money', basis)}, EBITDA of "
            f"{_v(is_.get('ebitda'), 'money', basis)} "
            f"({_v(dm.get('ebitda_margin'), 'pct')} margin). "
            f"Total debt stood at {_v(bs.get('total_debt'), 'money', basis)}, "
            f"implying leverage of {_v(dm.get('leverage_total_debt_to_ebitda'), 'x')} "
            f"and FCC of {_v(dm.get('fcc'), 'x')}. "
            f"Free cash flow was {_v(dm.get('free_cash_flow'), 'money', basis)} "
            f"with liquidity (cash) of {_v(bs.get('cash'), 'money', basis)}."
        ).font.size = Pt(10)

        # Credit risk box
        z = dm.get("altman_z_score")
        pd_s = dm.get("pd_score")
        rating = dm.get("credit_rating")
        if any(x is not None for x in [z, pd_s, rating]):
            doc.add_paragraph()
            _add_heading(doc, "Credit Risk Indicators", 2)
            risk_table = _add_metric_table(
                doc,
                [
                    ("Altman Z-Score", [_v(z, "raw")]),
                    ("PD Score (1–12)", [str(pd_s) if pd_s is not None else "—"]),
                    ("Credit Rating Category", [rating or "—"]),
                ],
                [period_name]
            )

    # ── Section 2: Income Statement ───────────────────────────────────────────
    doc.add_paragraph()
    _add_heading(doc, "2. Income Statement", 1)

    def _is_rows():
        return [
            ("Revenue",              [_v(p.get("income_statement", {}).get("revenue"), "money", basis) for p in periods]),
            ("Cost of Sales",        [_v(p.get("income_statement", {}).get("cost_of_sales"), "money", basis) for p in periods]),
            ("SG&A Expense",         [_v(p.get("income_statement", {}).get("sga_expense"), "money", basis) for p in periods]),
            ("Operating Income",     [_v(p.get("income_statement", {}).get("operating_income"), "money", basis) for p in periods]),
            ("D&A",                  [_v(p.get("income_statement", {}).get("depreciation_amortization"), "money", basis) for p in periods]),
            ("Rent/Lease Expense",   [_v(p.get("income_statement", {}).get("rent_expense"), "money", basis) for p in periods]),
            ("EBITDA",               [_v(p.get("income_statement", {}).get("ebitda"), "money", basis) for p in periods]),
            ("EBITDA Margin",        [_v(p.get("derived_metrics", {}).get("ebitda_margin"), "pct") for p in periods]),
            ("Net Income",           [_v(p.get("income_statement", {}).get("net_income"), "money", basis) for p in periods]),
        ]

    _add_metric_table(doc, _is_rows(), period_names)

    # ── Section 3: Balance Sheet ──────────────────────────────────────────────
    doc.add_paragraph()
    _add_heading(doc, "3. Balance Sheet", 1)

    def _bs_rows():
        return [
            ("Cash & Equivalents",      [_v(p.get("balance_sheet", {}).get("cash"), "money", basis) for p in periods]),
            ("Total Assets",            [_v(p.get("balance_sheet", {}).get("total_assets"), "money", basis) for p in periods]),
            ("Total Liabilities",       [_v(p.get("balance_sheet", {}).get("total_liabilities"), "money", basis) for p in periods]),
            ("Total Equity",            [_v(p.get("balance_sheet", {}).get("total_equity"), "money", basis) for p in periods]),
            ("Total Debt",              [_v(p.get("balance_sheet", {}).get("total_debt"), "money", basis) for p in periods]),
            ("Long-Term Debt",          [_v(p.get("balance_sheet", {}).get("long_term_debt"), "money", basis) for p in periods]),
            ("CPLTD",                   [_v(p.get("balance_sheet", {}).get("current_portion_long_term_debt"), "money", basis) for p in periods]),
            ("Revolver Facility Size",  [_v(p.get("balance_sheet", {}).get("revolver_facility_size"), "money", basis) for p in periods]),
            ("Revolver Borrowings",     [_v(p.get("balance_sheet", {}).get("revolver_borrowings"), "money", basis) for p in periods]),
            ("Revolver Availability",   [_v(p.get("balance_sheet", {}).get("revolver_availability"), "money", basis) for p in periods]),
        ]

    _add_metric_table(doc, _bs_rows(), period_names)

    # ── Section 4: Cash Flow ──────────────────────────────────────────────────
    doc.add_paragraph()
    _add_heading(doc, "4. Cash Flow", 1)

    def _cf_rows():
        return [
            ("Cash from Operations (CFO)", [_v(p.get("cash_flow", {}).get("cfo"), "money", basis) for p in periods]),
            ("Capital Expenditures",       [_v(p.get("cash_flow", {}).get("capex"), "money", basis) for p in periods]),
            ("Free Cash Flow",             [_v(p.get("derived_metrics", {}).get("free_cash_flow"), "money", basis) for p in periods]),
            ("Cash Interest Paid",         [_v(p.get("cash_flow", {}).get("cash_paid_for_interest"), "money", basis) for p in periods]),
            ("Cash Taxes Paid",            [_v(p.get("cash_flow", {}).get("cash_paid_for_income_taxes"), "money", basis) for p in periods]),
        ]

    _add_metric_table(doc, _cf_rows(), period_names)

    # ── Section 5: Credit Metrics ─────────────────────────────────────────────
    doc.add_paragraph()
    _add_heading(doc, "5. Credit Metrics", 1)

    def _cr_rows():
        return [
            ("EBITDA Margin",             [_v(p.get("derived_metrics", {}).get("ebitda_margin"), "pct") for p in periods]),
            ("Leverage (Debt / EBITDA)",  [_v(p.get("derived_metrics", {}).get("leverage_total_debt_to_ebitda"), "x") for p in periods]),
            ("FCC",                       [_v(p.get("derived_metrics", {}).get("fcc"), "x") for p in periods]),
            ("Altman Z-Score",            [_v(p.get("derived_metrics", {}).get("altman_z_score"), "raw") for p in periods]),
            ("PD Score (1–12)",           [str(p.get("derived_metrics", {}).get("pd_score") or "—") for p in periods]),
            ("Credit Rating Category",    [p.get("derived_metrics", {}).get("credit_rating") or "—" for p in periods]),
        ]

    _add_metric_table(doc, _cr_rows(), period_names)

    # ── Section 6: MD&A Summary ───────────────────────────────────────────────
    if mda_summary:
        doc.add_page_break()
        _add_heading(doc, "6. MD&A Segment & Driver Summary", 1)
        note = doc.add_paragraph()
        note.add_run("Source: Extracted from Management's Discussion & Analysis section of the 10-K. "
                     "AI-generated summary — verify against source document.").italic = True
        note.runs[0].font.size = Pt(8)
        note.runs[0].font.color.rgb = RGBColor(100, 100, 100)
        doc.add_paragraph()

        # Strip markdown headers for Word (convert ## → heading, ** → bold)
        import re
        for line in mda_summary.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                doc.add_paragraph()
                continue
            if line_stripped.startswith("## "):
                _add_heading(doc, line_stripped[3:], 2)
            elif line_stripped.startswith("### "):
                _add_heading(doc, line_stripped[4:], 3)
            elif line_stripped.startswith("- ") or line_stripped.startswith("* "):
                p = doc.add_paragraph(style="List Bullet")
                content = line_stripped[2:]
                # Handle inline bold (**text**)
                parts = re.split(r"\*\*(.+?)\*\*", content)
                for i, part in enumerate(parts):
                    r = p.add_run(part)
                    r.bold = (i % 2 == 1)
                    r.font.size = Pt(10)
            else:
                p = doc.add_paragraph()
                parts = re.split(r"\*\*(.+?)\*\*", line_stripped)
                for i, part in enumerate(parts):
                    r = p.add_run(part)
                    r.bold = (i % 2 == 1)
                    r.font.size = Pt(10)

    # ── Section 7: Underwriting Memo ─────────────────────────────────────────
    if memo_markdown:
        doc.add_page_break()
        _add_heading(doc, "7. Underwriting Memo", 1)
        note = doc.add_paragraph()
        note.add_run("AI-generated memo — analyst review and sign-off required before use.").italic = True
        note.runs[0].font.size = Pt(8)
        note.runs[0].font.color.rgb = RGBColor(180, 0, 0)
        doc.add_paragraph()

        import re
        for line in memo_markdown.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                doc.add_paragraph()
                continue
            if line_stripped.startswith("## "):
                _add_heading(doc, line_stripped[3:], 2)
            elif line_stripped.startswith("### "):
                _add_heading(doc, line_stripped[4:], 3)
            elif line_stripped.startswith("- ") or line_stripped.startswith("* "):
                p = doc.add_paragraph(style="List Bullet")
                content = line_stripped[2:]
                parts = re.split(r"\*\*(.+?)\*\*", content)
                for i, part in enumerate(parts):
                    r = p.add_run(part)
                    r.bold = (i % 2 == 1)
                    r.font.size = Pt(10)
            else:
                p = doc.add_paragraph()
                parts = re.split(r"\*\*(.+?)\*\*", line_stripped)
                for i, part in enumerate(parts):
                    r = p.add_run(part)
                    r.bold = (i % 2 == 1)
                    r.font.size = Pt(10)

    # ── Section 8: Validation Flags ───────────────────────────────────────────
    flags = getattr(run_record, "validation_flags", None) or []
    if flags:
        doc.add_paragraph()
        _add_heading(doc, "8. Extraction Notes & Flags", 1)
        for flag in flags:
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(flag).font.size = Pt(9)

    doc.save(output_path)
    return output_path


# =============================================================================
# Public entrypoint
# =============================================================================

def build_documents(
    run_record,
    borrower_name: str,
    output_dir: str = ".",
    memo_markdown: Optional[str] = None,
    mda_summary: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build both an Excel workbook and a Word document from a completed RunRecord.

    Args:
        run_record:    RunRecord from RunStore.get_run() — must have extracted_json
        borrower_name: Display name for the borrower (e.g. "Church & Dwight")
        output_dir:    Directory to write files into (created if absent)
        memo_markdown: Full underwriting memo markdown (optional)
        mda_summary:   MD&A segment driver summary (optional)

    Returns:
        {"excel": "<path>", "word": "<path>"}
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = borrower_name.replace(" ", "_").replace("/", "-")

    excel_path = os.path.join(output_dir, f"{safe_name}_credit_summary.xlsx")
    word_path  = os.path.join(output_dir, f"{safe_name}_credit_memo.docx")

    _build_excel(run_record, borrower_name, excel_path)
    _build_word(run_record, borrower_name, word_path,
                memo_markdown=memo_markdown,
                mda_summary=mda_summary)

    return {"excel": excel_path, "word": word_path}


# =============================================================================
# Narrative Word doc — delegates to build_narrative_doc.js
# =============================================================================

def build_word_narrative(
    run_record,
    borrower_name: str,
    output_path: str,
    memo_markdown: Optional[str] = None,
    mda_summary: Optional[str] = None,
    covenants: Optional[Dict] = None,
) -> str:
    """
    Produces the narrative-first analyst Word doc via the Node.js/docx-js builder.
    Requires: node + build_narrative_doc.js in the same directory as this file.
    """
    import subprocess, tempfile, json as _json

    payload = {
        "borrower_name": borrower_name,
        "statement_basis": getattr(run_record, "statement_basis", None)
            or (run_record.extracted_json or {}).get("statement_basis", "actual"),
        "periods": (run_record.extracted_json or {}).get("periods", []),
        "validation_flags": run_record.validation_flags or [],
        "completeness": run_record.completeness,
        "memo_markdown": memo_markdown,
        "mda_summary": mda_summary,
        "covenants": covenants or {},
    }

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_narrative_doc.js")
    if not os.path.exists(script):
        raise FileNotFoundError(f"build_narrative_doc.js not found at {script}")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        _json.dump(payload, tf)
        tf_path = tf.name

    try:
        result = subprocess.run(
            ["node", script, tf_path, output_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"build_narrative_doc.js failed:\n{result.stderr}")
    finally:
        os.unlink(tf_path)

    return output_path
