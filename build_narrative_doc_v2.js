/**
 * build_narrative_doc.js  (v2)
 * ----------------------------
 * Narrative-first analyst Word doc.
 *
 * Sections:
 *   1. Cover
 *   2. YoY Financial Narrative  (Revenue / COGS / Gross Margin tables + driver bullets,
 *                                 then EBITDA / Cash Flow / Debt bullets)
 *   3. Segment Analysis          (per-segment Revenue / COGS / Gross Margin table + drivers)
 *   4. Revolver & Liquidity
 *   5. Underwriting Memo
 *
 * input.json additions vs v1:
 *   "segment_data": {            <- prepared by Python before calling this script
 *     "consolidated_drivers": {
 *       "revenue":      { "key_drivers": [...], "offsetting_factors": [...] },
 *       "cogs":         { "cost_increasing": [...], "cost_reducing": [...] },
 *       "gross_margin": { "commentary": [...] }
 *     },
 *     "segments": [
 *       {
 *         "name": "Phosphates",
 *         "revenue": {
 *           "rows": [{ "label": "...", "curr": 3772.9, "prior": 3749.8 }, ...],
 *           "negative_drivers": [...],
 *           "positive_drivers": [...]
 *         },
 *         "cogs": {
 *           "rows": [{ "label": "Total COGS", "curr": 3924.8, "prior": 4022.2 }],
 *           "cost_increasing": [...],
 *           "cost_reducing": [...]
 *         }
 *       }, ...
 *     ]
 *   }
 */

"use strict";

const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, WidthType, ShadingType, BorderStyle,
  PageBreak, convertMillimetersToTwip,
} = require("docx");

// ── CLI ───────────────────────────────────────────────────────────────────────
const [,, inputPath, outputPath] = process.argv;
if (!inputPath || !outputPath) {
  console.error("Usage: node build_narrative_doc.js input.json output.docx");
  process.exit(1);
}

const payload = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const {
  borrower_name   = "Company",
  statement_basis = "actual",
  periods         = [],
  memo_markdown   = null,
  segment_data    = null,
  covenants       = {},
} = payload;

const unit = statement_basis === "millions" ? "M" : statement_basis === "thousands" ? "K" : "";
const today = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
const p0 = periods[0] || {};
const p1 = periods[1] || {};
const currName = p0.period_name || "Current";
const priorName = p1.period_name || "Prior";

// ── Colour constants ──────────────────────────────────────────────────────────
const NAVY    = "001F5B";
const BLUE_H  = "1F497D";   // heading blue
const AMBER   = "B8860B";
const LGRAY   = "F2F2F2";
const WHITE   = "FFFFFF";
const BLACK   = "000000";
const GREEN   = "1A6B1A";
const RED_C   = "CC0000";
const DGRAY   = "555555";
const GOLD    = "C09000";   // segment sub-section label

// ── Numbering (bullets) ───────────────────────────────────────────────────────
const BULLET_REF = "doc-bullets";
const numbering = {
  config: [{
    reference: BULLET_REF,
    levels: [{
      level: 0,
      format: "bullet",
      text: "\u2022",
      alignment: AlignmentType.LEFT,
      style: {
        paragraph: {
          indent: { left: convertMillimetersToTwip(7), hanging: convertMillimetersToTwip(7) },
          spacing: { after: 60 },
        },
        run: { font: "Arial", size: 20 },
      },
    }],
  }],
};

// ── Number formatting ─────────────────────────────────────────────────────────
function fmtM(x, allowNeg = true) {
  if (x == null) return "—";
  const n = parseFloat(x);
  if (isNaN(n)) return "—";
  const abs = Math.abs(n);
  const s = `$${abs.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}${unit}`;
  if (!allowNeg || n >= 0) return s;
  return `(${s})`;
}

function fmtChange(curr, prior) {
  if (curr == null || prior == null) return "—";
  const c = parseFloat(curr), p = parseFloat(prior);
  const delta = c - p;
  const abs = Math.abs(delta);
  const s = `$${abs.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}${unit}`;
  return delta >= 0 ? `+${s}` : `(${s})`;
}

function fmtPctChange(curr, prior) {
  if (curr == null || prior == null || parseFloat(prior) === 0) return "—";
  const pct = ((parseFloat(curr) - parseFloat(prior)) / Math.abs(parseFloat(prior))) * 100;
  const s = `${Math.abs(pct).toFixed(0)}%`;
  return pct >= 0 ? `+${s}` : `(${s})`;
}

function fmtPct(x) {
  if (x == null) return "—";
  const n = parseFloat(x);
  return `${(Math.abs(n) <= 1 ? n * 100 : n).toFixed(1)}%`;
}

function fmtX(x) {
  if (x == null) return "—";
  return `${parseFloat(x).toFixed(2)}x`;
}

function direction(curr, prior) {
  if (curr == null || prior == null) return "changed";
  return parseFloat(curr) >= parseFloat(prior) ? "increased" : "decreased";
}

function is_(p)  { return (p && p.income_statement)  || {}; }
function bs_(p)  { return (p && p.balance_sheet)     || {}; }
function cf_(p)  { return (p && p.cash_flow)         || {}; }
function dm_(p)  { return (p && p.derived_metrics)   || {}; }

// ── Paragraph helpers ─────────────────────────────────────────────────────────

function h1(text) {
  return new Paragraph({
    spacing: { before: 300, after: 80 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: NAVY } },
    children: [new TextRun({ text, font: "Arial", size: 28, bold: true, color: NAVY })],
  });
}

function h2(text) {
  return new Paragraph({
    spacing: { before: 200, after: 60 },
    children: [new TextRun({ text, font: "Arial", size: 24, bold: true, color: BLUE_H })],
  });
}

// Uppercase bold section label (REVENUE, COST OF GOODS SOLD, etc.)
function sectionLabel(text, color = NAVY) {
  return new Paragraph({
    spacing: { before: 160, after: 80 },
    children: [new TextRun({ text, font: "Arial", size: 22, bold: true, color, allCaps: true })],
  });
}

// "Key drivers:" / "Offsetting factors:" group header
function driverGroupLabel(text) {
  return new Paragraph({
    spacing: { before: 100, after: 40 },
    children: [new TextRun({ text, font: "Arial", size: 20, bold: true, color: BLACK })],
  });
}

function bullet(runs) {
  const children = (Array.isArray(runs) ? runs : [{ text: runs }]).map(r =>
    typeof r === "string"
      ? new TextRun({ text: r, font: "Arial", size: 20 })
      : new TextRun({ text: r.text, font: "Arial", size: 20, bold: !!r.bold, color: r.color || undefined })
  );
  return new Paragraph({
    numbering: { reference: BULLET_REF, level: 0 },
    spacing: { after: 60 },
    children,
  });
}

function prose(text, { italic = false, color = null, size = 20 } = {}) {
  return new Paragraph({
    spacing: { after: 80 },
    children: [new TextRun({ text, font: "Arial", size, italic, color: color || undefined })],
  });
}

function spacer(pts = 100) {
  return new Paragraph({ spacing: { after: pts } });
}

function divider() {
  return new Paragraph({
    spacing: { before: 40, after: 40 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" } },
  });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

// ── Mini data table (FY2024 / FY2023 / Change / % Change) ────────────────────
// Total width: 9360 DXA (full text area on Letter with 1.1" margins)
const TABLE_W   = 9000;
const COL_LABEL = 3400;
const COL_NUM   = 1400;  // × 4 = 5600   total = 9000

function cellShade(color) {
  return { type: ShadingType.CLEAR, color: "auto", fill: color };
}

function makeCell(text, { bold = false, color = BLACK, bg = WHITE, align = AlignmentType.RIGHT, size = 20 } = {}) {
  return new TableCell({
    width: { size: COL_NUM, type: WidthType.DXA },
    shading: cellShade(bg),
    margins: { top: 60, bottom: 60, left: 80, right: 80 },
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text: String(text), font: "Arial", size, bold, color })],
    })],
  });
}

function makeLabelCell(text, { bold = false, bg = WHITE, color = BLACK, size = 20 } = {}) {
  return new TableCell({
    width: { size: COL_LABEL, type: WidthType.DXA },
    shading: cellShade(bg),
    margins: { top: 60, bottom: 60, left: 100, right: 80 },
    children: [new Paragraph({
      alignment: AlignmentType.LEFT,
      children: [new TextRun({ text, font: "Arial", size, bold, color })],
    })],
  });
}

function buildMetricTable(rows) {
  /**
   * rows: [{ label, curr, prior, isTotalRow }]
   * Renders a 5-column table: Label | currName | priorName | Change | % Change
   */
  const headerRow = new TableRow({
    tableHeader: true,
    children: [
      makeLabelCell("", { bg: NAVY }),
      makeCell(currName,  { bold: true, color: WHITE, bg: NAVY, align: AlignmentType.CENTER }),
      makeCell(priorName, { bold: true, color: WHITE, bg: NAVY, align: AlignmentType.CENTER }),
      makeCell("Change",  { bold: true, color: WHITE, bg: NAVY, align: AlignmentType.CENTER }),
      makeCell("% Change",{ bold: true, color: WHITE, bg: NAVY, align: AlignmentType.CENTER }),
    ],
  });

  const dataRows = rows.map((r, i) => {
    const isTotal = !!r.isTotalRow;
    const bg      = isTotal ? LGRAY : WHITE;
    const chg     = fmtChange(r.curr, r.prior);
    const pct     = fmtPctChange(r.curr, r.prior);
    return new TableRow({
      children: [
        makeLabelCell(r.label, { bold: isTotal, bg }),
        makeCell(fmtM(r.curr),  { bold: isTotal, bg }),
        makeCell(fmtM(r.prior), { bold: isTotal, bg }),
        makeCell(chg,           { bold: isTotal, bg }),
        makeCell(pct,           { bold: isTotal, bg }),
      ],
    });
  });

  return new Table({
    width: { size: TABLE_W, type: WidthType.DXA },
    columnWidths: [COL_LABEL, COL_NUM, COL_NUM, COL_NUM, COL_NUM],
    rows: [headerRow, ...dataRows],
  });
}

function renderDrivers(keyDrivers = [], offsettingFactors = [], driverLabel = "Key drivers:", offsetLabel = "Offsetting factors:") {
  const out = [];
  if (keyDrivers.length) {
    out.push(driverGroupLabel(driverLabel));
    for (const d of keyDrivers) out.push(bullet(d));
  }
  if (offsettingFactors.length) {
    out.push(driverGroupLabel(offsetLabel));
    for (const d of offsettingFactors) out.push(bullet(d));
  }
  return out;
}

// ── Section builders ──────────────────────────────────────────────────────────

function buildCover() {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 80 },
      children: [new TextRun({ text: "CREDIT ANALYSIS — ANALYST WORKING DOCUMENT", font: "Arial", size: 32, bold: true, color: NAVY, allCaps: true })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 60 },
      children: [new TextRun({ text: borrower_name, font: "Arial", size: 28, bold: true, color: BLUE_H })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 40 },
      children: [new TextRun({
        text: `Periods: ${currName}, ${priorName}  |  Basis: ${statement_basis.charAt(0).toUpperCase() + statement_basis.slice(1)}  |  Generated: ${today}`,
        font: "Arial", size: 18, color: DGRAY,
      })],
    }),
    new Paragraph({
      border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: NAVY } },
      spacing: { after: 300 },
    }),
  ];
}

function buildYoYSection() {
  const is0 = is_(p0), is1 = is_(p1);
  const cf0 = cf_(p0), cf1 = cf_(p1);
  const bs0 = bs_(p0), bs1 = bs_(p1);
  const dm0 = dm_(p0), dm1 = dm_(p1);

  const cd = segment_data && segment_data.consolidated_drivers ? segment_data.consolidated_drivers : {};

  const rev0 = is0.revenue, rev1 = is1.revenue;
  const cos0 = is0.cost_of_sales, cos1 = is1.cost_of_sales;

  // Gross profit
  const gp0 = (rev0 != null && cos0 != null) ? rev0 - cos0 : null;
  const gp1 = (rev1 != null && cos1 != null) ? rev1 - cos1 : null;
  const gm0 = (gp0 != null && rev0) ? gp0 / rev0 : null;
  const gm1 = (gp1 != null && rev1) ? gp1 / rev1 : null;

  const ebi0 = is0.ebitda, ebi1 = is1.ebitda;
  const mar0 = dm0.ebitda_margin, mar1 = dm1.ebitda_margin;
  const oi0  = is0.operating_income, oi1 = is1.operating_income;
  const cfo0 = cf0.cfo,  cfo1 = cf1.cfo;
  const cpx0 = cf0.capex, cpx1 = cf1.capex;
  const fcf0 = dm0.free_cash_flow, fcf1 = dm1.free_cash_flow;
  const dbt0 = bs0.total_debt, dbt1 = bs1.total_debt;
  const csh0 = bs0.cash, csh1 = bs1.cash;
  const lev0 = dm0.leverage_total_debt_to_ebitda, lev1 = dm1.leverage_total_debt_to_ebitda;
  const fcc0 = dm0.fcc, fcc1 = dm1.fcc;

  const out = [
    h1("YoY Financial Narrative"),
    prose(`Comparison of ${currName} vs. ${priorName}. All figures in ${statement_basis === "actual" ? "USD" : statement_basis}.`, { italic: true, color: DGRAY }),
    spacer(),
  ];

  // ── REVENUE ────────────────────────────────────────────────────────────────
  out.push(sectionLabel("Revenue"));
  out.push(buildMetricTable([
    { label: "Total Net Sales", curr: rev0, prior: rev1, isTotalRow: true },
  ]));
  out.push(spacer(80));
  out.push(...renderDrivers(
    cd.revenue && cd.revenue.key_drivers        || [],
    cd.revenue && cd.revenue.offsetting_factors || [],
  ));
  out.push(divider());

  // ── COST OF GOODS SOLD ─────────────────────────────────────────────────────
  out.push(sectionLabel("Cost of Goods Sold"));
  out.push(buildMetricTable([
    { label: "Total Cost of Goods Sold", curr: cos0, prior: cos1, isTotalRow: true },
  ]));
  out.push(spacer(80));
  out.push(...renderDrivers(
    cd.cogs && cd.cogs.cost_increasing || [],
    cd.cogs && cd.cogs.cost_reducing   || [],
    "Cost-increasing factors:",
    "Cost-reducing factors:"
  ));
  out.push(divider());

  // ── GROSS MARGIN ───────────────────────────────────────────────────────────
  out.push(sectionLabel("Gross Margin"));
  out.push(buildMetricTable([
    { label: "Gross Profit",   curr: gp0, prior: gp1, isTotalRow: false },
    { label: "Gross Margin %",
      curr:  gm0 != null ? `${(gm0*100).toFixed(1)}%` : null,
      prior: gm1 != null ? `${(gm1*100).toFixed(1)}%` : null,
      isTotalRow: true },
  ]));
  out.push(spacer(80));
  if (cd.gross_margin && cd.gross_margin.commentary && cd.gross_margin.commentary.length) {
    out.push(driverGroupLabel("Commentary:"));
    for (const c of cd.gross_margin.commentary) out.push(bullet(c));
  }
  out.push(divider());

  // ── EBITDA ─────────────────────────────────────────────────────────────────
  out.push(sectionLabel("EBITDA & Profitability"));
  if (ebi0 != null && ebi1 != null) {
    out.push(bullet([
      { text: `EBITDA ${direction(ebi0, ebi1)} ${fmtPctChange(ebi0, ebi1)} YoY`, bold: true },
      { text: ` from ${fmtM(ebi1)} to ${fmtM(ebi0)}.` },
    ]));
  }
  if (mar0 != null && mar1 != null) {
    const ppChange = ((mar0 - mar1) * 100).toFixed(1);
    const dir = parseFloat(ppChange) >= 0 ? "expanded" : "contracted";
    out.push(bullet(`EBITDA margin ${dir} ${Math.abs(parseFloat(ppChange)).toFixed(1)}pp to ${fmtPct(mar0)} (${priorName}: ${fmtPct(mar1)}).`));
  }
  if (oi0 != null && oi1 != null) {
    out.push(bullet(`Operating income ${direction(oi0, oi1)} ${fmtPctChange(oi0, oi1)} to ${fmtM(oi0)}.`));
  }
  out.push(divider());

  // ── CASH FLOW ──────────────────────────────────────────────────────────────
  out.push(sectionLabel("Cash Flow"));
  if (cfo0 != null && cfo1 != null) {
    out.push(bullet([
      { text: `Operating cash flow (CFO) ${direction(cfo0, cfo1)} ${fmtPctChange(cfo0, cfo1)}`, bold: true },
      { text: ` from ${fmtM(cfo1)} to ${fmtM(cfo0)}.` },
    ]));
  }
  if (cpx0 != null) {
    const cpxAbs0 = Math.abs(parseFloat(cpx0)), cpxAbs1 = cpx1 != null ? Math.abs(parseFloat(cpx1)) : null;
    out.push(bullet(`CapEx ${cpxAbs1 != null ? direction(cpxAbs0, cpxAbs1) + " " + fmtPctChange(cpxAbs0, cpxAbs1) : "was"} ${fmtM(cpxAbs0)}${rev0 ? ` (${((cpxAbs0/rev0)*100).toFixed(1)}% of revenue)` : ""}.`));
  }
  if (fcf0 != null && fcf1 != null) {
    out.push(bullet([
      { text: `Free cash flow ${direction(fcf0, fcf1)} ${fmtPctChange(fcf0, fcf1)}`, bold: true },
      { text: ` to ${fmtM(fcf0)} (${priorName}: ${fmtM(fcf1)}).` },
    ]));
  }
  out.push(divider());

  // ── DEBT & CAPITAL STRUCTURE ───────────────────────────────────────────────
  out.push(sectionLabel("Debt & Capital Structure"));
  if (dbt0 != null && dbt1 != null) {
    out.push(bullet([
      { text: `Total debt ${direction(dbt0, dbt1)} ${fmtPctChange(dbt0, dbt1)}`, bold: true },
      { text: ` from ${fmtM(dbt1)} to ${fmtM(dbt0)}.` },
    ]));
  }
  if (lev0 != null && lev1 != null) {
    const levDir = parseFloat(lev0) <= parseFloat(lev1) ? "improved" : "deteriorated";
    out.push(bullet([
      { text: `Leverage (Debt/EBITDA) ${levDir} from ${fmtX(lev1)} to ${fmtX(lev0)}`, bold: true },
      { text: "." },
    ]));
  }
  if (fcc0 != null && fcc1 != null) {
    out.push(bullet(`FCC ${direction(fcc0, fcc1)} from ${fmtX(fcc1)} to ${fmtX(fcc0)}.`));
  }
  if (csh0 != null && csh1 != null) {
    out.push(bullet(`Cash ${direction(csh0, csh1)} ${fmtPctChange(csh0, csh1)} to ${fmtM(csh0)} (${priorName}: ${fmtM(csh1)}).`));
  }

  // Max leverage covenant check
  if (covenants.max_total_leverage != null && lev0 != null) {
    const ok = parseFloat(lev0) <= parseFloat(covenants.max_total_leverage);
    const cushion = ((covenants.max_total_leverage - parseFloat(lev0)) / covenants.max_total_leverage * 100).toFixed(1);
    out.push(bullet([
      { text: `Leverage covenant (max ${fmtX(covenants.max_total_leverage)}): ` },
      { text: ok ? `✓ In compliance — ${cushion}% headroom` : `✗ BREACH — leverage of ${fmtX(lev0)} exceeds maximum`, bold: true, color: ok ? GREEN : RED_C },
    ]));
  }
  if (covenants.min_fcc != null && fcc0 != null) {
    const ok = parseFloat(fcc0) >= parseFloat(covenants.min_fcc);
    out.push(bullet([
      { text: `FCC covenant (min ${fmtX(covenants.min_fcc)}): ` },
      { text: ok ? `✓ In compliance` : `✗ BREACH — FCC of ${fmtX(fcc0)} below minimum`, bold: true, color: ok ? GREEN : RED_C },
    ]));
  }

  return out;
}

function buildSegmentSection() {
  if (!segment_data || !segment_data.segments || !segment_data.segments.length) return [];

  const out = [
    pageBreak(),
    h1("Segment Analysis"),
    prose("Per-segment breakdown extracted from Management's Discussion & Analysis. Verify all figures against the source 10-K.", { italic: true, color: DGRAY }),
    spacer(),
  ];

  for (const seg of segment_data.segments) {
    // Segment header
    out.push(new Paragraph({
      spacing: { before: 200, after: 80 },
      shading: cellShade(NAVY),
      children: [new TextRun({ text: `SEGMENT: ${seg.name.toUpperCase()}`, font: "Arial", size: 22, bold: true, color: WHITE, allCaps: true })],
    }));

    // ── Revenue ──────────────────────────────────────────────────────────────
    out.push(sectionLabel("Revenue", GOLD));

    const revRows = (seg.revenue && seg.revenue.rows) || [];
    if (revRows.length) {
      out.push(buildMetricTable(
        revRows.map((r, i) => ({
          label: r.label,
          curr: r.curr,
          prior: r.prior,
          isTotalRow: i === revRows.length - 1 || r.label.toLowerCase().includes("total"),
        }))
      ));
      out.push(spacer(80));
    }

    out.push(...renderDrivers(
      seg.revenue && seg.revenue.negative_drivers || [],
      seg.revenue && seg.revenue.positive_drivers || [],
      "Drivers (negative):",
      "Offsetting factors (positive):"
    ));

    // ── COGS ─────────────────────────────────────────────────────────────────
    out.push(spacer(80));
    out.push(sectionLabel("Cost of Goods Sold", GOLD));

    const cogsRows = (seg.cogs && seg.cogs.rows) || [];
    if (cogsRows.length) {
      out.push(buildMetricTable(
        cogsRows.map((r, i) => ({
          label: r.label,
          curr: r.curr,
          prior: r.prior,
          isTotalRow: i === cogsRows.length - 1 || r.label.toLowerCase().includes("total"),
        }))
      ));
      out.push(spacer(80));
    }

    out.push(...renderDrivers(
      seg.cogs && seg.cogs.cost_increasing || [],
      seg.cogs && seg.cogs.cost_reducing   || [],
      "Drivers (cost-increasing):",
      "Offsetting factors (cost-reducing):"
    ));

    // ── Gross Margin (computed if we have the numbers) ────────────────────────
    const revTotal = revRows.find(r => r.label.toLowerCase().includes("total"));
    const cogsTotal = cogsRows.find(r => r.label.toLowerCase().includes("total"));
    if (revTotal && cogsTotal) {
      const gp0s  = (revTotal.curr  != null && cogsTotal.curr  != null) ? revTotal.curr  - cogsTotal.curr  : null;
      const gp1s  = (revTotal.prior != null && cogsTotal.prior != null) ? revTotal.prior - cogsTotal.prior : null;
      const gm0s  = (gp0s != null && revTotal.curr)  ? gp0s / revTotal.curr  : null;
      const gm1s  = (gp1s != null && revTotal.prior) ? gp1s / revTotal.prior : null;

      out.push(spacer(80));
      out.push(sectionLabel("Gross Margin", GOLD));
      out.push(buildMetricTable([
        { label: "Gross Profit",   curr: gp0s, prior: gp1s, isTotalRow: false },
        {
          label: "Gross Margin %",
          curr:  gm0s != null ? `${(gm0s*100).toFixed(1)}%` : null,
          prior: gm1s != null ? `${(gm1s*100).toFixed(1)}%` : null,
          isTotalRow: true,
        },
      ]));
    }

    out.push(divider(), spacer(120));
  }

  return out;
}

function buildRevolverSection() {
  const hasSomething = periods.some(p => {
    const b = bs_(p);
    return b.revolver_facility_size != null || b.cash != null;
  });
  if (!hasSomething) return [];

  const out = [
    pageBreak(),
    h1("Revolver & Liquidity"),
    spacer(),
  ];

  for (const p of periods) {
    const b = bs_(p);
    const size  = b.revolver_facility_size;
    const borr  = b.revolver_borrowings;
    const avail = b.revolver_availability;
    const cash  = b.cash;
    if (size == null && cash == null) continue;

    out.push(h2(p.period_name));
    if (cash  != null) out.push(bullet([{ text: "Cash on hand: ", bold: true }, { text: fmtM(cash) }]));
    if (size  != null) out.push(bullet([{ text: "Revolving credit facility size: ", bold: true }, { text: fmtM(size) }]));
    if (borr  != null) out.push(bullet([{ text: "Revolver drawn: ", bold: true }, { text: fmtM(borr) }]));
    if (avail != null) out.push(bullet([{ text: "Revolver availability: ", bold: true }, { text: fmtM(avail) }]));
    if (cash  != null && avail != null) {
      out.push(bullet([{ text: "Total liquidity (cash + availability): ", bold: true }, { text: fmtM(cash + avail) }]));
    }
    out.push(spacer(80));
  }

  return out;
}

function buildMemoSection() {
  if (!memo_markdown) return [];

  const out = [
    pageBreak(),
    h1("Underwriting Memo"),
    prose("AI-generated — analyst review and sign-off required before use in credit committee materials.", { italic: true, color: RED_C }),
    spacer(),
  ];

  for (const line of memo_markdown.split("\n")) {
    const s = line.trim();
    if (!s) { out.push(spacer(80)); continue; }
    if (s.startsWith("## "))  { out.push(h2(s.slice(3)));  continue; }
    if (s.startsWith("### ")) { out.push(h2(s.slice(4)));  continue; }
    if (s.startsWith("- ") || s.startsWith("* ")) {
      out.push(bullet(s.slice(2)));
      continue;
    }
    const parts = s.split(/\*\*(.+?)\*\*/);
    out.push(new Paragraph({
      spacing: { after: 80 },
      children: parts.map((part, i) => new TextRun({ text: part, font: "Arial", size: 20, bold: i % 2 === 1 })),
    }));
  }

  return out;
}

// ── Assemble & write ──────────────────────────────────────────────────────────
const children = [
  ...buildCover(),
  ...buildYoYSection(),
  ...buildSegmentSection(),
  ...buildRevolverSection(),
  ...buildMemoSection(),
];

const doc = new Document({
  numbering,
  styles: { default: { document: { run: { font: "Arial", size: 20 } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, bottom: 1080, left: 1260, right: 1260 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(outputPath, buf);
  console.log(`Written: ${outputPath}`);
}).catch(err => {
  console.error("Error:", err.message);
  process.exit(1);
});
