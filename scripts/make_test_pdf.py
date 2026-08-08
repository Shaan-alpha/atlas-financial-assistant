"""Generate a sample quarterly report for testing document intelligence.

The company is fictional so nothing here can be mistaken for a real filing.
The numbers are internally consistent and carry a deliberate tension — segment
growth alongside margin compression — because a summary-only reader will miss
it while a model that actually reads the tables will not.

    python scripts/make_test_pdf.py
"""

from fpdf import FPDF

COMPANY = "NovaChip Semiconductor Inc. (FICTIONAL SAMPLE)"


def _header(pdf, text, size=13):
    pdf.set_font("Helvetica", "B", size)
    pdf.set_text_color(20, 20, 60)
    pdf.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)


def _table(pdf, headers, rows, widths):
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(232, 236, 245)
    for head, width in zip(headers, widths, strict=True):
        pdf.cell(width, 7, head, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for row in rows:
        for value, width in zip(row, widths, strict=True):
            align = "L" if value is row[0] else "R"
            pdf.cell(width, 6.5, str(value), border=1, align=align)
        pdf.ln()
    pdf.ln(3)


def build(path: str = "NovaChip_Q3_FY2026_Results.pdf") -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "NovaChip Semiconductor Inc.", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(
        0, 7, "Third Quarter Fiscal 2026 Results - Unaudited",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(
        0, 6, "FICTIONAL SAMPLE DOCUMENT - created for software testing only.",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    _header(pdf, "Condensed Statement of Operations (USD millions)")
    _table(
        pdf,
        ["", "Q3 FY26", "Q2 FY26", "Q3 FY25", "YoY %"],
        [
            ["Revenue", "4,812", "4,190", "3,105", "+55.0%"],
            ["Cost of revenue", "2,743", "2,180", "1,512", "+81.4%"],
            ["Gross profit", "2,069", "2,010", "1,593", "+29.9%"],
            ["Research and development", "742", "690", "561", "+32.3%"],
            ["Sales, general and admin", "318", "301", "268", "+18.7%"],
            ["Operating income", "1,009", "1,019", "764", "+32.1%"],
            ["Net income", "846", "858", "631", "+34.1%"],
            ["Diluted EPS (USD)", "1.72", "1.75", "1.29", "+33.3%"],
        ],
        [58, 30, 30, 30, 32],
    )

    _header(pdf, "Revenue by Segment (USD millions)")
    _table(
        pdf,
        ["Segment", "Q3 FY26", "Q3 FY25", "YoY %", "Gross margin"],
        [
            ["Data Center", "3,104", "1,588", "+95.5%", "38.2%"],
            ["Client Computing", "902", "941", "-4.1%", "51.7%"],
            ["Embedded and IoT", "486", "402", "+20.9%", "58.4%"],
            ["Automotive", "320", "174", "+83.9%", "44.1%"],
            ["Total", "4,812", "3,105", "+55.0%", "43.0%"],
        ],
        [50, 30, 30, 30, 40],
    )

    _header(pdf, "Selected Balance Sheet Data (USD millions)")
    _table(
        pdf,
        ["", "Q3 FY26", "FY25 year end"],
        [
            ["Cash and equivalents", "6,240", "5,118"],
            ["Inventory", "2,905", "1,442"],
            ["Total debt", "3,100", "3,100"],
            ["Total shareholders' equity", "14,760", "12,890"],
        ],
        [80, 40, 45],
    )

    pdf.add_page()
    _header(pdf, "Management Discussion")
    pdf.multi_cell(
        0,
        5.5,
        "Revenue grew 55.0% year over year, led by Data Center, where accelerator "
        "shipments nearly doubled. Consolidated gross margin declined to 43.0% from "
        "51.3% in the prior-year quarter. The decline is attributable to segment mix: "
        "Data Center now represents 64.5% of revenue at a 38.2% gross margin, "
        "materially below the corporate average, and to elevated foundry wafer pricing "
        "under contracts renegotiated in the second quarter.\n\n"
        "Inventory rose 101.5% versus fiscal year end, ahead of revenue growth. "
        "Management attributes the build to advance procurement of substrate and "
        "high-bandwidth memory ahead of anticipated fourth quarter demand.",
    )
    pdf.ln(3)

    _header(pdf, "Guidance")
    pdf.multi_cell(
        0,
        5.5,
        "Fourth quarter revenue is expected to be 5,150 million USD, plus or minus "
        "150 million. Gross margin is expected to be approximately 41.5%, reflecting "
        "continued Data Center mix shift. Operating expenses are expected to be "
        "1,110 million USD.",
    )
    pdf.ln(3)

    _header(pdf, "Risk Factors")
    pdf.multi_cell(
        0,
        5.5,
        "1. Customer concentration. Three hyperscale customers accounted for 61% of "
        "Data Center revenue in the quarter. Loss of any one would materially affect "
        "results.\n"
        "2. Supply. The company relies on a single foundry partner for its 3nm and "
        "2nm nodes and has no qualified second source.\n"
        "3. Inventory risk. Should fourth quarter demand fall short of the advance "
        "procurement described above, the company may record write-downs.\n"
        "4. Export controls. Approximately 18% of Data Center revenue derives from "
        "customers in jurisdictions subject to evolving export licensing.",
    )

    pdf.output(path)
    return path


if __name__ == "__main__":
    print("wrote", build())
