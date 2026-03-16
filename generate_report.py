"""
generate_report.py
Generates the PDF report for CMPSC463 Project 1 using reportlab.
Run this AFTER main.py (so the plots/ directory exists).

Usage:
    python generate_report.py
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, Preformatted
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


# ─────────────────────────────────────────────
#  Styles
# ─────────────────────────────────────────────

styles = getSampleStyleSheet()

title_style = ParagraphStyle('Title2', parent=styles['Title'],
                              fontSize=20, spaceAfter=10)
h1_style    = ParagraphStyle('H1', parent=styles['Heading1'],
                              fontSize=14, spaceAfter=6, spaceBefore=14,
                              textColor=colors.HexColor('#1a3a5c'))
h2_style    = ParagraphStyle('H2', parent=styles['Heading2'],
                              fontSize=12, spaceAfter=4, spaceBefore=10,
                              textColor=colors.HexColor('#2c5f8a'))
body_style  = ParagraphStyle('Body2', parent=styles['Normal'],
                              fontSize=10, leading=14, spaceAfter=6,
                              alignment=TA_JUSTIFY)
bullet_style = ParagraphStyle('Bullet', parent=body_style,
                               leftIndent=20, bulletIndent=10)
code_style  = ParagraphStyle('Code', parent=styles['Code'],
                              fontSize=8, leading=11, leftIndent=20,
                              backColor=colors.HexColor('#f4f4f4'))
caption_style = ParagraphStyle('Caption', parent=body_style,
                                fontSize=9, alignment=TA_CENTER,
                                textColor=colors.gray, spaceAfter=12)


def P(text, style=None):
    return Paragraph(text, style or body_style)

def H1(text):
    return Paragraph(text, h1_style)

def H2(text):
    return Paragraph(text, h2_style)

def SP(n=6):
    return Spacer(1, n)

def HR():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor('#cccccc'), spaceAfter=4)

def code_block(text):
    return Preformatted(text, code_style)

def img(path, width=6.0*inch, caption_text=None):
    """Return [Image, caption] flowables if file exists, else a note."""
    items = []
    if os.path.exists(path):
        items.append(Image(path, width=width, height=width * 0.3))
        if caption_text:
            items.append(Paragraph(caption_text, caption_style))
    else:
        items.append(P(f"[Plot not found: {path} — run main.py first]",
                       caption_style))
    return items

def bullet(items):
    """Convert list of strings to bullet paragraphs."""
    return [Paragraph(f"• {t}", bullet_style) for t in items]

def make_table(header, rows, col_widths=None):
    data = [header] + rows
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3a5c')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#eef2f7')]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cccccc')),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


# ─────────────────────────────────────────────
#  Build document
# ─────────────────────────────────────────────

def build_report(out_path='report.pdf'):
    doc = SimpleDocTemplate(
        out_path,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    story = []

    # ── Title page ────────────────────────────────────────────────
    story += [
        SP(40),
        Paragraph("CMPSC463 Project 1", title_style),
        Paragraph("Time-Series Analysis of Water Pump Sensor Data",
                  ParagraphStyle('Sub', parent=h1_style, fontSize=13,
                                 alignment=TA_CENTER)),
        SP(8),
        Paragraph("Keval Patel", ParagraphStyle('Author', parent=body_style,
                                                alignment=TA_CENTER, fontSize=11)),
        Paragraph("March 2026", ParagraphStyle('Date', parent=body_style,
                                               alignment=TA_CENTER, fontSize=11)),
        PageBreak(),
    ]

    # ── 1. Project Overview ───────────────────────────────────────
    story += [
        H1("1. Project Overview"),
        HR(),
        P("This project applies three algorithmic techniques to the Water Pump RUL "
          "(Remaining Useful Life) predictive maintenance dataset from Kaggle. The dataset "
          "records readings from sensors on a water pump over time, along with a RUL value "
          "that tells you how many hours the pump has left before it fails."),
        SP(),
        P("The goal is to use divide-and-conquer and classic algorithms to find patterns in "
          "sensor data that relate to pump health. No machine learning libraries are used — "
          "everything is implemented from scratch using only pandas, numpy, and matplotlib."),
        SP(),
        H2("RUL Categories"),
        P("RUL values are split into four categories using the 10th, 50th, and 90th "
          "percentiles of the RUL column in the first 10,000 rows:"),
        SP(4),
        make_table(
            ["Category", "Condition", "Count (10,000 rows)"],
            [
                ["Extremely Low RUL",   "rul < Q10 (135.93)",          "1,000 (10%)"],
                ["Moderately Low RUL",  "Q10 <= rul < Q50 (202.59)",   "4,000 (40%)"],
                ["Moderately High RUL", "Q50 <= rul < Q90 (269.25)",   "4,000 (40%)"],
                ["Extremely High RUL",  "rul >= Q90",                  "1,000 (10%)"],
            ],
            col_widths=[2.0*inch, 2.2*inch, 2.0*inch],
        ),
        SP(12),
    ]

    # ── 2. Installation and Usage ─────────────────────────────────
    story += [
        H1("2. Installation and Usage"),
        HR(),
        *bullet([
            "Download rul_hrs.csv from Kaggle and place it in the project folder.",
            "Install dependencies:  pip install pandas numpy matplotlib",
            "Run everything:  python main.py",
            "Generate this report:  python generate_report.py",
            "Output plots are saved to the plots/ directory automatically.",
        ]),
        SP(12),
    ]

    # ── 3. Code Structure ─────────────────────────────────────────
    story += [
        H1("3. Code Structure"),
        HR(),
        P("The project is split into five Python files. Each file handles one specific job "
          "so it's easy to find where things are and change them without breaking everything else."),
        SP(6),
        make_table(
            ["File", "What it does"],
            [
                ["data_loading.py",       "Loads the CSV, drops the unnamed index column, and adds "
                                          "the rul_category column using the Q10/Q50/Q90 thresholds."],
                ["task1_segmentation.py", "Contains the recursive segmentation function, the "
                                          "complexity score calculation, and the plot function."],
                ["task2_clustering.py",   "Contains the divisive clustering algorithm — splits "
                                          "clusters by finding the highest-variance feature."],
                ["task3_kadane.py",       "Contains Kadane's algorithm and the per-sensor analysis "
                                          "pipeline (diff -> center -> Kadane -> RUL check)."],
                ["main.py",               "Ties everything together. Imports from the other four "
                                          "files and runs all three tasks in order."],
            ],
            col_widths=[1.8*inch, 4.4*inch],
        ),
        SP(8),
        P("Here is how the data flows through the program when you run main.py:"),
        SP(4),
        code_block(
            "main.py\n"
            "  |\n"
            "  |-- data_loading.py\n"
            "  |     load_data()        -> returns DataFrame (10,000 rows)\n"
            "  |     add_rul_category() -> adds rul_category column to DataFrame\n"
            "  |     get_sensor_columns() -> returns list of 50 sensor column names\n"
            "  |\n"
            "  |-- task1_segmentation.py\n"
            "  |     run_task1(df, sensor_columns)\n"
            "  |       -> randomly picks 10 sensors\n"
            "  |       -> calls segment_sensor() on each -> list of (start,end) segments\n"
            "  |       -> calls plot_segmentation() -> saves PNG to plots/\n"
            "  |       -> returns complexity scores dict\n"
            "  |\n"
            "  |-- task2_clustering.py\n"
            "  |     run_task2(df, sensor_columns)\n"
            "  |       -> builds X matrix (10000 x 50) from sensor columns\n"
            "  |       -> calls divisive_cluster(X, k=4) -> returns labels array\n"
            "  |       -> compares labels to rul_category, prints cluster stats\n"
            "  |\n"
            "  |-- task3_kadane.py\n"
            "        run_task3(df, sensor_columns)\n"
            "          -> loops over all 50 sensors\n"
            "          -> calls analyze_sensor() on each:\n"
            "               abs diff -> center -> kadane() -> dominant RUL\n"
            "          -> saves plot of top 10 sensors to plots/"
        ),
        SP(8),
        P("The DataFrame is the main data object that gets passed around. After "
          "data_loading.py processes it, the df has the original sensor columns plus "
          "two new columns: rul_category (0-3) and rul_category_name (text label). "
          "Tasks 1, 2, and 3 all read from this same df — none of them modify it."),
        SP(12),
    ]

    # ── 4. Dataset Description ────────────────────────────────────
    story += [
        H1("4. Dataset Description"),
        HR(),
        P("The dataset is the Water Pump RUL Predictive Maintenance dataset from Kaggle "
          "(uploaded by Ansel D'Souza). It contains time-series readings from sensors "
          "on a single water pump, recorded at 1-minute intervals. Each row is one "
          "minute of operation."),
        SP(),
        H2("Variables"),
        *bullet([
            "timestamp — date and time of the reading, format: YYYY-MM-DD HH:MM:SS. "
            "Starts at 2018-04-01 00:00:00. Not used directly in the algorithms "
            "(we treat time as a simple integer index).",
            "sensor_00 through sensor_51 — numeric sensor readings. The sensors "
            "measure things like temperature, pressure, and flow rates, but the "
            "dataset does not label what each sensor physically measures. "
            "Values are continuous floats.",
            "rul — Remaining Useful Life in hours. This is how many hours the pump "
            "has left before it needs maintenance or replacement. The range in the "
            "first 10,000 rows goes from roughly 100 to 300 hours.",
        ]),
        SP(6),
        H2("Basic Statistics (first 10,000 rows)"),
        make_table(
            ["Statistic", "Value"],
            [
                ["Total rows used",         "10,000"],
                ["Timestamp range",         "2018-04-01 00:00 to roughly 2018-04-08"],
                ["RUL minimum",             "~100 hours"],
                ["RUL maximum",             "~300 hours"],
                ["Q10 (Extremely Low cutoff)", "135.93 hours"],
                ["Q50 (median)",            "202.59 hours"],
                ["Q90 (Extremely High cutoff)", "269.25 hours"],
                ["Sensor columns",          "50 (see note below)"],
            ],
            col_widths=[3.0*inch, 3.2*inch],
        ),
        SP(8),
        P("<b>Note on sensor count:</b> The assignment description says the dataset has "
          "52 sensors, but the actual CSV file only contains 50 sensor columns. "
          "Specifically, sensor_15 and sensor_50 are not present in this version of "
          "the dataset. This was discovered when loading the data — the shape came out "
          "as (10000, 52) after dropping the unnamed index column, meaning "
          "timestamp + 50 sensors + rul = 52 columns total, not 54. "
          "All three tasks were run on the 50 sensors that are actually available."),
        SP(12),
    ]

    # ── 5. Algorithm Descriptions ─────────────────────────────────
    story += [
        H1("5. Algorithm Descriptions"),
        HR(),

        H2("Task 1: Divide-and-Conquer Variance Segmentation"),
        P("The idea here is pretty simple — if a chunk of a sensor's time series has "
          "high variance, it means something interesting is happening and we should "
          "look closer. If the variance is low, the sensor is stable and we can leave "
          "that chunk alone."),
        SP(4),
        code_block(
            "def segment(values, start, end, threshold, min_size):\n"
            "    if end - start < min_size:          # too small to split\n"
            "        record [start, end] as stable\n"
            "        return\n"
            "    if variance(values[start:end]) <= threshold:\n"
            "        record [start, end] as stable   # low variance = stable\n"
            "    else:\n"
            "        mid = (start + end) // 2        # split in half\n"
            "        segment(values, start, mid, threshold, min_size)\n"
            "        segment(values, mid, end, threshold, min_size)"
        ),
        SP(4),
        P("The threshold is set to 10% of the overall variance of each sensor. "
          "Minimum segment size is 50 time steps. The Complexity Score is just the "
          "total number of segments produced — more segments means more volatile sensor."),
        SP(),
        P("<b>Toy Example</b> — 10-element array:"),
        code_block(
            "arr = [1.0, 1.1, 1.0, 1.2, 5.0, 9.0, 8.5, 9.2, 1.1, 1.0]\n"
            "threshold = 1.0,  min_size = 2\n"
            "\n"
            "Segments produced:\n"
            "  [0:2]  = [1.0, 1.1]         var=0.0025  -> stable\n"
            "  [2:3]  = [1.0]              var=0.0000  -> stable (too small)\n"
            "  [3:4]  = [1.2]              var=0.0000  -> stable (too small)\n"
            "  [4:5]  = [5.0]              var=0.0000  -> stable (too small)\n"
            "  [5:7]  = [9.0, 8.5]         var=0.0625  -> stable\n"
            "  [7:8]  = [9.2]              var=0.0000  -> stable (too small)\n"
            "  [8:10] = [1.1, 1.0]         var=0.0025  -> stable\n"
            "Complexity score = 7"
        ),
        SP(10),

        H2("Task 2: Divisive (Top-Down) Clustering"),
        P("Instead of starting with individual points and merging them (bottom-up), "
          "we start with all 10,000 rows in one big group and keep splitting the "
          "messiest group until we have 4 clusters. This is the divide-and-conquer "
          "approach to clustering."),
        SP(4),
        code_block(
            "def divisive_cluster(X, k=4):\n"
            "    clusters = [all row indices]      # start with everything in 1 group\n"
            "    while len(clusters) < k:\n"
            "        pick cluster with largest spread (sum of variances)\n"
            "        find feature with highest variance in that cluster\n"
            "        split at the median of that feature -> two new clusters\n"
            "    assign labels 0..k-1\n"
            "    return labels"
        ),
        SP(4),
        P("<b>Toy Example</b> — 8 points, 2 features, k=4:"),
        code_block(
            "X = [[1.0,2.0], [1.5,1.8], [1.2,2.2],\n"
            "     [8.0,9.0], [8.5,8.8], [9.0,9.5],\n"
            "     [1.1,8.0], [1.3,7.8]]\n"
            "\n"
            "After bisection 1 (split on x-axis at median ~4.5):\n"
            "  Group A: points with x <= 4.5 -> rows 0,1,2,6,7\n"
            "  Group B: points with x > 4.5  -> rows 3,4,5\n"
            "\n"
            "After bisection 2 (split Group A on y-axis at median ~2.2):\n"
            "  Cluster 0: rows 0,2,6,7  (low-x, mixed-y)\n"
            "  Cluster 2: row 1         (low-x, low-y)\n"
            "  ... and so on until k=4\n"
            "\n"
            "Final labels: [0, 2, 0, 1, 3, 1, 0, 0]"
        ),
        SP(10),

        H2("Task 3: Kadane's Maximum Subarray Algorithm"),
        P("Kadane's algorithm finds the contiguous subarray with the largest sum in O(n) time. "
          "We use it to find the time window where a sensor's variability is most concentrated. "
          "A sensor that spikes a lot right before failure is a good early-warning indicator."),
        SP(4),
        P("<b>Pre-processing steps:</b>"),
        *bullet([
            "Step 1 — Absolute first difference:  d[i] = |sensor[i] - sensor[i-1]|",
            "Step 2 — Center it:  x[i] = d[i] - mean(d)   "
            "(positive = more variable than average, negative = calmer than average)",
            "Step 3 — Run Kadane's on x[] to find the max-sum window",
            "Step 4 — Check which RUL category dominates the rows in that window",
        ]),
        SP(4),
        code_block(
            "def kadane(arr):\n"
            "    max_sum = arr[0];  cur = arr[0]\n"
            "    best_s = 0;  best_e = 0;  cur_s = 0\n"
            "    for i in range(1, len(arr)):\n"
            "        if cur + arr[i] < arr[i]:   # better to start fresh\n"
            "            cur = arr[i];  cur_s = i\n"
            "        else:\n"
            "            cur += arr[i]\n"
            "        if cur > max_sum:\n"
            "            max_sum = cur           # total deviation of this window\n"
            "            best_s = cur_s;  best_e = i\n"
            "    return max_sum, best_s, best_e  # O(n)"
        ),
        SP(4),
        P("<b>Toy Example — Kadane's on a simple array:</b>"),
        code_block(
            "arr = [-2.0, 1.0, -3.0, 4.0, -1.0, 2.0, 1.0, -5.0, 4.0]\n"
            "Max subarray: [4.0, -1.0, 2.0, 1.0]  at indices [3, 6]\n"
            "Total deviation = 6.0"
        ),
        SP(4),
        P("<b>Full sensor pipeline example:</b>"),
        code_block(
            "sensor values:  [10.0, 10.1, 10.2, 15.0, 20.0, 19.8, 10.3, 10.1, 10.2]\n"
            "RUL categories: [3,    3,    3,    2,    1,    0,    0,    0,    0   ]\n"
            "\n"
            "Step 1 - abs diff: [0.1,  0.1,  4.8,  5.0,  0.2,  9.5,  0.2,  0.1]\n"
            "Step 2 - centered: [-2.4, -2.4,  2.3,  2.5, -2.3,  7.0, -2.3, -2.4]\n"
            "Step 3 - Kadane:   max window = [2, 5],  total deviation = 9.5\n"
            "Step 4 - dominant RUL in rows [2..6]: Extremely Low\n"
            "\n"
            "=> This sensor spikes most when RUL is very low — good early warning!"
        ),
        PageBreak(),
    ]

    # ── 6. Execution Results ──────────────────────────────────────
    story += [
        H1("6. Execution Results"),
        HR(),

        H2("Task 1 — Segmentation Results"),
        P("Ten sensors were randomly selected using seed=42: sensor_41, sensor_07, "
          "sensor_01, sensor_18, sensor_16, sensor_14, sensor_08, sensor_06, "
          "sensor_35, sensor_05."),
        SP(4),
        P("<b>Complexity Scores:</b>"),
        SP(4),
        make_table(
            ["Sensor", "Segments (Complexity Score)"],
            [
                ["sensor_18", "256"], ["sensor_14", "256"], ["sensor_08", "256"],
                ["sensor_05", "251"], ["sensor_35", "248"], ["sensor_41", "174"],
                ["sensor_06", "109"], ["sensor_01",  "80"], ["sensor_16",  "65"],
                ["sensor_07",  "61"],
            ],
            col_widths=[2.5*inch, 2.5*inch],
        ),
        SP(8),
        P("Sensors like sensor_18, sensor_14, and sensor_08 hit the maximum possible "
          "complexity (256 segments), which means they were volatile enough to keep "
          "splitting all the way down to the minimum segment size of 50 time steps. "
          "Sensor_07 is the calmest with only 61 segments."),
        SP(4),
        P("Most segments across all sensors were dominated by Moderately Low or "
          "Moderately High RUL, which makes sense since those two categories make up "
          "80% of the data. Sensors with high complexity scores (18, 14, 08) tended "
          "to have more segments dominated by Moderately Low RUL, which suggests that "
          "more volatile sensors correlate with the pump being in a slightly more "
          "degraded state."),
        SP(4),
        P("<b>Temporal dynamics:</b> Looking at the segmentation plots, the high-complexity "
          "sensors do not show a clean trend where complexity increases toward the end "
          "of the time series. The volatility appears fairly spread out across the full "
          "10,000 time steps rather than concentrating near the end. This is probably "
          "because the RUL values in this dataset change gradually and the Moderately Low "
          "category covers a wide range (rows ~1000 to ~5000). Sensors with lower "
          "complexity scores like sensor_07 and sensor_16 stay relatively stable throughout "
          "the entire series."),
        SP(8),
    ]

    for sensor_name, caption in [
        ("sensor_07", "sensor_07 — low complexity (61 segments). Background color "
                      "shows dominant RUL category per segment."),
        ("sensor_18", "sensor_18 — high complexity (256 segments). Highly volatile throughout."),
    ]:
        path = f"plots/seg_{sensor_name}.png"
        story += img(path, caption_text=caption)
        story += [SP(4)]

    story += [
        SP(8),
        H2("Task 2 — Clustering Results"),
        P("The divisive algorithm clustered all 10,000 rows (50 sensor features each) "
          "into exactly 4 clusters:"),
        SP(4),
        make_table(
            ["Cluster", "Size", "Majority RUL Category", "Majority Count", "Purity"],
            [
                ["0", "5,000", "Moderately Low",  "1,639", "32.8%"],
                ["1", "2,500", "Moderately High", "1,488", "59.5%"],
                ["2", "1,250", "Moderately Low",    "779", "62.3%"],
                ["3", "1,250", "Moderately High",   "628", "50.2%"],
            ],
            col_widths=[1.0*inch, 1.0*inch, 1.8*inch, 1.5*inch, 1.0*inch],
        ),
        SP(6),
        P("<b>Overall purity: 45.3%</b> (random chance would give ~25% for 4 equal categories)."),
        SP(4),
        P("The clusters are very unequal in size (5000/2500/1250/1250) because the "
          "divisive algorithm always bisects the largest/messiest cluster, so each split "
          "roughly halves that cluster. Cluster 0 ended up being the big catch-all group "
          "from the first split — it's the least pure (32.8%) because it still contains "
          "a mix of everything. Clusters 1 and 2 are better separated (~60% purity) "
          "because the second and third splits happened in regions with clearer "
          "feature differences. Overall the clusters don't map perfectly to RUL classes, "
          "which makes sense — RUL changes continuously so there's no sharp boundary "
          "in sensor space that perfectly separates the four categories."),
        SP(8),
        H2("Task 3 — Kadane's Algorithm Results"),
        P("Kadane's algorithm was applied to all 50 sensors. The table below shows the "
          "total deviation (max-sum subarray value), the time interval, and the dominant "
          "RUL category for the top sensors:"),
        SP(4),
        make_table(
            ["Sensor", "Total Deviation", "Interval", "Dominant RUL"],
            [
                ["sensor_31", "16,527",  "[830, 4075]",  "Moderately High"],
                ["sensor_27", "10,840",  "[334, 622]",   "Extremely High"],
                ["sensor_30", "10,405",  "[882, 2442]",  "Moderately High"],
                ["sensor_32",  "5,165",  "[4197, 8179]", "Moderately Low"],
                ["sensor_29",  "4,936",  "[5061, 9998]", "Moderately Low"],
                ["sensor_17",  "4,372",  "[7630, 9998]", "Moderately Low"],
                ["sensor_37",  "4,353",  "[8808, 9998]", "Extremely Low"],
                ["sensor_25",  "4,153",  "[5332, 9997]", "Moderately Low"],
                ["sensor_36",  "4,095",  "[3545, 9998]", "Moderately Low"],
                ["sensor_24",  "3,683",  "[8600, 9998]", "Extremely Low"],
            ],
            col_widths=[1.3*inch, 1.3*inch, 1.7*inch, 1.9*inch],
        ),
        SP(8),
        P("Out of 50 sensors, <b>29 sensors</b> had their highest-deviation window "
          "fall in a Low RUL region (Extremely Low or Moderately Low). The best "
          "early-warning candidates are sensor_37, sensor_24, sensor_22, and sensor_19, "
          "which all spike in the Extremely Low RUL window (rows ~8600-9998) — meaning "
          "they become most volatile right before the pump fails. sensor_17, sensor_25, "
          "and sensor_29 are also good since their peak deviation intervals overlap "
          "heavily with Moderately Low RUL."),
    ]

    story += img("plots/kadane_top_sensors.png",
                 caption_text="Top 10 sensors by total deviation. Shaded region = max-deviation window. "
                              "Color: red = Extremely Low RUL, orange = Moderately Low, "
                              "blue = Moderately High, green = Extremely High.")
    story += [SP(4)]

    # ── 7. Discussion ─────────────────────────────────────────────
    story += [
        PageBreak(),
        H1("7. Discussion and Conclusions"),
        HR(),

        H2("Key Findings"),
        *bullet([
            "Segmentation complexity loosely correlates with RUL — more volatile sensors "
            "tend toward lower RUL segments — but the signal is weak because 80% of the "
            "data is in the two middle categories.",
            "Clustering gets 45% overall purity, better than the 25% random baseline, "
            "but the four clusters don't cleanly separate the four RUL classes. The "
            "divisive approach works but creates very unequal cluster sizes.",
            "Kadane's algorithm is the most useful tool here: 29 out of 50 sensors "
            "have their peak-deviation window in a low-RUL period. A handful of sensors "
            "(37, 24, 22, 19) are particularly strong early-warning candidates.",
        ]),
        SP(8),

        H2("Challenges"),
        *bullet([
            "NaN handling — some sensor columns had missing values (NaN). We handled "
            "them with forward-fill (use the previous value) for Tasks 1 and 3, and "
            "column mean imputation for Task 2. Neither approach is perfect but it was "
            "good enough to keep things running.",
            "Choosing the variance threshold for segmentation — there's no obvious "
            "right answer. We went with 10% of each sensor's overall variance as a "
            "reasonable default. Different thresholds give very different segment counts, "
            "so this choice has a big effect on the results.",
            "Sensor column discrepancy — the assignment says 52 sensors but the actual "
            "CSV only has 50. Figuring this out required loading the file and counting "
            "the columns manually. It's not a big deal, but it means the results are "
            "based on 50 sensors instead of 52.",
            "Making the clustering produce balanced clusters — the divisive approach "
            "naturally creates clusters of sizes 5000/2500/1250/1250 because it always "
            "bisects the biggest group. It's hard to get equal-sized clusters with this "
            "algorithm without adding more complexity.",
            "Mapping Kadane's result back to RUL categories — the absolute first difference "
            "has length N-1 (one shorter than the original series), so the index mapping "
            "back to the original rows needed a +1 offset. Easy to get wrong.",
        ]),
        SP(8),

        H2("Limitations"),
        *bullet([
            "The segmentation threshold (10% of global variance) is fixed. An adaptive "
            "threshold per sensor might give more consistent segment sizes.",
            "Divisive clustering with median bisection produces geometrically nested "
            "cluster sizes. A different split criterion (like maximizing inter-cluster "
            "distance) would likely give more meaningful clusters.",
            "Kadane's algorithm finds only the single most extreme window. Looking at "
            "the top-K windows would give a more complete picture.",
            "All three algorithms treat timestamps as simple integer indices. Actual "
            "time structure (daily patterns, weekends, etc.) is ignored.",
            "The dataset is from a single pump. Results may not generalize to other "
            "pumps with different sensor configurations.",
        ]),
        SP(8),

        H2("Possible Improvements"),
        *bullet([
            "For segmentation: try different thresholds (like the 5th or 25th percentile "
            "of variance) and compare complexity scores to see which threshold gives "
            "the most useful segmentation.",
            "For clustering: add a refinement step after divisive clustering — reassign "
            "points to their nearest cluster centroid to clean up the boundaries.",
            "For Kadane's: run it on a sliding window instead of the full series to "
            "find multiple high-deviation windows, not just the biggest one.",
            "Better missing value handling — interpolation or a learned imputation "
            "would be more accurate than forward-fill or column mean.",
            "Using the actual timestamps to align sensor behavior with time of day or "
            "day of week could reveal patterns that are invisible with index-based analysis.",
        ]),
        SP(8),

        H1("8. Conclusion"),
        HR(),
        P("This project showed that basic algorithmic techniques can find useful patterns "
          "in industrial sensor data without any machine learning. The most practical "
          "finding is that a small group of sensors (especially sensor_37, sensor_24, "
          "sensor_22, and sensor_17) become most volatile right before the pump is "
          "about to fail. In a real system, you could monitor just these sensors and "
          "trigger an alert when their variability spikes above a threshold. That would "
          "be a simple, interpretable early-warning system with no ML required."),
    ]

    doc.build(story)
    print(f"Report saved -> {out_path}")


if __name__ == '__main__':
    build_report('report.pdf')
