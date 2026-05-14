# -*- coding: utf-8 -*-
"""
Country ranking figure (1990–2020 pooled) — softened display with stretched y-axis
+ quadrant marker shapes
+ 8 pie charts embedded inside blank areas of each country-ranking quartile

IMPORTANT:
Main plot = COUNTRY-level ranking
Inset pie charts = INDEPENDENT CITY-level ranking composition

Logic:
1. Main plot:
   - Aggregate data to country level.
   - Rank countries by green-prioritized co-evolution score.
   - Plot countries along the x-axis.

2. Pie charts:
   - Aggregate data independently to city level.
   - Rank cities by their own green-prioritized co-evolution score.
   - Divide cities into Top 25%, 25–50%, 50–75%, Bottom 25%.
   - For each city-level quartile, draw:
       a) city quadrant composition: Q1/Q2/Q3/Q4
       b) city continent composition: NA/EU/AS/SA/OC/AF

Therefore:
- Country points and background bands belong to country ranking.
- Pie charts belong to independent city ranking.
- City pie chart n should be approximately equal across four quartiles.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba


# =========================
# CONFIG
# =========================
DATA_PATH = r"C:\Users\steve\Desktop\最新结论\Result 3 四象限\LOG_outputs_quadrants_椭圆阈值1.0\labels_yearly_UGSrel_NLIrel_1.00.csv"

OUT_DIR = r"C:\Users\steve\Desktop\最新结论\Result 3 四象限\3.3.2 国家排序_INDEPENDENT_CITYPies_0.60.4"
os.makedirs(OUT_DIR, exist_ok=True)

YEAR_COL    = "Year"
ID_COL      = "ID_HDC_G0"
CITY_COL    = "city"
COUNTRY_COL = "country"
CONT_COL    = "continent_UGS"
X_COL       = "UGSrel"
Y_COL       = "NLIrel"

# Ranking weights
W_BAL = 0.6
W_UGS = 0.4

# Quartile background bands for country ranking
QTL_BAND_COLORS = {
    1: "#F2F2F2",
    2: "#E6E6E6",
    3: "#D9D9D9",
    4: "#CCCCCC"
}
QTL_BAND_ALPHA = 0.40

# Continent colors
CONT_COLORS = {
    "AF": "#B22222",
    "AS": "#8B4513",
    "EU": "#4F81BD",
    "OC": "#708090",
    "SA": "#F4A460",
    "NorthAmerica": "#2E8B57",
    "Other": "#B0B0B0",
}

CONT_ORDER = ["NorthAmerica", "EU", "AS", "SA", "OC", "AF", "Other"]

CONT_LABEL = {
    "NorthAmerica": "NA",
    "EU": "EU",
    "AS": "AS",
    "SA": "SA",
    "OC": "OC",
    "AF": "AF",
    "Other": "Other"
}

# Quadrant marker shapes for country points
QUAD_MARKERS = {
    "Q1": "o",   # circle
    "Q2": "D",   # diamond
    "Q3": "^",   # triangle
    "Q4": "s",   # square
}

# Quadrant pie colors
QUAD_COLORS = {
    "Q1": "#9E9E9E",
    "Q2": "#BDBDBD",
    "Q3": "#D9D9D9",
    "Q4": "#7A7A7A",
}

# Point size settings
MIN_MS = 80
MAX_MS = 2000
POINT_ALPHA = 0.58

# Smooth line settings
SHOW_CONNECT_LINE = True
LINE_ALPHA = 0.18
LINE_WIDTH = 1.45

# Display curve controls for country plot only
ROLL_WINDOW = 13
BLEND_W_RAW = 0.2
BLEND_W_SMOOTH = 0.8
UPPER_STRETCH_GAMMA = 2
EXTREME_COMPRESS_POWER = 0.90


# =========================
# Helper functions
# =========================
def mode_or_first(s: pd.Series):
    """Return the modal value of a Series; if no mode, return the first non-null value."""
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) > 0 else s.iloc[0]


def normalize_0_1(arr):
    """Normalize an array to [0, 1]."""
    arr = np.asarray(arr, dtype=float)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)

    if not np.isfinite(amin) or not np.isfinite(amax) or (amax - amin) < 1e-12:
        return np.zeros_like(arr)

    return (arr - amin) / (amax - amin + 1e-12)


def continent_key(raw):
    """Normalize continent labels into fixed keys."""
    if pd.isna(raw):
        return "Other"

    s = str(raw).strip()
    s_low = s.lower().replace(" ", "").replace("_", "")

    if s_low in ["af", "africa"]:
        return "AF"
    if s_low in ["as", "asia"]:
        return "AS"
    if s_low in ["eu", "europe"]:
        return "EU"
    if s_low in ["oc", "oceania"]:
        return "OC"
    if s_low in ["sa", "southamerica"]:
        return "SA"
    if s_low in ["na", "usa", "northamerica"]:
        return "NorthAmerica"
    if ("north" in s_low) and ("america" in s_low):
        return "NorthAmerica"

    return "Other"


def get_quadrant(x, y):
    """Classify UGSrel–NLIrel position into Q1–Q4."""
    if pd.isna(x) or pd.isna(y):
        return np.nan

    if x > 0 and y > 0:
        return "Q1"
    elif x <= 0 and y > 0:
        return "Q2"
    elif x <= 0 and y <= 0:
        return "Q3"
    else:
        return "Q4"


def scale_sizes_log(values, min_s=MIN_MS, max_s=MAX_MS):
    """Scale country point size by number of cities."""
    vals = np.asarray(values, dtype=float)
    vals = np.where(vals < 1, 1, vals)

    lv = np.log1p(vals) ** 2.5

    vmin = np.nanmin(lv)
    vmax = np.nanmax(lv)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-12:
        return np.full_like(vals, (min_s + max_s) / 2.0, dtype=float)

    return min_s + (lv - vmin) * (max_s - min_s) / (vmax - vmin)


def soften_series_monotone(
    y,
    roll_window=ROLL_WINDOW,
    w_raw=BLEND_W_RAW,
    w_smooth=BLEND_W_SMOOTH,
    power=EXTREME_COMPRESS_POWER
):
    """Smooth display score only for visual presentation."""
    s = pd.Series(np.asarray(y, dtype=float))
    sm = s.rolling(window=roll_window, center=True, min_periods=1).median()

    y_blend = w_raw * s + w_smooth * sm

    ymin = float(np.nanmin(y_blend))
    ymax = float(np.nanmax(y_blend))
    yrng = ymax - ymin + 1e-12

    z = (y_blend - ymin) / yrng
    z2 = z ** power

    y_soft = ymin + z2 * yrng
    y_soft = pd.Series(y_soft).rolling(window=5, center=True, min_periods=1).mean()

    return np.asarray(y_soft, dtype=float)


def stretch_upper_tail(y, gamma=UPPER_STRETCH_GAMMA):
    """Stretch upper display range for visual readability."""
    y = np.asarray(y, dtype=float)

    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    yrng = ymax - ymin + 1e-12

    z = (y - ymin) / yrng
    z2 = z ** gamma

    return ymin + z2 * yrng


def add_composition_pie(
    ax,
    sub_df,
    rect,
    group_col,
    color_map,
    order,
    title,
    label_map=None,
    alpha=0.78,
    min_pct_show=6,
    panel_fc=(1, 1, 1, 0.32)
):
    """
    Add one inset pie chart.
    sub_df is city-level data for one independent city-ranking quartile.
    """
    axp = ax.inset_axes(rect)
    axp.set_aspect("equal")
    axp.set_xticks([])
    axp.set_yticks([])
    axp.set_facecolor(panel_fc)

    for sp in axp.spines.values():
        sp.set_visible(False)

    if sub_df.empty:
        axp.text(
            0.5, 0.5,
            f"{title}\n(no data)",
            ha="center", va="center",
            fontsize=8
        )
        return

    counts = sub_df[group_col].value_counts(dropna=False).to_dict()

    labels, sizes, colors = [], [], []

    for k in order:
        v = counts.get(k, 0)
        if v <= 0:
            continue

        labels.append(label_map.get(k, k) if label_map else k)
        sizes.append(v)
        colors.append(to_rgba(color_map.get(k, "#B0B0B0"), alpha))

    if len(sizes) == 0:
        axp.text(
            0.5, 0.5,
            f"{title}\n(no data)",
            ha="center", va="center",
            fontsize=8
        )
        return

    wedges, texts, autotexts = axp.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda p: f"{p:.0f}%" if p >= min_pct_show else "",
        pctdistance=0.72,
        labeldistance=1.05,
        wedgeprops=dict(linewidth=0.7, edgecolor="white")
    )

    for t in texts:
        t.set_fontsize(7)

    for t in autotexts:
        t.set_fontsize(7)

    # Add city count under each pie.
    axp.text(
        0.5, -0.13,
        f"n={len(sub_df):,} cities",
        ha="center",
        va="center",
        fontsize=7,
        transform=axp.transAxes
    )


# =========================
# Build pooled tables
# =========================
def build_country_pooled_table(df):
    """
    Country-level pooled table.
    Each country appears once.
    This is used for the main country-ranking plot.
    """
    pooled = (
        df.groupby(COUNTRY_COL, as_index=False)
          .agg(
              x_med=(X_COL, "median"),
              y_med=(Y_COL, "median"),
              ugs_mean=(X_COL, "mean"),
              continent=(CONT_COL, mode_or_first),
              n_obs=(YEAR_COL, "size"),
              n_city=(ID_COL, pd.Series.nunique)
          )
    )

    pooled["continent_key"] = pooled["continent"].apply(continent_key)
    pooled["quadrant"] = pooled.apply(
        lambda r: get_quadrant(r["x_med"], r["y_med"]),
        axis=1
    )

    return pooled


def build_city_pooled_table(df):
    """
    City-level pooled table.
    Each city appears once.
    This is used for independent city ranking and city pie charts.
    """
    pooled = (
        df.groupby([ID_COL, CITY_COL], as_index=False)
          .agg(
              x_med=(X_COL, "median"),
              y_med=(Y_COL, "median"),
              ugs_mean=(X_COL, "mean"),
              country=(COUNTRY_COL, mode_or_first),
              continent=(CONT_COL, mode_or_first),
              n_obs=(YEAR_COL, "size")
          )
    )

    pooled["continent_key"] = pooled["continent"].apply(continent_key)
    pooled["quadrant"] = pooled.apply(
        lambda r: get_quadrant(r["x_med"], r["y_med"]),
        axis=1
    )

    pooled = pooled.dropna(subset=["quadrant"]).copy()

    return pooled


# =========================
# Ranking functions
# =========================
def rank_countries_composite(pooled):
    """
    Country-level composite ranking for main plot.
    """
    r = pooled.copy()

    r["dist45"] = np.abs(r["y_med"] - r["x_med"]) / np.sqrt(2.0)
    r["balance_score"] = 1.0 - normalize_0_1(r["dist45"].values)
    r["ugs_score"] = normalize_0_1(r["ugs_mean"].values)
    r["composite_score"] = W_BAL * r["balance_score"] + W_UGS * r["ugs_score"]

    r = r.sort_values(
        ["composite_score", "balance_score", "ugs_mean", COUNTRY_COL],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    r["rank"] = np.arange(1, len(r) + 1)

    y_soft = soften_series_monotone(r["composite_score"].values)
    r["display_score"] = stretch_upper_tail(
        y_soft,
        gamma=UPPER_STRETCH_GAMMA
    )

    return r


def rank_cities_composite(city_pooled):
    """
    CITY-level independent composite ranking for pie charts.
    Each city is ranked by its own balance + UGS score.
    This is independent from the country-level ranking used in the main plot.
    """
    r = city_pooled.copy()

    r["dist45"] = np.abs(r["y_med"] - r["x_med"]) / np.sqrt(2.0)
    r["balance_score"] = 1.0 - normalize_0_1(r["dist45"].values)
    r["ugs_score"] = normalize_0_1(r["ugs_mean"].values)
    r["composite_score"] = W_BAL * r["balance_score"] + W_UGS * r["ugs_score"]

    r = r.sort_values(
        ["composite_score", "balance_score", "ugs_mean", CITY_COL],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    r["city_rank"] = np.arange(1, len(r) + 1)

    return r


def add_rank_quartile_label(ranked):
    """
    Divide ranked countries into:
    1 = Top 25%
    2 = 25–50%
    3 = 50–75%
    4 = Bottom 25%
    """
    n = len(ranked)

    q1_end = int(np.floor(0.25 * n))
    q2_end = int(np.floor(0.50 * n))
    q3_end = int(np.floor(0.75 * n))

    q1_end = max(1, q1_end)
    q2_end = max(q1_end + 1, q2_end)
    q3_end = max(q2_end + 1, q3_end)

    qtl = np.full(n, 4, dtype=int)
    qtl[:q1_end] = 1
    qtl[q1_end:q2_end] = 2
    qtl[q2_end:q3_end] = 3
    qtl[q3_end:] = 4

    out = ranked.copy()
    out["rank_quartile"] = qtl

    bounds = (q1_end, q2_end, q3_end, n)

    return out, bounds


def add_city_rank_quartile_label(city_ranked):
    """
    Divide independently ranked cities into four nearly equal city quartiles:
    1 = Top 25%
    2 = 25–50%
    3 = 50–75%
    4 = Bottom 25%
    """
    n = len(city_ranked)

    q1_end = int(np.floor(0.25 * n))
    q2_end = int(np.floor(0.50 * n))
    q3_end = int(np.floor(0.75 * n))

    q1_end = max(1, q1_end)
    q2_end = max(q1_end + 1, q2_end)
    q3_end = max(q2_end + 1, q3_end)

    qtl = np.full(n, 4, dtype=int)
    qtl[:q1_end] = 1
    qtl[q1_end:q2_end] = 2
    qtl[q2_end:q3_end] = 3
    qtl[q3_end:] = 4

    out = city_ranked.copy()
    out["rank_quartile"] = qtl

    city_bounds = (q1_end, q2_end, q3_end, n)

    return out, city_bounds


# =========================
# Plot
# =========================
def plot_country_ranking_soft(ranked, city_for_pies, country_bounds, title, out_base):
    n_total = len(ranked)
    q1_end, q2_end, q3_end, _ = country_bounds

    fig = plt.figure(figsize=(17.5, 9.2), constrained_layout=False)
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.82])

    yvals = ranked["display_score"].values

    ymin = float(np.nanmin(yvals))
    ymax = float(np.nanmax(yvals))
    yrng = ymax - ymin + 1e-12
    ypad = 0.05 * yrng

    ax.set_ylim(ymin - ypad, ymax + ypad)

    xpad = max(3, int(0.01 * n_total))
    ax.set_xlim(1 - xpad, n_total + xpad)

    # Background country-ranking quartile bands
    spans = [
        (1, q1_end, 1),
        (q1_end + 1, q2_end, 2),
        (q2_end + 1, q3_end, 3),
        (q3_end + 1, n_total, 4),
    ]

    for x0, x1, part in spans:
        ax.axvspan(
            x0, x1,
            facecolor=QTL_BAND_COLORS[part],
            alpha=QTL_BAND_ALPHA,
            zorder=0
        )

    # Country-ranking quartile boundary lines
    for xb in [q1_end + 0.5, q2_end + 0.5, q3_end + 0.5]:
        ax.vlines(
            xb,
            ymin=ax.get_ylim()[0],
            ymax=ax.get_ylim()[1],
            colors="black",
            linestyles="--",
            linewidth=1.3,
            alpha=0.75,
            zorder=1
        )

    # Soft trend line
    if SHOW_CONNECT_LINE:
        ax.plot(
            ranked["rank"].values,
            ranked["display_score"].values,
            color="#6F6F6F",
            lw=LINE_WIDTH,
            alpha=LINE_ALPHA,
            zorder=2
        )

    # Country points
    ranked = ranked.copy()

    ranked["point_color"] = ranked["continent_key"].map(
        lambda k: to_rgba(CONT_COLORS.get(k, "#B0B0B0"), POINT_ALPHA)
    )

    ranked["point_size"] = scale_sizes_log(
        ranked["n_city"].values,
        min_s=MIN_MS,
        max_s=MAX_MS
    )

    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        sub = ranked[ranked["quadrant"] == quad].copy()

        if sub.empty:
            continue

        ax.scatter(
            sub["rank"].values,
            sub["display_score"].values,
            s=sub["point_size"].values,
            c=sub["point_color"].tolist(),
            marker=QUAD_MARKERS[quad],
            edgecolors=(0, 0, 0, 0.26),
            linewidths=0.35,
            zorder=3
        )

    # Axis settings
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    raw_ticks = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])

    y_raw_min = np.nanmin(ranked["composite_score"].values)
    y_raw_max = np.nanmax(ranked["composite_score"].values)
    y_raw_rng = y_raw_max - y_raw_min + 1e-12

    z = (raw_ticks - y_raw_min) / y_raw_rng
    z = np.clip(z, 0, 1)

    tick_pos = np.nanmin(yvals) + (z ** UPPER_STRETCH_GAMMA) * (
        np.nanmax(yvals) - np.nanmin(yvals)
    )

    ax.set_yticks(tick_pos)
    ax.set_yticklabels([f"{t:.1f}" for t in raw_ticks])

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Country rank (left = better, right = worse)", fontsize=15)
    ax.set_ylabel("Composite score", fontsize=15)

    ax.grid(True, which="major", axis="y", alpha=0.14)
    ax.grid(False, axis="x")

    # -------------------------
    # INDEPENDENT CITY pies
    # These pies are NOT based on country quartile membership.
    # They are based on city-level ranking quartiles.
    # -------------------------
    quart_sub = {
        1: city_for_pies[city_for_pies["rank_quartile"] == 1].copy(),
        2: city_for_pies[city_for_pies["rank_quartile"] == 2].copy(),
        3: city_for_pies[city_for_pies["rank_quartile"] == 3].copy(),
        4: city_for_pies[city_for_pies["rank_quartile"] == 4].copy(),
    }

    pie_rects = {
        "Q1_quad": [0.05, 0.28, 0.15, 0.24],
        "Q1_cont": [0.05, 0.05, 0.15, 0.24],
        "Q2_quad": [0.31, 0.29, 0.15, 0.24],
        "Q2_cont": [0.31, 0.06, 0.15, 0.24],
        "Q3_quad": [0.54, 0.25, 0.15, 0.24],
        "Q3_cont": [0.54, 0.02, 0.15, 0.24],
        "Q4_quad": [0.80, 0.66, 0.17, 0.24],
        "Q4_cont": [0.80, 0.43, 0.17, 0.24],
    }

    quad_label_map = {
        "Q1": "Q1",
        "Q2": "Q2",
        "Q3": "Q3",
        "Q4": "Q4"
    }

    # Top 25%
    add_composition_pie(
        ax,
        quart_sub[1],
        pie_rects["Q1_quad"],
        group_col="quadrant",
        color_map=QUAD_COLORS,
        order=["Q1", "Q2", "Q3", "Q4"],
        title="Top 25%\nCity-ranked quadrant",
        label_map=quad_label_map
    )

    add_composition_pie(
        ax,
        quart_sub[1],
        pie_rects["Q1_cont"],
        group_col="continent_key",
        color_map=CONT_COLORS,
        order=CONT_ORDER,
        title="Top 25%\nCity-ranked continent",
        label_map=CONT_LABEL
    )

    # 25–50%
    add_composition_pie(
        ax,
        quart_sub[2],
        pie_rects["Q2_quad"],
        group_col="quadrant",
        color_map=QUAD_COLORS,
        order=["Q1", "Q2", "Q3", "Q4"],
        title="25–50%\nCity-ranked quadrant",
        label_map=quad_label_map
    )

    add_composition_pie(
        ax,
        quart_sub[2],
        pie_rects["Q2_cont"],
        group_col="continent_key",
        color_map=CONT_COLORS,
        order=CONT_ORDER,
        title="25–50%\nCity-ranked continent",
        label_map=CONT_LABEL
    )

    # 50–75%
    add_composition_pie(
        ax,
        quart_sub[3],
        pie_rects["Q3_quad"],
        group_col="quadrant",
        color_map=QUAD_COLORS,
        order=["Q1", "Q2", "Q3", "Q4"],
        title="50–75%\nCity-ranked quadrant",
        label_map=quad_label_map
    )

    add_composition_pie(
        ax,
        quart_sub[3],
        pie_rects["Q3_cont"],
        group_col="continent_key",
        color_map=CONT_COLORS,
        order=CONT_ORDER,
        title="50–75%\nCity-ranked continent",
        label_map=CONT_LABEL
    )

    # Bottom 25%
    add_composition_pie(
        ax,
        quart_sub[4],
        pie_rects["Q4_quad"],
        group_col="quadrant",
        color_map=QUAD_COLORS,
        order=["Q1", "Q2", "Q3", "Q4"],
        title="Bottom 25%\nCity-ranked quadrant",
        label_map=quad_label_map
    )

    add_composition_pie(
        ax,
        quart_sub[4],
        pie_rects["Q4_cont"],
        group_col="continent_key",
        color_map=CONT_COLORS,
        order=CONT_ORDER,
        title="Bottom 25%\nCity-ranked continent",
        label_map=CONT_LABEL
    )

    # Legends
    cont_handles = []

    for k in CONT_ORDER:
        if k in ranked["continent_key"].unique():
            cont_handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    color="none",
                    markerfacecolor=to_rgba(CONT_COLORS.get(k, "#B0B0B0"), POINT_ALPHA),
                    markeredgecolor=(0, 0, 0, 0.35),
                    markeredgewidth=0.5,
                    markersize=8,
                    label=CONT_LABEL.get(k, k)
                )
            )

    quad_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7,
            label="Q1"
        ),
        Line2D(
            [0], [0],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7,
            label="Q2"
        ),
        Line2D(
            [0], [0],
            marker="^",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7,
            label="Q3"
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7,
            label="Q4"
        ),
    ]

    leg1 = ax.legend(
        handles=cont_handles,
        title="Country continent",
        loc="upper left",
        bbox_to_anchor=(0, 0.8),
        frameon=True,
        fontsize=12,
        title_fontsize=13
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=quad_handles,
        title="Country quadrant",
        loc="upper center",
        bbox_to_anchor=(0.57, 1.01),
        ncol=4,
        frameon=True,
        fontsize=11,
        title_fontsize=12
    )
    ax.add_artist(leg2)

    svg = out_base + ".svg"
    pdf = out_base + ".pdf"
    png = out_base + ".png"

    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")

    plt.close(fig)

    return svg, pdf, png


# =========================
# Export tables
# =========================
def export_tables(ranked, city_for_pies, out_dir):
    full_csv = os.path.join(out_dir, "POOLED_country_ranking_full.csv")
    pie_csv = os.path.join(out_dir, "POOLED_independent_city_ranking_for_pies.csv")
    summary_csv = os.path.join(out_dir, "POOLED_independent_city_pie_summary_by_quartile.csv")
    xlsx = os.path.join(out_dir, "POOLED_country_ranking_INDEPENDENT_CITY_pie_tables.xlsx")

    ranked.to_csv(full_csv, index=False, encoding="utf-8-sig")
    city_for_pies.to_csv(pie_csv, index=False, encoding="utf-8-sig")

    # Summary table for direct writing in Results
    rows = []

    for q in [1, 2, 3, 4]:
        sub = city_for_pies[city_for_pies["rank_quartile"] == q].copy()
        n = len(sub)

        quad_counts = sub["quadrant"].value_counts().to_dict()
        cont_counts = sub["continent_key"].value_counts().to_dict()

        row = {
            "rank_quartile": q,
            "quartile_label": {
                1: "Top 25%",
                2: "25-50%",
                3: "50-75%",
                4: "Bottom 25%"
            }[q],
            "n_cities": n,
        }

        for qq in ["Q1", "Q2", "Q3", "Q4"]:
            c = quad_counts.get(qq, 0)
            row[f"{qq}_n"] = c
            row[f"{qq}_pct"] = 100 * c / n if n > 0 else np.nan

        for cc in CONT_ORDER:
            c = cont_counts.get(cc, 0)
            label = CONT_LABEL.get(cc, cc)
            row[f"{label}_n"] = c
            row[f"{label}_pct"] = 100 * c / n if n > 0 else np.nan

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        ranked.to_excel(w, sheet_name="country_ranking", index=False)
        city_for_pies.to_excel(w, sheet_name="independent_city_ranking_for_pies", index=False)
        summary_df.to_excel(w, sheet_name="city_pie_summary", index=False)

    return full_csv, pie_csv, summary_csv, xlsx


# =========================
# Main
# =========================
def main():
    if abs((W_BAL + W_UGS) - 1.0) > 1e-9:
        raise ValueError("W_BAL + W_UGS must equal 1.0")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    for c in [YEAR_COL, X_COL, Y_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    need_cols = [
        YEAR_COL,
        ID_COL,
        CITY_COL,
        COUNTRY_COL,
        CONT_COL,
        X_COL,
        Y_COL
    ]

    df = df.dropna(subset=need_cols).copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)

    pooled_raw = df[
        (df[YEAR_COL] >= 1990) &
        (df[YEAR_COL] <= 2020)
    ].copy()

    # =========================
    # COUNTRY-level main plot
    # =========================
    pooled_country = build_country_pooled_table(pooled_raw)

    ranked = rank_countries_composite(pooled_country)
    ranked, country_bounds = add_rank_quartile_label(ranked)

    # =========================
    # INDEPENDENT CITY-level pies
    # =========================
    city_pooled = build_city_pooled_table(pooled_raw)
    city_ranked = rank_cities_composite(city_pooled)
    city_for_pies, city_bounds = add_city_rank_quartile_label(city_ranked)

    out_base = os.path.join(
        OUT_DIR,
        "POOLED_1990_2020_country_ranking_balance45_meanUGS_soft_yaxis_quadrantShapes_INDEPENDENT_CITYPies_insideQuarters"
    )

    title = ""

    svg, pdf, png = plot_country_ranking_soft(
        ranked,
        city_for_pies,
        country_bounds,
        title,
        out_base
    )

    full_csv, pie_csv, summary_csv, xlsx = export_tables(
        ranked,
        city_for_pies,
        OUT_DIR
    )

    print("DONE.")
    print("Country quartile boundaries (end ranks):", country_bounds)
    print("Independent city quartile boundaries (end ranks):", city_bounds)
    print("FIG SVG:", svg)
    print("FIG PDF:", pdf)
    print("FIG PNG:", png)
    print("COUNTRY CSV:", full_csv)
    print("INDEPENDENT CITY PIE CSV:", pie_csv)
    print("INDEPENDENT CITY PIE SUMMARY CSV:", summary_csv)
    print("XLSX:", xlsx)

    print("\nIndependent city composition by city-ranking quartile:")
    summary = pd.read_csv(summary_csv)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()