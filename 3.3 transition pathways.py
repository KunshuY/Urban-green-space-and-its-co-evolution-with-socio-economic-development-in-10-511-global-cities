# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# =========================
# CONFIG
# =========================
DATA_PATH = r"C:\Users\steve\Desktop\最新结论\Result 3 四象限\LOG_outputs_quadrants_椭圆阈值1.0\labels_yearly_UGSrel_NLIrel_1.00.csv"
OUT_DIR   = r"C:\Users\steve\Desktop\最新结论\Result 3 四象限\1990_2020_跨象限_箭头+堆叠柱\最终改颜色"
os.makedirs(OUT_DIR, exist_ok=True)

ID_COL       = "ID_HDC_G0"
CITY_COL     = "city"
COUNTRY_COL  = "country"
CONT_COL     = "continent_UGS"
YEAR_COL     = "Year"
X_COL        = "UGSrel"
Y_COL        = "NLIrel"

YEAR_START = 1990
YEAR_END   = 2020
USE_SIGNED_LOG = True
CROSS_ONLY = True

# significant filter (optional)
USE_SIGNIFICANT_ONLY = False
SIG_METHOD   = "quantile"      # "quantile" or "std"
SIG_QUANTILE = 0.85
SIG_K_MAD    = 1.5
SIG_METRIC   = "delta_nli"

# =========================
# Enhanced contrast (WIDTH + ALPHA + ARROWHEAD)
# =========================
ARROW_FIGSIZE = (12, 10)

# linewidth mapping
MIN_LW = 0.7
MAX_LW = 24.0
P_CLIP_LW = 99.0
POWER_LW  = 2.2

# alpha mapping (make small flows less faint)
ALPHA_MIN = 0.45
ALPHA_MAX = 0.98
ALPHA_POWER = 0.65

# arrowhead scaling (avoid "tube" look for thick arrows)
MS_BASE = 10.0        # base mutation scale
MS_PER_LW = 1.6       # head grows with linewidth
SHRINK_AB = 24        # push head outside node

# bar plot
BAR_FIGSIZE = (16, 8)

# continent colors
CONTINENT_COLORS = {
    "AF":  "#DC143C",
    "AS":  "#A0522D",
    "EU":  "#4F81BD",
    "OC":  "#708090",
    "SA":  "#F4A460",
    "NorthAmerica": "#2E8B57",
    "USA": "#2E8B57",
    "NA":  "#2E8B57",
}
DEFAULT_COLOR = "#999999"
BASE_CONTS = ["AF", "AS", "EU", "OC", "SA", "NorthAmerica"]

# quadrant positions
Q_POS = {
    "Q1": ( 1.0,  1.0),
    "Q2": (-1.0,  1.0),
    "Q3": (-1.0, -1.0),
    "Q4": ( 1.0, -1.0),
}
Q_ORDER = ["Q1","Q2","Q3","Q4"]

# to separate same transition across continents
CONT_RAD = {
    "AF": -0.24,
    "AS": -0.14,
    "EU": -0.05,
    "OC":  0.05,
    "SA":  0.14,
    "NorthAmerica": 0.24
}

# =========================
# Helpers
# =========================
def signed_log1p(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))

def compute_quadrant(x, y):
    q = np.empty(len(x), dtype=object)
    q[(x >= 0) & (y >= 0)] = "Q1"
    q[(x <  0) & (y >= 0)] = "Q2"
    q[(x <  0) & (y <  0)] = "Q3"
    q[(x >= 0) & (y <  0)] = "Q4"
    return q

def mode_or_first(s):
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) > 0 else s.iloc[0]

def robust_sigma_mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad if mad > 0 else np.nan

def imbalance_d45(x, y):
    return np.abs(y - x) / np.sqrt(2.0)

def normalize_continent(c):
    c = str(c)
    if c in ("USA", "NA"):
        return "NorthAmerica"
    return c

def color_of_cont(c):
    return CONTINENT_COLORS.get(str(c), DEFAULT_COLOR)

def width_map(v, min_lw=MIN_LW, max_lw=MAX_LW, power=POWER_LW, p_clip=P_CLIP_LW):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return v
    v_f = v[np.isfinite(v)]
    if len(v_f) == 0:
        return np.full_like(v, (min_lw + max_lw) / 2.0)

    lo = np.nanmin(v_f)
    hi = np.nanpercentile(v_f, p_clip)
    if hi - lo < 1e-12:
        return np.full_like(v, (min_lw + max_lw) / 2.0)

    vv = np.clip(v, lo, hi)
    z = (vv - lo) / (hi - lo + 1e-12)
    z = np.power(z, power)  # head emphasis
    return min_lw + (max_lw - min_lw) * z

def alpha_from_lw(lw, min_lw=MIN_LW, max_lw=MAX_LW,
                  a_min=ALPHA_MIN, a_max=ALPHA_MAX, a_power=ALPHA_POWER):
    z = (lw - min_lw) / (max_lw - min_lw + 1e-12)
    z = float(np.clip(z, 0.0, 1.0))
    z = z ** a_power
    return a_min + (a_max - a_min) * z

# =========================
# Build independent-city table (1990 -> 2020)
# =========================
def build_independent_city_table(df):
    d = df.copy()
    d[YEAR_COL] = pd.to_numeric(d[YEAR_COL], errors="coerce")
    d[X_COL] = pd.to_numeric(d[X_COL], errors="coerce")
    d[Y_COL] = pd.to_numeric(d[Y_COL], errors="coerce")

    d = d.dropna(subset=[ID_COL, YEAR_COL, X_COL, Y_COL, CITY_COL, COUNTRY_COL, CONT_COL]).copy()
    d[YEAR_COL] = d[YEAR_COL].astype(int)
    d = d[d[YEAR_COL].isin([YEAR_START, YEAR_END])].copy()

    if USE_SIGNED_LOG:
        d["x_t"] = signed_log1p(d[X_COL].values)
        d["y_t"] = signed_log1p(d[Y_COL].values)
    else:
        d["x_t"] = d[X_COL].values
        d["y_t"] = d[Y_COL].values

    d["quad"] = compute_quadrant(d["x_t"].values, d["y_t"].values)

    city_year = (d.groupby([ID_COL, YEAR_COL], as_index=False)
                  .agg(x_t=("x_t","median"),
                       y_t=("y_t","median"),
                       city=(CITY_COL, mode_or_first),
                       country=(COUNTRY_COL, mode_or_first),
                       continent=(CONT_COL, mode_or_first),
                       quad=("quad", mode_or_first)))

    s = city_year[city_year[YEAR_COL] == YEAR_START].copy()
    e = city_year[city_year[YEAR_COL] == YEAR_END].copy()

    m = s.merge(e, on=ID_COL, suffixes=("_s","_e"), how="inner")

    m["city"]      = m["city_s"].combine_first(m["city_e"])
    m["country"]   = m["country_s"].combine_first(m["country_e"])
    m["continent"] = m["continent_s"].combine_first(m["continent_e"]).map(normalize_continent)

    m["x0"] = m["x_t_s"]
    m["y0"] = m["y_t_s"]
    m["x1"] = m["x_t_e"]
    m["y1"] = m["y_t_e"]

    m["q_start"] = m["quad_s"]
    m["q_end"]   = m["quad_e"]
    m["transition"] = m["q_start"].astype(str) + "->" + m["q_end"].astype(str)

    m["delta_ugs"] = m["x1"] - m["x0"]
    m["delta_nli"] = m["y1"] - m["y0"]
    m["d0"] = imbalance_d45(m["x0"].values, m["y0"].values)
    m["d1"] = imbalance_d45(m["x1"].values, m["y1"].values)
    m["delta_balance"] = m["d1"] - m["d0"]

    if CROSS_ONLY:
        m = m[m["q_start"] != m["q_end"]].copy()

    return m

def apply_significant_filter(m):
    if not USE_SIGNIFICANT_ONLY:
        return m.copy(), "all independent cross-quadrant cities"

    x = m[SIG_METRIC].values.astype(float)

    if SIG_METHOD == "quantile":
        thr = float(np.nanquantile(np.abs(x), SIG_QUANTILE))
        keep = np.abs(x) >= thr
        rule = f"|{SIG_METRIC}| >= q{SIG_QUANTILE:.2f} ({thr:.4f})"
    elif SIG_METHOD == "std":
        med = float(np.nanmedian(x))
        rs = robust_sigma_mad(x)
        if not np.isfinite(rs) or rs == 0:
            rs = float(np.nanstd(x))
        thr = SIG_K_MAD * rs
        keep = np.abs(x - med) >= thr
        rule = f"|{SIG_METRIC}-median| >= {SIG_K_MAD:.2f}*MADsigma ({thr:.4f})"
    else:
        raise ValueError("SIG_METHOD must be quantile or std.")

    return m[keep].copy(), rule

# =========================
# Flow tables
# =========================
def build_flow(df_city):
    flow = (df_city.groupby(["continent","q_start","q_end"], as_index=False)
              .size()
              .rename(columns={"size":"n_city"}))
    flow["transition"] = flow["q_start"] + "->" + flow["q_end"]
    return flow

def build_transition_by_continent(df_city):
    mat = (df_city.groupby(["transition","continent"])
                 .size()
                 .unstack(fill_value=0))

    for alt in ["USA", "NA"]:
        if alt in mat.columns:
            mat["NorthAmerica"] = mat.get("NorthAmerica", 0) + mat[alt]
            mat = mat.drop(columns=[alt])

    for c in BASE_CONTS:
        if c not in mat.columns:
            mat[c] = 0
    mat = mat[BASE_CONTS]

    mat["__total__"] = mat.sum(axis=1)
    mat = mat.sort_values("__total__", ascending=False)
    totals = mat["__total__"].values.copy()
    mat = mat.drop(columns="__total__")
    return mat, totals

# =========================
# Plot 1: enhanced arrows (NO numbers on paths)
# =========================
def plot_continent_arrows(flow, label_rule, out_base):
    fig, ax = plt.subplots(figsize=ARROW_FIGSIZE, constrained_layout=True)

    ax.axvline(0, color="black", lw=1.0)
    ax.axhline(0, color="black", lw=1.0)
    ax.plot([-1.7, 1.7], [-1.7, 1.7], "--", color="black", lw=1.0, alpha=0.55)

    for q in Q_ORDER:
        x, y = Q_POS[q]
        ax.scatter([x], [y], s=1600, c="white", edgecolors="#333333", linewidths=1.2, zorder=4)
        ax.text(x, y, q, ha="center", va="center", fontsize=11, weight="bold", zorder=5)

    if flow.empty:
        ax.set_title("No flows under current filters.")
        ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    else:
        flow = flow.copy()
        flow["lw"] = width_map(flow["n_city"].values)
        flow = flow.sort_values("lw", ascending=True).reset_index(drop=True)

        for _, r in flow.iterrows():
            cont = r["continent"]
            q1, q2 = r["q_start"], r["q_end"]
            lw = float(r["lw"])

            x1, y1 = Q_POS[q1]
            x2, y2 = Q_POS[q2]

            rad = CONT_RAD.get(cont, 0.0)
            col = color_of_cont(cont)

            # alpha increases with linewidth => "more vivid" for big flows
            a = alpha_from_lw(lw)

            # arrowhead size grows with linewidth => avoids "tube" look
            ms = MS_BASE + MS_PER_LW * lw

            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='-|>',
                mutation_scale=ms,
                linewidth=lw,
                color=col,
                alpha=a,
                shrinkA=SHRINK_AB, shrinkB=SHRINK_AB,
                connectionstyle=f"arc3,rad={rad}",
                zorder=3
            )
            ax.add_patch(arrow)

        handles = [Line2D([0],[0], color=color_of_cont(c), lw=4) for c in BASE_CONTS]
        ax.legend(handles, BASE_CONTS, title="Continent color", loc="upper right", frameon=True)

        # width demo (keep numbers here only)
        qmin, qmax = int(flow["n_city"].min()), int(flow["n_city"].max())
        q25, q75 = np.percentile(flow["n_city"].values, [25, 75])
        demo_vals = sorted(list(dict.fromkeys([qmin, int(round(q25)), int(round(q75)), qmax])))
        demo_lws = width_map(np.array(demo_vals, dtype=float))

        x0, y0 = -1.62, -1.58
        ax.text(x0, y0+0.20, "Width ~ city count (enhanced contrast)", fontsize=9, ha="left", va="bottom")
        for i, (v, lw_demo) in enumerate(zip(demo_vals, demo_lws)):
            yy = y0 - i*0.11
            ax.plot([x0, x0+0.33], [yy, yy], color="#444444", lw=lw_demo, solid_capstyle="round")
            ax.text(x0+0.37, yy, f"{v}", fontsize=8.5, va="center", ha="left")

        subtitle = "cross-quadrant only"
        if USE_SIGNIFICANT_ONLY:
            subtitle += f" | significant-only: {label_rule}"
        else:
            subtitle += " | all independent cities"

        ax.set_title(
            "1990→2020 Continent arrows (enhanced width contrast + alpha scaling + bigger arrowheads)\n"
            f"Color=continent; Width~count; Alpha scales with width; {subtitle}",
            fontsize=12.5
        )

        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    p_svg = out_base + "_plot1_continent_arrows_ENHANCED.svg"
    p_pdf = out_base + "_plot1_continent_arrows_ENHANCED.pdf"
    p_png = out_base + "_plot1_continent_arrows_ENHANCED.png"
    fig.savefig(p_svg, format="svg", bbox_inches="tight")
    fig.savefig(p_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(p_png, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return p_svg, p_pdf, p_png

# =========================
# Plot 2: stacked bar (unchanged)
# =========================
def plot_stacked_bar(mat, totals, out_base):
    total_all = int(np.sum(totals))

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE, constrained_layout=True)

    x = np.arange(len(mat.index))
    bottom = np.zeros(len(mat.index), dtype=float)

    for c in BASE_CONTS:
        vals = mat[c].values.astype(float)
        ax.bar(x, vals, bottom=bottom, width=0.75, color=color_of_cont(c), label=c)

        for i, v in enumerate(vals):
            t = totals[i]
            if v <= 0 or t <= 0:
                continue
            pct = 100.0 * v / t
            if pct >= 9:
                ax.text(i, bottom[i] + v/2.0, f"{pct:.0f}%", ha="center", va="center",
                        fontsize=8, color="white")
        bottom += vals

    for i, t in enumerate(totals):
        if t <= 0:
            continue
        pct_all = 100.0 * t / total_all if total_all > 0 else 0.0
        ax.text(i + 0.42, t, f"{pct_all:.1f}% of all", ha="left", va="center", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(mat.index.tolist(), rotation=45, ha="right", fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Number of cities")
    ax.set_xlabel("Transition type (sorted)")

    title = "1990→2020 Independent-city transitions (stacked by continent, sorted high→low)"
    if USE_SIGNIFICANT_ONLY:
        title = "1990→2020 Significant independent-city transitions (stacked by continent, sorted high→low)"
    ax.set_title(title, fontsize=13)
    ax.legend(title="Continent", loc="upper right", frameon=True)

    p_svg = out_base + "_plot2_transition_stacked_sorted.svg"
    p_pdf = out_base + "_plot2_transition_stacked_sorted.pdf"
    p_png = out_base + "_plot2_transition_stacked_sorted.png"
    fig.savefig(p_svg, format="svg", bbox_inches="tight")
    fig.savefig(p_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(p_png, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return p_svg, p_pdf, p_png, total_all

# =========================
# Main
# =========================
def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    city_all = build_independent_city_table(df)
    city_used, label_rule = apply_significant_filter(city_all)

    flow = build_flow(city_used)
    mat, totals = build_transition_by_continent(city_used)

    out_base = os.path.join(OUT_DIR, "1990_2020")

    p1 = plot_continent_arrows(flow, label_rule, out_base)
    p2 = plot_stacked_bar(mat, totals, out_base)

    print("DONE.")
    print("OUT_DIR:", OUT_DIR)
    print("All cross-quadrant cities:", len(city_all))
    print("Used cities:", len(city_used))
    print("Arrow plot:", p1)
    print("Stacked plot:", p2[:3])

if __name__ == "__main__":
    main()