# -*- coding: utf-8 -*-
"""
Scheme2 (POOLED 1990–2020):
- Country center = median(UGSrel), median(NLIrel)
- Bubble size = number of cities per country (n)
- Labels = union of:
    (1) top by dispersion (r50)
    (2) top by city count (n)
    (3) closest to 45° line (balanced)
    (4) high UGSrel (x_med) but not necessarily balanced
- Balanced + High-UGS markers: SAME style (black ring + bold black label)
- No extra legend for rings
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================
# CONFIG
# =========================
DATA_PATH = r"C:\Users\steve\Desktop\最新结论\Result 3 四象限\LOG_outputs_quadrants_椭圆阈值1.0\labels_yearly_UGSrel_NLIrel_1.00.csv"
OUT_DIR   = r"C:\Users\steve\Desktop\最新结论\Result 3 四象限\泡泡图（中心 + 异质性半径）\最终——调节颜色"
os.makedirs(OUT_DIR, exist_ok=True)

YEAR_COL    = "Year"
COUNTRY_COL = "country"
CONT_COL    = "continent_UGS"
X_COL       = "UGSrel"
Y_COL       = "NLIrel"

MIN_N = 30

ALPHA_BUBBLE = 0.6
TOP_LABEL_N = 50
TOP_LABEL_N_BY_CITYCOUNT = 30

# NEW
TOP_LABEL_BALANCED = 15     # closest to 45° line
TOP_LABEL_HIGH_UGS = 15     # highest x_med (UGSrel median)
RING_SCALE = 1.10           # ring size multiplier
RING_COLOR = "black"        # SAME marker style for both

P_CLIP = 99.0
GAMMA  = 0.70
S_MIN  = 26
S_MAX  = 1700

SAVE_PNG = True
DPI_PNG  = 300

CONT_COLORS = {
    "AF":  "#B22222",
    "AS":  "#8B4513",
    "EU":  "#4F81BD",
    "OC":  "#708090",
    "SA":  "#F4A460",
    "NorthAmerica": "#2E8B57",
}
def color_for_cont(c):
    return CONT_COLORS.get(str(c), "#999999")

# =========================
# Helpers
# =========================
def signed_log1p(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))

def add_screen_45deg_line(ax):
    ox, oy = ax.transAxes.inverted().transform(ax.transData.transform((0, 0)))
    b = oy - ox
    pts = []
    y_at_x0 = b
    y_at_x1 = 1 + b
    if 0 <= y_at_x0 <= 1: pts.append((0, y_at_x0))
    if 0 <= y_at_x1 <= 1: pts.append((1, y_at_x1))
    x_at_y0 = -b
    x_at_y1 = 1 - b
    if 0 <= x_at_y0 <= 1: pts.append((x_at_y0, 0))
    if 0 <= x_at_y1 <= 1: pts.append((x_at_y1, 1))
    if len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[0], pts[1]
        ax.plot([x1, x2], [y1, y2], transform=ax.transAxes,
                linestyle="--", linewidth=1)

def country_centers_and_stats(df_any):
    d = df_any[[COUNTRY_COL, CONT_COL, X_COL, Y_COL]].copy()
    d[X_COL] = pd.to_numeric(d[X_COL], errors="coerce")
    d[Y_COL] = pd.to_numeric(d[Y_COL], errors="coerce")
    d = d.dropna()

    centers = (
        d.groupby([COUNTRY_COL, CONT_COL], as_index=False)
         .agg(n=(X_COL, "size"),
              x_med=(X_COL, "median"),
              y_med=(Y_COL, "median"))
    )

    d = d.merge(
        centers[[COUNTRY_COL, CONT_COL, "x_med", "y_med"]],
        on=[COUNTRY_COL, CONT_COL], how="left"
    )
    d["dist"] = np.sqrt((d[X_COL] - d["x_med"])**2 + (d[Y_COL] - d["y_med"])**2)

    r50 = (
        d.groupby([COUNTRY_COL, CONT_COL], as_index=False)["dist"]
         .median()
         .rename(columns={"dist": "r50"})
    )

    out = centers.merge(r50, on=[COUNTRY_COL, CONT_COL], how="left")
    out["x_t"] = signed_log1p(out["x_med"].values)
    out["y_t"] = signed_log1p(out["y_med"].values)
    return out

def bubble_area_from_n(n_arr, p_clip=P_CLIP, gamma=GAMMA, s_min=S_MIN, s_max=S_MAX):
    n_arr = np.asarray(n_arr, dtype=float)
    s = np.full_like(n_arr, s_min, dtype=float)

    m = np.isfinite(n_arr) & (n_arr >= 0)
    if not np.any(m):
        return s

    x = n_arr[m]
    lo = np.nanmin(x)
    hi = np.nanpercentile(x, p_clip)

    if hi - lo < 1e-12:
        s[m] = (s_min + s_max) / 2.0
        return s

    x = np.clip(x, lo, hi)
    z = (x - lo) / (hi - lo + 1e-12)
    z = z ** gamma
    s[m] = s_min + (s_max - s_min) * z
    return s

def select_labels_union(tab):
    a = tab.sort_values("r50", ascending=False).head(TOP_LABEL_N)
    b = tab.sort_values("n", ascending=False).head(TOP_LABEL_N_BY_CITYCOUNT)
    c = tab.sort_values("d45", ascending=True).head(TOP_LABEL_BALANCED)
    d = tab.sort_values("x_med", ascending=False).head(TOP_LABEL_HIGH_UGS)
    lab = pd.concat([a, b, c, d], axis=0).drop_duplicates(subset=[COUNTRY_COL]).copy()
    lab = lab.sort_values(["n", "r50"], ascending=[False, False]).reset_index(drop=True)
    return lab

def continent_box_stats(tab):
    rows = []
    for cont, g in tab.groupby(CONT_COL):
        vals = g["r50"].dropna().values
        if len(vals) == 0:
            continue
        q1 = np.percentile(vals, 25)
        med = np.percentile(vals, 50)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1
        low_thr = q1 - 1.5 * iqr
        high_thr = q3 + 1.5 * iqr
        low = np.min(vals[vals >= low_thr]) if np.any(vals >= low_thr) else np.min(vals)
        high = np.max(vals[vals <= high_thr]) if np.any(vals <= high_thr) else np.max(vals)
        rows.append({
            CONT_COL: cont,
            "q1": q1, "med": med, "q3": q3,
            "low": low, "high": high,
            "n_country": len(vals)
        })
    return pd.DataFrame(rows)

def draw_box_legend_on_ax(ax, box_df):
    if box_df.empty:
        return

    iax = ax.inset_axes([1.02, 0.10, 0.25, 0.82])
    iax.set_title("Continent dispersion\n(box on country r50)", fontsize=9)

    box_df = box_df.sort_values("med", ascending=False).reset_index(drop=True)
    ymin = np.nanmin(box_df["low"].values)
    ymax = np.nanmax(box_df["high"].values)
    pad = 0.08 * (ymax - ymin + 1e-12)
    iax.set_ylim(ymin - pad, ymax + pad)

    x_positions = np.arange(len(box_df))
    box_w = 0.52

    for i, r in box_df.iterrows():
        c = r[CONT_COL]
        col = color_for_cont(c)

        iax.plot([i, i], [r["low"], r["high"]], color=col, linewidth=1.2)

        rect = Rectangle((i - box_w / 2, r["q1"]),
                         box_w, r["q3"] - r["q1"],
                         facecolor=col, edgecolor=col, alpha=0.35, linewidth=1.2)
        iax.add_patch(rect)

        iax.plot([i - box_w / 2, i + box_w / 2], [r["med"], r["med"]],
                 color=col, linewidth=2)

    iax.set_xticks(x_positions)
    iax.set_xticklabels(box_df[CONT_COL].tolist(), fontsize=8)
    iax.set_ylabel("r50 dispersion", fontsize=8)
    iax.tick_params(axis="y", labelsize=8)

def draw_bubble_size_legend(ax, n_values=(30, 100, 300, 700)):
    n_values = np.array(n_values, dtype=float)
    s_values = bubble_area_from_n(n_values)

    handles, labels = [], []
    for n, s in zip(n_values, s_values):
        h = ax.scatter([], [], s=s, facecolors='none',
                       edgecolors='black', linewidths=1.0, alpha=0.9)
        handles.append(h)
        labels.append(f"n={int(n)}")

    leg = ax.legend(
        handles, labels,
        title="Bubble size\n(city count)",
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        fontsize=8,
        title_fontsize=8,
        borderpad=0.8,
        labelspacing=1.2,
        handletextpad=1.0,
        scatterpoints=1
    )
    ax.add_artist(leg)

# =========================
# Plot
# =========================
def plot_scheme2_pooled(df):
    tab = country_centers_and_stats(df)
    tab["msize"] = bubble_area_from_n(tab["n"].values)

    fig, ax = plt.subplots(figsize=(12.4, 8.2), constrained_layout=True)

    xmin, xmax = np.nanpercentile(tab["x_t"], [1, 99])
    ymin, ymax = np.nanpercentile(tab["y_t"], [1, 99])
    dx = xmax - xmin
    dy = ymax - ymin
    xlim0, xlim1 = (xmin - 0.10 * dx, xmax + 0.10 * dx)
    ylim0, ylim1 = (ymin - 0.10 * dy, ymax + 0.10 * dy)
    ax.set_xlim(xlim0, xlim1)
    ax.set_ylim(ylim0, ylim1)

    # d45 in axes-normalized space
    x_norm = (tab["x_t"].values - xlim0) / (xlim1 - xlim0 + 1e-12)
    y_norm = (tab["y_t"].values - ylim0) / (ylim1 - ylim0 + 1e-12)
    tab["d45"] = np.abs(y_norm - x_norm) / np.sqrt(2.0)

    ax.axvline(0, linewidth=1)
    ax.axhline(0, linewidth=1)
    add_screen_45deg_line(ax)

    # scatter
    for cont, g in tab.groupby(CONT_COL):
        col = color_for_cont(cont)
        big = g[g["n"] >= MIN_N]
        small = g[g["n"] < MIN_N]

        if not big.empty:
            ax.scatter(big["x_t"], big["y_t"], s=big["msize"],
                       c=col, alpha=ALPHA_BUBBLE, edgecolors=col, linewidths=0.8)
        if not small.empty:
            ax.scatter(small["x_t"], small["y_t"], s=18,
                       c=col, alpha=ALPHA_BUBBLE, edgecolors="none")

    # balanced + high-UGS sets
    balanced = tab.sort_values("d45", ascending=True).head(TOP_LABEL_BALANCED).copy()

    high_ugs_pos = tab[tab["x_med"] > 0].sort_values("x_med", ascending=False)
    if len(high_ugs_pos) >= TOP_LABEL_HIGH_UGS:
        high_ugs = high_ugs_pos.head(TOP_LABEL_HIGH_UGS).copy()
    else:
        high_ugs = tab.sort_values("x_med", ascending=False).head(TOP_LABEL_HIGH_UGS).copy()

    # SAME ring style for both (union)
    ring_set = pd.concat([balanced, high_ugs], axis=0).drop_duplicates(subset=[COUNTRY_COL]).copy()
    if not ring_set.empty:
        ax.scatter(ring_set["x_t"], ring_set["y_t"],
                   s=ring_set["msize"] * RING_SCALE,
                   facecolors="none", edgecolors=RING_COLOR, linewidths=1.5, alpha=0.95)

    # labels union
    lab = select_labels_union(tab)

    top_n_countries = set(tab.sort_values("n", ascending=False)
                          .head(TOP_LABEL_N_BY_CITYCOUNT)[COUNTRY_COL].tolist())
    ring_countries = set(ring_set[COUNTRY_COL].tolist())

    for _, r in lab.iterrows():
        ctry = r[COUNTRY_COL]
        fw = "bold" if (ctry in top_n_countries or ctry in ring_countries) else "normal"
        ax.text(r["x_t"] + 0.02, r["y_t"] + 0.02, str(ctry),
                fontsize=8, fontweight=fw, color="black" if ctry in ring_countries else "#111111",
                ha="left", va="bottom")

    ax.set_title(
        "scheme2_POOLED_1990_2020_scatter_box_legend\n"
        "labels=top r50 ∪ top n ∪ closest to 45° ∪ high UGS (same black ring; no ring legend)",
        fontsize=12
    )
    ax.set_xlabel("signed-log1p(UGSrel median)")
    ax.set_ylabel("signed-log1p(NLIrel median)")

    # Continent legend (keep)
    conts = sorted(tab[CONT_COL].dropna().unique().tolist())
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color_for_cont(c), markersize=8)
        for c in conts
    ]
    leg1 = ax.legend(handles, conts, title="Continent", loc="upper left", frameon=True)
    ax.add_artist(leg1)

    # Bubble size legend (keep)
    draw_bubble_size_legend(ax, n_values=(30, 100, 300, 700))

    # Box legend (keep)
    box_df = continent_box_stats(tab)
    draw_box_legend_on_ax(ax, box_df)

    # Save
    base = os.path.join(OUT_DIR, "scheme2_POOLED_1990_2020_scatter_box_legend")
    fig.savefig(base + ".svg", format="svg", bbox_inches="tight")
    fig.savefig(base + ".pdf", format="pdf", bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(base + ".png", dpi=DPI_PNG, bbox_inches="tight")

    # Export tables
    tab.sort_values(["n", "r50"], ascending=[False, False]).to_csv(
        base + "_country_table.csv", index=False, encoding="utf-8-sig"
    )
    ring_set[[COUNTRY_COL, CONT_COL, "n", "x_med", "y_med", "r50", "d45", "x_t", "y_t"]].to_csv(
        base + "_marked_balanced_or_highUGS.csv", index=False, encoding="utf-8-sig"
    )

    plt.close(fig)

def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce")
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")
    df = df.dropna(subset=[YEAR_COL, COUNTRY_COL, CONT_COL, X_COL, Y_COL]).copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)

    pooled = df[(df[YEAR_COL] >= 1990) & (df[YEAR_COL] <= 2020)].copy()
    plot_scheme2_pooled(pooled)
    print("DONE.")

if __name__ == "__main__":
    main()