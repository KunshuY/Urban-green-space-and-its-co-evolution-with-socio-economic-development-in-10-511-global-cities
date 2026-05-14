# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.scale import FuncScale

# =========================
# 0) PATHS / SETTINGS
# =========================
INPUT_CSV = r"C:\Users\steve\Desktop\最新结论\Result 3 四象限\LOG_outputs_quadrants_椭圆阈值1.0\labels_yearly_UGSrel_NLIrel_1.00.csv"
OUT_DIR   = r"C:\Users\steve\Desktop\最新结论\3.1 & 3.2\compress_only_upper_tail_positive"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_YEAR = 2020
ALPHA = 0.5
FIG_SIZE = (7.2, 7.2)

BASE_COLORS = {
    "AF": "#B22222",
    "AS": "#8B5A2B",
    "EU": "#4F81BD",
    "OC": "#4D4D4D",
    "SA": "#F39C12",
    "NA": "#2E8B57",
}
CONT_ALLOWED = ["AF", "AS", "EU", "OC", "SA", "NA"]

# -------------------------
# 只压缩正方向极高值
# -------------------------
COMPRESS_Y_FOR = ["NLIrel"]

HIGH_Y_THRESHOLD = {
    "NLIrel": 12.0,
}

HIGH_Y_COMPRESS_FACTOR = {
    "NLIrel": 4.0,
}

YTICKS_RAW = {
    "NLIrel": [-5, 0, 5, 10, 15, 20, 30, 40, 50, 60]
}

# =========================
# 1) LOAD & CLEAN
# =========================
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)

for c in ["Year", "UGSrel", "NLIrel"]:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

continent_candidates = ["continent_UGS", "continent", "continent_NLI"]
continent_col = next((c for c in continent_candidates if c in df.columns), None)
if continent_col is None:
    raise ValueError(
        f"Cannot find continent column in {continent_candidates}. Existing: {list(df.columns)}"
    )

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year", "UGSrel", "NLIrel", continent_col]).copy()
df["Year"] = df["Year"].astype(int)

df["continent"] = df[continent_col].astype(str).str.strip()
df["continent"] = df["continent"].replace({
    "NorthAmerica": "NA",
    "North America": "NA",
    "northamerica": "NA",
    "NORTHAMERICA": "NA",
    "Americas": "NA",
    "USA": "NA"
})

df = df[df["Year"] == TARGET_YEAR].copy()
df = df[df["continent"].isin(CONT_ALLOWED)].copy()

palette_alpha = {k: to_rgba(v, ALPHA) for k, v in BASE_COLORS.items()}

# =========================
# 2) UTILITIES
# =========================
sns.set_style("whitegrid")

def pretty_var_label(var_name):
    return "NTL" if var_name == "NLIrel" else "UGS"

def order_by_median_desc(data, group_col, value_col):
    return (
        data.groupby(group_col)[value_col]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
    )

def make_upper_tail_compress_scale(threshold, factor):
    """
    仅压缩 y > threshold 的正向高值：
    - y <= threshold: 保持原样（包括所有负值、0附近、主体部分）
    - y > threshold : 用 log1p 压缩
    """
    def forward(y):
        y = np.asarray(y, dtype=float)
        out = y.copy()
        mask = y > threshold
        out[mask] = threshold + factor * np.log1p((y[mask] - threshold) / factor)
        return out

    def inverse(z):
        z = np.asarray(z, dtype=float)
        out = z.copy()
        mask = z > threshold
        out[mask] = threshold + factor * np.expm1((z[mask] - threshold) / factor)
        return out

    return forward, inverse

def apply_compressed_yaxis(ax, data, value_col):
    """
    只压缩正向极高值：
    - 0线以下正常显示
    - 0附近不拉伸
    - 主体部分尽量保持线性
    """
    if value_col not in COMPRESS_Y_FOR:
        return

    threshold = HIGH_Y_THRESHOLD.get(value_col, 12.0)
    factor = HIGH_Y_COMPRESS_FACTOR.get(value_col, 4.0)

    forward, inverse = make_upper_tail_compress_scale(threshold, factor)
    ax.set_yscale("function", functions=(forward, inverse))

    raw_ticks = YTICKS_RAW.get(value_col, None)
    if raw_ticks is not None:
        ax.set_yticks(raw_ticks)
        ax.set_yticklabels([str(t) for t in raw_ticks])

def plot_continent_violin_box(data, value_col, outbase):
    d = data.copy()
    order = order_by_median_desc(d, "continent", value_col)
    var_label = pretty_var_label(value_col)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    sns.violinplot(
        data=d,
        x="continent",
        y=value_col,
        order=order,
        hue="continent",
        dodge=False,
        palette=palette_alpha,
        inner=None,
        cut=1.2,
        linewidth=0.35,
        width=0.28,
        ax=ax
    )

    sns.boxplot(
        data=d,
        x="continent",
        y=value_col,
        order=order,
        hue="continent",
        dodge=False,
        palette=palette_alpha,
        width=0.045,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 2.2,
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "alpha": 0.35,
            "markeredgewidth": 0.2
        },
        boxprops={"facecolor": "none", "linewidth": 0.9},
        whiskerprops={"linewidth": 0.85},
        capprops={"linewidth": 0.85},
        medianprops={"linewidth": 1.05, "color": "black"},
        ax=ax
    )

    apply_compressed_yaxis(ax, d, value_col)

    ax.axhline(0, linestyle="--", linewidth=0.9, color="black", alpha=0.75)

    if value_col in COMPRESS_Y_FOR:
        ax.set_title(
            f"{TARGET_YEAR} {var_label} by continent (sorted high→low; upper positive tail compressed)"
        )
    else:
        ax.set_title(f"{TARGET_YEAR} {var_label} by continent (sorted high→low)")

    ax.set_xlabel("")
    ax.set_ylabel(var_label)
    ax.tick_params(axis="x", rotation=0)
    ax.set_box_aspect(1)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    handles = [
        Patch(facecolor=palette_alpha[c], edgecolor="black", label=c)
        for c in order
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)

    plt.tight_layout()
    plt.savefig(outbase + ".pdf", bbox_inches="tight")
    plt.savefig(outbase + ".svg", bbox_inches="tight")
    plt.close(fig)

# =========================
# 3) OUTPUTS
# =========================
for var in ["UGSrel", "NLIrel"]:
    label = pretty_var_label(var)
    outbase = os.path.join(
        OUT_DIR,
        f"{TARGET_YEAR}_Continent_violin_box_{label}_sorted_upperPositiveTailCompressed"
    )
    plot_continent_violin_box(df, var, outbase)

print("Done. Saved vector files (.pdf and .svg) to:", OUT_DIR)
print("continent column used:", continent_col)