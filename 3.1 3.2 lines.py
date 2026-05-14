# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

# =========================
# 0) PATHS / SETTINGS
# =========================
INPUT_CSV = r"C:\Users\steve\Desktop\最新结论\Result 3 四象限\LOG_outputs_quadrants_椭圆阈值1.0\labels_yearly_UGSrel_NLIrel_1.00.csv"
OUT_DIR   = r"C:\Users\steve\Desktop\最新结论\3.1 & 3.2\supplementary"
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
ALPHA = 0.5  # 50% transparency for fills

# cluster 图太乱：默认只画城市数最多的前 26 个 cluster；要画全部设为 None
TOP_N_CLUSTERS = 26

# 你的大洲配色（50%透明）
BASE_COLORS = {
    "AF":"#B22222",
    "AS":"#DAA520",
    "EU":"#4F81BD",
    "OC":"#708090",
    "SA":"#F4A460",
    "USA":"#2E8B57",
}

# =========================
# 1) LOAD & DETECT COLUMNS
# =========================
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)

# required targets
for c in ["Year", "UGSrel", "NLIrel"]:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# detect continent column
continent_candidates = ["continent_UGS", "continent", "continent_NLI"]
continent_col = next((c for c in continent_candidates if c in df.columns), None)
if continent_col is None:
    raise ValueError(f"Cannot find continent column in {continent_candidates}. Existing: {list(df.columns)}")

# detect cluster column
cluster_candidates = ["cluster_number", "cluster", "cluster_UGS", "cluster_NLI", "cluster_id"]
cluster_col = next((c for c in cluster_candidates if c in df.columns), None)
if cluster_col is None:
    raise ValueError(f"Cannot find cluster column in {cluster_candidates}. Existing: {list(df.columns)}")

# clean
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
df = df.dropna(subset=["Year", "UGSrel", "NLIrel", continent_col, cluster_col]).copy()
df["Year"] = df["Year"].astype(int)
df = df[df["Year"].isin(YEARS)].copy()

df["continent"] = df[continent_col].astype(str).str.strip()
df["cluster"]   = df[cluster_col].astype(str).str.strip()

# Map NA/NorthAmerica to USA color if needed
if "NA" in df["continent"].unique() and "USA" in BASE_COLORS:
    BASE_COLORS["NA"] = BASE_COLORS["USA"]
if "NorthAmerica" in df["continent"].unique() and "USA" in BASE_COLORS:
    BASE_COLORS["NorthAmerica"] = BASE_COLORS["USA"]

PALETTE_ALPHA = {k: to_rgba(v, ALPHA) for k, v in BASE_COLORS.items()}

# =========================
# 2) UTILITIES
# =========================
sns.set_style("whitegrid")

def order_groups_by_1990_median(d, group_col, value_col):
    d1990 = d[d["Year"] == 1990]
    order = (d1990.groupby(group_col)[value_col]
                  .median()
                  .sort_values(ascending=False)
                  .index.tolist())
    return order

def plot_1990_violin_box_points(d, group_col, value_col, title, outpath, hue_col=None, palette=None):
    """
    1990 spatial differences: violin + box + points, sorted high->low.
    - group_col: "continent" or "cluster"
    - hue_col: for coloring. For continent plot, hue_col = continent. For cluster plot, hue_col = continent too.
    """
    d1990 = d[d["Year"] == 1990].copy()
    order = order_groups_by_1990_median(d1990, group_col, value_col)

    plt.figure(figsize=(18, 6))
    # violin (light, translucent)
    sns.violinplot(
        data=d1990, x=group_col, y=value_col,
        order=order,
        hue=hue_col if hue_col else None,
        dodge=False,
        palette=palette,
        inner=None, cut=0, linewidth=0.8
    )
    # box overlay (no fill)
    sns.boxplot(
        data=d1990, x=group_col, y=value_col,
        order=order,
        hue=hue_col if hue_col else None,
        dodge=False,
        palette=palette,
        width=0.18, showfliers=False,
        boxprops={"facecolor":"none", "linewidth":1.2},
        whiskerprops={"linewidth":1.2},
        medianprops={"linewidth":1.5}
    )
    # points overlay (strip)
    sns.stripplot(
        data=d1990, x=group_col, y=value_col,
        order=order,
        color="black", alpha=0.18, size=1.5, jitter=0.25
    )

    plt.axhline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.8)
    plt.title(title)
    plt.xlabel("")
    plt.ylabel(value_col)
    plt.xticks(rotation=90)
    # remove repeated legend created by violin+box
    if plt.gca().get_legend() is not None:
        plt.gca().get_legend().remove()

    # custom legend (continents)
    if hue_col and palette:
        conts = [c for c in BASE_COLORS.keys() if c in d1990[hue_col].unique()]
        handles = [Patch(facecolor=palette[c], edgecolor="black", label=c) for c in conts]
        plt.legend(handles=handles, loc="upper right", frameon=False)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def compute_yearly_stats(d, group_col, value_col):
    """
    per year × group: mean, sd, q25,q75,q10,q90 -> IQR and P90-P10
    """
    g = d.groupby(["Year", group_col])[value_col]
    stats = g.agg(
        mean="mean",
        sd="std",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        q10=lambda s: s.quantile(0.10),
        q90=lambda s: s.quantile(0.90),
        n="count"
    ).reset_index()
    stats["IQR"] = stats["q75"] - stats["q25"]
    stats["P90_P10"] = stats["q90"] - stats["q10"]
    return stats

def plot_time_inequality_panel(stats, group_col, title_prefix, outpath, palette):
    """
    2x2 panel: IQR, P90-P10, mean, sd lines over time; one line per group colored by palette
    """
    metrics = [("IQR", "IQR"), ("P90_P10", "P90–P10"), ("mean", "Mean"), ("sd", "SD")]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    groups = stats[group_col].astype(str).unique().tolist()
    groups = sorted(groups)

    for ax, (col, label) in zip(axes, metrics):
        for g in groups:
            sub = stats[stats[group_col].astype(str) == g].sort_values("Year")
            color = palette.get(g, "#666666")
            ax.plot(sub["Year"], sub[col], linewidth=1.6, alpha=0.9, color=color)
        ax.set_title(label)
        ax.set_xlabel("Year")
        ax.set_ylabel(label)
        ax.set_xticks(YEARS)

    # Legend at bottom, not covering titles
    handles = [Patch(facecolor=palette.get(g, "#666666"), edgecolor="none", label=g) for g in groups if g in palette]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title_prefix, y=1.02, fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# =========================
# 3) PREP: continent & cluster palettes
# =========================
# continent palette (RGBA)
cont_palette = PALETTE_ALPHA.copy()

# cluster -> continent mapping (mode)
cluster_to_cont = (df.groupby("cluster")["continent"]
                     .agg(lambda s: s.value_counts().index[0])
                     .to_dict())

# decide cluster list
if TOP_N_CLUSTERS is None:
    cluster_keep = sorted(df["cluster"].unique().tolist())
else:
    # top clusters by #unique cities (if city column exists), else by count
    if "city" in df.columns:
        counts = df.groupby("cluster")["city"].nunique().sort_values(ascending=False)
    else:
        counts = df["cluster"].value_counts()
    cluster_keep = counts.head(TOP_N_CLUSTERS).index.tolist()

df_cluster = df[df["cluster"].isin(cluster_keep)].copy()

# cluster palette (each cluster colored by its continent color)
cluster_palette = {cl: cont_palette.get(cluster_to_cont.get(cl, ""), to_rgba("#999999", ALPHA)) for cl in cluster_keep}

# =========================
# 4) OUTPUTS
# =========================

# ---- A) CONTINENT-BASED: 1990 spatial + time inequality lines ----
for var in ["UGSrel", "NLIrel"]:
    out1 = os.path.join(OUT_DIR, f"Continent_1990_violin_box_points_{var}.png")
    plot_1990_violin_box_points(
        df, group_col="continent", value_col=var,
        title=f"1990 {var} spatial differences by continent (sorted high→low; 50% alpha)",
        outpath=out1,
        hue_col="continent",
        palette=cont_palette
    )

    stats = compute_yearly_stats(df, "continent", var)
    out2 = os.path.join(OUT_DIR, f"Continent_TimeMetrics_{var}_IQR_P90P10_Mean_SD.png")
    plot_time_inequality_panel(
        stats, "continent",
        title_prefix=f"{var} over time by continent: IQR / P90–P10 / Mean / SD",
        outpath=out2,
        palette={k: cont_palette[k] for k in cont_palette if k in df["continent"].unique()}
    )

# ---- B) CLUSTER-BASED: 1990 spatial + time inequality lines ----
# Spatial plot: x=cluster (sorted high->low), colored by continent (one color per cluster)
for var in ["UGSrel", "NLIrel"]:
    out3 = os.path.join(OUT_DIR, f"Cluster_1990_violin_box_points_{var}_coloredByContinent.png")
    # IMPORTANT: to color each cluster by its continent, we set hue to continent and dodge=False
    plot_1990_violin_box_points(
        df_cluster, group_col="cluster", value_col=var,
        title=f"1990 {var} spatial differences by cluster (sorted high→low; colored by continent; 50% alpha)",
        outpath=out3,
        hue_col="continent",
        palette=cont_palette
    )

    stats_cl = compute_yearly_stats(df_cluster, "cluster", var)
    out4 = os.path.join(OUT_DIR, f"Cluster_TimeMetrics_{var}_IQR_P90P10_Mean_SD_TOP{TOP_N_CLUSTERS}.png")
    # Note: legend would be huge for 26 clusters; here we color lines by continent but do not legend every cluster.
    # We'll plot with continent colors using alpha; keep legend off for readability.
    # Build a plotting palette for clusters (cluster -> RGBA)
    plot_time_inequality_panel(
        stats_cl, "cluster",
        title_prefix=f"{var} over time by cluster: IQR / P90–P10 / Mean / SD (top {TOP_N_CLUSTERS})",
        outpath=out4,
        palette=cluster_palette
    )

print("Done. Saved to:", OUT_DIR)
print("continent column:", continent_col, "cluster column:", cluster_col)
