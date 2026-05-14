# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ========================
# I/O paths
# ========================
IN_PATH = r"C:\Users\steve\Desktop\最新结论\3.4回归数据2_加入rel.csv"
OUT_DIR = r"C:\Users\steve\Desktop\3.4\3.4.1rel_G+U"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_XLSX   = os.path.join(OUT_DIR, "01_Level_Model_GU_parallel_OLS.xlsx")
OUT_TXT    = os.path.join(OUT_DIR, "01_Level_Model_GU_parallel_OLS_summary.txt")
OUT_STDCSV = os.path.join(OUT_DIR, "01_Level_Model_GU_stdcoef.csv")
OUT_CLEAN  = os.path.join(OUT_DIR, "01_Level_Model_GU_cleaned_used.csv")

# Vector/raster figure output base names
OUT_FIG_ALL_BASE  = os.path.join(OUT_DIR, "Fig_3_4A_Level_GU_stdcoef")
OUT_FIG_ZOOM_BASE = os.path.join(OUT_DIR, "Fig_3_4A_Level_GU_stdcoef_zoom")

# ========================
# Matplotlib vector export settings
# ========================
# Keep text editable in PDF/SVG where possible
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

# ========================
# Columns
# ========================
CITY_ID = "ID_UGS"
PERIOD  = "period"

# -------- Level DV choice --------
# Use "t1" (recommended) or "t0"
LEVEL_WHICH = "t1"

# Candidate DV names (auto-detected below)
Y_G_CAND = [f"UGSrel_{LEVEL_WHICH}", "UGSrel"]
Y_U_CAND = [f"NLIrel_{LEVEL_WHICH}", "NLIrel"]

X_CAT = ["biome", "Soil", "Level", "catchment", "development", "income", "climate"]
X_CONT_BASE = ["bucap_mean", "gdpp_mean", "temp_mean", "rain_mean"]

ALPHA_SIG = 0.05

# ========================
# Helper: PCA pollution_pc1
# ========================
def add_pollution_pc1(df, poll_vars=("pm_mean", "c_mean"), out_col="pollution_pc1"):
    if out_col in df.columns and df[out_col].notna().any():
        return df, None

    df2 = df.dropna(subset=list(poll_vars)).copy()
    X = df2[list(poll_vars)].astype(float).values
    Xz = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=0).fit(Xz)
    pc1 = pca.transform(Xz)[:, 0]

    corr_pm = np.corrcoef(pc1, df2[poll_vars[0]].values)[0, 1]
    corr_c  = np.corrcoef(pc1, df2[poll_vars[1]].values)[0, 1]

    if (corr_pm < 0) and (corr_c < 0):
        pc1 = -pc1

    df2[out_col] = pc1
    poll_city = df2.groupby(CITY_ID)[out_col].mean()
    df[out_col] = df[CITY_ID].map(poll_city)

    return df, pca

# ========================
# OLS with city-cluster SE
# ========================
def fit_ols(df, y_col, formula_rhs):
    formula = f"{y_col} ~ {formula_rhs}"
    model = sm.OLS.from_formula(formula, data=df)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df[CITY_ID]})
    return res, formula

def coef_table(res):
    return pd.DataFrame({
        "coef": res.params,
        "se_cluster_city": res.bse,
        "t": res.tvalues,
        "p": res.pvalues
    })

# ========================
# Standardized coefficient table: beta*
# ========================
def standardized_effects(res, y_series, cont_names):
    y_sd = float(np.nanstd(y_series.astype(float).values, ddof=1))

    if not np.isfinite(y_sd) or y_sd == 0:
        raise ValueError("y has zero/invalid standard deviation; cannot standardize.")

    X_design = pd.DataFrame(res.model.exog, columns=res.model.exog_names)

    rows = []

    for term in res.params.index:
        if term == "Intercept" or term.startswith(f"C({PERIOD})"):
            continue

        beta = float(res.params[term])
        se   = float(res.bse[term])

        lo = beta - 1.96 * se
        hi = beta + 1.96 * se

        if term in cont_names:
            sd_x = float(np.nanstd(X_design[term].values, ddof=1))
        else:
            if term in X_design.columns:
                p = float(np.nanmean(X_design[term].values))
                sd_x = float(np.sqrt(p * (1 - p))) if np.isfinite(p) else np.nan
            else:
                sd_x = np.nan

        if not np.isfinite(sd_x) or sd_x == 0:
            beta_star = lo_star = hi_star = np.nan
        else:
            s = sd_x / y_sd
            beta_star = beta * s
            lo_star = lo * s
            hi_star = hi * s

        pval = float(res.pvalues[term])

        rows.append({
            "term": term,
            "beta": beta,
            "se_cluster_city": se,
            "p": pval,
            "beta_star": beta_star,
            "ci_low_star": lo_star,
            "ci_high_star": hi_star,
            "sig": (pval < ALPHA_SIG)
        })

    return pd.DataFrame(rows)

# ========================
# Order matching your S/M figure
# ========================
def build_sm_like_order(all_terms):
    order = []

    for b in ["3.0", "4.0", "5.0", "6.0", "7.0", "8.0"]:
        order.append(f"C(biome)[T.{b}]")

    for s in ["2.0", "3.0"]:
        order.append(f"C(Soil)[T.{s}]")

    for lv in ["2", "3", "4", "5", "6"]:
        order.append(f"C(Level)[T.{lv}]")

    order.append("C(catchment)[T.1]")

    for inc in ["2.0", "3.0", "4.0"]:
        order.append(f"C(income)[T.{inc}]")

    for dv in ["2", "3"]:
        order.append(f"C(development)[T.{dv}]")

    order.append("C(climate)[T.continental]")
    order.append("C(climate)[T.temperate]")
    order.append("C(climate)[T.tropical]")

    order += [
        "bucap_mean",
        "gdpp_mean",
        "temp_mean",
        "rain_mean",
        "pollution_pc1"
    ]

    seen = set(order)
    extras = [t for t in all_terms if t not in seen]

    return order + extras

# ========================
# Forest plot: Model G vs Model U
# Export PNG + PDF + SVG
# ========================
def forest_plot_GU(stdG, stdU, out_base, title, zoom_terms=None, labG="Model G", labU="Model U"):
    g = stdG[["term", "beta_star", "ci_low_star", "ci_high_star", "p", "sig"]].copy()
    u = stdU[["term", "beta_star", "ci_low_star", "ci_high_star", "p", "sig"]].copy()

    g = g.rename(columns={
        "beta_star": "betaG",
        "ci_low_star": "loG",
        "ci_high_star": "hiG",
        "sig": "sigG"
    })

    u = u.rename(columns={
        "beta_star": "betaU",
        "ci_low_star": "loU",
        "ci_high_star": "hiU",
        "sig": "sigU"
    })

    m = pd.merge(g, u, on="term", how="outer")

    if zoom_terms is not None:
        m = m[m["term"].isin(zoom_terms)].copy()

    m = m.dropna(subset=["betaG", "betaU"], how="all").copy()

    sm_order = build_sm_like_order(m["term"].tolist())
    order_map = {t: i for i, t in enumerate(sm_order)}

    m["ord_sm"] = m["term"].map(order_map).fillna(10**9).astype(int)
    m = m.sort_values(["ord_sm", "term"], ascending=[True, True]).reset_index(drop=True)

    y = np.arange(len(m))

    fig, ax = plt.subplots(figsize=(11, max(6, 0.30 * len(m))))

    def plot_one(beta, lo, hi, sig, yshift, label, marker):
        beta = beta.values
        lo = lo.values
        hi = hi.values
        sig = sig.fillna(False).values

        yy = y + yshift
        ok = np.isfinite(beta) & np.isfinite(lo) & np.isfinite(hi)

        # Significant coefficients: black line + marker
        idx = np.where(sig & ok)[0]

        if idx.size > 0:
            ax.errorbar(
                beta[idx],
                yy[idx],
                xerr=[beta[idx] - lo[idx], hi[idx] - beta[idx]],
                fmt=marker,
                linestyle="none",
                capsize=2,
                color="k",
                ecolor="k",
                elinewidth=1.2,
                markerfacecolor="k",
                markeredgecolor="k",
                label=label
            )

        # Non-significant coefficients: gray dashed CI bars
        idx = np.where((~sig) & ok)[0]

        if idx.size > 0:
            eb = ax.errorbar(
                beta[idx],
                yy[idx],
                xerr=[beta[idx] - lo[idx], hi[idx] - beta[idx]],
                fmt=marker,
                linestyle="none",
                capsize=0,
                color="0.6",
                ecolor="0.6",
                elinewidth=1.2,
                markerfacecolor="0.6",
                markeredgecolor="0.6"
            )

            for barcol in eb[2]:
                barcol.set_linestyle("--")

    plot_one(
        m["betaG"],
        m["loG"],
        m["hiG"],
        m["sigG"],
        yshift=-0.12,
        label=labG,
        marker="^"
    )

    plot_one(
        m["betaU"],
        m["loU"],
        m["hiU"],
        m["sigU"],
        yshift=+0.12,
        label=labU,
        marker="o"
    )

    ax.axvline(0, linestyle="--", color="0.3", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(m["term"], fontsize=8)
    ax.invert_yaxis()

    ax.set_xlabel("Standardized coefficient (beta*) with 95% CI")
    ax.set_title(title)
    ax.legend(frameon=False)

    ax.text(
        0.99,
        0.01,
        f"Gray dashed error bars: p ≥ {ALPHA_SIG}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.6"
    )

    fig.tight_layout()

    # ========================
    # Export figures
    # ========================
    fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")

    plt.close(fig)

# ========================
# Load + prep
# ========================
df = pd.read_csv(IN_PATH)
df, pca = add_pollution_pc1(df)

# ========================
# Auto-detect DV columns
# ========================
def pick_first_existing(candidates, columns):
    for c in candidates:
        if c in columns:
            return c
    return None

Y_G = pick_first_existing(Y_G_CAND, df.columns)
Y_U = pick_first_existing(Y_U_CAND, df.columns)

if (Y_G is None) or (Y_U is None):
    raise ValueError(
        f"Cannot find DV columns. Need one of {Y_G_CAND} and one of {Y_U_CAND}. "
        f"Available columns: {list(df.columns)[:30]} ..."
    )

X_CONT = X_CONT_BASE + ["pollution_pc1"]

need_cols = [CITY_ID, PERIOD, Y_G, Y_U] + X_CAT + X_CONT
df = df.dropna(subset=need_cols).copy()

# RHS = X + period FE
rhs_terms = [f"C({v})" for v in X_CAT] + X_CONT + [f"C({PERIOD})"]
rhs = " + ".join(rhs_terms)

resG, fG = fit_ols(df, Y_G, rhs)
resU, fU = fit_ols(df, Y_U, rhs)

# ========================
# Save regression outputs
# ========================
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
    coef_table(resG).to_excel(w, sheet_name=f"ModelG_{Y_G}")
    coef_table(resU).to_excel(w, sheet_name=f"ModelU_{Y_U}")

    if pca is not None:
        loadings = pd.DataFrame(
            pca.components_.T,
            index=["pm_mean", "c_mean"],
            columns=["PC1_loading", "PC2_loading"]
        )

        evr = pd.DataFrame(
            {"explained_variance_ratio": pca.explained_variance_ratio_},
            index=["PC1", "PC2"]
        )

        loadings.to_excel(w, sheet_name="PCA_loadings")
        evr.to_excel(w, sheet_name="PCA_EVR")

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(f"=== Model G ({Y_G}) ===\nFormula: {fG}\n\n")
    f.write(resG.summary().as_text())

    f.write(f"\n\n=== Model U ({Y_U}) ===\nFormula: {fU}\n\n")
    f.write(resU.summary().as_text())

# ========================
# Standardized outputs
# ========================
stdG = standardized_effects(resG, df[Y_G], cont_names=X_CONT)
stdU = standardized_effects(resU, df[Y_U], cont_names=X_CONT)

g = stdG.rename(columns={
    "beta_star": "beta_star_G",
    "ci_low_star": "ci_low_G",
    "ci_high_star": "ci_high_G",
    "p": "p_G",
    "sig": "sig_G"
})

u = stdU.rename(columns={
    "beta_star": "beta_star_U",
    "ci_low_star": "ci_low_U",
    "ci_high_star": "ci_high_U",
    "p": "p_U",
    "sig": "sig_U"
})

std_merge = pd.merge(
    g[["term", "beta_star_G", "ci_low_G", "ci_high_G", "p_G", "sig_G"]],
    u[["term", "beta_star_U", "ci_low_U", "ci_high_U", "p_U", "sig_U"]],
    on="term",
    how="outer"
)

std_merge.to_csv(OUT_STDCSV, index=False, encoding="utf-8-sig")
df.to_csv(OUT_CLEAN, index=False, encoding="utf-8-sig")

# ========================
# Plot all predictors
# ========================
forest_plot_GU(
    stdG,
    stdU,
    OUT_FIG_ALL_BASE,
    title=(
        f"Fig 3.4A (LEVEL) — Standardized effects (beta*) with 95% CI: "
        f"{Y_G} vs {Y_U}\n"
        f"Non-significant: gray dashed error bars"
    ),
    labG=f"Model G: {Y_G}",
    labU=f"Model U: {Y_U}"
)

# ========================
# Plot zoom version
# ========================
zoom_terms = [
    "bucap_mean",
    "gdpp_mean",
    "temp_mean",
    "rain_mean",
    "pollution_pc1",
    "C(income)[T.2.0]",
    "C(income)[T.3.0]",
    "C(income)[T.4.0]",
    "C(development)[T.2]",
    "C(development)[T.3]"
]

forest_plot_GU(
    stdG,
    stdU,
    OUT_FIG_ZOOM_BASE,
    title=(
        f"Fig 3.4A (LEVEL, zoom) — Key predictors: "
        f"{Y_G} vs {Y_U}\n"
        f"Non-significant: gray dashed error bars"
    ),
    zoom_terms=zoom_terms,
    labG=f"Model G: {Y_G}",
    labU=f"Model U: {Y_U}"
)

# ========================
# Print saved files
# ========================
print("Saved:")
print(" -", OUT_XLSX)
print(" -", OUT_TXT)
print(" -", OUT_STDCSV)
print(" -", OUT_CLEAN)

print(" -", OUT_FIG_ALL_BASE + ".png")
print(" -", OUT_FIG_ALL_BASE + ".pdf")
print(" -", OUT_FIG_ALL_BASE + ".svg")

print(" -", OUT_FIG_ZOOM_BASE + ".png")
print(" -", OUT_FIG_ZOOM_BASE + ".pdf")
print(" -", OUT_FIG_ZOOM_BASE + ".svg")

print("DV used:", Y_G, "and", Y_U)
print("Rows used:", df.shape[0])