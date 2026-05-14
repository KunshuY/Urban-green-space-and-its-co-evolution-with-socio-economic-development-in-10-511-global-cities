# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# Matplotlib vector export settings
# =========================
# Keep text editable in PDF/SVG where possible
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

# =========================
# Paths
# =========================
IN_PATH = r"C:\Users\steve\Desktop\最新结论\3.4回归数据2_加入rel.csv"
OUT_DIR = r"C:\Users\steve\Desktop\3.4\model2"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_XLSX  = os.path.join(OUT_DIR, "02_Model_SM_jointsystem_outputs.xlsx")
OUT_CLEAN = os.path.join(OUT_DIR, "02_Model_SM_cleaned_used.csv")

# Figure output base name: will export .png, .pdf, .svg
OUT_FIG_BASE = os.path.join(OUT_DIR, "Fig_A1_thetaS_thetaM_forest_siggray")

# =========================
# Columns
# =========================
CITY   = "ID_UGS"
PERIOD = "period"
DUGS   = "dUGSrel"
DNLI   = "dNLIrel"

X_CAT = ["biome", "Soil", "Level", "catchment", "income", "development", "climate"]
X_CONT = ["bucap_mean", "gdpp_mean", "temp_mean", "rain_mean"]
POLL_VARS = ["pm_mean", "c_mean"]

# =========================
# Helpers
# =========================
def mad_z(s: pd.Series, c=1.4826):
    """Robust z-score using MAD."""
    x = s.astype(float).values
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    denom = c * mad if (np.isfinite(mad) and mad > 0) else np.nan
    return (s - med) / denom


def add_pollution_pc1(df, out_col="pollution_pc1"):
    """
    Build pollution_pc1 from pm_mean & c_mean via PCA city-mean PC1,
    unless df already contains a non-missing pollution_pc1.
    """
    if out_col in df.columns and df[out_col].notna().any():
        return df, None

    d = df.dropna(subset=POLL_VARS).copy()

    X = d[POLL_VARS].astype(float).values
    Xz = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=0).fit(Xz)
    pc1 = pca.transform(Xz)[:, 0]

    # Align sign: higher pc1 should mean higher pm and higher c
    corr1 = np.corrcoef(pc1, d["pm_mean"].values)[0, 1]
    corr2 = np.corrcoef(pc1, d["c_mean"].values)[0, 1]

    if (corr1 < 0) and (corr2 < 0):
        pc1 = -pc1

    d[out_col] = pc1

    poll_city = d.groupby(CITY)[out_col].mean()
    df[out_col] = df[CITY].map(poll_city)

    return df, pca


def build_design(df):
    """Design matrix with identical RHS used for both equations + period FE."""
    rhs = " + ".join(
        [f"C({v})" for v in X_CAT]
        + X_CONT
        + ["pollution_pc1"]
        + [f"C({PERIOD})"]
    )

    W = patsy.dmatrix(f"1 + {rhs}", data=df, return_type="dataframe")

    return W, rhs


def cluster_robust_cov(X, u, groups):
    """
    Liang-Zeger / Cameron-Miller cluster-robust covariance:
    V = (X'X)^-1 (sum_g X_g' u_g u_g' X_g) (X'X)^-1
    """
    X = np.asarray(X)
    u = np.asarray(u).reshape(-1, 1)
    groups = np.asarray(groups)

    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        Xg = X[idx, :]
        ug = u[idx, :]

        Sg = Xg.T @ ug
        meat += Sg @ Sg.T

    return XtX_inv @ meat @ XtX_inv


def build_sm_like_order(all_terms):
    """
    Optional: make order stable & consistent with your Model 1 figure.
    Unmatched terms are appended at the end.
    """
    order = []

    # Biome
    for b in ["3.0", "4.0", "5.0", "6.0", "7.0", "8.0"]:
        order.append(f"C(biome)[T.{b}]")

    # Soil
    for s in ["2.0", "3.0"]:
        order.append(f"C(Soil)[T.{s}]")

    # Level
    for lv in ["2", "3", "4", "5", "6"]:
        order.append(f"C(Level)[T.{lv}]")

    # Catchment
    order.append("C(catchment)[T.1]")

    # Income
    for inc in ["2.0", "3.0", "4.0"]:
        order.append(f"C(income)[T.{inc}]")

    # Development
    for dv in ["2", "3"]:
        order.append(f"C(development)[T.{dv}]")

    # Climate
    order.append("C(climate)[T.continental]")
    order.append("C(climate)[T.temperate]")
    order.append("C(climate)[T.tropical]")

    # Continuous variables
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


def forest_plot_two_siggray(ax, names, bS, loS, hiS, bM, loM, hiM, sigS, sigM, title):
    """
    Unified style:
    - theta_S: black triangle '^' if significant; gray triangle if non-significant
    - theta_M: black circle 'o' if significant; gray circle if non-significant
    - Non-significant error bars: gray dashed bars only, no caps
    """
    y = np.arange(len(names))

    ax.axvline(0, linestyle="--", color="0.3", linewidth=1)

    def plot_series(b, lo, hi, sig, yshift, label, marker):
        b = np.asarray(b)
        lo = np.asarray(lo)
        hi = np.asarray(hi)
        sig = np.asarray(sig).astype(bool)

        yy = y + yshift
        ok = np.isfinite(b) & np.isfinite(lo) & np.isfinite(hi)

        # Significant: black
        idx = np.where(sig & ok)[0]

        if idx.size:
            ax.errorbar(
                b[idx],
                yy[idx],
                xerr=[b[idx] - lo[idx], hi[idx] - b[idx]],
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

        # Non-significant: gray dashed bars only, no caps
        idx = np.where((~sig) & ok)[0]

        if idx.size:
            eb = ax.errorbar(
                b[idx],
                yy[idx],
                xerr=[b[idx] - lo[idx], hi[idx] - b[idx]],
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

    plot_series(
        bS,
        loS,
        hiS,
        sigS,
        yshift=-0.12,
        label="theta_S (synergy intensity)",
        marker="^"
    )

    plot_series(
        bM,
        loM,
        hiM,
        sigM,
        yshift=+0.12,
        label="theta_M (mismatch orientation)",
        marker="o"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()

    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8, frameon=False)

    ax.text(
        0.99,
        0.01,
        "Non-significant: gray dashed error bars",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.6"
    )


# =========================
# Load + prep
# =========================
df = pd.read_csv(IN_PATH)
df, pca = add_pollution_pc1(df)

need = [CITY, PERIOD, DUGS, DNLI, "pollution_pc1"] + X_CAT + X_CONT
df = df.dropna(subset=need).copy()

# Optional: remove exact zeros if you treat them as boundary noise
df = df[(df[DUGS] != 0) & (df[DNLI] != 0)].copy()

# Robust standardization for joint system
df["x"] = mad_z(df[DUGS])
df["y"] = mad_z(df[DNLI])

W, rhs = build_design(df)

X = W.values
names = W.columns.tolist()

# =========================
# Joint system estimation: stacked OLS + city-cluster covariance
# =========================
n = X.shape[0]
k = X.shape[1]

# Stack:
# [x] = [W 0] [beta_x] + u_x
# [y]   [0 W] [beta_y] + u_y
Xstack = np.zeros((2 * n, 2 * k))
Xstack[:n, :k] = X
Xstack[n:, k:] = X

ystack = np.concatenate([df["x"].values, df["y"].values], axis=0)

beta = np.linalg.lstsq(Xstack, ystack, rcond=None)[0]

beta_x = beta[:k]
beta_y = beta[k:]

u = ystack - Xstack @ beta

groups = np.concatenate([df[CITY].values, df[CITY].values], axis=0)

V = cluster_robust_cov(Xstack, u, groups)

se = np.sqrt(np.diag(V))

se_x = se[:k]
se_y = se[k:]

ci_x = np.vstack([
    beta_x - 1.96 * se_x,
    beta_x + 1.96 * se_x
]).T

ci_y = np.vstack([
    beta_y - 1.96 * se_y,
    beta_y + 1.96 * se_y
]).T

# =========================
# Derived S/M coefficients via exact linear transform
# =========================
sqrt2 = np.sqrt(2)

theta_S = (beta_x + beta_y) / sqrt2
theta_M = (beta_x - beta_y) / sqrt2

A = np.zeros((2 * k, 2 * k))

A[:k, :k] = np.eye(k) / sqrt2
A[:k, k:] = np.eye(k) / sqrt2

A[k:, :k] = np.eye(k) / sqrt2
A[k:, k:] = -np.eye(k) / sqrt2

V_theta = A @ V @ A.T

se_S = np.sqrt(np.diag(V_theta)[:k])
se_M = np.sqrt(np.diag(V_theta)[k:])

ci_S = np.vstack([
    theta_S - 1.96 * se_S,
    theta_S + 1.96 * se_S
]).T

ci_M = np.vstack([
    theta_M - 1.96 * se_M,
    theta_M + 1.96 * se_M
]).T

# Significance flags: CI not crossing zero
sig_S = ~((ci_S[:, 0] <= 0) & (ci_S[:, 1] >= 0))
sig_M = ~((ci_M[:, 0] <= 0) & (ci_M[:, 1] >= 0))

# =========================
# Save tables
# =========================
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
    pd.DataFrame({
        "term": names,
        "beta_x": beta_x,
        "se_x": se_x,
        "lo_x": ci_x[:, 0],
        "hi_x": ci_x[:, 1]
    }).to_excel(w, sheet_name="beta_x", index=False)

    pd.DataFrame({
        "term": names,
        "beta_y": beta_y,
        "se_y": se_y,
        "lo_y": ci_y[:, 0],
        "hi_y": ci_y[:, 1]
    }).to_excel(w, sheet_name="beta_y", index=False)

    pd.DataFrame({
        "term": names,
        "theta_S": theta_S,
        "se_S": se_S,
        "lo_S": ci_S[:, 0],
        "hi_S": ci_S[:, 1],
        "sig_S": sig_S
    }).to_excel(w, sheet_name="theta_S", index=False)

    pd.DataFrame({
        "term": names,
        "theta_M": theta_M,
        "se_M": se_M,
        "lo_M": ci_M[:, 0],
        "hi_M": ci_M[:, 1],
        "sig_M": sig_M
    }).to_excel(w, sheet_name="theta_M", index=False)

    if pca is not None:
        loadings = pd.DataFrame(
            pca.components_.T,
            index=POLL_VARS,
            columns=["PC1_loading", "PC2_loading"]
        )

        evr = pd.DataFrame(
            {"explained_variance_ratio": pca.explained_variance_ratio_},
            index=["PC1", "PC2"]
        )

        loadings.to_excel(w, sheet_name="PCA_loadings")
        evr.to_excel(w, sheet_name="PCA_EVR")

# Save cleaned data used
df.to_csv(OUT_CLEAN, index=False, encoding="utf-8-sig")

print("Saved:", OUT_XLSX)
print("Saved cleaned:", OUT_CLEAN)

# =========================
# Figure: theta_S / theta_M forest
# Exclude intercept and period FE
# =========================
mask = ~pd.Series(names).str.startswith(("Intercept", f"C({PERIOD})"))

terms_plot = list(pd.Series(names)[mask])

S2 = theta_S[mask.values]
M2 = theta_M[mask.values]

Slo = ci_S[mask.values, 0]
Shi = ci_S[mask.values, 1]

Mlo = ci_M[mask.values, 0]
Mhi = ci_M[mask.values, 1]

sigS2 = sig_S[mask.values]
sigM2 = sig_M[mask.values]

# Enforce stable order matching Model 1
sm_order = build_sm_like_order(terms_plot)
order_map = {t: i for i, t in enumerate(sm_order)}

ord_idx = np.argsort([
    order_map.get(t, 10**9) for t in terms_plot
])

terms_plot = [terms_plot[i] for i in ord_idx]

S2 = S2[ord_idx]
Slo = Slo[ord_idx]
Shi = Shi[ord_idx]
sigS2 = sigS2[ord_idx]

M2 = M2[ord_idx]
Mlo = Mlo[ord_idx]
Mhi = Mhi[ord_idx]
sigM2 = sigM2[ord_idx]

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(10, max(6, 0.22 * len(terms_plot))))

forest_plot_two_siggray(
    ax,
    terms_plot,
    S2,
    Slo,
    Shi,
    M2,
    Mlo,
    Mhi,
    sigS2,
    sigM2,
    (
        "Fig A1 — Derived effects on S (synergy intensity) and M (mismatch orientation)\n"
        "Non-significant: gray dashed error bars"
    )
)

fig.tight_layout()

# =========================
# Export raster + vector figures
# =========================
fig.savefig(OUT_FIG_BASE + ".png", dpi=600, bbox_inches="tight")
fig.savefig(OUT_FIG_BASE + ".pdf", bbox_inches="tight")
fig.savefig(OUT_FIG_BASE + ".svg", bbox_inches="tight")

plt.close(fig)

print("Saved figure:")
print(" -", OUT_FIG_BASE + ".png")
print(" -", OUT_FIG_BASE + ".pdf")
print(" -", OUT_FIG_BASE + ".svg")

print("Rows used:", df.shape[0])
print("Number of predictors plotted:", len(terms_plot))