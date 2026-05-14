# -*- coding: utf-8 -*-
import os, re
import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================================================
# PATHS
# =========================================================
IN_PATH = r"C:\Users\steve\Desktop\最新结论\3.4回归数据2_加入rel.csv"
OUT_DIR = r"C:\Users\steve\Desktop\3.4\model1_Rcontrib_dualtrack"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# COLUMNS
# =========================================================
CITY   = "ID_UGS"
PERIOD = "period"
CONT   = "continent_UGS"

LEVEL_WHICH = "t1"  # prefer t1 if exists
Y_G_CAND = [f"UGSrel_{LEVEL_WHICH}", "UGSrel"]
Y_U_CAND = [f"NLIrel_{LEVEL_WHICH}", "NLIrel"]

X_CAT  = ["biome", "Soil", "Level", "catchment", "income", "development", "climate"]
X_CONT = ["bucap_mean", "gdpp_mean", "temp_mean", "rain_mean"]
POLL_VARS = ["pm_mean", "c_mean"]
POLL_PC   = "pollution_pc1"

INCLUDE_PERIOD_FE = True
RANK_PERIOD_FE_AS_FACTOR = False  # period FE作为控制项，不做“主导机制”排名

MIN_ROWS   = 80
MIN_CITIES = 40
ALPHA_SIG  = 0.05

# =========================================================
# HELPERS
# =========================================================
def safe_name(x):
    s = str(x).strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s

def pick_first_existing(candidates, cols):
    for c in candidates:
        if c in cols:
            return c
    return None

def add_pollution_pc1(df, out_col=POLL_PC):
    if out_col in df.columns and df[out_col].notna().any():
        return df, None
    if not all(v in df.columns for v in POLL_VARS):
        df[out_col] = np.nan
        return df, None

    d = df.dropna(subset=POLL_VARS).copy()
    if d.empty:
        df[out_col] = np.nan
        return df, None

    X = d[POLL_VARS].astype(float).values
    Xz = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0).fit(Xz)
    pc1 = pca.transform(Xz)[:, 0]

    corr1 = np.corrcoef(pc1, d[POLL_VARS[0]].values)[0, 1]
    corr2 = np.corrcoef(pc1, d[POLL_VARS[1]].values)[0, 1]
    if (corr1 < 0) and (corr2 < 0):
        pc1 = -pc1

    d[out_col] = pc1
    poll_city = d.groupby(CITY)[out_col].mean()
    df[out_col] = df[CITY].map(poll_city)
    return df, pca

def build_rhs():
    rhs_terms = [f"C({v})" for v in X_CAT] + X_CONT + [POLL_PC]
    if INCLUDE_PERIOD_FE:
        rhs_terms += [f"C({PERIOD})"]
    return " + ".join(rhs_terms)

def build_design(df, rhs):
    return patsy.dmatrix(f"1 + {rhs}", data=df, return_type="dataframe")

def ols_r2(y, W):
    y = np.asarray(y).reshape(-1)
    X = np.asarray(W)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[ok]
    X = X[ok, :]
    n = X.shape[0]
    k = X.shape[1]
    if n <= k + 5:
        return np.nan, np.nan
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    ssr = np.sum((y - yhat)**2)
    tss = np.sum((y - np.mean(y))**2)
    R2 = 1.0 - ssr/tss if tss > 0 else np.nan
    AdjR2 = 1.0 - (1.0 - R2) * (n - 1) / (n - k) if (np.isfinite(R2) and (n > k)) else np.nan
    return R2, AdjR2

def factor_groups_from_design_columns(cols):
    groups = {}
    for v in X_CAT:
        groups[v] = []
    for v in X_CONT:
        groups[v] = []
    groups[POLL_PC] = []
    if INCLUDE_PERIOD_FE:
        groups[PERIOD] = []

    for c in cols:
        if c == "Intercept":
            continue

        matched = False
        for v in X_CAT:
            if c.startswith(f"C({v})["):
                groups[v].append(c)
                matched = True
                break
        if matched:
            continue

        if INCLUDE_PERIOD_FE and c.startswith(f"C({PERIOD})["):
            groups[PERIOD].append(c)
            continue

        if c in X_CONT:
            groups[c].append(c)
            continue

        if c == POLL_PC:
            groups[POLL_PC].append(c)
            continue

        groups.setdefault("other", []).append(c)

    groups = {k: v for k, v in groups.items() if len(v) > 0}
    return groups

def drop_columns(Wdf, cols_to_drop):
    keep = [c for c in Wdf.columns if c not in set(cols_to_drop)]
    return Wdf[keep]

def compute_contrib(y, Wdf, groups_rank):
    R2_full, AdjR2_full = ols_r2(y, Wdf.values)
    rows = []
    for gname, gcols in groups_rank.items():
        W_drop = drop_columns(Wdf, gcols)
        R2_drop, _ = ols_r2(y, W_drop.values)
        dR2 = (R2_full - R2_drop) if (np.isfinite(R2_full) and np.isfinite(R2_drop)) else np.nan
        share = (dR2 / R2_full) if (np.isfinite(dR2) and np.isfinite(R2_full) and R2_full != 0) else np.nan
        rows.append({"factor": gname, "dR2": dR2, "share": share})
    out = pd.DataFrame(rows).sort_values("dR2", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["R2_full"] = R2_full
    out["AdjR2_full"] = AdjR2_full
    return out

def fit_cluster_ols(df, ycol, rhs):
    formula = f"{ycol} ~ {rhs}"
    res = sm.OLS.from_formula(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[CITY]}
    )
    return res, formula

def coef_dualtrack(res, y_series):
    """
    Dual-track table:
      beta, se(cluster), p, CI
      beta_star, CI_star  (computed using design SDs)
    """
    y = pd.to_numeric(y_series, errors="coerce").values.astype(float)
    y_sd = float(np.nanstd(y, ddof=1))
    if not np.isfinite(y_sd) or y_sd == 0:
        y_sd = np.nan

    X_design = pd.DataFrame(res.model.exog, columns=res.model.exog_names)

    rows = []
    for term in res.params.index:
        beta = float(res.params[term])
        se   = float(res.bse[term])
        p    = float(res.pvalues[term])
        lo, hi = beta - 1.96*se, beta + 1.96*se

        # sd_x from design column
        if term in X_design.columns:
            sd_x = float(np.nanstd(X_design[term].values, ddof=1))
        else:
            sd_x = np.nan

        # standardized
        if (np.isfinite(sd_x) and sd_x > 0 and np.isfinite(y_sd) and y_sd > 0):
            scale = sd_x / y_sd
            beta_star = beta * scale
            lo_star   = lo * scale
            hi_star   = hi * scale
        else:
            beta_star = lo_star = hi_star = np.nan

        rows.append({
            "term": term,
            "beta": beta,
            "se_cluster": se,
            "p": p,
            "ci_low": lo,
            "ci_high": hi,
            "beta_star": beta_star,
            "ci_low_star": lo_star,
            "ci_high_star": hi_star,
            "sig": (p < ALPHA_SIG)
        })

    out = pd.DataFrame(rows)

    # optional: remove intercept for reporting convenience
    return out

def group_direction_from_coefstar(coef_df, W_cols, groups_rank):
    """
    Direction index per factor group using standardized coefficients:
      Dir_g = sum(beta*) / sum(|beta*|) in [-1,1]
    Only uses terms belonging to that group.
    """
    term2bstar = dict(zip(coef_df["term"], coef_df["beta_star"]))
    rows = []
    for g, cols in groups_rank.items():
        # map design columns to corresponding term names in regression
        # NOTE: statsmodels names usually align with design columns + "Intercept"
        vals = []
        for c in cols:
            if c in term2bstar:
                vals.append(term2bstar[c])
        vals = np.array(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            dir_g = np.nan
            mag_g = np.nan
        else:
            dir_g = float(np.sum(vals) / np.sum(np.abs(vals))) if np.sum(np.abs(vals)) > 0 else np.nan
            mag_g = float(np.sum(np.abs(vals)))  # “方向加权强度”（可选）
        rows.append({"factor": g, "Dir_g": dir_g, "AbsSum_betaStar": mag_g})
    return pd.DataFrame(rows)

def plot_topk_bar(contrib_df, title, out_svg, out_png, topk=10):
    d = contrib_df.sort_values("dR2", ascending=False).head(topk).copy()
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.35*len(d))))
    y = np.arange(len(d))
    ax.barh(y, d["dR2"].values)
    ax.set_yticks(y)
    ax.set_yticklabels(d["factor"])
    ax.invert_yaxis()
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("ΔR² = R²(full) − R²(drop factor group)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)  # vector
    plt.close(fig)

# =========================================================
# LOAD
# =========================================================
df0 = pd.read_csv(IN_PATH)
df0, _ = add_pollution_pc1(df0)

Y_G = pick_first_existing(Y_G_CAND, df0.columns)
Y_U = pick_first_existing(Y_U_CAND, df0.columns)
if (Y_G is None) or (Y_U is None):
    raise ValueError(f"Cannot find DV columns. Need {Y_G_CAND} and {Y_U_CAND}.")

if CONT not in df0.columns:
    raise ValueError(f"Cannot find continent column '{CONT}'.")

df0[CONT] = df0[CONT].astype(str).str.strip()
df0.loc[df0[CONT].isin(["", "nan", "None"]), CONT] = np.nan
df0 = df0.dropna(subset=[CONT]).copy()

rhs = build_rhs()

# =========================================================
# RUN PER CONTINENT
# =========================================================
all_detail = []
r2_summary = []
skipped = []

for cont in sorted(df0[CONT].unique()):
    d = df0[df0[CONT] == cont].copy()
    need = [CITY, PERIOD, CONT, Y_G, Y_U, POLL_PC] + X_CAT + X_CONT
    d = d.dropna(subset=need).copy()

    n_rows = d.shape[0]
    n_cities = d[CITY].nunique()
    if n_rows < MIN_ROWS or n_cities < MIN_CITIES:
        skipped.append((cont, n_rows, n_cities))
        continue

    cont_dir = os.path.join(OUT_DIR, safe_name(cont))
    os.makedirs(cont_dir, exist_ok=True)

    # design for ΔR² (same RHS as regression)
    Wdf = build_design(d, rhs)
    groups = factor_groups_from_design_columns(Wdf.columns.tolist())
    if (not RANK_PERIOD_FE_AS_FACTOR) and (PERIOD in groups):
        groups_rank = {k: v for k, v in groups.items() if k != PERIOD}
    else:
        groups_rank = groups.copy()

    # --- fit OLS with cluster SE (for beta, p, etc.)
    resG, fG = fit_cluster_ols(d, Y_G, rhs)
    resU, fU = fit_cluster_ols(d, Y_U, rhs)

    # --- dual-track coefficient tables
    coefG = coef_dualtrack(resG, d[Y_G])
    coefU = coef_dualtrack(resU, d[Y_U])

    # --- R contribution (ΔR²)
    y_ugs = pd.to_numeric(d[Y_G], errors="coerce").values
    y_nli = pd.to_numeric(d[Y_U], errors="coerce").values
    contribG = compute_contrib(y_ugs, Wdf, groups_rank)
    contribU = compute_contrib(y_nli, Wdf, groups_rank)

    # --- group direction indices from beta_star
    dirG = group_direction_from_coefstar(coefG, Wdf.columns.tolist(), groups_rank)
    dirU = group_direction_from_coefstar(coefU, Wdf.columns.tolist(), groups_rank)

    # merge: contrib + direction
    contribG2 = contribG.merge(dirG, on="factor", how="left")
    contribU2 = contribU.merge(dirU, on="factor", how="left")

    # R2 summary
    r2_summary.append({
        "continent": cont,
        "n_rows": n_rows,
        "n_cities": n_cities,
        "DV_UGS": Y_G,
        "DV_NLI": Y_U,
        "R2_UGS": float(contribG2["R2_full"].iloc[0]),
        "AdjR2_UGS": float(contribG2["AdjR2_full"].iloc[0]),
        "R2_NLI": float(contribU2["R2_full"].iloc[0]),
        "AdjR2_NLI": float(contribU2["AdjR2_full"].iloc[0]),
        "rhs": rhs
    })

    # save per-continent excel
    out_xlsx = os.path.join(cont_dir, f"Model1_DualTrack_Rcontrib__{safe_name(cont)}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        pd.DataFrame({"formula_UGS": [fG], "formula_NLI": [fU]}).to_excel(w, "formulas", index=False)
        coefG.to_excel(w, "coef_dualtrack_UGS", index=False)
        coefU.to_excel(w, "coef_dualtrack_NLI", index=False)
        contribG2.to_excel(w, "Rcontrib_rank_UGS", index=False)
        contribU2.to_excel(w, "Rcontrib_rank_NLI", index=False)

    # save plots (Top10 ΔR²)
    plot_topk_bar(contribG2,
                  title=f"{cont} — UGS level: Top ΔR² contributors",
                  out_svg=os.path.join(cont_dir, f"Fig_Rcontrib__{safe_name(cont)}__UGS.svg"),
                  out_png=os.path.join(cont_dir, f"Fig_Rcontrib__{safe_name(cont)}__UGS.png"),
                  topk=10)
    plot_topk_bar(contribU2,
                  title=f"{cont} — NLI level: Top ΔR² contributors",
                  out_svg=os.path.join(cont_dir, f"Fig_Rcontrib__{safe_name(cont)}__NLI.svg"),
                  out_png=os.path.join(cont_dir, f"Fig_Rcontrib__{safe_name(cont)}__NLI.png"),
                  topk=10)

    # save cleaned used
    d.to_csv(os.path.join(cont_dir, f"Model1_cleaned_used__{safe_name(cont)}.csv"),
             index=False, encoding="utf-8-sig")

    # append global detail
    cg = contribG2.copy(); cg.insert(0, "continent", cont); cg.insert(1, "target", "UGS_level")
    cu = contribU2.copy(); cu.insert(0, "continent", cont); cu.insert(1, "target", "NLI_level")
    all_detail.append(cg)
    all_detail.append(cu)

    print(f"[DONE] {cont} | saved -> {out_xlsx}")

# =========================================================
# SAVE GLOBAL SUMMARY
# =========================================================
detail_df = pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame()
r2_df = pd.DataFrame(r2_summary).sort_values("continent") if r2_summary else pd.DataFrame()

out_all_xlsx = os.path.join(OUT_DIR, "00_Model1_DualTrack_Rcontrib_AllContinents.xlsx")
out_all_csv  = os.path.join(OUT_DIR, "00_Model1_DualTrack_Rcontrib_AllContinents_detail.csv")
out_skip_csv = os.path.join(OUT_DIR, "00_Model1_DualTrack_skipped.csv")

with pd.ExcelWriter(out_all_xlsx, engine="openpyxl") as w:
    r2_df.to_excel(w, "R2_summary", index=False)
    detail_df.to_excel(w, "Rcontrib_detail", index=False)

    if not detail_df.empty:
        for tgt in ["UGS_level", "NLI_level"]:
            dtop = detail_df[detail_df["target"] == tgt].sort_values(["continent","dR2"], ascending=[True, False])
            dtop = dtop.groupby("continent").head(10).reset_index(drop=True)
            dtop.to_excel(w, f"Top10_{tgt}", index=False)

detail_df.to_csv(out_all_csv, index=False, encoding="utf-8-sig")

if skipped:
    pd.DataFrame(skipped, columns=["continent","rows","cities"]).to_csv(out_skip_csv, index=False, encoding="utf-8-sig")

print("========================================")
print("Saved:", out_all_xlsx)
print("Detail csv:", out_all_csv)
if skipped:
    print("Skipped:", out_skip_csv)
print("OUT_DIR:", OUT_DIR)
print("DV used:", Y_G, "and", Y_U)