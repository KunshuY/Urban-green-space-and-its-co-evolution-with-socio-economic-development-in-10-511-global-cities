# -*- coding: utf-8 -*-
import os, re, glob
import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt
from scipy.stats import norm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================================================
# PATHS
# =========================================================
BASE_DIR  = r"C:\Users\steve\Desktop\3.4\model2_byContinent"              # 输入：找csv
OUT_DIR   = r"C:\Users\steve\Desktop\3.4\model2_byContinent\Rcontribution" # 输出：写结果
os.makedirs(OUT_DIR, exist_ok=True)

pattern = os.path.join(BASE_DIR, "**", "02_Model_SM_cleaned_used__*.csv")  # 仍然在 BASE_DIR 里找
files = sorted(glob.glob(pattern, recursive=True))

# =========================================================
# COLUMNS (Model2)
# =========================================================
CITY   = "ID_UGS"
PERIOD = "period"
DUGS   = "dUGSrel"
DNLI   = "dNLIrel"
CONT_COL = "continent_UGS"  # if present in cleaned_used, use it; else infer from filename/folder

X_CAT  = ["biome", "Soil", "Level", "catchment", "income", "development", "climate"]
X_CONT = ["bucap_mean", "gdpp_mean", "temp_mean", "rain_mean"]
POLL_VARS = ["pm_mean", "c_mean"]
POLL_PC   = "pollution_pc1"

INCLUDE_PERIOD_FE = True
RANK_PERIOD_FE_AS_FACTOR = False  # keep period FE as control, not "main driver"

DROP_ZERO_CHANGE = True

MIN_ROWS   = 80
MIN_CITIES = 40

ALPHA_SIG = 0.05  # for a "sig" flag from p-values

# =========================================================
# HELPERS
# =========================================================
def safe_name(x):
    s = str(x).strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s

def infer_continent_from_path(path):
    fn = os.path.basename(path)
    m = re.search(r"__([A-Za-z0-9_]+)\.csv$", fn)
    if m:
        return m.group(1)
    return os.path.basename(os.path.dirname(path))

def mad_z(s: pd.Series, c=1.4826):
    x = pd.to_numeric(s, errors="coerce").astype(float).values
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    denom = c * mad if (np.isfinite(mad) and mad > 0) else np.nan
    return (pd.to_numeric(s, errors="coerce").astype(float) - med) / denom

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
    W = patsy.dmatrix(f"1 + {rhs}", data=df, return_type="dataframe")
    return W

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

def ols_r2(y, X):
    y = np.asarray(y).reshape(-1)
    X = np.asarray(X)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[ok]
    X = X[ok, :]
    n = X.shape[0]
    k = X.shape[1]
    if n <= k + 5:
        return np.nan, np.nan
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ b
    ssr = np.sum((y - yhat)**2)
    tss = np.sum((y - np.mean(y))**2)
    R2 = 1.0 - ssr/tss if tss > 0 else np.nan
    AdjR2 = 1.0 - (1.0 - R2) * (n - 1) / (n - k) if (np.isfinite(R2) and (n > k)) else np.nan
    return R2, AdjR2

def cluster_robust_cov(X, u, groups, df_correction=True):
    """
    Liang-Zeger cluster-robust covariance with pinv + finite-sample correction.
    """
    X = np.asarray(X)
    u = np.asarray(u).reshape(-1, 1)
    groups = np.asarray(groups)

    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))

    uniq = np.unique(groups)
    G = len(uniq)
    N = X.shape[0]
    K = X.shape[1]

    for g in uniq:
        idx = np.where(groups == g)[0]
        Xg = X[idx, :]
        ug = u[idx, :]
        Sg = Xg.T @ ug
        meat += Sg @ Sg.T

    V = XtX_inv @ meat @ XtX_inv
    if df_correction and (G > 1) and (N > K):
        V = V * (G/(G-1)) * ((N-1)/(N-K))
    return V

def dualtrack_table_from_joint(names, coef, se, y_sd, X_sd):
    """
    Build per-term dual-track table:
      beta, se, z, p, CI
      beta_star, CI_star
    """
    coef = np.asarray(coef)
    se   = np.asarray(se)
    z = coef / se
    p = 2.0 * (1.0 - norm.cdf(np.abs(z)))

    lo = coef - 1.96*se
    hi = coef + 1.96*se

    # standardized
    beta_star = coef * (X_sd / y_sd)
    lo_star   = lo   * (X_sd / y_sd)
    hi_star   = hi   * (X_sd / y_sd)

    df = pd.DataFrame({
        "term": names,
        "beta": coef,
        "se_cluster": se,
        "z": z,
        "p": p,
        "ci_low": lo,
        "ci_high": hi,
        "beta_star": beta_star,
        "ci_low_star": lo_star,
        "ci_high_star": hi_star,
        "sig": (p < ALPHA_SIG)
    })
    return df

def group_direction_from_theta_star(theta_star_df, groups_rank):
    """
    Dir_S_g / Dir_M_g based on standardized theta values:
      Dir_g = sum(theta*) / sum(|theta*|) in [-1,1]
    """
    term2val = dict(zip(theta_star_df["term"], theta_star_df["beta_star"]))
    rows = []
    for g, cols in groups_rank.items():
        vals = []
        for c in cols:
            if c in term2val and np.isfinite(term2val[c]):
                vals.append(term2val[c])
        vals = np.array(vals, dtype=float)
        if vals.size == 0 or np.sum(np.abs(vals)) == 0:
            rows.append({"factor": g, "Dir_g": np.nan, "AbsSum_thetaStar": np.nan})
        else:
            rows.append({
                "factor": g,
                "Dir_g": float(np.sum(vals) / np.sum(np.abs(vals))),
                "AbsSum_thetaStar": float(np.sum(np.abs(vals)))
            })
    return pd.DataFrame(rows)

def plot_topk_bar(contrib_df, title, out_svg, out_png, value_col, topk=10):
    d = contrib_df.sort_values(value_col, ascending=False).head(topk).copy()
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.35*len(d))))
    y = np.arange(len(d))
    ax.barh(y, d[value_col].values)
    ax.set_yticks(y)
    ax.set_yticklabels(d["factor"])
    ax.invert_yaxis()
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("ΔR² = R²(full) − R²(drop factor group)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    plt.close(fig)

# =========================================================
# CORE: run one continent file
# =========================================================
def run_one_file(fp):
    df = pd.read_csv(fp)
    df, _ = add_pollution_pc1(df)

    # infer continent
    cont = None
    if CONT_COL in df.columns and df[CONT_COL].notna().any():
        cont = str(df[CONT_COL].dropna().iloc[0]).strip()
    if (cont is None) or (cont == "") or (cont.lower() in ["nan","none"]):
        cont = infer_continent_from_path(fp)

    # required
    need = [CITY, PERIOD, DUGS, DNLI, POLL_PC] + X_CAT + X_CONT
    for c in need:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    d = df.dropna(subset=need).copy()
    if DROP_ZERO_CHANGE:
        d = d[(d[DUGS] != 0) & (d[DNLI] != 0)].copy()

    n_rows = d.shape[0]
    n_cities = d[CITY].nunique()
    if n_rows < MIN_ROWS or n_cities < MIN_CITIES:
        raise ValueError(f"too small: rows={n_rows}, cities={n_cities}")

    # x,y
    if ("x" in d.columns) and ("y" in d.columns) and d["x"].notna().any() and d["y"].notna().any():
        x = pd.to_numeric(d["x"], errors="coerce").values.astype(float)
        y = pd.to_numeric(d["y"], errors="coerce").values.astype(float)
    else:
        x = mad_z(d[DUGS]).values.astype(float)
        y = mad_z(d[DNLI]).values.astype(float)

    # S/M observed
    s_obs = (x + y) / np.sqrt(2)
    m_obs = (x - y) / np.sqrt(2)

    rhs = build_rhs()
    Wdf = build_design(d, rhs)
    X = Wdf.values
    names = Wdf.columns.tolist()

    # SDs for standardization (design-column SD)
    X_sd = np.array([np.nanstd(Wdf[c].values, ddof=1) for c in names], dtype=float)
    # Y SDs
    x_sd = float(np.nanstd(x, ddof=1))
    y_sd = float(np.nanstd(y, ddof=1))
    s_sd = float(np.nanstd(s_obs, ddof=1))
    m_sd = float(np.nanstd(m_obs, ddof=1))

    # stacked system
    n = X.shape[0]
    k = X.shape[1]
    Xstack = np.zeros((2*n, 2*k))
    Xstack[:n, :k] = X
    Xstack[n:, k:] = X
    ystack = np.concatenate([x, y], axis=0)

    beta = np.linalg.lstsq(Xstack, ystack, rcond=None)[0]
    beta_x = beta[:k]
    beta_y = beta[k:]

    u = ystack - Xstack @ beta
    groups = np.concatenate([d[CITY].values, d[CITY].values], axis=0)
    V = cluster_robust_cov(Xstack, u, groups, df_correction=True)
    se = np.sqrt(np.diag(V))
    se_x = se[:k]
    se_y = se[k:]

    # dual-track per equation
    tab_x = dualtrack_table_from_joint(names, beta_x, se_x, y_sd=x_sd, X_sd=X_sd)
    tab_y = dualtrack_table_from_joint(names, beta_y, se_y, y_sd=y_sd, X_sd=X_sd)

    # theta_S/M raw
    sqrt2 = np.sqrt(2)
    theta_S = (beta_x + beta_y) / sqrt2
    theta_M = (beta_x - beta_y) / sqrt2

    # theta SE via linear transform
    A = np.zeros((2*k, 2*k))
    A[:k, :k] = np.eye(k)/sqrt2
    A[:k, k:] = np.eye(k)/sqrt2
    A[k:, :k] = np.eye(k)/sqrt2
    A[k:, k:] = -np.eye(k)/sqrt2

    V_theta = A @ V @ A.T
    se_S = np.sqrt(np.diag(V_theta)[:k])
    se_M = np.sqrt(np.diag(V_theta)[k:])

    tab_S = dualtrack_table_from_joint(names, theta_S, se_S, y_sd=s_sd, X_sd=X_sd)
    tab_M = dualtrack_table_from_joint(names, theta_M, se_M, y_sd=m_sd, X_sd=X_sd)

    # --- R2 for x,y,S,M (fit using W only; same RHS)
    R2_x, AdjR2_x = ols_r2(x, X)
    R2_y, AdjR2_y = ols_r2(y, X)
    R2_s, AdjR2_s = ols_r2(s_obs, X)
    R2_m, AdjR2_m = ols_r2(m_obs, X)

    # --- R contribution (ΔR²) on S and M via leave-one-group-out
    groups = factor_groups_from_design_columns(names)
    if (not RANK_PERIOD_FE_AS_FACTOR) and (PERIOD in groups):
        groups_rank = {k: v for k, v in groups.items() if k != PERIOD}
    else:
        groups_rank = groups.copy()

    # full R2
    R2S_full, _ = ols_r2(s_obs, X)
    R2M_full, _ = ols_r2(m_obs, X)

    contrib_rows = []
    for gname, gcols in groups_rank.items():
        W_drop = drop_columns(Wdf, gcols)
        Xd = W_drop.values

        R2S_drop, _ = ols_r2(s_obs, Xd)
        R2M_drop, _ = ols_r2(m_obs, Xd)

        dR2S = (R2S_full - R2S_drop) if (np.isfinite(R2S_full) and np.isfinite(R2S_drop)) else np.nan
        dR2M = (R2M_full - R2M_drop) if (np.isfinite(R2M_full) and np.isfinite(R2M_drop)) else np.nan

        contrib_rows.append({
            "factor": gname,
            "dR2_S": dR2S,
            "share_S": (dR2S / R2S_full) if (np.isfinite(dR2S) and np.isfinite(R2S_full) and R2S_full != 0) else np.nan,
            "dR2_M": dR2M,
            "share_M": (dR2M / R2M_full) if (np.isfinite(dR2M) and np.isfinite(R2M_full) and R2M_full != 0) else np.nan
        })

    contrib = pd.DataFrame(contrib_rows)
    contrib["rank_S"] = contrib["dR2_S"].rank(ascending=False, method="dense")
    contrib["rank_M"] = contrib["dR2_M"].rank(ascending=False, method="dense")

    # direction indices based on standardized theta tables
    dirS = group_direction_from_theta_star(tab_S[["term","beta_star"]], groups_rank).rename(columns={"Dir_g":"Dir_S_g","AbsSum_thetaStar":"AbsSum_thetaS_star"})
    dirM = group_direction_from_theta_star(tab_M[["term","beta_star"]], groups_rank).rename(columns={"Dir_g":"Dir_M_g","AbsSum_thetaStar":"AbsSum_thetaM_star"})
    contrib = contrib.merge(dirS, on="factor", how="left").merge(dirM, on="factor", how="left")

    # summary
    summ = pd.DataFrame([{
        "continent": cont,
        "source_file": fp,
        "n_rows": n_rows,
        "n_cities": n_cities,
        "R2_x": R2_x, "AdjR2_x": AdjR2_x,
        "R2_y": R2_y, "AdjR2_y": AdjR2_y,
        "R2_S": R2_s, "AdjR2_S": AdjR2_s,
        "R2_M": R2_m, "AdjR2_M": AdjR2_m,
        "k_params": k,
        "rhs": rhs
    }])

    return cont, d, summ, tab_x, tab_y, tab_S, tab_M, contrib

# =========================================================
# RUN ALL CONTINENT FILES
# =========================================================
pattern = os.path.join(BASE_DIR, "**", "02_Model_SM_cleaned_used__*.csv")
files = sorted(glob.glob(pattern, recursive=True))
if not files:
    raise FileNotFoundError(f"No files found: {pattern}")

all_summary = []
all_contrib = []
skipped = []

for fp in files:
    try:
        cont, d_used, summ, tab_x, tab_y, tab_S, tab_M, contrib = run_one_file(fp)
        cont_safe = safe_name(cont)
        cont_dir = os.path.join(OUT_DIR, cont_safe)
        os.makedirs(cont_dir, exist_ok=True)

        # save per-continent Excel
        out_xlsx = os.path.join(cont_dir, f"Model2_DualTrack_Rcontrib__{cont_safe}.xlsx")
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            summ.to_excel(w, "R2_summary", index=False)
            tab_x.to_excel(w, "beta_x_dualtrack", index=False)
            tab_y.to_excel(w, "beta_y_dualtrack", index=False)
            tab_S.to_excel(w, "theta_S_dualtrack", index=False)
            tab_M.to_excel(w, "theta_M_dualtrack", index=False)
            contrib.sort_values("rank_S").to_excel(w, "Rcontrib_S_M", index=False)

        # save plots (Top10 ΔR² for S and M)
        plot_topk_bar(contrib,
                      title=f"{cont} — Top ΔR² contributors (Synergy axis S: Q1↔Q3)",
                      out_svg=os.path.join(cont_dir, f"Fig_Model2_Rcontrib__{cont_safe}__S.svg"),
                      out_png=os.path.join(cont_dir, f"Fig_Model2_Rcontrib__{cont_safe}__S.png"),
                      value_col="dR2_S",
                      topk=10)

        plot_topk_bar(contrib,
                      title=f"{cont} — Top ΔR² contributors (Mismatch axis M: Q4↔Q2)",
                      out_svg=os.path.join(cont_dir, f"Fig_Model2_Rcontrib__{cont_safe}__M.svg"),
                      out_png=os.path.join(cont_dir, f"Fig_Model2_Rcontrib__{cont_safe}__M.png"),
                      value_col="dR2_M",
                      topk=10)

        # save cleaned used (optional)
        d_used.to_csv(os.path.join(cont_dir, f"Model2_cleaned_used__{cont_safe}.csv"),
                      index=False, encoding="utf-8-sig")

        # collect global
        all_summary.append(summ)
        c2 = contrib.copy()
        c2.insert(0, "continent", cont)
        c2.insert(1, "source_file", fp)
        all_contrib.append(c2)

        print(f"[DONE] {cont} -> {out_xlsx}")

    except Exception as e:
        skipped.append((fp, str(e)))
        print(f"[SKIP] {fp}: {e}")

# =========================================================
# SAVE GLOBAL SUMMARY
# =========================================================
if all_summary:
    sum_df = pd.concat(all_summary, ignore_index=True).sort_values("continent")
    contrib_df = pd.concat(all_contrib, ignore_index=True)

    out_all = os.path.join(OUT_DIR, "00_Model2_DualTrack_Rcontrib_AllContinents.xlsx")
    out_all_csv = os.path.join(OUT_DIR, "00_Model2_DualTrack_Rcontrib_AllContinents_detail.csv")
    out_skip = os.path.join(OUT_DIR, "00_Model2_DualTrack_skipped.csv")

    with pd.ExcelWriter(out_all, engine="openpyxl") as w:
        sum_df.to_excel(w, "R2_summary", index=False)
        contrib_df.to_excel(w, "Rcontrib_detail", index=False)

        # top10 panels
        topS = contrib_df.sort_values(["continent","dR2_S"], ascending=[True, False]).groupby("continent").head(10)
        topM = contrib_df.sort_values(["continent","dR2_M"], ascending=[True, False]).groupby("continent").head(10)
        topS.to_excel(w, "Top10_S", index=False)
        topM.to_excel(w, "Top10_M", index=False)

    contrib_df.to_csv(out_all_csv, index=False, encoding="utf-8-sig")

    if skipped:
        pd.DataFrame(skipped, columns=["file","reason"]).to_csv(out_skip, index=False, encoding="utf-8-sig")

    print("========================================")
    print("Saved global summary:", out_all)
    print("Saved detail csv:", out_all_csv)
    if skipped:
        print("Saved skipped list:", out_skip)
else:
    print("No continent outputs generated. Check skipped messages.")