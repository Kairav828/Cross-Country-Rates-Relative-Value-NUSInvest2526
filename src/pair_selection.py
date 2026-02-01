"""
Pair Eligibility Selection

Goal:
- Combine rolling correlation stability (pair-level),
  clustering persistence (system-level),
  PCA factor neutrality + eigenvector instability (factor-level)
  into a single eligibility table + defendable decisions.

Inputs:
- results/rolling_corr_pair_metrics.csv
- results/cluster_pair_persistence.csv
- results/pca_5Y_fullsample_loadings.csv
- results/pca_5Y_pc1_loadings_low_vol.csv
- results/pca_5Y_pc1_loadings_high_vol.csv
- results/pca_5Y_rolling_pc1_metrics.csv

Outputs:
- results/pair_eligibility_table.csv
- results/pair_eligibility_summary.md

Run:
    python -m src.pair_selection_task6
"""

from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"

ROLLING_CORR_FILE = RESULTS / "rolling_corr_pair_metrics.csv"
CLUSTER_FILE = RESULTS / "cluster_pair_persistence.csv"
PCA_FULL_FILE = RESULTS / "pca_5Y_fullsample_loadings.csv"
PCA_LOW_FILE = RESULTS / "pca_5Y_pc1_loadings_low_vol.csv"
PCA_HIGH_FILE = RESULTS / "pca_5Y_pc1_loadings_high_vol.csv"
PCA_ROLL_METRICS_FILE = RESULTS / "pca_5Y_rolling_pc1_metrics.csv"

OUT_TABLE = RESULTS / "pair_eligibility_table.csv"
OUT_MD = RESULTS / "pair_eligibility_summary.md"


# -----------------------------
# Helpers
# -----------------------------
def clean_asset_to_ccy(asset: str) -> str:
    """
    'GTUSD5Y Govt'  -> 'USD'
    'GTEUR10Y Govt' -> 'EUR'
    """
    s = asset.replace("GT", "").replace(" Govt", "")
    # strip tenor
    if s.endswith("5Y"):
        return s[:-2]
    if s.endswith("10Y"):
        return s[:-3]
    return s

def load_pc1_series(pca_full: pd.DataFrame) -> dict:
    """
    Map asset -> PC1 loading (full sample PCA, 5Y only in this repo).
    """
    # The PCA file has column 'Unnamed: 0' for asset names, and 'PC1' for loadings
    return dict(zip(pca_full["Unnamed: 0"], pca_full["PC1"]))


def load_regime_pc1_map(regime_file: Path) -> dict:
    """
    The low/high vol PC1 files have columns:
      Unnamed: 0  (asset)
      0          (loading)
    """
    df = pd.read_csv(regime_file)
    return dict(zip(df["Unnamed: 0"], df["0"]))


# -----------------------------
# Gate thresholds (tuneable)
# -----------------------------
THRESH = {
    # Gate 1: correlation stability (252d rolling corr summary)
    "corr_mean_min": 0.30,      # require at least moderate average co-movement
    "corr_std_max": 0.20,       # too jumpy => unstable
    "corr_p05_min": 0.00,       # if 5th percentile is negative -> sign flips risk

    # Gate 2: clustering persistence (10Y clustering; applied as bloc sanity check)
    "cocluster_min": 60.0,      # % of windows same cluster

    # Gate 3: PCA factor neutrality (equal-weight spread exposure to PC1)
    # For spread (i - j), PC1 exposure is proportional to |l_i - l_j|
    "pc1_diff_full_max": 0.10,
    "pc1_diff_low_max": 0.15,
    "pc1_diff_high_max": 0.15,

    # Gate 4: PCA structural stability
    # If PC1 rotates frequently, "global factor" meaning is unstable.
    "pc1_cosine_good": 0.80,     # "aligned with full sample" threshold
    "pc1_cosine_good_frac": 0.55 # fraction of time above threshold required (repo reality is ~0.57)
}


def main():
    # -----------------------------
    # Load all required inputs
    # -----------------------------
    corr = pd.read_csv(ROLLING_CORR_FILE)
    clusters = pd.read_csv(CLUSTER_FILE)
    pca_full = pd.read_csv(PCA_FULL_FILE)
    pca_roll = pd.read_csv(PCA_ROLL_METRICS_FILE)

    pc1_full = load_pc1_series(pca_full)
    pc1_low = load_regime_pc1_map(PCA_LOW_FILE)
    pc1_high = load_regime_pc1_map(PCA_HIGH_FILE)

    # -----------------------------
    # Prepare cluster map (country-level)
    # -----------------------------
    # cluster file is by Country_A / Country_B
    # build symmetric lookup
    cocluster = {}
    for _, r in clusters.iterrows():
        a, b = r["Country_A"], r["Country_B"]
        cocluster[(a, b)] = float(r["CoCluster_Pct"])
        cocluster[(b, a)] = float(r["CoCluster_Pct"])

    # -----------------------------
    # PCA structural stability summary (global gate, not pair-specific)
    # -----------------------------
    cosine = pca_roll["pc1_cosine_to_fullsample"].dropna()
    cosine_good_frac = float((cosine > THRESH["pc1_cosine_good"]).mean())

    structural_note = (
        f"Rolling PCA PC1 cosine-to-fullsample: "
        f"{cosine_good_frac:.3f} of windows > {THRESH['pc1_cosine_good']}. "
        f"(Low values imply frequent eigenvector rotation.)"
    )

    # -----------------------------
    # Build pair table from rolling corr (this defines the universe)
    # -----------------------------
    rows = []
    for _, r in corr.iterrows():
        tenor = r["Tenor"]
        a_asset = r["Asset_A"]
        b_asset = r["Asset_B"]

        a_ccy = clean_asset_to_ccy(a_asset)
        b_ccy = clean_asset_to_ccy(b_asset)

        # Gate 1 metrics
        corr_mean = float(r["mean"])
        corr_std = float(r["std"])
        corr_p05 = float(r["p05"])

        gate1_pass = (
            (corr_mean >= THRESH["corr_mean_min"]) and
            (corr_std <= THRESH["corr_std_max"]) and
            (corr_p05 >= THRESH["corr_p05_min"])
        )

        # Gate 2 metric (clustering persistence from 10Y clustering)
        coc = cocluster.get((a_ccy, b_ccy), np.nan)
        gate2_pass = (not np.isnan(coc)) and (coc >= THRESH["cocluster_min"])

        # Gate 3 metrics (PCA neutrality, 5Y PCA used as factor proxy)
        # If PCA file doesn't have this asset (should exist for 5Y), mark NaN.
        l_full_a = pc1_full.get(a_asset, np.nan)
        l_full_b = pc1_full.get(b_asset, np.nan)
        pc1_diff_full = float(abs(l_full_a - l_full_b)) if (not np.isnan(l_full_a) and not np.isnan(l_full_b)) else np.nan

        l_low_a = pc1_low.get(a_asset, np.nan)
        l_low_b = pc1_low.get(b_asset, np.nan)
        pc1_diff_low = float(abs(l_low_a - l_low_b)) if (not np.isnan(l_low_a) and not np.isnan(l_low_b)) else np.nan

        l_high_a = pc1_high.get(a_asset, np.nan)
        l_high_b = pc1_high.get(b_asset, np.nan)
        pc1_diff_high = float(abs(l_high_a - l_high_b)) if (not np.isnan(l_high_a) and not np.isnan(l_high_b)) else np.nan

        # Gate 3 pass requires neutrality in full sample AND both regime splits (if available)
        # If a regime split metric is NaN (e.g., if PCA assets missing), treat as fail: we can't defend it.
        gate3_pass = (
            (not np.isnan(pc1_diff_full)) and pc1_diff_full <= THRESH["pc1_diff_full_max"] and
            (not np.isnan(pc1_diff_low)) and pc1_diff_low <= THRESH["pc1_diff_low_max"] and
            (not np.isnan(pc1_diff_high)) and pc1_diff_high <= THRESH["pc1_diff_high_max"]
        )

        # Gate 4: structural stability is global; we treat this as "regime restriction risk"
        gate4_pass = cosine_good_frac >= THRESH["pc1_cosine_good_frac"]

        # -----------------------------
        # Decision logic
        # -----------------------------
        # Eligible:
        #   must pass Gates 1–3
        # Structural stability (Gate 4) controls whether we label it "eligible" vs "eligible but regime-watch".
        #
        # Conditionally eligible:
        #   passes bloc + factor neutrality, but correlation stability is weaker (Gate 1 fails)
        #
        # Rejected:
        #   fails clustering persistence OR fails PCA neutrality
        if gate2_pass and gate3_pass and gate1_pass:
            decision = "Eligible" if gate4_pass else "Eligible (watch factor rotation)"
        elif gate2_pass and gate3_pass and (not gate1_pass):
            decision = "Conditionally eligible (needs regime gating)"
        else:
            decision = "Rejected"

        rows.append({
            "tenor": tenor,
            "asset_a": a_asset,
            "asset_b": b_asset,
            "ccy_a": a_ccy,
            "ccy_b": b_ccy,

            # Gate 1 evidence
            "corr_mean": corr_mean,
            "corr_std": corr_std,
            "corr_p05": corr_p05,
            "gate1_corr_stable": gate1_pass,

            # Gate 2 evidence
            "cocluster_pct_10y": coc,
            "gate2_cluster_persistent": gate2_pass,

            # Gate 3 evidence
            "pc1_diff_full": pc1_diff_full,
            "pc1_diff_lowvol": pc1_diff_low,
            "pc1_diff_highvol": pc1_diff_high,
            "gate3_pc1_neutral": gate3_pass,

            # Gate 4 evidence (global)
            "pc1_cosine_good_frac": cosine_good_frac,
            "gate4_factor_stable_global": gate4_pass,

            "decision": decision
        })

    out = pd.DataFrame(rows).sort_values(["decision", "tenor"], ascending=[True, True])
    out.to_csv(OUT_TABLE, index=False)

    # -----------------------------
    # Defendable markdown summary
    # -----------------------------
    eligible = out[out["decision"].str.contains("Eligible")].copy()
    cond = out[out["decision"].str.contains("Conditionally")].copy()
    rej = out[out["decision"] == "Rejected"].copy()

    def pair_name(r):
        return f"{r['ccy_a']}-{r['ccy_b']} ({r['tenor']})"

    lines = []
    lines.append("# Task 6 — Pair Eligibility Summary\n")
    lines.append("## Global structural note (Gate 4)\n")
    lines.append(f"- {structural_note}\n")
    lines.append("Interpretation: even if a pair is RV-neutral under full-sample PCA, frequent PC1 rotation means we must expect regime-dependent behaviour.\n")

    lines.append("## Eligible pairs\n")
    if len(eligible) == 0:
        lines.append("- None\n")
    else:
        for _, r in eligible.iterrows():
            lines.append(
                f"- **{pair_name(r)}** — "
                f"G1 corr(mean={r['corr_mean']:.2f}, std={r['corr_std']:.2f}, p05={r['corr_p05']:.2f}); "
                f"G2 cocluster={r['cocluster_pct_10y']:.1f}%; "
                f"G3 pc1_diff(full={r['pc1_diff_full']:.3f}, low={r['pc1_diff_lowvol']:.3f}, high={r['pc1_diff_highvol']:.3f})."
            )

    lines.append("\n## Conditionally eligible pairs\n")
    if len(cond) == 0:
        lines.append("- None\n")
    else:
        for _, r in cond.iterrows():
            lines.append(
                f"- **{pair_name(r)}** — passes bloc+neutrality but weaker corr stability; "
                f"G1 corr(mean={r['corr_mean']:.2f}, std={r['corr_std']:.2f}, p05={r['corr_p05']:.2f}); "
                f"G2 cocluster={r['cocluster_pct_10y']:.1f}%; "
                f"G3 pc1_diff(full={r['pc1_diff_full']:.3f}, low={r['pc1_diff_lowvol']:.3f}, high={r['pc1_diff_highvol']:.3f}). "
                f"Treat as regime-restricted candidate."
            )

    lines.append("\n## Rejected pairs (reason codes)\n")
    lines.append("- Any pair failing clustering persistence (Gate 2) is rejected (not the same structural bloc).\n")
    lines.append("- Any pair failing PCA neutrality across regimes (Gate 3) is rejected (directional disguise risk).\n")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] wrote {OUT_TABLE}")
    print(f"[OK] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
