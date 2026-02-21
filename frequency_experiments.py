"""Experiment 1: Care-Frequency Quartile Evaluation

Trains two XGBoost models and evaluates each per care-frequency quartile:
  - Model 1: Trained on all patients (all quartiles)
  - Model 2: Trained on Q4 (best-care) patients only

Both models are tested on 30% held-out data from EACH quartile (Q1–Q4),
revealing the fairness gap between care quality tiers.

Uses final_cohort_with_race.csv produced by merge_race.py.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "data", "final_cohort_with_race.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "results", "frequency")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
MIN_POS_FOR_AUC = 30

APS3_FEATURES = [
    "apsiii_heartrate", "apsiii_meanbp", "apsiii_temp", "apsiii_resprate",
    "apsiii_pao2_aado2", "apsiii_hematocrit", "apsiii_wbc", "apsiii_creatinine",
    "apsiii_uo", "apsiii_bun", "apsiii_sodium", "apsiii_albumin",
    "apsiii_bilirubin", "apsiii_glucose", "apsiii_acidbase", "apsiii_gcs", "apsiii",
]
LODS_FEATURES = [
    "lods_neurologic", "lods_cardiovascular", "lods_renal",
    "lods_pulmonary", "lods_hematologic", "lods_hepatic",
]
ALL_FEATURES = APS3_FEATURES + LODS_FEATURES
TARGET = "hospital_expire_flag"
QUARTILE_ORDER = ["Q1", "Q2", "Q3", "Q4"]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def bootstrap_auc_ci(y_true, y_score, n_bootstrap=1000, ci=0.95):
    y_true, y_score = np.array(y_true), np.array(y_score)
    rng = np.random.default_rng(RANDOM_STATE)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    return np.mean(aucs), np.percentile(aucs, 100 * alpha), np.percentile(aucs, 100 * (1 - alpha))


def random_downsample(X, y):
    data = X.copy()
    data[TARGET] = y.values
    dead = data[data[TARGET] == 1]
    alive = data[data[TARGET] == 0]
    n_dead = len(dead)
    if n_dead == 0 or len(alive) == 0:
        return X, y
    alive_sample = alive.sample(n=min(n_dead, len(alive)), random_state=RANDOM_STATE)
    balanced = pd.concat([dead, alive_sample]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return balanced[ALL_FEATURES], balanced[TARGET]


def train_xgboost(X_train, y_train):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=1.0, colsample_bytree=1.0,
        gamma=0, min_child_weight=1,
        eval_metric="logloss", use_label_encoder=False,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    grid = GridSearchCV(model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV AUC: {grid.best_score_:.3f}")
    return grid.best_estimator_


def compute_full_metrics(y_true, y_prob):
    from sklearn.metrics import confusion_matrix
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos < 2 or n_neg < 2:
        return None
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(y_true, y_prob)
    return {
        "AUC": round(float(auc_mean), 4),
        "CI_low": round(float(auc_lo), 4),
        "CI_high": round(float(auc_hi), 4),
        "Sensitivity": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0, 4),
        "Specificity": round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0, 4),
        "PPV": round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0, 4),
        "NPV": round(float(tn / (tn + fn)) if (tn + fn) > 0 else 0, 4),
        "Accuracy": round(float((tp + tn) / len(y_true)), 4),
        "n": int(len(y_true)), "n_pos": n_pos,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    print(f"Loading {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df):,} rows, {df['stay_id'].nunique():,} patients")

    for col in ALL_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def assign_quartiles(df):
    patient = df.groupby("stay_id")["average_item_interval"].first().reset_index()
    patient["quartile"] = pd.qcut(
        patient["average_item_interval"],
        q=4,
        labels=["Q4", "Q3", "Q2", "Q1"],
    )
    df = df.merge(patient[["stay_id", "quartile"]], on="stay_id", how="left")
    print(f"\n  Quartile distribution (patients):")
    for q in QUARTILE_ORDER:
        n = (patient["quartile"] == q).sum()
        print(f"    {q}: {n:,} patients")
    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model_1(df):
    """Train Model 1 on all patients (all quartiles)."""
    print("\n" + "=" * 60)
    print("Training Model 1 (all patients, all quartiles)")
    print("=" * 60)

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    X_bal, y_bal = random_downsample(X, y)
    print(f"  Balanced: {len(X_bal):,} rows")

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, stratify=y_bal, random_state=RANDOM_STATE,
    )

    return train_xgboost(X_train, y_train)


def train_model_2(df):
    """Train Model 2 on Q4 (best-care) patients only."""
    print("\n" + "=" * 60)
    print("Training Model 2 (Q4 only)")
    print("=" * 60)

    q4 = df[df["quartile"] == "Q4"].copy()
    print(f"  Q4 subset: {len(q4):,} rows, {q4['stay_id'].nunique():,} patients")
    print(f"  Q4 deaths: {q4[TARGET].sum():,} ({q4[TARGET].mean()*100:.1f}%)")

    X_q4 = q4[ALL_FEATURES]
    y_q4 = q4[TARGET]
    X_bal, y_bal = random_downsample(X_q4, y_q4)
    print(f"  Balanced Q4: {len(X_bal):,} rows")

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, stratify=y_bal, random_state=RANDOM_STATE,
    )

    return train_xgboost(X_train, y_train)


# ---------------------------------------------------------------------------
# Per-Quartile Evaluation
# ---------------------------------------------------------------------------

def quartile_holdout_table(df, model, model_name):
    """Evaluate a trained model on 30% held-out from each quartile."""
    print(f"\n{'=' * 100}")
    print(f"{model_name} RESULTS BY QUARTILE")
    print(f"{'=' * 100}")

    fmt_header = (f"{'Test Set':>40s} {'N':>6s} {'Deaths':>7s} {'Mortality Rate':>15s} "
                  f"{'AUC':>6s} {'AUC 95% CI':>14s} {'Sensitivity':>12s} "
                  f"{'Specificity':>12s} {'PPV':>6s} {'NPV':>6s} {'Accuracy':>9s}")
    print(fmt_header)

    results = {}
    all_y_true = []
    all_y_prob = []

    for q in QUARTILE_ORDER:
        q_mask = df["quartile"] == q
        q_data = df[q_mask].copy()
        X_q = q_data[ALL_FEATURES]
        y_q = q_data[TARGET]

        if y_q.nunique() < 2 or y_q.sum() < MIN_POS_FOR_AUC:
            continue

        _, X_test, _, y_test = train_test_split(
            X_q, y_q, test_size=0.3, stratify=y_q, random_state=RANDOM_STATE,
        )
        label = f"{model_name} – Test {q} (30%)"

        y_prob = model.predict_proba(X_test)[:, 1]
        m = compute_full_metrics(np.array(y_test), y_prob)
        if m:
            results[q] = m
            all_y_true.extend(y_test.values)
            all_y_prob.extend(y_prob)

            n = m["n"]
            deaths = m["n_pos"]
            mort_rate = deaths / n * 100
            ci_str = f"{m['CI_low']:.3f}-{m['CI_high']:.3f}"

            print(f"{label:>40s} {n:>6d} {deaths:>7d} {mort_rate:>14.1f}% "
                  f"{m['AUC']:>6.3f} {ci_str:>14s} {m['Sensitivity']:>12.3f} "
                  f"{m['Specificity']:>12.3f} {m['PPV']:>6.3f} {m['NPV']:>6.3f} {m['Accuracy']:>9.3f}")

    if all_y_true:
        combined = compute_full_metrics(np.array(all_y_true), np.array(all_y_prob))
        if combined:
            results["combined"] = combined
            n = combined["n"]
            deaths = combined["n_pos"]
            mort_rate = deaths / n * 100
            ci_str = f"{combined['CI_low']:.3f}-{combined['CI_high']:.3f}"
            comb_label = f"{model_name} – All Quartiles Combined (30%)"
            print(f"{comb_label:>40s} {n:>6d} {deaths:>7d} {mort_rate:>14.1f}% "
                  f"{combined['AUC']:>6.3f} {ci_str:>14s} {combined['Sensitivity']:>12.3f} "
                  f"{combined['Specificity']:>12.3f} {combined['PPV']:>6.3f} {combined['NPV']:>6.3f} {combined['Accuracy']:>9.3f}")

    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_auc_by_quartile(results_m1, results_m2):
    """Grouped bar chart: AUC by quartile for Model 1 vs Model 2."""
    quartiles = [q for q in QUARTILE_ORDER if q in results_m1 and q in results_m2]
    auc_m1 = [results_m1[q]["AUC"] for q in quartiles]
    auc_m2 = [results_m2[q]["AUC"] for q in quartiles]
    ci_lo_m1 = [results_m1[q]["AUC"] - results_m1[q]["CI_low"] for q in quartiles]
    ci_hi_m1 = [results_m1[q]["CI_high"] - results_m1[q]["AUC"] for q in quartiles]
    ci_lo_m2 = [results_m2[q]["AUC"] - results_m2[q]["CI_low"] for q in quartiles]
    ci_hi_m2 = [results_m2[q]["CI_high"] - results_m2[q]["AUC"] for q in quartiles]

    x = np.arange(len(quartiles))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, auc_m1, width, yerr=[ci_lo_m1, ci_hi_m1],
           label="Model 1 (All Quartiles)", color="#457B9D", capsize=4)
    ax.bar(x + width / 2, auc_m2, width, yerr=[ci_lo_m2, ci_hi_m2],
           label="Model 2 (Q4 Only)", color="#E63946", capsize=4)

    ax.set_xlabel("Care-Frequency Quartile")
    ax.set_ylabel("AUC")
    ax.set_title("AUC by Care-Frequency Quartile — Model 1 vs Model 2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q}\n({'worst' if q == 'Q1' else 'best' if q == 'Q4' else ''})" for q in quartiles])
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out_path = os.path.join(OUT_DIR, "auc_by_quartile.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {out_path}")


def plot_fairness_gap_by_quartile(results_m1, results_m2):
    """Bar chart: AUC delta (Model 1 - Model 2) per quartile."""
    quartiles = [q for q in QUARTILE_ORDER if q in results_m1 and q in results_m2]
    gaps = [results_m1[q]["AUC"] - results_m2[q]["AUC"] for q in quartiles]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(quartiles, gaps, color="#E63946", width=0.5)

    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{gap:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_xlabel("Care-Frequency Quartile")
    ax.set_ylabel("AUC Gap (Model 1 - Model 2)")
    ax.set_title("Fairness Gap by Care-Frequency Quartile")
    ax.set_ylim(0, max(gaps) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    out_path = os.path.join(OUT_DIR, "fairness_gap_by_quartile.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_sensitivity_specificity_by_quartile(results_m2):
    """Grouped bar: Sensitivity vs Specificity for Model 2 across quartiles."""
    quartiles = [q for q in QUARTILE_ORDER if q in results_m2]
    sens = [results_m2[q]["Sensitivity"] for q in quartiles]
    spec = [results_m2[q]["Specificity"] for q in quartiles]

    x = np.arange(len(quartiles))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, sens, width, label="Sensitivity", color="#F4A261")
    ax.bar(x + width / 2, spec, width, label="Specificity", color="#2A9D8F")

    ax.set_xlabel("Care-Frequency Quartile")
    ax.set_ylabel("Score")
    ax.set_title("Model 2 (Q4-Trained) — Sensitivity vs Specificity by Quartile")
    ax.set_xticks(x)
    ax.set_xticklabels(quartiles)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out_path = os.path.join(OUT_DIR, "sensitivity_specificity_by_quartile.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def save_csvs(results_m1, results_m2):
    for model_num, results in [(1, results_m1), (2, results_m2)]:
        if not results:
            continue
        rows = []
        for key, m in results.items():
            rows.append({"Test Set": key, "Model": f"Model {model_num}", **m})
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, f"quartile_holdout_model_{model_num}.csv"),
            index=False,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()
    df = assign_quartiles(df)

    model_1 = train_model_1(df)
    model_2 = train_model_2(df)

    results_m1 = quartile_holdout_table(df, model_1, "Model 1")
    results_m2 = quartile_holdout_table(df, model_2, "Model 2")

    plot_auc_by_quartile(results_m1, results_m2)
    plot_fairness_gap_by_quartile(results_m1, results_m2)
    plot_sensitivity_specificity_by_quartile(results_m2)
    save_csvs(results_m1, results_m2)

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 DONE")
    print(f"Results saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
