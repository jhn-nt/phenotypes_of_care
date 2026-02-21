"""Replication of Pang et al. 2022 ICU Mortality Prediction
Implements:
  - Random downsampling
  - 70/30 train/test split
  - 5-fold CV + grid search
  - XGBoost only
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.calibration import calibration_curve
import pickle
import json

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "data", "final_cohort.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42

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


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Raw cohort: {len(df):,} patients")
    print(f"Deaths: {df[TARGET].sum():,} ({df[TARGET].mean()*100:.1f}%)")
    
    missing_cols = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    
    if X.isnull().any().any():
        print("Imputing missing values with median")
        X = X.fillna(X.median())
    
    return X, y, df


def random_downsample(X, y):
    data = X.copy()
    data[TARGET] = y
    dead = data[data[TARGET] == 1]
    alive = data[data[TARGET] == 0]
    n_dead = len(dead)
    alive_sample = alive.sample(n=n_dead, random_state=RANDOM_STATE)
    balanced = pd.concat([dead, alive_sample]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    X_bal = balanced[ALL_FEATURES]
    y_bal = balanced[TARGET]
    print(f"Balanced dataset: {len(X_bal):,} patients ({n_dead} dead / {n_dead} alive)")
    return X_bal, y_bal


def bootstrap_auc_ci(y_true, y_score, n_bootstrap=1000, ci=0.95):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
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
    return np.mean(aucs), np.percentile(aucs, 100*alpha), np.percentile(aucs, 100*(1-alpha))


def compute_metrics(y_true, y_prob, threshold=None):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / len(y_true)
    auc_mean, auc_low, auc_high = bootstrap_auc_ci(y_true, y_prob)
    
    return {
        "AUC": float(auc_mean),
        "AUC_CI_low": float(auc_low),
        "AUC_CI_high": float(auc_high),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "PPV": float(ppv),
        "NPV": float(npv),
        "Accuracy": float(accuracy),
        "Threshold": float(threshold),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def train_xgboost(X_train, y_train):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=1.0, colsample_bytree=1.0,
        gamma=0, min_child_weight=1,
        eval_metric="logloss", use_label_encoder=False,
        random_state=RANDOM_STATE, n_jobs=-1
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
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best CV AUC: {grid.best_score_:.3f}")
    return grid.best_estimator_


def plot_roc_curve(y_test, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_mean, auc_low, auc_high = bootstrap_auc_ci(y_test, y_prob)
    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, color="#E63946", lw=2.5, label=f"XGBoost\nAUC={auc_mean:.3f} ({auc_low:.3f}-{auc_high:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("1 − Specificity")
    plt.ylabel("Sensitivity")
    plt.title("ROC Curve — ICU Mortality Prediction")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_calibration_curve(y_test, y_prob, out_path):
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
    plt.figure(figsize=(8, 7))
    plt.plot(mean_pred, frac_pos, "o-", color="#E63946", label="XGBoost")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve — ICU Mortality Prediction")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_feature_importance(model, feature_names, out_path):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(vals)), vals[::-1], color=plt.cm.viridis(np.linspace(0.2,0.9,len(vals)))[::-1])
    plt.yticks(range(len(vals)), names[::-1])
    plt.xlabel("Feature Importance (Gain)")
    plt.title("XGBoost Feature Importance")
    plt.grid(axis="x", alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_markdown_summary(metrics, out_path):
    m = metrics
    md = f"""
# ICU Mortality Prediction — XGBoost

**Dataset**: Turning Care Cohort (balanced via random downsampling)  
**Train/Test Split**: 70/30

## Performance Metrics

| Metric | Value |
|--------|-------|
| AUC | {m['AUC']:.3f} ({m['AUC_CI_low']:.3f}–{m['AUC_CI_high']:.3f}) |
| Accuracy | {m['Accuracy']:.3f} |
| Sensitivity | {m['Sensitivity']:.3f} |
| Specificity | {m['Specificity']:.3f} |
| PPV | {m['PPV']:.3f} |
| NPV | {m['NPV']:.3f} |
| Threshold | {m['Threshold']:.3f} |
| TP / TN / FP / FN | {m['TP']} / {m['TN']} / {m['FP']} / {m['FN']} |

## Plots

- ROC Curve: `figure2_roc_curves.png`  
- Calibration Curve: `figure3_calibration.png`  
- Feature Importance: `figure4_feature_importance.png`
"""
    with open(out_path, "w") as f:
        f.write(md)
    print(f"Saved Markdown summary: {out_path}")


def main():
    X, y, _ = load_and_prepare_data(CSV_PATH)
    X_bal, y_bal = random_downsample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, stratify=y_bal, random_state=RANDOM_STATE
    )
    
    xgb_model = train_xgboost(X_train, y_train)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob)
    
    # Plots
    plot_roc_curve(y_test, y_prob, os.path.join(OUT_DIR, "replication_roc_curves.png"))
    plot_calibration_curve(y_test, y_prob, os.path.join(OUT_DIR, "replication_calibration.png"))
    plot_feature_importance(xgb_model, ALL_FEATURES, os.path.join(OUT_DIR, "replication_feature_importance.png"))
    
    
    df_pred = pd.DataFrame({
        "y_true": y_test,
        "xgb_prob": y_prob,
        "xgb_pred": (y_prob >= metrics["Threshold"]).astype(int)
    })
    df_pred.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)
    
    with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    generate_markdown_summary(metrics, os.path.join(OUT_DIR, "summary.md"))
    
    print("Replication complete. All outputs saved to:", OUT_DIR)
    return metrics


if __name__ == "__main__":
    results = main()