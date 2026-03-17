"""
Model Training Module — Frugal AI Comparison
=============================================
Trains 4 models of increasing complexity, tracks carbon emissions
with CodeCarbon, and compares performance vs resource usage.
"""

import pandas as pd
import numpy as np
import time
import json
import os
import warnings
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler

# Try importing CodeCarbon & XGBoost
try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False
    print("⚠️  CodeCarbon not installed. Carbon tracking disabled.")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Using GradientBoosting instead.")

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def get_models():
    """Return dictionary of models ordered by complexity (frugal → complex)."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=7, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric="logloss", scale_pos_weight=2
        )
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )

    return models


def train_and_evaluate(X, y, feature_names):
    """
    Train all models, track carbon, and compare results.

    Returns:
        results: dict with metrics for each model
        best_model: the trained best model
        scaler: the fitted scaler
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    print(f"📊 Features: {len(feature_names)}")

    models = get_models()
    results = {}
    trained_models = {}

    print("\n" + "=" * 70)
    print("🏋️  FRUGAL AI — MODEL COMPARISON")
    print("=" * 70)

    for name, model in models.items():
        print(f"\n{'─' * 50}")
        print(f"🔄 Training: {name}")
        print(f"{'─' * 50}")

        # Track carbon emissions
        emissions = 0.0
        energy = 0.0
        if HAS_CODECARBON:
            tracker = EmissionsTracker(
                project_name=f"HR-AI-{name}",
                measure_power_secs=1,
                save_to_file=False,
                log_level="error"
            )
            tracker.start()

        # Train with timing
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time

        if HAS_CODECARBON:
            emissions = tracker.stop()
            if emissions is None:
                emissions = 0.0
            energy = getattr(tracker, '_total_energy', 0.0)
            if hasattr(tracker, '_total_energy'):
                energy = tracker._total_energy.kWh if hasattr(tracker._total_energy, 'kWh') else 0.0

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        try:
            auc_roc = roc_auc_score(y_test, y_prob)
        except:
            auc_roc = 0.0

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="f1")

        # Model complexity metrics
        n_params = _count_parameters(model, name)

        results[name] = {
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "auc_roc": round(auc_roc, 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
            "train_time_seconds": round(train_time, 4),
            "carbon_emissions_kg": round(emissions * 1000, 6) if emissions else 0.0,
            "energy_kwh": round(energy, 8) if energy else 0.0,
            "n_parameters": n_params,
        }

        trained_models[name] = model

        # Print results
        print(f"  ✅ Accuracy:    {accuracy:.4f}")
        print(f"  ✅ F1-Score:    {f1:.4f}")
        print(f"  ✅ Precision:   {precision:.4f}")
        print(f"  ✅ Recall:      {recall:.4f}")
        print(f"  ✅ AUC-ROC:     {auc_roc:.4f}")
        print(f"  ✅ CV F1 Mean:  {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"  ⏱️  Train time:  {train_time:.4f}s")
        if emissions:
            print(f"  🌱 CO₂ (kg):    {emissions*1000:.6f}")
        print(f"  📐 Parameters:  {n_params}")

    # Select best frugal model (best F1/complexity ratio)
    best_name = _select_frugal_model(results)
    best_model = trained_models[best_name]

    print(f"\n{'=' * 70}")
    print(f"🏆 BEST FRUGAL MODEL: {best_name}")
    print(f"   F1={results[best_name]['f1_score']}, AUC={results[best_name]['auc_roc']}")
    print(f"   Time={results[best_name]['train_time_seconds']}s")
    print(f"{'=' * 70}")

    # Save best model and scaler
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    # Save all models
    for name, model in trained_models.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODEL_DIR, f"{safe_name}.pkl"))

    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))

    with open(os.path.join(RESULTS_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved models to: {MODEL_DIR}")
    print(f"✅ Saved results to: {RESULTS_DIR}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("📊 FRUGAL AI — COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model':<25} {'F1':>6} {'AUC':>6} {'Time(s)':>8} {'Params':>8}")
    print("-" * 60)
    for name, r in results.items():
        marker = " 🏆" if name == best_name else ""
        print(f"{name:<25} {r['f1_score']:>6.4f} {r['auc_roc']:>6.4f} {r['train_time_seconds']:>8.4f} {r['n_parameters']:>8}{marker}")

    return results, best_model, best_name, scaler, trained_models, X_test, y_test, X_test_scaled


def _count_parameters(model, name):
    """Estimate number of parameters for model complexity comparison."""
    if "Logistic" in name:
        return len(model.coef_.flatten()) + len(model.intercept_)
    elif "Decision Tree" in name:
        return model.tree_.node_count
    elif "Random Forest" in name:
        return sum(est.tree_.node_count for est in model.estimators_)
    elif "XGBoost" in name or "Gradient" in name:
        try:
            return model.n_estimators * 50  # approximate
        except:
            return 5000
    return 0


def _select_frugal_model(results):
    """
    Select the best frugal model — prioritize performance
    but penalize unnecessary complexity.
    """
    scores = {}
    for name, r in results.items():
        # Frugal score: high F1 with low complexity
        f1 = r["f1_score"]
        params = max(r["n_parameters"], 1)
        time_factor = max(r["train_time_seconds"], 0.001)

        # Efficiency score: performance per unit of complexity
        efficiency = f1 / np.log1p(params)
        scores[name] = efficiency

    # If simplest model is within 5% of best F1, prefer it
    f1_scores = {n: r["f1_score"] for n, r in results.items()}
    best_f1_name = max(f1_scores, key=f1_scores.get)
    best_f1 = f1_scores[best_f1_name]

    model_names = list(results.keys())
    for name in model_names:  # ordered from simple to complex
        if f1_scores[name] >= best_f1 * 0.95:
            return name

    return max(scores, key=scores.get)


if __name__ == "__main__":
    from data_processing import load_raw_data, anonymize_data, engineer_features, get_model_features, run_pipeline
    # Run full pipeline (generates all files including synthetic)
    run_pipeline()
    # But train models on ORIGINAL data only for best performance
    import pandas as pd
    from data_processing import ANONYMIZED_PATH
    df_orig = pd.read_csv(ANONYMIZED_PATH)
    X, y, feature_names = get_model_features(df_orig)
    print(f"\n🎯 Training models on ORIGINAL data ({len(df_orig)} employees) for best quality...")
    train_and_evaluate(X, y, feature_names)

