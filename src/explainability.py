"""
Explainability Module — SHAP + LIME
====================================
Provides global and local explanations for the HR turnover
prediction model using SHAP and LIME.
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def create_shap_explainer(model, X_train, feature_names, model_name="model"):
    """
    Create SHAP explainer appropriate for the model type.
    Returns explainer and shap_values.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"\n🔍 Computing SHAP values for: {model_name}")

    # Use TreeExplainer for tree-based models, otherwise KernelExplainer
    model_type = type(model).__name__
    tree_models = ["DecisionTreeClassifier", "RandomForestClassifier",
                   "GradientBoostingClassifier", "XGBClassifier"]

    if model_type in tree_models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # For binary classification, tree explainer may return list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 (terminated)
    else:
        # Use KernelExplainer for logistic regression
        # Sample background data for speed
        bg_data = shap.sample(pd.DataFrame(X_train, columns=feature_names), 50)
        explainer = shap.KernelExplainer(model.predict_proba, bg_data)
        shap_values = explainer.shap_values(
            pd.DataFrame(X_train[:100], columns=feature_names)
        )
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        X_train = X_train[:100]  # Match the subset

    print(f"  ✅ SHAP values computed: shape={np.array(shap_values).shape}")
    return explainer, shap_values


def plot_shap_summary(shap_values, X_data, feature_names, save_name="shap_summary"):
    """Generate and save SHAP summary plot (global feature importance)."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        pd.DataFrame(X_data, columns=feature_names),
        show=False,
        plot_size=(12, 8)
    )
    plt.title("SHAP Feature Importance — Employee Turnover Risk Factors", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{save_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")
    return path


def plot_shap_bar(shap_values, X_data, feature_names, save_name="shap_bar"):
    """Generate and save SHAP bar plot (mean absolute importance)."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        pd.DataFrame(X_data, columns=feature_names),
        plot_type="bar",
        show=False,
        plot_size=(10, 7)
    )
    plt.title("Mean |SHAP| — Feature Importance Ranking", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{save_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")
    return path


def plot_shap_force_single(explainer, shap_values, X_data, feature_names, idx=0,
                           save_name="shap_force_single"):
    """Generate SHAP force plot for a single prediction."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Use matplotlib-based force plot
    plt.figure(figsize=(14, 3))
    if hasattr(explainer, "expected_value"):
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = ev[1] if len(ev) > 1 else ev[0]
    else:
        ev = 0.5

    shap.force_plot(
        ev,
        shap_values[idx],
        pd.DataFrame(X_data, columns=feature_names).iloc[idx],
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Explanation — Employee #{idx}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{save_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")
    return path


def plot_shap_waterfall(explainer, shap_values, X_data, feature_names, idx=0,
                         save_name="shap_waterfall"):
    """Generate SHAP waterfall plot for a single prediction."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if hasattr(explainer, "expected_value"):
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = ev[1] if len(ev) > 1 else ev[0]
    else:
        ev = 0.5

    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=ev,
        data=X_data[idx] if isinstance(X_data, np.ndarray) else X_data.iloc[idx].values,
        feature_names=feature_names
    )

    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"SHAP Waterfall — Employee #{idx}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{save_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")
    return path


def explain_with_lime(model, X_train, X_instance, feature_names, class_names=None,
                      idx=0, save_name="lime_explanation"):
    """
    Generate LIME explanation for a single prediction.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if class_names is None:
        class_names = ["Active", "Terminated"]

    lime_explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=42
    )

    instance = X_instance[idx] if isinstance(X_instance, np.ndarray) else X_instance.iloc[idx].values

    exp = lime_explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=10,
        num_samples=500
    )

    # Save as HTML
    html_path = os.path.join(PLOTS_DIR, f"{save_name}.html")
    exp.save_to_file(html_path)

    # Save as image
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(12, 6)
    plt.title(f"LIME Explanation — Employee #{idx}", fontsize=14)
    plt.tight_layout()
    img_path = os.path.join(PLOTS_DIR, f"{save_name}.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✅ Saved LIME: {img_path}")
    print(f"  ✅ Saved LIME HTML: {html_path}")
    return exp, img_path


def generate_explanation_text(shap_values, feature_names, X_instance, idx=0, top_n=5):
    """
    Generate a human-readable, HR-friendly explanation of why an employee
    is predicted to leave (or stay). Uses plain language instead of code.
    """
    # Friendly name mapping
    friendly = {
        "Salary": "Salary",
        "Age": "Age",
        "Tenure_Years": "Time at the company",
        "EngagementSurvey": "Engagement score",
        "EmpSatisfaction": "Job satisfaction",
        "SpecialProjectsCount": "Special projects involvement",
        "DaysLateLast30": "Recent tardiness",
        "Absences": "Number of absences",
        "PerfScore_Numeric": "Performance rating",
        "Sex_Binary": "Gender",
        "MaritalStatus_Num": "Marital status",
        "Is_Manager": "Management role",
        "FromDiversityJobFairID": "Recruitment source",
    }

    sv = shap_values[idx]
    feature_impacts = list(zip(feature_names, sv))

    # Sort by absolute impact
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    risk_factors = []
    protective_factors = []

    for feat_name, impact in feature_impacts[:top_n]:
        name = friendly.get(feat_name, feat_name.replace("_", " ").replace("Dept ", "Department: "))
        if impact > 0:
            risk_factors.append(name)
        else:
            protective_factors.append(name)

    lines = []
    if risk_factors:
        lines.append("⚠️ **Factors that increase this employee's risk of leaving:**")
        for f in risk_factors:
            lines.append(f"  • {f}")
        lines.append("")

    if protective_factors:
        lines.append("✅ **Factors that help retain this employee:**")
        for f in protective_factors:
            lines.append(f"  • {f}")
        lines.append("")

    # Add actionable summary
    if len(risk_factors) > len(protective_factors):
        lines.append("📌 **Bottom line:** More factors are pushing this employee toward leaving than keeping them. HR should proactively address the risk factors listed above.")
    elif len(risk_factors) == 0:
        lines.append("📌 **Bottom line:** No significant risk factors detected. This employee appears well-retained under current conditions.")
    else:
        lines.append("📌 **Bottom line:** The protective factors outweigh the risks, but it's still worth monitoring the warning signs listed above.")

    return "\n".join(lines)


def run_explainability(model, X_train, X_test, y_test, feature_names, model_name="Best Model"):
    """Run the full explainability pipeline."""
    print("\n" + "=" * 60)
    print("🔍 EXPLAINABILITY PIPELINE")
    print("=" * 60)

    # 1. SHAP
    explainer, shap_values = create_shap_explainer(model, X_train, feature_names, model_name)
    plot_shap_summary(shap_values, X_train, feature_names)
    plot_shap_bar(shap_values, X_train, feature_names)

    # 2. Individual explanations for terminated employees
    terminated_indices = np.where(y_test == 1)[0]
    active_indices = np.where(y_test == 0)[0]

    if len(terminated_indices) > 0:
        idx = terminated_indices[0]
        explainer_test, shap_values_test = create_shap_explainer(
            model, X_test, feature_names, f"{model_name}_test"
        )
        plot_shap_waterfall(explainer_test, shap_values_test, X_test, feature_names,
                            idx=0, save_name="shap_waterfall_terminated")

        # SHAP force plot
        plot_shap_force_single(explainer_test, shap_values_test, X_test, feature_names,
                               idx=0, save_name="shap_force_terminated")

        # Text explanation
        text = generate_explanation_text(shap_values_test, feature_names, X_test, idx=0)
        print(f"\n{text}")

    # 3. LIME
    explain_with_lime(model, X_train, X_test, feature_names, idx=0,
                      save_name="lime_terminated")

    if len(active_indices) > 0:
        explain_with_lime(model, X_train, X_test, feature_names,
                          idx=active_indices[0], save_name="lime_active")

    print(f"\n✅ Explainability pipeline complete! Plots saved to: {PLOTS_DIR}")
    return explainer, shap_values


if __name__ == "__main__":
    import joblib
    from data_processing import run_pipeline
    from model_training import train_and_evaluate

    df_feat, X, y, feature_names, _ = run_pipeline()
    results, best_model, best_name, scaler, trained_models, X_test, y_test, X_test_scaled = train_and_evaluate(X, y, feature_names)

    X_scaled = scaler.transform(X)
    run_explainability(best_model, X_scaled, X_test_scaled, y_test, feature_names, best_name)
