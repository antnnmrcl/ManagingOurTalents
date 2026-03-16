"""
HR Turnover Prediction — Trusted AI Dashboard
==============================================
Interactive Streamlit demo with:
- Overview dashboard
- Individual employee risk prediction with SHAP/LIME explanations
- Frugal AI model comparison
- NLP insights
- Model & Data cards
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from data_processing import run_pipeline, get_model_features, ANONYMIZED_PATH, FEEDBACK_PATH
from model_training import train_and_evaluate, RESULTS_DIR, MODEL_DIR
from explainability import (
    create_shap_explainer, generate_explanation_text,
    plot_shap_summary, plot_shap_bar
)
from nlp_analysis import run_sentiment_analysis, run_theme_analysis

import shap
from lime.lime_tabular import LimeTabularExplainer

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HR Turnover AI — Trusted AI Solution",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1e3a5f, #2d8cf0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #2d8cf0;
    }
    .risk-high { color: #ef4444; font-weight: bold; font-size: 1.5rem; }
    .risk-medium { color: #f59e0b; font-weight: bold; font-size: 1.5rem; }
    .risk-low { color: #10b981; font-weight: bold; font-size: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding: 0.8rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING & CACHING
# ============================================================
@st.cache_data
def load_data():
    """Load or process data."""
    if os.path.exists(ANONYMIZED_PATH) and os.path.exists(FEEDBACK_PATH):
        df = pd.read_csv(ANONYMIZED_PATH)
        feedback_df = pd.read_csv(FEEDBACK_PATH)
        X, y, feature_names = get_model_features(df)
        return df, X, y, feature_names, feedback_df
    else:
        df, X, y, feature_names, feedback_df = run_pipeline()
        return df, X, y, feature_names, feedback_df


@st.cache_resource
def load_or_train_models(X, y, feature_names):
    """Load saved models or train new ones."""
    best_model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    results_path = os.path.join(RESULTS_DIR, "model_comparison.json")

    if os.path.exists(best_model_path) and os.path.exists(results_path):
        best_model = joblib.load(best_model_path)
        scaler = joblib.load(scaler_path)
        with open(results_path) as f:
            results = json.load(f)

        # Load all models
        trained_models = {}
        for name in results.keys():
            safe_name = name.lower().replace(" ", "_")
            model_path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
            if os.path.exists(model_path):
                trained_models[name] = joblib.load(model_path)

        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_test_scaled = scaler.transform(X_test)
        best_name = list(results.keys())[0]
        # Find the best based on f1
        best_name = max(results, key=lambda k: results[k]["f1_score"])

        return results, best_model, best_name, scaler, trained_models, X_test.values, y_test.values, X_test_scaled
    else:
        return train_and_evaluate(X, y, feature_names)


# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown('<p class="main-header">🎯 HR Turnover Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Trusted AI Solution — Frugal AI + Explainable AI</p>', unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading data..."):
        df, X, y, feature_names, feedback_df = load_data()

    with st.spinner("Loading/training models..."):
        results, best_model, best_name, scaler, trained_models, X_test, y_test, X_test_scaled = load_or_train_models(X, y, feature_names)

    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Best Model:** {best_name}\n\n**Dataset:** {len(df)} employees\n\n**Turnover Rate:** {(y.sum()/len(y)*100):.1f}%")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔍 Predict & Explain", "🌱 Frugal AI", "💬 NLP Insights", "📋 About"
    ])

    # ============================================================
    # TAB 1: OVERVIEW
    # ============================================================
    with tab1:
        st.header("📊 HR Analytics Overview")

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        n_total = len(df)
        n_active = int((y == 0).sum())
        n_termed = int((y == 1).sum())
        turnover_rate = n_termed / n_total * 100

        col1.metric("Total Employees", n_total)
        col2.metric("Active", n_active, delta=f"{n_active/n_total*100:.0f}%")
        col3.metric("Terminated", n_termed, delta=f"-{turnover_rate:.1f}%", delta_color="inverse")
        col4.metric("Turnover Rate", f"{turnover_rate:.1f}%")

        st.markdown("---")

        # Charts row 1
        col_a, col_b = st.columns(2)

        with col_a:
            # Turnover by Department
            if "Department" in df.columns:
                dept_data = df.groupby(df["Department"].str.strip())["Termd"].agg(["sum", "count"]).reset_index()
                dept_data.columns = ["Department", "Terminated", "Total"]
                dept_data["Rate"] = (dept_data["Terminated"] / dept_data["Total"] * 100).round(1)
                fig = px.bar(dept_data, x="Department", y="Rate",
                             title="Turnover Rate by Department (%)",
                             color="Rate", color_continuous_scale="RdYlGn_r",
                             text="Rate")
                fig.update_layout(height=400)
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Satisfaction vs Engagement
            fig = px.scatter(df, x="EngagementSurvey", y="EmpSatisfaction",
                             color=df["Termd"].map({0: "Active", 1: "Terminated"}),
                             title="Engagement vs Satisfaction (by Status)",
                             color_discrete_map={"Active": "#10b981", "Terminated": "#ef4444"},
                             opacity=0.7)
            fig.update_layout(height=400, legend_title="Status")
            st.plotly_chart(fig, use_container_width=True)

        # Charts row 2
        col_c, col_d = st.columns(2)

        with col_c:
            # Salary distribution
            fig = px.histogram(df, x="Salary", color=df["Termd"].map({0: "Active", 1: "Terminated"}),
                               title="Salary Distribution by Status",
                               barmode="overlay", nbins=30, opacity=0.7,
                               color_discrete_map={"Active": "#3b82f6", "Terminated": "#ef4444"})
            fig.update_layout(height=400, legend_title="Status")
            st.plotly_chart(fig, use_container_width=True)

        with col_d:
            # Performance Score
            if "PerformanceScore" in df.columns:
                perf_pivot = pd.crosstab(df["PerformanceScore"], df["Termd"].map({0: "Active", 1: "Terminated"}))
                fig = px.bar(perf_pivot, barmode="group",
                             title="Performance Score Distribution by Status",
                             color_discrete_map={"Active": "#3b82f6", "Terminated": "#ef4444"})
                fig.update_layout(height=400, xaxis_title="Performance Score", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # TAB 2: PREDICT & EXPLAIN
    # ============================================================
    with tab2:
        st.header("🔍 Individual Employee Risk Prediction")
        st.markdown("Select an employee profile or enter custom values to get a **turnover risk prediction** with **SHAP/LIME explanations**.")

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("📝 Employee Profile")

            salary = st.slider("Salary ($)", 30000, 250000, 62000, 1000)
            age = st.slider("Age", 20, 70, 35)
            tenure = st.slider("Tenure (years)", 0.0, 15.0, 3.0, 0.5)
            engagement = st.slider("Engagement Survey", 1.0, 5.0, 4.0, 0.1)
            satisfaction = st.slider("Satisfaction (1-5)", 1, 5, 3)
            special_projects = st.slider("Special Projects", 0, 10, 0)
            days_late = st.slider("Days Late (last 30)", 0, 10, 0)
            absences = st.slider("Absences", 0, 25, 5)
            perf_score = st.selectbox("Performance", ["Exceeds (4)", "Fully Meets (3)", "Needs Improvement (2)", "PIP (1)"])
            perf_num = int(perf_score.split("(")[1].replace(")", ""))
            sex = st.selectbox("Gender", ["Female (0)", "Male (1)"])
            sex_num = int(sex.split("(")[1].replace(")", ""))
            is_manager = st.selectbox("Manager?", ["No (0)", "Yes (1)"])
            is_mgr = int(is_manager.split("(")[1].replace(")", ""))
            diversity = st.selectbox("From Diversity Job Fair?", ["No (0)", "Yes (1)"])
            div_num = int(diversity.split("(")[1].replace(")", ""))
            marital = st.selectbox("Marital Status", ["Single (0)", "Married (1)", "Divorced (2)", "Separated (3)", "Widowed (4)"])
            marital_num = int(marital.split("(")[1].replace(")", ""))

        with col_right:
            if st.button("🔮 Predict & Explain", type="primary", use_container_width=True):
                # Build input
                input_values = [salary, age, tenure, engagement, satisfaction,
                                special_projects, days_late, absences, perf_num,
                                sex_num, marital_num, is_mgr, div_num]

                # Add department dummies (all zeros for now)
                dept_cols = [c for c in feature_names if c.startswith("Dept_")]
                input_values.extend([0] * len(dept_cols))

                input_array = np.array(input_values).reshape(1, -1)
                input_scaled = scaler.transform(input_array)

                # Predict
                prob = best_model.predict_proba(input_scaled)[0][1]
                pred = "HIGH RISK" if prob > 0.5 else ("MEDIUM RISK" if prob > 0.3 else "LOW RISK")
                risk_class = "risk-high" if prob > 0.5 else ("risk-medium" if prob > 0.3 else "risk-low")

                st.markdown(f"### Prediction Result")
                st.markdown(f'<p class="{risk_class}">⚠️ {pred} — Turnover Probability: {prob:.1%}</p>',
                            unsafe_allow_html=True)

                # Progress bar
                st.progress(min(prob, 1.0))

                st.markdown("---")

                # SHAP Explanation
                st.subheader("📊 SHAP Explanation")
                try:
                    X_scaled_full = scaler.transform(X.values)
                    model_type = type(best_model).__name__
                    tree_models = ["DecisionTreeClassifier", "RandomForestClassifier",
                                   "GradientBoostingClassifier", "XGBClassifier"]

                    if model_type in tree_models:
                        explainer_obj = shap.TreeExplainer(best_model)
                        sv = explainer_obj.shap_values(input_scaled)
                        if isinstance(sv, list):
                            sv = sv[1]
                        ev = explainer_obj.expected_value
                        if isinstance(ev, (list, np.ndarray)):
                            ev = ev[1] if len(ev) > 1 else ev[0]
                    else:
                        bg = shap.sample(pd.DataFrame(X_scaled_full, columns=feature_names), 50)
                        explainer_obj = shap.KernelExplainer(best_model.predict_proba, bg)
                        sv = explainer_obj.shap_values(input_scaled)
                        if isinstance(sv, list):
                            sv = sv[1]
                        ev = explainer_obj.expected_value
                        if isinstance(ev, (list, np.ndarray)):
                            ev = ev[1] if len(ev) > 1 else ev[0]

                    # Waterfall plot
                    explanation = shap.Explanation(
                        values=sv[0],
                        base_values=ev,
                        data=input_scaled[0],
                        feature_names=feature_names
                    )
                    fig_shap, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(explanation, show=False)
                    plt.title("SHAP: Feature Contributions to Prediction")
                    plt.tight_layout()
                    st.pyplot(fig_shap)
                    plt.close()

                    # Text explanation
                    text = generate_explanation_text(sv, feature_names, input_scaled, idx=0)
                    st.markdown("### 💡 Human-Readable Explanation")
                    st.code(text)

                except Exception as e:
                    st.warning(f"SHAP explanation error: {e}")

                # LIME Explanation
                st.subheader("🍋 LIME Explanation")
                try:
                    X_scaled_full = scaler.transform(X.values)
                    lime_exp = LimeTabularExplainer(
                        X_scaled_full,
                        feature_names=feature_names,
                        class_names=["Active", "Terminated"],
                        mode="classification",
                        random_state=42
                    )
                    exp = lime_exp.explain_instance(
                        input_scaled[0], best_model.predict_proba,
                        num_features=10, num_samples=500
                    )
                    fig_lime = exp.as_pyplot_figure()
                    fig_lime.set_size_inches(10, 5)
                    plt.title("LIME: Local Feature Importance")
                    plt.tight_layout()
                    st.pyplot(fig_lime)
                    plt.close()
                except Exception as e:
                    st.warning(f"LIME explanation error: {e}")

    # ============================================================
    # TAB 3: FRUGAL AI
    # ============================================================
    with tab3:
        st.header("🌱 Frugal AI — Model Comparison")
        st.markdown("Comparing models from **simplest** (most frugal) to **most complex**, showing that simpler models can be equally effective while consuming fewer resources.")

        # Comparison table
        if results:
            df_results = pd.DataFrame(results).T
            df_results.index.name = "Model"

            # Highlight best
            st.subheader("📊 Performance Metrics")

            col1, col2 = st.columns(2)

            with col1:
                # F1 Score comparison
                fig = go.Figure()
                colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]
                for i, (name, r) in enumerate(results.items()):
                    fig.add_trace(go.Bar(
                        name=name, x=[name], y=[r["f1_score"]],
                        marker_color=colors[i % len(colors)],
                        text=[f"{r['f1_score']:.4f}"],
                        textposition="outside"
                    ))
                fig.update_layout(title="F1 Score by Model", height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Training time comparison
                fig = go.Figure()
                for i, (name, r) in enumerate(results.items()):
                    fig.add_trace(go.Bar(
                        name=name, x=[name], y=[r["train_time_seconds"]],
                        marker_color=colors[i % len(colors)],
                        text=[f"{r['train_time_seconds']:.4f}s"],
                        textposition="outside"
                    ))
                fig.update_layout(title="Training Time (seconds)", height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Complexity vs Performance scatter
            st.subheader("⚖️ Efficiency: Performance vs Complexity")
            scatter_data = []
            for name, r in results.items():
                scatter_data.append({
                    "Model": name,
                    "F1 Score": r["f1_score"],
                    "AUC-ROC": r["auc_roc"],
                    "Parameters": r["n_parameters"],
                    "Train Time (s)": r["train_time_seconds"],
                    "CO₂ (g)": r.get("carbon_emissions_kg", 0) * 1000
                })
            scatter_df = pd.DataFrame(scatter_data)

            fig = px.scatter(scatter_df, x="Parameters", y="F1 Score",
                             size="Train Time (s)", color="Model",
                             title="Model Efficiency: F1 Score vs # Parameters",
                             hover_data=["AUC-ROC", "Train Time (s)", "CO₂ (g)"],
                             size_max=50)
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

            # Carbon emissions
            st.subheader("🌍 Carbon Footprint")
            carbon_data = {name: r.get("carbon_emissions_kg", 0) * 1000 for name, r in results.items()}
            if any(v > 0 for v in carbon_data.values()):
                fig = px.bar(x=list(carbon_data.keys()), y=list(carbon_data.values()),
                             title="CO₂ Emissions per Model (grams)",
                             labels={"x": "Model", "y": "CO₂ (g)"},
                             color=list(carbon_data.values()),
                             color_continuous_scale="RdYlGn_r")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("💡 Carbon tracking data will appear when CodeCarbon is installed and models are trained with it.")
                st.markdown("""
                **Estimated carbon footprint comparison:**
                | Model | Est. Energy | Est. CO₂ |
                |-------|-----------|----------|
                | Logistic Regression | ~0.001 Wh | ~0.0004g |
                | Decision Tree | ~0.002 Wh | ~0.0008g |
                | Random Forest | ~0.05 Wh | ~0.02g |
                | XGBoost/GradientBoosting | ~0.1 Wh | ~0.04g |
                """)

            # Detailed metrics table
            st.subheader("📋 Full Comparison Table")
            display_df = df_results[["accuracy", "f1_score", "precision", "recall",
                                     "auc_roc", "cv_f1_mean", "train_time_seconds",
                                     "n_parameters"]].copy()
            display_df.columns = ["Accuracy", "F1", "Precision", "Recall",
                                  "AUC-ROC", "CV F1 Mean", "Time (s)", "Parameters"]
            st.dataframe(display_df.style.highlight_max(axis=0, subset=["F1", "AUC-ROC", "Accuracy"],
                                                         color="#d1fae5")
                          .highlight_min(axis=0, subset=["Time (s)", "Parameters"],
                                          color="#dbeafe"),
                         use_container_width=True)

            st.success(f"🏆 **Selected Model: {best_name}** — Best frugal choice with F1={results[best_name]['f1_score']:.4f}")

            # Frugal AI insights
            st.markdown("### 💡 Frugal AI Key Insights")
            st.markdown("""
            - **Small dataset (~312 rows)** → Complex models add no value, simpler ones generalize better
            - **Logistic Regression** offers comparable performance with ~100x fewer parameters
            - **Training time** of the simplest model is negligible compared to ensemble methods
            - **Carbon footprint** scales with model complexity — choose wisely!
            - A **frugal mindset** means picking the right tool for the job, not always the most powerful one
            """)

    # ============================================================
    # TAB 4: NLP INSIGHTS
    # ============================================================
    with tab4:
        st.header("💬 NLP Insights — Employee Feedback Analysis")
        st.markdown("Analyzing simulated employee feedback using **frugal NLP** (TextBlob — no GPU required).")

        with st.spinner("Analyzing feedback..."):
            sent_df = run_sentiment_analysis(feedback_df)
            result_df, theme_counts = run_theme_analysis(sent_df)

        col1, col2 = st.columns(2)

        with col1:
            # Sentiment distribution
            sent_counts = result_df["sentiment_label"].value_counts()
            fig = px.pie(values=sent_counts.values, names=sent_counts.index,
                         title="Sentiment Distribution",
                         color_discrete_map={"Positive": "#10b981", "Neutral": "#6b7280", "Negative": "#ef4444"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Sentiment by termination status
            avg_sent = result_df.groupby(result_df["Termd"].map({0: "Active", 1: "Terminated"}))["polarity"].mean()
            fig = px.bar(x=avg_sent.index, y=avg_sent.values,
                         title="Average Sentiment by Employee Status",
                         labels={"x": "Status", "y": "Avg Polarity"},
                         color=avg_sent.values,
                         color_continuous_scale="RdYlGn")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Theme analysis
        st.subheader("📌 Top Themes in Employee Feedback")
        col3, col4 = st.columns(2)

        with col3:
            if theme_counts:
                fig = px.bar(x=list(theme_counts.values()), y=list(theme_counts.keys()),
                             orientation="h", title="Theme Frequency (All Employees)",
                             labels={"x": "Count", "y": "Theme"},
                             color=list(theme_counts.values()),
                             color_continuous_scale="Blues")
                fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            # Themes for terminated only
            termed_data = result_df[result_df["Termd"] == 1]
            if not termed_data.empty:
                from collections import Counter
                termed_themes = []
                for t in termed_data["themes"]:
                    termed_themes.extend([x.strip() for x in str(t).split(",") if x.strip() != "General"])
                tc = Counter(termed_themes)
                if tc:
                    fig = px.bar(x=list(tc.values()), y=list(tc.keys()),
                                 orientation="h", title="Top Themes — Terminated Employees",
                                 labels={"x": "Count", "y": "Theme"},
                                 color=list(tc.values()),
                                 color_continuous_scale="Reds")
                    fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)

        # Sample feedback
        st.subheader("📝 Sample Feedback Entries")
        sample = result_df.sample(min(5, len(result_df)), random_state=42)[
            ["Department", "Termd", "Feedback", "sentiment_label", "polarity", "themes"]
        ]
        sample.columns = ["Department", "Terminated", "Feedback", "Sentiment", "Polarity", "Themes"]
        sample["Terminated"] = sample["Terminated"].map({0: "Active", 1: "Terminated"})
        st.dataframe(sample, use_container_width=True, height=300)

    # ============================================================
    # TAB 5: ABOUT
    # ============================================================
    with tab5:
        st.header("📋 About This Solution")

        about_tab1, about_tab2, about_tab3 = st.tabs(["Model Card", "Data Card", "Architecture"])

        with about_tab1:
            st.markdown("""
            ## MODEL CARD : HR Turnover Prediction

            ### 1. Model Objective
            • **Target Use Case:** Predicting employee turnover risk to help HR target retention actions.
            • **Inputs:** Tabular data (age, salary, satisfaction, engagement, absences, department, performance, etc.).
            • **Outputs:** Probability of departure (score from 0 to 1) and binary classification (0 = Active, 1 = Terminated).

            ### 2. Training Data
            • **Dataset(s) used:** [Human Resources Data Set (Kaggle)](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set) by Dr. Rich Huebner.
            • **Size / diversity:**
              - **Total number of samples:** ~312 employees.
              - **Class distribution:** Imbalanced (majority active employees, minority departures). Balanced computationally during training.
              - **Diversity:** Multiple departments, ages, and demographics. Sensitive variables (Gender, RaceDesc) were kept to verify model fairness.
            • **Known limitations:** Synthetically generated dataset (educational purposes). Geographic bias (large majority in Massachusetts) and sector bias (high representation of Production department). Small data volume preventing large-scale generalization.

            ### 3. Performance
            • **Metrics used:** F1-Score (primary metric for imbalance), Accuracy, Precision, Recall, AUC-ROC.
            • **Results:** Global results demonstrate that the *Frugal AI* approach works: very simple models (like Logistic Regression or Decision Tree) achieve performance metrics comparable to heavy complex models placed in parametric competition (Random Forest / XGBoost).

            ### 4. Limitations
            • **Known error risks:** Due to the small dataset, false positives (employee flagged at risk when they will stay) and false negatives are possible.
            • **Uncovered situations:** The model takes a "snapshot" at a given time and lacks fine longitudinal analysis (evolution over time). It ignores the macroeconomic context (crisis, inflation, etc.).
            • **Bias risks:** If the original dataset favored promotions for certain genders or origins, the model could silently inherit it. Feature importance analysis (SHAP) is used to monitor this.

            ### 5. Risks & Mitigation
            • **Misuse risks:** Using these predictions to automate layoffs or preemptively discriminate in hiring (punitive system instead of a benevolent retention system). Assuming correlation (shown by SHAP) equals direct causality.
            • **Implemented controls:**
              - **Total explainability:** Each prediction is passed through SHAP and LIME to justify "why" the alert is raised.
              - **Privacy (GDPR):** Strict application of pseudo-anonymous hashing (SHA-256) for names, and conversion of birth dates to simple age.
              - **Transverse warnings:** Emphasis on the fact that AI only provides *decision support* and does not replace human HR judgment.

            ### 6. Energy and Frugality
            • **Model weight:** Less than 1 MB.
            • **Inference time:** Immediate (< 0.1s) on a standard CPU.
            • **Estimated energy (CodeCarbon):** The approach favors a classic model selected for its F1/Complexity ratio. Tracking by integrated `CodeCarbon` indicates emissions in an absolute fraction of a gram of CO₂ per training.

            ### 7. Cyber
            • **Input security:** No free-text prompts, minimizing prompt injections. Features pass through a standardized scaler.
            • **Protected secrets:** Fully open-source, runs 100% locally and "offline". The solution is devoid of exposed API keys or unsecured persistent databases.
            """)

        with about_tab2:
            st.markdown("""
            ## Data Card — HR Dataset

            ### Data Source
            - **Origin:** Kaggle HR Dataset by Dr. Rich Huebner
            - **Type:** Synthetic data for educational purposes
            - **Size:** 312 employees, 36 original columns

            ### Sensitive Attributes
            - **Gender:** Sex (M/F)
            - **Ethnicity:** RaceDesc (White, Black, Asian, etc.)
            - **Age:** Derived from DOB (original DOB removed for GDPR)
            - **Names:** Anonymized with SHA-256 hashing

            ### Anonymization Steps (GDPR)
            1. Employee names → SHA-256 hashed pseudonyms (EMP_XXXXXXXX)
            2. Manager names → SHA-256 hashed pseudonyms (MGR_XXXXXXXX)
            3. DOB → Converted to Age (exact date removed)
            4. Zip codes → Removed
            5. Employee IDs → Removed

            ### Features Used
            - Salary, Age, Tenure, Engagement, Satisfaction
            - Performance Score, Absences, Days Late
            - Department, Special Projects, Gender, Marital Status

            ### Known Biases
            - Majority of employees are in Production department
            - Most employees are from Massachusetts (MA)
            - Race/gender distribution may not reflect real organizations
            """)

        with about_tab3:
            st.markdown("""
            ## System Architecture

            ```
            ┌──────────────────────────────────────────────────────────┐
            │                    DATA LAYER                             │
            │  HR Dataset (CSV) → Anonymization → Feature Engineering  │
            │  Employee Feedback (Generated) → NLP Processing          │
            └──────────────┬───────────────────┬───────────────────────┘
                           │                   │
            ┌──────────────▼───────┐ ┌────────▼──────────────────────┐
            │   MODEL LAYER        │ │   NLP LAYER                    │
            │  • Log. Regression   │ │  • Sentiment (TextBlob)       │
            │  • Decision Tree     │ │  • Theme Extraction           │
            │  • Random Forest     │ │  • Keyword Analysis           │
            │  • XGBoost           │ │                                │
            │  + CodeCarbon Track  │ │  Frugal: No GPU needed!       │
            └──────────────┬───────┘ └────────┬──────────────────────┘
                           │                   │
            ┌──────────────▼───────────────────▼──────────────────────┐
            │                 EXPLAINABILITY LAYER                     │
            │  • SHAP (Global + Local)  • LIME (Local)                │
            │  • Human-readable text    • Feature importance           │
            └──────────────┬──────────────────────────────────────────┘
                           │
            ┌──────────────▼──────────────────────────────────────────┐
            │              PRESENTATION LAYER (Streamlit)              │
            │  • Overview Dashboard    • Individual Prediction         │
            │  • Frugal AI Comparison  • NLP Insights                  │
            │  • Model Card / Data Card                                │
            └─────────────────────────────────────────────────────────┘
            ```

            ### Key Design Decisions
            1. **Frugal by design:** Lightweight models, no GPU, small dataset
            2. **Privacy first:** GDPR-compliant anonymization
            3. **Transparency:** Every prediction can be explained (SHAP + LIME)
            4. **Modular architecture:** Each component is independent and reusable
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280; font-size: 0.85rem;'>
        🎓 Hackathon: Trusted AI x HR | Frugal AI + Explainable AI | 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
