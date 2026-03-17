# 🎯 HR Turnover Prediction — Trusted AI Solution

> **Hackathon: AI x RH** | Themes: **Frugal AI** + **Explainable AI**

> **Scope choice (per hackathon instructions):** Our team intentionally focused on **2 themes out of 4** — **Frugal AI** and **Explainable AI**.

## 📋 Table of Contents
- [Objective](#objective)
- [Persona & Use Case](#persona--use-case)
- [Dataset](#dataset)
- [Installation](#installation)
- [Hackathon Compliance Checklist](#hackathon-compliance-checklist)
- [Jury Quickstart (2 minutes)](#jury-quickstart-2-minutes)
- [Usage](#usage)
- [Architecture](#architecture)
- [Frugal AI Approach](#frugal-ai-approach)
- [Explainable AI Approach](#explainable-ai-approach)
- [NLP Analysis](#nlp-analysis)
- [10-Minute Demo Flow](#10-minute-demo-flow)
- [Known Limits & Responsible Use](#known-limits--responsible-use)
- [Deliverables](#deliverables)
- [Team](#team)

---

## 🎯 Objective

An imaginary company faces a **high resignation rate** and wants to use AI to:
1. **Predict** which employees are at risk of leaving
2. **Explain** why (transparent, interpretable predictions)
3. **Suggest** preventive actions for HR
4. Do all of this in a **frugal** (resource-efficient) and **responsible** manner

---

## 👤 Persona & Use Case

**Use Case:** Employee Resignation Analysis

**Persona — The Client (HR Director):**
> *"I need an AI solution to understand why my employees leave and how to retain them. But I need to trust the AI — it must be transparent, fair, and not waste resources."*

**Persona — The Solution Provider (HR-AI):**
> *"We provide a frugal, explainable AI solution that predicts turnover risk and gives HR actionable, transparent insights — without the environmental cost of over-engineered models."*

---

## 📊 Dataset

- **Source:** [Kaggle HR Dataset](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set) by Dr. Rich Huebner
- **Size:** ~312 employees, 36 features
- **Key Variables:** Salary, Performance, Satisfaction, Engagement, Absences, Department, Demographics
- **Target:** `Termd` (0 = Active, 1 = Terminated)
- **Sensitive Attributes:** Sex (M/F), RaceDesc, Age — handled with care for fairness
- **Supplementary:** Simulated employee exit interview feedback (NLP analysis)

### GDPR Compliance
All personal data has been **anonymized**:
- Names → SHA-256 pseudonyms
- DOB → Age only
- Zip codes & IDs → Removed

---

## 🚀 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "explainability ai"

# Install dependencies
pip install -r requirements.txt

# Download TextBlob corpora (for NLP)
python -m textblob.download_corpora
```

### Requirements
- Python 3.9+
- No GPU required (Frugal AI!)

---

## ✅ Hackathon Compliance Checklist

| Required item | Status | Evidence |
|---|---|---|
| Clear README (objectives, scope, persona, instructions) | ✅ | `README.md` |
| Technical documentation | ✅ | `docs/` |
| Architecture scheme | ✅ | `docs/architecture.md` |
| Data card | ✅ | `docs/data_card.md` |
| Model card | ✅ | `docs/model_card.md` |
| Executive summary | ✅ | `docs/executive_summary.md` |
| Demo (live) | ✅ | `streamlit run app.py` |
| Slides | ⏳ | Add `slides.pptx` or `slides.pdf` before final submission |

---

## ⚡ Jury Quickstart (2 minutes)

Run from project root:

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
streamlit run app.py
```

Open: `http://localhost:8501`

### Reproducibility note
- For the **live demo**, the app uses files already present in `data/`, `models/`, and `results/`.
- Full regeneration scripts (`src/data_processing.py`, `src/model_training.py`, `src/explainability.py`) expect the original Kaggle raw file path configured in `src/data_processing.py`.

---

## ▶️ Usage

### 1. Run the Data Pipeline
```bash
python src/data_processing.py
```
Outputs: `data/hr_anonymized.csv`, `data/employee_feedback.csv`

### 2. Train & Compare Models
```bash
python src/model_training.py
```
Outputs: Trained models in `models/`, comparison metrics in `results/`

### 3. Run Explainability Analysis
```bash
python src/explainability.py
```
Outputs: SHAP/LIME plots in `results/plots/`

### 4. Run NLP Analysis
```bash
python src/nlp_analysis.py
```
Outputs: Sentiment & theme analysis in `results/`

### 5. Launch the Demo Dashboard
```bash
streamlit run app.py
```
Opens an interactive dashboard at `http://localhost:8501`

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      DATA LAYER                            │
│  HR Dataset (CSV) → Anonymization → Feature Engineering    │
│  Employee Feedback (Simulated) → NLP Processing            │
└──────────────┬────────────────────┬────────────────────────┘
               │                    │
┌──────────────▼────────┐ ┌────────▼───────────────────────┐
│    MODEL LAYER        │ │    NLP LAYER                   │
│  • Logistic Regr.     │ │  • Sentiment (TextBlob)        │
│  • Decision Tree      │ │  • Theme Extraction            │
│  • Random Forest      │ │  • Keyword Analysis            │
│  • XGBoost            │ │                                │
│  + CodeCarbon Tracker │ │  Frugal: No GPU needed!        │
└──────────────┬────────┘ └────────┬───────────────────────┘
               │                    │
┌──────────────▼────────────────────▼────────────────────────┐
│               EXPLAINABILITY LAYER                         │
│  • SHAP (Global + Local)    • LIME (Local)                 │
│  • Human-readable text      • Feature importance plots     │
└──────────────┬─────────────────────────────────────────────┘
               │
┌──────────────▼─────────────────────────────────────────────┐
│            PRESENTATION LAYER (Streamlit)                  │
│  • Overview Dashboard      • Individual Prediction         │
│  • Frugal AI Comparison    • NLP Insights                  │
│  • Model Card / Data Card                                  │
└────────────────────────────────────────────────────────────┘
```

---

## 🌱 Frugal AI Approach

| Aspect | Our Approach |
|--------|-------------|
| **Data Frugality** | ~312 rows — no need for big data infrastructure |
| **Model Frugality** | Compare 4 models: simple → complex, prove simple is enough |
| **NLP Frugality** | TextBlob (CPU, <1MB) instead of GPT/BERT (GPU, >1GB) |
| **No GPU** | Entire solution runs on any laptop |
| **Carbon Tracking** | CodeCarbon measures CO₂ emissions per model |
| **Key Insight** | Logistic Regression ≈ XGBoost performance on this dataset |

---

## 🔍 Explainable AI Approach

| Tool | Type | Purpose |
|------|------|---------|
| **SHAP** | Global | Feature importance ranking across all employees |
| **SHAP** | Local | "Why is Employee X predicted to leave?" |
| **LIME** | Local | Alternative local explanations for individual predictions |
| **Text** | Human-readable | Plain language explanations for HR managers |

### Example Explanation
> *"Employee X is at HIGH RISK of leaving because:*
> - *Low satisfaction score (2/5) → increases risk*
> - *High absences (18 days) → increases risk*
> - *Below-market salary ($45K) → increases risk"*

---

## 💬 NLP Analysis

- **Simulated Data:** Exit interview feedback generated based on real TermReason patterns
- **Sentiment Analysis:** TextBlob polarity/subjectivity scoring (frugal, no GPU)
- **Theme Extraction:** Keyword-based topic detection (8 themes: Compensation, Work-Life Balance, Career Growth, Management, Culture, Personal, Performance, Satisfaction)
- **Key Finding:** Terminated employees show significantly lower average sentiment

---

## 🎤 10-Minute Demo Flow

1. **Problem & scope (1 min):** turnover challenge + selected themes (**Frugal AI** and **Explainable AI**).
2. **Dashboard overview (2 min):** KPIs, turnover by department, engagement/satisfaction trends.
3. **Predict & explain (3 min):** one employee scenario + SHAP/LIME explanation.
4. **Frugal AI evidence (2 min):** model comparison and why simpler models are sufficient.
5. **NLP insights + actions (2 min):** sentiment/themes and concrete HR recommendations.

---

## ⚠️ Known Limits & Responsible Use

- This is a **decision-support** tool, not an automated HR decision engine.
- Predictions must not be used alone for critical actions (e.g., termination/promotion decisions).
- Dataset is synthetic and modest in size; results should be interpreted with caution.
- Correlation-based explanations (SHAP/LIME) do not prove direct causality.

---

## 📦 Deliverables

| # | Deliverable | Location |
|---|------------|----------|
| 1 | README | `README.md` (this file) |
| 2 | Technical Documentation | `docs/` folder |
| 3 | Architecture Diagram | `docs/architecture.md` |
| 4 | Data Card | `docs/data_card.md` |
| 5 | Model Card | `docs/model_card.md` |
| 6 | Demo | `streamlit run app.py` |
| 7 | Executive Summary | `docs/executive_summary.md` |

---

## 🏫 Team

- Hackathon: Trusted AI x HR
- Course: Explainability AI
- Date: March 2026

---

*Built with ❤️ and 🌱 minimal carbon footprint*
