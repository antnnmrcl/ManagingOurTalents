# Architecture — HR Turnover Prediction System

## System Overview

The system follows a **modular, layered architecture** designed for maximum transparency and minimal resource usage.

```
                        ┌─────────────────┐
                        │   User (HR)     │
                        │   via Browser   │
                        └────────┬────────┘
                                 │
                        ┌────────▼───────┐
                        │   Streamlit    │
                        │   Dashboard    │
                        │   (app.py)     │
                        └────┬───┬───┬───┘
                             │   │   │
              ┌──────────────┘   │   └──────────────┐
              │                  │                  │
     ┌────────▼──────┐  ┌──────▼───────┐  ┌────────▼───────┐
     │ Explainability│  │ Model Layer  │  │  NLP Layer     │
     │ (SHAP/LIME)   │  │ (ML Models)  │  │  (TextBlob)    │
     └────────┬──────┘  └──────┬───────┘  └────────┬───────┘
              │                │                   │
              └────────┬───────┘                   │
                       │                           │
              ┌────────▼───────────────────────────▼──┐
              │         Data Processing Layer         │
              │  (Anonymization + Feature Engineering)│
              └────────────────┬──────────────────────┘
                               │
                      ┌────────▼────────┐
                      │   Raw CSV Data  │
                      │  (HR Dataset)   │
                      └─────────────────┘
```

## Component Details

### 1. Data Layer (`src/data_processing.py`)
- **Input:** Raw HR CSV dataset
- **Processing:** GDPR anonymization, feature engineering
- **Output:** Clean, anonymized dataset + simulated feedback

### 2. Model Layer (`src/model_training.py`)
- **Input:** Processed features + target variable
- **Processing:** Train 4 models, CodeCarbon tracking, cross-validation
- **Output:** Trained models, comparison metrics, saved artifacts

### 3. Explainability Layer (`src/explainability.py`)
- **Input:** Trained model + data
- **Processing:** SHAP values (global/local), LIME explanations
- **Output:** Visualization plots, text explanations

### 4. NLP Layer (`src/nlp_analysis.py`)
- **Input:** Employee feedback text
- **Processing:** Sentiment analysis (TextBlob), theme extraction
- **Output:** Sentiment scores, theme distributions

### 5. Presentation Layer (`app.py`)
- **Framework:** Streamlit
- **Tabs:** Overview, Predict & Explain, Frugal AI, NLP Insights, About
- **Interactive:** Real-time predictions with explanations

## Technology Stack

| Component | Technology | Why This Choice |
|-----------|-----------|----------------|
| ML Models | scikit-learn, XGBoost | Industry standard, frugal |
| Explainability | SHAP, LIME | Gold standard for XAI |
| NLP | TextBlob | Lightweight, no GPU (frugal) |
| Carbon Tracking | CodeCarbon | Measure environmental impact |
| Dashboard | Streamlit | Rapid prototyping, interactive |
| Visualization | Plotly, Matplotlib | Rich, interactive charts |
| Data | Pandas, NumPy | Standard data science stack |

## Data Flow

```
1. HR CSV → data_processing.py → Anonymized CSV + Feedback CSV
2. Anonymized CSV → model_training.py → 4 Trained Models + Metrics
3. Trained Models → explainability.py → SHAP/LIME Plots
4. Feedback CSV → nlp_analysis.py → Sentiment + Themes
5. All outputs → app.py (Streamlit) → Interactive Dashboard
```

## Frugal Design Principles

1. **No cloud dependency:** Runs entirely on a local machine
2. **No GPU:** All models and NLP run on CPU
3. **Minimal dependencies:** Core libraries only
4. **Small footprint:** Dataset < 100KB, models < 10MB total
5. **Fast training:** Complete pipeline in < 30 seconds
