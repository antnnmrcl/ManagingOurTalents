# Model Card — HR Turnover Prediction

## Model Details

| Field | Value |
|-------|-------|
| **Task** | Binary Classification — Predict Employee Turnover |
| **Models Compared** | Logistic Regression, Decision Tree, Random Forest, XGBoost |
| **Selected Model** | Best frugal choice (highest F1/complexity ratio) |
| **Framework** | scikit-learn 1.5.1, XGBoost 2.0.3 |
| **Training Data** | 80% split (~250 employees) |
| **Test Data** | 20% split (~62 employees) |
| **Carbon Tracking** | CodeCarbon 2.4.1 |

## Intended Use

### Primary Use
- HR analytics tool to **identify employees at risk of leaving**
- Support tool for HR managers in **talent retention strategies**
- Educational demonstration of **Frugal AI** and **Explainable AI**

### Users
- HR Directors and People Analytics Teams
- Decision support (advisory) — **NOT automated decision-making**

### Out of Scope
- ❌ Automated hiring/firing decisions
- ❌ Individual employee surveillance
- ❌ Performance-based penalties
- ❌ Use without human oversight

## Training Data

- **Source:** Kaggle HR Dataset (synthetic, educational)
- **Size:** ~312 employees
- **Features:** 13 selected features + department dummies
- **Target:** `Termd` (0 = Active, 1 = Terminated)
- **Class Balance:** Imbalanced (more active than terminated)
- **Handling:** Class weighting (balanced) in model training

## Performance Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **F1 Score** | Harmonic mean of precision/recall (primary metric) |
| **Precision** | Of predicted terminations, how many actually left |
| **Recall** | Of actual terminations, how many were caught |
| **AUC-ROC** | Area under ROC curve — discrimination ability |
| **CV F1** | 5-fold cross-validated F1 score |

*(See Frugal AI tab in demo for exact values)*

## Frugal AI Considerations

1. **Model Selection:** Deliberate comparison from simplest to most complex
2. **Resource Tracking:** CO₂ emissions measured with CodeCarbon
3. **Key Insight:** Simple models achieve comparable performance
4. **No GPU Required:** Entire pipeline runs on CPU
5. **Small Dataset:** No need for distributed computing or big data tools

## Explainability

| Method | Scope | Purpose |
|--------|-------|---------|
| SHAP | Global | Overall feature importance ranking |
| SHAP | Local | Individual prediction explanations |
| LIME | Local | Alternative local interpretability |
| Text | Local | Human-readable explanations for HR |

## Limitations

- **Data Size:** Only ~312 employees — limited statistical power
- **Synthetic Data:** May not reflect real-world patterns
- **Temporal:** Snapshot in time, not longitudinal
- **Features:** Limited feature set — real HR data would have more signals
- **Bias Risk:** Historical data patterns may encode past biases

## Ethical Considerations

### Fairness
- Sensitive attributes (gender, race) are present in training data
- Model should be **regularly audited** for disparate impact
- Feature importance ≠ causal relationship (correlation ≠ causation)

### Privacy (GDPR)
- All personal identifiers anonymized (SHA-256 hashing)
- DOB → Age conversion (no exact dates)
- Names, zip codes, employee IDs removed

### Transparency
- Every prediction can be explained with SHAP/LIME
- Model card and data card provided
- Open-source tools used

### Recommendations
- ⚠️ Always pair AI predictions with human judgment
- ⚠️ Regularly retrain and audit the model
- ⚠️ Do NOT use for automated employment decisions
- ✅ Use as a decision-support tool for proactive HR strategies
