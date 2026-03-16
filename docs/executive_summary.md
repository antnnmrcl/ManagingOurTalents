# Executive Summary — HR Turnover Prediction (Trusted AI)

## The Problem

Employee turnover is a critical challenge for organizations. Replacing an employee costs **50–200% of their annual salary** (SHRM). Our client faces a **high resignation rate** and needs to understand why employees leave and how to retain them.

## Our Solution

We built an **AI-powered HR analytics solution** that:

1. **Predicts** which employees are at risk of leaving
2. **Explains** the reasons behind each prediction in plain language
3. **Respects** data privacy (GDPR-compliant anonymization)
4. **Minimizes** environmental impact (Frugal AI approach)

## Key Results

### Prediction Performance
- Trained and compared **4 machine learning models** from simplest to most complex
- Achieved reliable turnover predictions with **balanced precision and recall**
- Demonstrated that **simpler models perform comparably** to complex ones on this dataset

### Top Turnover Risk Factors Identified
1. **Low employee satisfaction** — strongest predictor of departure
2. **High absences** — signal of disengagement
3. **Below-market salary** — financial motivation to leave
4. **Low engagement survey scores** — emotional disconnection
5. **Poor performance scores** — both cause and consequence

### NLP Insights from Employee Feedback
- Terminated employees show **significantly more negative sentiment** than active employees
- Top themes: **Compensation** (44%), **Work-Life Balance** (23%), **Career Growth** (18%)
- Actionable insight: Address compensation gaps and provide clear career paths

## Responsible AI Framework

### 🌱 Frugal AI
- **No GPU required** — entire solution runs on any laptop
- **Carbon tracked** — CO₂ emissions measured with CodeCarbon
- **Simple models preferred** — Logistic Regression achieves comparable results to XGBoost
- **Lightweight NLP** — TextBlob (CPU, <1MB) instead of large language models

### 🔍 Explainable AI
- **SHAP analysis** — global and local feature importance
- **LIME explanations** — individual prediction interpretability
- **Plain language** — "Employee X is at risk because their satisfaction score is low and absences are high"
- **No black boxes** — HR managers can understand and trust every prediction

### 🔒 Data Privacy
- **GDPR-compliant** anonymization (names hashed, DOB → age, identifiers removed)
- **Sensitive attributes** (gender, race) monitored for fairness
- **No personal data** exposed in the application

## Business Value

| Benefit | Impact |
|---------|--------|
| **Early warning** | Identify at-risk employees before they resign |
| **Targeted retention** | Focus resources on high-risk, high-value employees |
| **Cost savings** | Reduce turnover costs (50-200% of salary per employee) |
| **Data-driven HR** | Move from gut feeling to evidence-based decisions |
| **Trust** | Transparent AI that HR managers can understand and explain |

## Recommendation

Deploy this solution as an **advisory tool** for HR managers. Key actions:
1. Use the dashboard to monitor turnover risk across the organization
2. Investigate individual at-risk employees with SHAP/LIME explanations
3. Address the top factors: satisfaction, compensation, engagement
4. Regularly retrain the model as new data becomes available
5. Audit for fairness monthly

---

*This solution demonstrates that effective AI doesn't need massive computing power. A frugal, explainable approach delivers the same insights with a fraction of the environmental and financial cost.*
