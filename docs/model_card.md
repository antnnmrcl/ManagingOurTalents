# MODEL CARD : HR Turnover Prediction

## Scope Note (Hackathon)
This model card is aligned with the team's selected themes: **Frugal AI** and **Explainable AI** (2 themes required by the challenge).

## 1. Model Objective
• **Target Use Case:** Predicting employee turnover risk to help HR target retention actions.
• **Inputs:** Tabular data (age, salary, satisfaction, engagement, absences, department, performance, etc.).
• **Outputs:** Probability of departure (score from 0 to 1) and binary classification (0 = Active, 1 = Terminated).

## 2. Training Data
• **Dataset(s) used:** [Human Resources Data Set (Kaggle)](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set) by Dr. Rich Huebner.
• **Size / diversity:**
  - **Total number of samples:** ~312 employees.
  - **Class distribution:** Imbalanced (majority active employees, minority departures). Balanced computationally during training.
  - **Diversity:** Multiple departments, ages, and demographics. Sensitive variables (Gender, RaceDesc) were kept to verify model fairness.
• **Known limitations:** Synthetically generated dataset (educational purposes). Geographic bias (large majority in Massachusetts) and sector bias (high representation of Production department). Small data volume preventing large-scale generalization.

## 3. Performance
• **Metrics used:** F1-Score (primary metric for imbalance), Accuracy, Precision, Recall, AUC-ROC.
• **Results:** Global results demonstrate that the *Frugal AI* approach works: very simple models (like Logistic Regression or Decision Tree) achieve performance metrics comparable to heavy complex models placed in parametric competition (Random Forest / XGBoost).

## 4. Limitations
• **Known error risks:** Due to the small dataset, false positives (employee flagged at risk when they will stay) and false negatives are possible.
• **Uncovered situations:** The model takes a "snapshot" at a given time and lacks fine longitudinal analysis (evolution over time). It ignores the macroeconomic context (crisis, inflation, etc.).
• **Bias risks:** If the original dataset favored promotions for certain genders or origins, the model could silently inherit it. Feature importance analysis (SHAP) is used to monitor this.

## 5. Risks & Mitigation
• **Misuse risks:** Using these predictions to automate layoffs or preemptively discriminate in hiring (punitive system instead of a benevolent retention system). Assuming correlation (shown by SHAP) equals direct causality.
• **Implemented controls:**
  - **Total explainability:** Each prediction is passed through SHAP and LIME to justify "why" the alert is raised.
  - **Privacy (GDPR):** Strict application of pseudo-anonymous hashing (SHA-256) for names, and conversion of birth dates to simple age.
  - **Transverse warnings:** Emphasis on the fact that AI only provides *decision support* and does not replace human HR judgment.

## 6. Energy and Frugality
• **Model weight:** Less than 1 MB.
• **Inference time:** Immediate (< 0.1s) on a standard CPU.
• **Estimated energy (CodeCarbon):** The approach favors a classic model selected for its F1/Complexity ratio. Tracking by integrated `CodeCarbon` indicates emissions in an absolute fraction of a gram of CO₂ per training.

## 7. Cyber
• **Input security:** No free-text prompts, minimizing prompt injections. Features pass through a standardized scaler.
• **Protected secrets:** Fully open-source, runs 100% locally and "offline". The solution is devoid of exposed API keys or unsecured persistent databases.
