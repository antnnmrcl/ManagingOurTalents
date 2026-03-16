# Data Card — HR Dataset

## Dataset Overview

| Field | Value |
|-------|-------|
| **Name** | Human Resources Data Set |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set) |
| **Authors** | Dr. Rich Huebner & Dr. Carla Patalano |
| **Type** | Synthetic (educational purposes) |
| **Size** | 312 employees, 36 columns |
| **Format** | CSV |
| **License** | Open source (Kaggle) |

## Data Description

### Target Variable
- **Termd:** Binary (0 = Active/Still Employed, 1 = Terminated/Resigned)
- **TermReason:** Textual reason for departure (e.g., "unhappy", "more money", "career change")

### Key Features

| Feature | Type | Description |
|---------|------|-------------|
| Salary | Numeric | Annual salary ($) |
| Age | Numeric | Derived from DOB (original DOB anonymized) |
| EngagementSurvey | Numeric | Employee engagement score (1.0–5.0) |
| EmpSatisfaction | Numeric | Satisfaction level (1–5) |
| PerformanceScore | Categorical | Exceeds / Fully Meets / Needs Improvement / PIP |
| Absences | Numeric | Number of absences |
| DaysLateLast30 | Numeric | Days late in last 30 days |
| SpecialProjectsCount | Numeric | Number of special projects participated in |
| Department | Categorical | Production, IT/IS, Sales, Software Engineering, Admin, Executive |
| Position | Categorical | Job title |
| Tenure_Years | Numeric | Calculated from hire/termination dates |

### Sensitive Attributes

| Attribute | Values | Handling |
|-----------|--------|----------|
| **Sex** | M / F | Kept for fairness analysis, encoded as binary |
| **RaceDesc** | White, Black, Asian, Hispanic, American Indian, Two or more | Kept for fairness audit |
| **MaritalDesc** | Single, Married, Divorced, Separated, Widowed | Used as feature |
| **CitizenDesc** | US Citizen, Eligible NonCitizen, Non-Citizen | Available but not used |
| **HispanicLatino** | Yes / No | Available but not used |

## Data Processing

### Anonymization Steps (GDPR Compliance)

1. **Employee Names** → SHA-256 hashed pseudonyms (`EMP_XXXXXXXX`)
2. **Manager Names** → SHA-256 hashed pseudonyms (`MGR_XXXXXXXX`)
3. **Date of Birth** → Converted to Age (exact date removed)
4. **Zip Codes** → Removed entirely
5. **Employee IDs** → Removed entirely

### Feature Engineering

1. **Tenure_Years:** Calculated from hire date to termination date (or reference date for active)
2. **PerfScore_Numeric:** Performance mapped to ordinal scale (PIP=1, NI=2, FM=3, Exceeds=4)
3. **Sex_Binary:** Gender encoded as 0/1
4. **MaritalStatus_Num:** Marital status encoded as ordinal
5. **Department dummies:** One-hot encoded
6. **Is_Manager:** Binary flag from position title

## Supplementary Data: Simulated Feedback

### Why?
Real employee feedback is confidential. We generated **simulated exit interview feedback** based on the `TermReason` field to demonstrate NLP capabilities.

### How?
- Templates created for each termination reason (13 categories)
- Contextual details added based on satisfaction/engagement scores
- 312 feedback entries generated (one per employee)

### Limitations of Simulated Data
- Does not capture real linguistic patterns
- Limited vocabulary diversity
- Sentiment may be more uniform than real feedback

## Known Biases & Limitations

| Issue | Description |
|-------|-------------|
| **Small size** | 312 rows — limited for complex modeling |
| **Synthetic** | Created for educational use, not from real HR systems |
| **Geographic bias** | Most employees from Massachusetts (MA) |
| **Department imbalance** | Production department is majority |
| **Temporal snapshot** | Data represents a fixed time period |
| **Missing values** | Some manager IDs are missing |

## Ethical Use Guidelines

- ⚠️ This dataset is **synthetic** — do not treat findings as real HR insights
- ⚠️ Sensitive attributes present — monitor for bias in predictions
- ✅ Suitable for educational and demonstration purposes
- ✅ GDPR-compliant after anonymization processing
