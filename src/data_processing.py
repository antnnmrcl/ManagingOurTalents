"""
Data Processing & GDPR Anonymization Module
============================================
Loads the HR dataset, anonymizes sensitive fields (GDPR compliance),
performs feature engineering, and prepares data for modeling.
"""

import pandas as pd
import numpy as np
import hashlib
import os
import json
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "archive (4)", "HRDataset_v14.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data")
ANONYMIZED_PATH = os.path.join(PROCESSED_DATA_DIR, "hr_anonymized.csv")
FEEDBACK_PATH = os.path.join(PROCESSED_DATA_DIR, "employee_feedback.csv")

# Sensitive columns to anonymize/remove
SENSITIVE_COLUMNS = ["Employee_Name", "Zip", "DOB"]
# Columns that are identifiers but needed in hashed form
ID_COLUMNS = ["ManagerName"]


def load_raw_data():
    """Load the raw HR dataset."""
    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")
    print(f"✅ Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def anonymize_data(df):
    """
    Anonymize the dataset for GDPR compliance:
    - Hash employee names to pseudonymous IDs
    - Remove direct identifiers (DOB → keep age only, remove zip)
    - Hash manager names
    """
    df_anon = df.copy()

    # 1. Hash employee names → pseudonymous ID
    df_anon["EmployeeID_Anon"] = df_anon["Employee_Name"].apply(
        lambda x: "EMP_" + hashlib.sha256(str(x).encode()).hexdigest()[:8].upper()
    )

    # 2. Hash manager names
    df_anon["ManagerID_Anon"] = df_anon["ManagerName"].apply(
        lambda x: "MGR_" + hashlib.sha256(str(x).encode()).hexdigest()[:8].upper()
        if pd.notna(x) else "UNKNOWN"
    )

    # 3. Convert DOB to Age (remove exact birthdate)
    df_anon["DOB"] = pd.to_datetime(df_anon["DOB"], format="mixed", dayfirst=False)
    reference_date = pd.Timestamp("2019-03-01")  # approximate dataset end date
    df_anon["Age"] = ((reference_date - df_anon["DOB"]).dt.days / 365.25).astype(int)

    # 4. Drop sensitive columns
    cols_to_drop = ["Employee_Name", "DOB", "Zip", "ManagerName", "ManagerID", "EmpID"]
    df_anon = df_anon.drop(columns=[c for c in cols_to_drop if c in df_anon.columns])

    print(f"✅ Anonymized dataset: removed {len(cols_to_drop)} sensitive columns")
    print(f"   Added: EmployeeID_Anon, ManagerID_Anon, Age")
    return df_anon


def engineer_features(df):
    """
    Feature engineering for turnover prediction.
    """
    df_feat = df.copy()

    # 1. Calculate tenure (years)
    df_feat["DateofHire"] = pd.to_datetime(df_feat["DateofHire"], format="mixed")
    df_feat["DateofTermination"] = pd.to_datetime(df_feat["DateofTermination"], format="mixed", errors="coerce")
    reference_date = pd.Timestamp("2019-03-01")
    end_date = df_feat["DateofTermination"].fillna(reference_date)
    df_feat["Tenure_Years"] = ((end_date - df_feat["DateofHire"]).dt.days / 365.25).round(2)

    # 2. Encode performance score as numeric
    perf_map = {"Exceeds": 4, "Fully Meets": 3, "Needs Improvement": 2, "PIP": 1}
    df_feat["PerfScore_Numeric"] = df_feat["PerformanceScore"].map(perf_map).fillna(2)

    # 3. Binary encode categorical features
    df_feat["Sex_Binary"] = (df_feat["Sex"].str.strip() == "M").astype(int)  # 1=Male, 0=Female

    # 4. Encode marital status
    marital_map = {"Single": 0, "Married": 1, "Divorced": 2, "Separated": 3, "Widowed": 4}
    df_feat["MaritalStatus_Num"] = df_feat["MaritalDesc"].map(marital_map).fillna(0)

    # 5. Department encoding
    dept_dummies = pd.get_dummies(df_feat["Department"].str.strip(), prefix="Dept", drop_first=True)
    df_feat = pd.concat([df_feat, dept_dummies], axis=1)

    # 6. Recruitment source encoding
    source_dummies = pd.get_dummies(df_feat["RecruitmentSource"], prefix="Source", drop_first=True)
    df_feat = pd.concat([df_feat, source_dummies], axis=1)

    # 7. Race encoding for fairness analysis (keep original for audit)
    race_dummies = pd.get_dummies(df_feat["RaceDesc"], prefix="Race", drop_first=True)
    df_feat = pd.concat([df_feat, race_dummies], axis=1)

    # 8. Position level (extract seniority indicator)
    df_feat["Is_Manager"] = df_feat["Position"].str.contains(
        "Manager|Director|CEO|CIO|VP", case=False, na=False
    ).astype(int)

    print(f"✅ Feature engineering complete: {df_feat.shape[1]} total columns")
    return df_feat


def get_model_features(df):
    """
    Select features for the ML model and return X, y.
    """
    feature_cols = [
        "Salary", "Age", "Tenure_Years", "EngagementSurvey", "EmpSatisfaction",
        "SpecialProjectsCount", "DaysLateLast30", "Absences",
        "PerfScore_Numeric", "Sex_Binary", "MaritalStatus_Num",
        "Is_Manager", "FromDiversityJobFairID"
    ]

    # Add department dummies
    dept_cols = [c for c in df.columns if c.startswith("Dept_")]
    feature_cols.extend(dept_cols)

    # Ensure all columns exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df["Termd"].astype(int)

    print(f"✅ Model features: {len(feature_cols)} features, target: Termd")
    print(f"   Class distribution: Active={sum(y==0)}, Terminated={sum(y==1)}")
    return X, y, feature_cols


def generate_simulated_feedback(df):
    """
    Generate simulated employee exit interview feedback based on TermReason.
    This enriches the dataset with text data for NLP analysis.
    """
    # Templates based on common term reasons
    feedback_templates = {
        "unhappy": [
            "I felt undervalued in my role. The work environment was not supportive and I didn't see a future here.",
            "Management didn't listen to our concerns. I tried to raise issues multiple times but nothing changed.",
            "The company culture deteriorated over time. I no longer felt motivated to come to work.",
            "Work-life balance was terrible. I was constantly stressed and burned out.",
            "I didn't feel appreciated despite working long hours. Recognition was non-existent.",
        ],
        "more money": [
            "I received a much better offer elsewhere. My salary here was below market rate for my experience.",
            "Despite good performance reviews, my compensation didn't reflect my contributions.",
            "I asked for a raise multiple times but was told the budget didn't allow it. Meanwhile, new hires were getting more.",
            "The benefits package was not competitive. I found a company offering better total compensation.",
            "I love the team but financially it didn't make sense to stay. The pay gap was just too large.",
        ],
        "Another position": [
            "I found an opportunity that better aligned with my career goals and offered more growth potential.",
            "A recruiter reached out with an amazing opportunity I couldn't pass up.",
            "I wanted to explore a different industry and this was the right time to make the switch.",
            "The new role offers more responsibility and leadership opportunities.",
            "I was headhunted for a position that perfectly matches my long-term career plan.",
        ],
        "career change": [
            "I decided to pursue a completely different career path that I'm more passionate about.",
            "After much reflection, I realized this industry wasn't for me. I want to follow my true calling.",
            "I'm going back to school to retrain in a field I've always been interested in.",
            "This was a great stepping stone but I need to move in a different direction professionally.",
            "I've discovered a new passion and want to build a career around it.",
        ],
        "hours": [
            "The long hours were taking a toll on my health and family life.",
            "I couldn't sustain the overtime expectations. Work-life balance is essential for me.",
            "The shift schedule was incompatible with my personal commitments.",
            "I was working 60+ hour weeks regularly with no extra compensation.",
            "The rigid schedule didn't allow any flexibility for personal needs.",
        ],
        "attendance": [
            "I had some personal issues that affected my attendance but felt the company wasn't understanding.",
            "Health problems caused me to miss days but I didn't feel supported by management.",
            "Transportation issues made it difficult to maintain perfect attendance.",
            "Family emergencies kept coming up and the company's absence policy was too strict.",
            "I was dealing with a chronic condition that the company refused to accommodate.",
        ],
        "return to school": [
            "I decided to pursue a graduate degree to advance my qualifications.",
            "I got accepted into a program I've been wanting to attend for years.",
            "Education is a priority for me and I want to invest in my future.",
            "I'm going back to school full-time to complete my degree.",
            "An educational opportunity came up that will significantly boost my career prospects.",
        ],
        "relocation out of area": [
            "My spouse got transferred and we had to move to a different state.",
            "Family circumstances required me to relocate closer to my parents.",
            "I decided to move to be closer to family. Remote work wasn't an option.",
            "We bought a house in a different area and the commute was too long.",
            "Personal reasons required me to relocate to a different region.",
        ],
        "retiring": [
            "After many productive years, it's time for me to enjoy retirement.",
            "I've reached the stage in my life where I want to slow down and spend time with family.",
            "It's been a wonderful career here but I'm ready for the next chapter.",
            "I'm grateful for the opportunities but it's time to retire and pursue personal interests.",
            "Health considerations and age have led me to decide it's time to step back.",
        ],
        "performance": [
            "I felt the performance expectations were unrealistic given the resources provided.",
            "The performance review process seemed subjective and unfair.",
            "I struggled to meet targets because of inadequate training and support.",
            "The performance criteria kept changing without proper communication.",
            "I disagree with my performance assessment. I believe I was not given a fair chance.",
        ],
        "military": [
            "I'm being deployed for military service. I'm proud to serve my country.",
            "My National Guard unit has been activated and I need to fulfill my duty.",
            "I'm joining the armed forces to serve and plan to return to civilian work afterward.",
            "Military obligations require me to leave. I hope to return when my service is complete.",
            "I've decided to pursue a full-time military career.",
        ],
        "medical issues": [
            "A health condition has made it impossible for me to continue working.",
            "I need to take extended time off for medical treatment that the company cannot accommodate.",
            "A chronic illness requires me to step back from full-time work.",
            "Medical reasons prevent me to continue in this physically demanding role.",
            "I need to focus on my health recovery and cannot maintain my work schedule.",
        ],
        "no-call, no-show": [
            "I was going through a crisis and didn't know how to communicate it.",
            "Personal issues overwhelmed me and I couldn't face coming in.",
            "I had an emergency that prevented me from contacting anyone.",
            "Mental health challenges made it impossible for me to function normally.",
            "I'm sorry for the lack of communication. Things spiraled out of control.",
        ],
        "N/A-StillEmployed": [
            "I enjoy working here. The team is great and I feel valued.",
            "The company has been supportive of my growth. I appreciate the opportunities.",
            "I'm satisfied with my role and compensation. Management is fair.",
            "Good work-life balance and collaborative environment keep me here.",
            "I feel my contributions are recognized and I see a clear career path.",
            "The benefits and culture are excellent. I'm happy with my decision to stay.",
            "My manager is supportive and the projects are interesting and challenging.",
            "I've grown a lot professionally here and continue to learn new things.",
        ],
    }

    # Default feedback for unmapped reasons
    default_feedback = [
        "I left for personal reasons that I'd prefer not to discuss in detail.",
        "Various factors contributed to my decision to leave the company.",
        "It was a difficult decision but I felt it was the right time to move on.",
    ]

    feedbacks = []
    np.random.seed(42)

    for _, row in df.iterrows():
        reason = str(row.get("TermReason", "N/A-StillEmployed")).strip().lower()

        # Find matching template
        matched = False
        for key, templates in feedback_templates.items():
            if key.lower() in reason:
                feedback = np.random.choice(templates)
                matched = True
                break

        if not matched:
            feedback = np.random.choice(default_feedback)

        # Add some contextual details
        satisfaction = row.get("EmpSatisfaction", 3)
        engagement = row.get("EngagementSurvey", 3.0)

        if satisfaction <= 2:
            feedback += " Overall, my satisfaction with the job was quite low."
        elif satisfaction >= 5:
            feedback += " Despite this, I had many positive experiences here."

        if engagement < 3.0:
            feedback += " I felt disconnected from the company's mission."
        elif engagement > 4.5:
            feedback += " I always felt engaged with the work itself."

        feedbacks.append({
            "EmployeeID_Anon": "EMP_" + hashlib.sha256(
                str(row.get("Employee_Name", "")).encode()
            ).hexdigest()[:8].upper(),
            "Termd": row.get("Termd", 0),
            "TermReason": row.get("TermReason", "N/A-StillEmployed"),
            "Department": str(row.get("Department", "Unknown")).strip(),
            "Feedback": feedback,
            "Satisfaction": satisfaction,
            "Engagement": engagement,
        })

    feedback_df = pd.DataFrame(feedbacks)
    print(f"✅ Generated {len(feedback_df)} simulated feedback entries")
    return feedback_df


def run_pipeline():
    """Run the full data processing pipeline."""
    print("=" * 60)
    print("🔄 DATA PROCESSING PIPELINE")
    print("=" * 60)

    # Create output directory
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Step 1: Load
    df = load_raw_data()

    # Step 2: Anonymize
    df_anon = anonymize_data(df)

    # Step 3: Feature Engineering
    df_feat = engineer_features(df_anon)

    # Step 4: Save anonymized dataset
    df_feat.to_csv(ANONYMIZED_PATH, index=False)
    print(f"✅ Saved anonymized dataset to: {ANONYMIZED_PATH}")

    # Step 5: Generate simulated feedback
    feedback_df = generate_simulated_feedback(df)
    feedback_df.to_csv(FEEDBACK_PATH, index=False)
    print(f"✅ Saved feedback data to: {FEEDBACK_PATH}")

    # Step 6: Get model features
    X, y, feature_names = get_model_features(df_feat)
    print(f"\n✅ Pipeline complete!")
    print(f"   Dataset shape: {X.shape}")
    print(f"   Target distribution: {dict(y.value_counts())}")

    return df_feat, X, y, feature_names, feedback_df


if __name__ == "__main__":
    run_pipeline()
