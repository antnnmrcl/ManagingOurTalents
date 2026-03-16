"""
NLP Analysis Module — Employee Feedback
========================================
Performs sentiment analysis and theme extraction on
simulated employee feedback data using lightweight/frugal models.
"""

import pandas as pd
import numpy as np
import os
import re
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")


# ============================================================
# SENTIMENT ANALYSIS (Frugal: TextBlob — no GPU needed)
# ============================================================

def analyze_sentiment(text):
    """Analyze sentiment of a text using TextBlob (lightweight/frugal)."""
    if not HAS_TEXTBLOB:
        # Fallback: simple keyword-based sentiment
        return _keyword_sentiment(text)

    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity      # -1 to 1
    subjectivity = blob.sentiment.subjectivity  # 0 to 1

    if polarity > 0.1:
        label = "Positive"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "polarity": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "sentiment_label": label
    }


def _keyword_sentiment(text):
    """Fallback keyword-based sentiment analysis."""
    text_lower = str(text).lower()
    pos_words = ["enjoy", "great", "excellent", "happy", "love", "appreciate",
                 "supportive", "satisfied", "positive", "wonderful", "growth",
                 "valued", "recognized", "opportunity"]
    neg_words = ["unhappy", "stressed", "frustrated", "unfair", "terrible",
                 "burned", "disconnected", "undervalued", "crisis", "overwhelmed",
                 "difficult", "impossible", "inadequate", "strict", "low"]

    pos_count = sum(1 for w in pos_words if w in text_lower)
    neg_count = sum(1 for w in neg_words if w in text_lower)

    if pos_count > neg_count:
        polarity = min(0.5, pos_count * 0.15)
        label = "Positive"
    elif neg_count > pos_count:
        polarity = max(-0.5, -neg_count * 0.15)
        label = "Negative"
    else:
        polarity = 0.0
        label = "Neutral"

    return {"polarity": polarity, "subjectivity": 0.5, "sentiment_label": label}


def run_sentiment_analysis(feedback_df):
    """Run sentiment analysis on all feedback entries."""
    print("\n📝 Running Sentiment Analysis (TextBlob — Frugal NLP)...")

    sentiments = feedback_df["Feedback"].apply(analyze_sentiment)
    sent_df = pd.DataFrame(sentiments.tolist())
    result_df = pd.concat([feedback_df, sent_df], axis=1)

    # Summary stats
    print(f"\n  📊 Sentiment Distribution:")
    for label, count in result_df["sentiment_label"].value_counts().items():
        pct = count / len(result_df) * 100
        print(f"     {label}: {count} ({pct:.1f}%)")

    # By termination status
    print(f"\n  📊 Average Sentiment by Status:")
    for status, group in result_df.groupby("Termd"):
        status_label = "Terminated" if status == 1 else "Active"
        avg_pol = group["polarity"].mean()
        print(f"     {status_label}: avg polarity = {avg_pol:.4f}")

    return result_df


# ============================================================
# THEME EXTRACTION (Frugal: keyword-based, no LLM needed)
# ============================================================

THEME_KEYWORDS = {
    "Compensation": ["salary", "pay", "money", "compensation", "raise", "offer", "benefits",
                     "budget", "financial", "market rate", "package"],
    "Work-Life Balance": ["hours", "overtime", "balance", "schedule", "flexibility",
                          "work-life", "stress", "burnout", "burned", "weekends"],
    "Career Growth": ["career", "growth", "opportunity", "advancement", "promotion",
                      "leadership", "responsibility", "development", "potential"],
    "Management": ["manager", "management", "leadership", "listen", "supportive",
                   "communication", "recognition", "appreciated", "valued"],
    "Company Culture": ["culture", "environment", "team", "morale", "values",
                        "mission", "toxic", "collaborative", "engaged"],
    "Personal Reasons": ["family", "personal", "relocat", "health", "medical",
                         "military", "school", "education", "retire"],
    "Performance": ["performance", "expectations", "training", "support", "feedback",
                    "review", "targets", "criteria", "assessment"],
    "Job Satisfaction": ["satisfied", "enjoy", "happy", "unhappy", "motivated",
                         "fulfilled", "disconnect", "passionate"],
}


def extract_themes(text):
    """Extract themes from feedback text using keyword matching."""
    text_lower = str(text).lower()
    themes = []
    theme_scores = {}

    for theme, keywords in THEME_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            themes.append(theme)
            theme_scores[theme] = count

    return themes, theme_scores


def run_theme_analysis(feedback_df):
    """Run theme extraction on all feedback entries."""
    print("\n📝 Running Theme Extraction (Keyword-based — Frugal NLP)...")

    all_themes = []
    theme_data = []

    for _, row in feedback_df.iterrows():
        themes, scores = extract_themes(row["Feedback"])
        all_themes.extend(themes)
        theme_data.append({
            "themes": ", ".join(themes) if themes else "General",
            "primary_theme": themes[0] if themes else "General",
            "n_themes": len(themes)
        })

    theme_df = pd.DataFrame(theme_data)
    result_df = pd.concat([feedback_df.reset_index(drop=True), theme_df], axis=1)

    # Theme frequency
    theme_counts = Counter(all_themes)
    print(f"\n  📊 Theme Frequency (All Feedback):")
    for theme, count in theme_counts.most_common():
        print(f"     {theme}: {count}")

    # Themes by termination status
    print(f"\n  📊 Top Themes — Terminated Employees:")
    termed = result_df[result_df["Termd"] == 1]
    if not termed.empty:
        termed_themes = []
        for themes_str in termed["themes"]:
            termed_themes.extend([t.strip() for t in themes_str.split(",") if t.strip() != "General"])
        for theme, count in Counter(termed_themes).most_common(5):
            print(f"     {theme}: {count}")

    return result_df, theme_counts


def run_nlp_pipeline(feedback_path=None):
    """Run the full NLP pipeline."""
    print("\n" + "=" * 60)
    print("📝 NLP ANALYSIS PIPELINE")
    print("=" * 60)

    if feedback_path is None:
        feedback_path = os.path.join(DATA_DIR, "employee_feedback.csv")

    feedback_df = pd.read_csv(feedback_path)
    print(f"✅ Loaded {len(feedback_df)} feedback entries")

    # 1. Sentiment Analysis
    sent_df = run_sentiment_analysis(feedback_df)

    # 2. Theme Extraction
    result_df, theme_counts = run_theme_analysis(sent_df)

    # 3. Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "nlp_analysis_results.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\n✅ NLP results saved to: {output_path}")

    return result_df, theme_counts


if __name__ == "__main__":
    run_nlp_pipeline()
