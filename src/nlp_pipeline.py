"""
NLP feature extraction pipeline using HuggingFace Transformers.

This module converts tabular student features into natural language
descriptions, then uses a pre-trained DistilBERT model to extract
sentiment and confidence signals. These NLP-derived features augment
the tabular feature set for the hybrid prediction model.

Pipeline:
1. Convert each student's feature vector to a natural language sentence
2. Run DistilBERT sentiment classification on the generated sentences
3. Extract label and confidence as numeric features
4. Combine with forum engagement proxy features
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Student Profile Text Generation
# ---------------------------------------------------------------------------

def student_to_text(row: pd.Series, feature_stats: Dict[str, float]) -> str:
    """
    Convert a student's feature row into a natural language description.

    The generated sentence describes the student's engagement level,
    learning pace, and assessment performance in qualitative terms.
    This serves as input to the sentiment classifier.

    Parameters
    ----------
    row : pd.Series
        A single student's feature values.
    feature_stats : dict
        Dictionary with 'mean_engagement', 'mean_pace', 'mean_score'
        used as thresholds for qualitative classification.

    Returns
    -------
    str
        Natural language description of the student profile.
    """
    # Engagement level
    engagement_rate = row.get("module_engagement_rate", 0)
    if engagement_rate > 1.5:
        engagement = "very high"
    elif engagement_rate > 0.8:
        engagement = "moderate"
    elif engagement_rate > 0.3:
        engagement = "low"
    else:
        engagement = "minimal"

    # Learning pace
    pace = row.get("learning_pace", 0)
    mean_pace = feature_stats.get("mean_pace", 0)
    if pace < mean_pace - 5:
        pace_desc = "ahead of schedule"
    elif pace < mean_pace + 5:
        pace_desc = "on schedule"
    elif pace < mean_pace + 15:
        pace_desc = "slightly behind schedule"
    else:
        pace_desc = "significantly behind schedule"

    # Assessment performance
    score = row.get("average_score", 0)
    if score > 80:
        performance = "excellent"
    elif score > 60:
        performance = "satisfactory"
    elif score > 40:
        performance = "below average"
    else:
        performance = "poor"

    # Completion ratio
    completion = row.get("completed_assessments_ratio", 0)
    if completion > 0.9:
        completion_desc = "has completed nearly all assessments"
    elif completion > 0.6:
        completion_desc = "has completed most assessments"
    elif completion > 0.3:
        completion_desc = "has completed some assessments"
    else:
        completion_desc = "has completed few or no assessments"

    # Forum engagement
    forum_ratio = row.get("forum_engagement_ratio", 0)
    if forum_ratio > 0.15:
        forum_desc = "active participation in discussion forums"
    elif forum_ratio > 0.05:
        forum_desc = "occasional forum participation"
    else:
        forum_desc = "little to no forum activity"

    text = (
        f"This student shows {engagement} engagement with the course material, "
        f"is {pace_desc} in assessment submissions, and demonstrates "
        f"{performance} academic performance. The student {completion_desc} "
        f"and shows {forum_desc}."
    )

    return text


def generate_student_descriptions(
    df: pd.DataFrame,
    feature_stats: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Generate natural language descriptions for all students.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with one row per student.
    feature_stats : dict, optional
        Precomputed statistics for thresholding. If None, computed from df.

    Returns
    -------
    pd.Series
        Series of text descriptions, indexed like the input DataFrame.
    """
    if feature_stats is None:
        feature_stats = {
            "mean_engagement": df["module_engagement_rate"].mean()
            if "module_engagement_rate" in df.columns else 1.0,
            "mean_pace": df["learning_pace"].mean()
            if "learning_pace" in df.columns else 0.0,
            "mean_score": df["average_score"].mean()
            if "average_score" in df.columns else 50.0,
        }

    descriptions = df.apply(
        lambda row: student_to_text(row, feature_stats), axis=1
    )

    return descriptions


# ---------------------------------------------------------------------------
# DistilBERT Sentiment Classification
# ---------------------------------------------------------------------------

def load_sentiment_classifier(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
):
    """
    Load the HuggingFace sentiment classification pipeline.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.

    Returns
    -------
    transformers.Pipeline
        Ready-to-use text classification pipeline.
    """
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model=model_name,
        truncation=True,
        max_length=512,
    )

    print(f"Loaded sentiment classifier: {model_name}")
    return classifier


def extract_nlp_features(
    descriptions: pd.Series,
    classifier=None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Run sentiment classification on student descriptions and extract features.

    Parameters
    ----------
    descriptions : pd.Series
        Natural language descriptions of student profiles.
    classifier : transformers.Pipeline, optional
        Pre-loaded classifier. If None, loads the default model.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - nlp_sentiment : predicted label (POSITIVE / NEGATIVE)
        - nlp_confidence : model confidence score
        - nlp_label : numeric encoding (POSITIVE=1, NEGATIVE=0)
    """
    if classifier is None:
        classifier = load_sentiment_classifier()

    texts = descriptions.tolist()

    # Process in batches to manage memory
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = classifier(batch, truncation=True)
        all_results.extend(results)

    nlp_df = pd.DataFrame({
        "nlp_sentiment": [r["label"] for r in all_results],
        "nlp_confidence": [r["score"] for r in all_results],
    })

    # Numeric encoding
    nlp_df["nlp_label"] = (nlp_df["nlp_sentiment"] == "POSITIVE").astype(int)

    # Signed confidence: positive for POSITIVE, negative for NEGATIVE
    nlp_df["nlp_signed_confidence"] = np.where(
        nlp_df["nlp_label"] == 1,
        nlp_df["nlp_confidence"],
        -nlp_df["nlp_confidence"],
    )

    nlp_df.index = descriptions.index

    return nlp_df


# ---------------------------------------------------------------------------
# Forum Engagement Proxy Features
# ---------------------------------------------------------------------------

def build_forum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build forum engagement proxy features.

    Since raw forum text is not publicly available in OULAD,
    we use behavioral signals from VLE interaction patterns
    as proxies for social learning engagement.

    Features created:
    - forum_engagement_ratio : forum clicks / total clicks
    - forum_silence          : binary flag for zero forum activity
    - late_forum_dropoff     : below 25th percentile of forum engagement
    - forum_vs_content_ratio : forum clicks / content clicks
    """
    result = pd.DataFrame(index=df.index)

    total_clicks = df.get("total_clicks", pd.Series(0, index=df.index))
    forum_clicks = df.get("engagement_forumng", pd.Series(0, index=df.index))
    content_clicks = df.get("engagement_oucontent", pd.Series(0, index=df.index))

    result["forum_engagement_ratio"] = forum_clicks / (total_clicks + 1)
    result["forum_silence"] = (forum_clicks == 0).astype(int)

    q25 = forum_clicks[forum_clicks > 0].quantile(0.25) if (forum_clicks > 0).any() else 0
    result["late_forum_dropoff"] = (forum_clicks < q25).astype(int)

    result["forum_vs_content_ratio"] = forum_clicks / (content_clicks + 1)

    return result


# ---------------------------------------------------------------------------
# Full NLP Feature Pipeline
# ---------------------------------------------------------------------------

def run_nlp_pipeline(
    df: pd.DataFrame,
    classifier=None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Run the complete NLP feature extraction pipeline.

    Steps:
    1. Generate natural language descriptions from tabular features
    2. Run DistilBERT sentiment classification
    3. Extract forum engagement proxy features
    4. Combine all NLP-derived features

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with tabular features.
    classifier : transformers.Pipeline, optional
        Pre-loaded classifier.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    pd.DataFrame
        Original DataFrame augmented with NLP features.
    """
    print("Step 1/3: Generating student descriptions...")
    descriptions = generate_student_descriptions(df)
    print(f"  Generated {len(descriptions)} descriptions")

    print("Step 2/3: Running DistilBERT sentiment classification...")
    nlp_features = extract_nlp_features(descriptions, classifier, batch_size)
    print(f"  Sentiment distribution:")
    print(f"    POSITIVE: {(nlp_features['nlp_label'] == 1).sum()}")
    print(f"    NEGATIVE: {(nlp_features['nlp_label'] == 0).sum()}")

    print("Step 3/3: Building forum engagement features...")
    forum_features = build_forum_features(df)

    # Combine
    result = df.copy()
    for col in nlp_features.columns:
        result[col] = nlp_features[col].values
    for col in forum_features.columns:
        if col not in result.columns:
            result[col] = forum_features[col].values

    print(f"\nNLP pipeline complete. Added {len(nlp_features.columns) + len(forum_features.columns)} features.")

    return result
