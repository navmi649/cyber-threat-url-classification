"""
Lexical Feature Extraction and Feature Fusion Utilities
Part of the Real-Time Cyber Threat & Malicious URL Classification System.
"""

import re
import numpy as np
from typing import Dict, Any, List, Union
from scipy.sparse import hstack, csr_matrix, issparse


# Core 4 Lexical Feature Names specified in capstone requirements
CORE_LEXICAL_FEATURES = [
    "url_length",
    "dot_count",
    "hyphen_count",
    "digit_ratio",
]

# Label encodings
CLASS_LABELS = {
    0: "SAFE",
    1: "MALICIOUS",
    2: "PHISHING"
}

LABEL_TO_ID = {
    "SAFE": 0,
    "MALICIOUS": 1,
    "PHISHING": 2
}


def sanitize_url(url: str) -> str:
    """
    Standardize the URL string for parsing and feature extraction.
    Removes leading/trailing whitespaces while preserving the structural path.
    """
    if not isinstance(url, str):
        return ""
    return url.strip()


def extract_lexical_features(url: str) -> Dict[str, Any]:
    """
    Extracts structural numerical lexical features for any given URL string:
    - URL Total Length: len(url)
    - Dot Frequency Count: count of '.'
    - Hyphen Count: count of '-'
    - Digit Ratio: count of numeric digits / total length
    
    Also computes secondary forensic metrics for enriched dashboard insights.
    """
    url_clean = sanitize_url(url)
    length = len(url_clean)

    if length == 0:
        return {
            "url_length": 0,
            "dot_count": 0,
            "hyphen_count": 0,
            "digit_ratio": 0.0,
            # Secondary diagnostic metrics
            "digit_count": 0,
            "at_count": 0,
            "slash_count": 0,
            "question_count": 0,
            "equal_count": 0,
            "has_ip": 0,
            "has_https": 0,
        }

    # Core required features
    dot_count = url_clean.count('.')
    hyphen_count = url_clean.count('-')
    digits = sum(c.isdigit() for c in url_clean)
    digit_ratio = round(digits / length, 4)

    # Forensic indicators
    at_count = url_clean.count('@')
    slash_count = url_clean.count('/')
    question_count = url_clean.count('?')
    equal_count = url_clean.count('=')
    
    # Check if host is an IPv4 address
    ip_pattern = r"(?:https?://)?(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?"
    has_ip = 1 if re.search(ip_pattern, url_clean) else 0
    has_https = 1 if url_clean.lower().startswith("https://") else 0

    return {
        "url_length": length,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "digit_ratio": digit_ratio,
        # Secondary diagnostic metrics
        "digit_count": digits,
        "at_count": at_count,
        "slash_count": slash_count,
        "question_count": question_count,
        "equal_count": equal_count,
        "has_ip": has_ip,
        "has_https": has_https,
    }


def extract_lexical_vector(url: str) -> np.ndarray:
    """
    Returns a 2D numpy array of shape (1, 4) for the 4 core lexical features:
    [url_length, dot_count, hyphen_count, digit_ratio]
    """
    features = extract_lexical_features(url)
    vec = [
        features["url_length"],
        features["dot_count"],
        features["hyphen_count"],
        features["digit_ratio"],
    ]
    return np.array([vec], dtype=np.float64)


def extract_lexical_matrix(urls: List[str]) -> np.ndarray:
    """
    Extracts lexical feature matrix of shape (n_samples, 4) for a list of URLs.
    """
    rows = []
    for u in urls:
        f = extract_lexical_features(u)
        rows.append([
            f["url_length"],
            f["dot_count"],
            f["hyphen_count"],
            f["digit_ratio"],
        ])
    return np.array(rows, dtype=np.float64)


def fuse_features(lexical_matrix: Union[np.ndarray, csr_matrix], tfidf_matrix) -> csr_matrix:
    """
    Mathematical Feature Fusion:
    Horizontally stacks the dense numerical lexical feature array
    with the high-dimensional sparse TF-IDF matrix using `scipy.sparse.hstack`.
    
    Returns a unified scipy CSR sparse matrix.
    """
    if not issparse(lexical_matrix):
        lexical_sparse = csr_matrix(lexical_matrix)
    else:
        lexical_sparse = lexical_matrix

    if not issparse(tfidf_matrix):
        tfidf_sparse = csr_matrix(tfidf_matrix)
    else:
        tfidf_sparse = tfidf_matrix

    fused = hstack([lexical_sparse, tfidf_sparse], format="csr")
    return fused


def get_risk_assessment(prediction: str, confidence: float, features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provides human-readable threat diagnostics based on extracted metrics and prediction.
    """
    reasons = []
    
    if features.get("has_ip", 0) == 1:
        reasons.append("Raw IPv4 address detected instead of registered domain name.")
    if features.get("digit_ratio", 0) > 0.20:
        reasons.append(f"High digit ratio ({features['digit_ratio'] * 100:.1f}%), often associated with DGA or hex obfuscation.")
    if features.get("dot_count", 0) >= 4:
        reasons.append(f"Excessive dot count ({features['dot_count']}), indicative of complex subdomain spoofing.")
    if features.get("hyphen_count", 0) >= 3:
        reasons.append(f"Multiple hyphens ({features['hyphen_count']}) commonly used to brand-jack legitimate targets.")
    if features.get("at_count", 0) >= 1:
        reasons.append("Contains '@' character used for credential injection or URL disguise.")
    if features.get("url_length", 0) > 85:
        reasons.append(f"Abnormally long URL ({features['url_length']} chars), concealing payload parameters.")

    if prediction == "MALICIOUS":
        severity = "CRITICAL"
        color = "#dc3545"
        description = "Identified as malicious infrastructure (payload delivery, trojan, or C2 string)."
    elif prediction == "PHISHING":
        severity = "HIGH"
        color = "#fd7e14"
        description = "Identified as a phishing link attempting credential or financial theft."
    else:
        severity = "LOW"
        color = "#28a745"
        description = "Structural and lexical properties align with standard legitimate web services."

    return {
        "severity": severity,
        "color": color,
        "description": description,
        "anomalies": reasons if reasons else ["No high-risk lexical anomalies flagged."]
    }
