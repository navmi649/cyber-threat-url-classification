"""
FastAPI Backend Application for Real-Time Cyber Threat & Malicious URL Classification
Part of the 3rd-year B.Tech CSE Capstone Project.
"""

import os
import sys
import time

# Ensure backend directory is in sys.path regardless of execution working directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import joblib
import numpy as np
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils import (
    extract_lexical_features,
    extract_lexical_vector,
    extract_lexical_matrix,
    fuse_features,
    get_risk_assessment,
    CLASS_LABELS,
    LABEL_TO_ID
)


# Global containers for model artifacts
MODEL_CACHE = {
    "model": None,
    "vectorizer": None,
    "scaler": None,
    "is_loaded": False,
    "model_path": None,
    "vectorizer_path": None
}


def load_artifacts():
    """Load serialized Logistic Regression model, TF-IDF vectorizer, and feature scaler."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.normpath(os.path.join(base_dir, "..", "models"))
    
    model_path = os.path.join(models_dir, "saved_model.pkl")
    vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please execute train_model.py first.")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}. Please execute train_model.py first.")

    MODEL_CACHE["model"] = joblib.load(model_path)
    MODEL_CACHE["vectorizer"] = joblib.load(vectorizer_path)
    if os.path.exists(scaler_path):
        MODEL_CACHE["scaler"] = joblib.load(scaler_path)
    elif hasattr(MODEL_CACHE["vectorizer"], "scaler_"):
        MODEL_CACHE["scaler"] = MODEL_CACHE["vectorizer"].scaler_
    else:
        MODEL_CACHE["scaler"] = None

    MODEL_CACHE["model_path"] = model_path
    MODEL_CACHE["vectorizer_path"] = vectorizer_path
    MODEL_CACHE["is_loaded"] = True
    print(f"[FastAPI Engine] Successfully loaded model and vectorizer from {models_dir}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load models once on startup."""
    try:
        load_artifacts()
    except Exception as e:
        print(f"[FastAPI Warning] Artifact loading deferred: {e}")
    yield


app = FastAPI(
    title="Cyber Threat & Malicious URL Classification API",
    description=(
        "Production-ready REST API implementing Feature Fusion (Lexical + Char-level TF-IDF) "
        "and Multinomial Logistic Regression (Softmax) for real-time cyber threat detection."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable Cross-Origin Resource Sharing (CORS) for Streamlit frontend & external callers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request and Response Schemas ---

class URLRequest(BaseModel):
    url: str = Field(..., example="http://paypal-security-account-verification.com/login")


class BatchURLRequest(BaseModel):
    urls: List[str] = Field(..., example=[
        "https://www.google.com",
        "http://185.220.101.45:8080/payload.exe",
        "http://netflix-billing-update-center.org/account/verify"
    ])


class LexicalFeaturesResponse(BaseModel):
    url_length: int
    dot_count: int
    hyphen_count: int
    digit_ratio: float
    digit_count: Optional[int] = 0
    at_count: Optional[int] = 0
    slash_count: Optional[int] = 0
    question_count: Optional[int] = 0
    equal_count: Optional[int] = 0
    has_ip: Optional[int] = 0
    has_https: Optional[int] = 0


class RiskAssessmentResponse(BaseModel):
    severity: str
    color: str
    description: str
    anomalies: List[str]


class PredictionResponse(BaseModel):
    url: str
    prediction: str
    class_id: int
    confidence: float
    probabilities: Dict[str, float]
    lexical_features: LexicalFeaturesResponse
    risk_assessment: RiskAssessmentResponse
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    model_type: str
    classes: Dict[int, str]


# --- Helper Prediction Function ---

def run_single_inference(url: str) -> Dict[str, Any]:
    """Extracts features, executes feature fusion, and computes softmax predictions."""
    if not MODEL_CACHE["is_loaded"]:
        load_artifacts()

    model = MODEL_CACHE["model"]
    vectorizer = MODEL_CACHE["vectorizer"]

    start_time = time.perf_counter()

    # 1. Lexical Feature Extraction
    lex_dict = extract_lexical_features(url)
    lex_vec = extract_lexical_vector(url)  # Shape (1, 4)

    # Standardize lexical vector if scaler is present
    scaler = MODEL_CACHE.get("scaler") or getattr(vectorizer, "scaler_", None)
    if scaler is not None:
        lex_scaled = scaler.transform(lex_vec)
    else:
        lex_scaled = lex_vec

    # 2. Character-level TF-IDF Transformation
    tfidf_vec = vectorizer.transform([url])

    # 3. Mathematical Feature Fusion via scipy.sparse.hstack
    fused_features = fuse_features(lex_scaled, tfidf_vec)

    # 4. Multi-Class Logistic Regression Inference
    pred_id = int(model.predict(fused_features)[0])
    pred_label = CLASS_LABELS.get(pred_id, "UNKNOWN")
    
    # 5. Softmax Probabilities
    probs = model.predict_proba(fused_features)[0]
    prob_dict = {
        CLASS_LABELS[i]: round(float(p), 4)
        for i, p in enumerate(probs)
    }
    confidence = round(float(np.max(probs)), 4)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    risk = get_risk_assessment(pred_label, confidence, lex_dict)

    return {
        "url": url,
        "prediction": pred_label,
        "class_id": pred_id,
        "confidence": confidence,
        "probabilities": prob_dict,
        "lexical_features": lex_dict,
        "risk_assessment": risk,
        "inference_time_ms": elapsed_ms
    }


# --- API Routes ---

@app.get("/", tags=["General"])
async def root():
    return {
        "project": "Cyber Threat & Malicious URL Classification System",
        "author": "3rd-Year B.Tech CSE Capstone Project",
        "status": "Online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "batch_predict": "/predict/batch (POST)",
            "samples": "/samples (GET)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return {
        "status": "HEALTHY",
        "models_loaded": MODEL_CACHE["is_loaded"],
        "model_type": "Multi-Class Logistic Regression with Character TF-IDF & Lexical Fusion",
        "classes": CLASS_LABELS
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_url(payload: URLRequest):
    """
    Accepts a single raw URL string, extracts lexical and TF-IDF features,
    fuses them horizontally, and returns classification and softmax probabilities.
    """
    if not payload.url or not payload.url.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The URL string cannot be empty."
        )

    try:
        result = run_single_inference(payload.url)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Inference"])
async def predict_batch(payload: BatchURLRequest):
    """
    Accepts a list of URLs and returns predictions for all samples.
    """
    if not payload.urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL list cannot be empty."
        )

    results = []
    for u in payload.urls:
        if u and u.strip():
            results.append(run_single_inference(u))
    return results


@app.get("/samples", tags=["Testing"])
async def get_sample_urls():
    """Provides curated sample test URLs across all three classes for testing and presentation."""
    return {
        "SAFE": [
            "https://www.google.com",
            "https://github.com/torvalds/linux",
            "https://en.wikipedia.org/wiki/Cybersecurity",
            "https://stackoverflow.com/questions/tagged/python",
            "https://docs.python.org/3/library/scipy.html"
        ],
        "MALICIOUS": [
            "http://185.220.101.45:8080/payload.exe",
            "http://194.26.29.112/c2/beacon.php?id=9942",
            "http://botnet-c2-command.ru/gate.php?aff=99281&exec=true",
            "http://dark-shell-backdoor.info/shell.php?cmd=whoami",
            "http://ransomware-distro.su/payment/decrypt_tool.exe"
        ],
        "PHISHING": [
            "http://paypal-security-account-verification.com/login",
            "http://netflix-billing-update-center.org/account/verify",
            "http://apple-id-credential-reset.info/manage/auth",
            "http://wellsfargo-secure-banking-portal.online/signon",
            "http://chase-bank-verify-identity-login.com/auth/step2"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
