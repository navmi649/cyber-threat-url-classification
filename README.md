# Real-Time Cyber Threat & Malicious URL Classification System
### 3rd-Year B.Tech CSE Capstone Project

An end-to-end Machine Learning cyber defense platform engineered to detect, inspect, and classify malicious URLs, command-and-control (C2) gateways, and credential phishing portals in real-time.

The system utilizes a **Hybrid Feature Fusion Pipeline** that combines structural dense lexical analysis with high-dimensional character-level TF-IDF n-grams ($n \in [3, 5]$), driving a **Multinomial Logistic Regression Classifier** with softmax activation for precise probability vector distribution.

---

## 🏛️ System Architecture

```
                       ┌────────────────────────────────────────┐
                       │            Incoming Raw URL            │
                       └───────────────────┬────────────────────┘
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   ▼                                               ▼
     ┌────────────────────────────┐                 ┌─────────────────────────────┐
     │  Lexical Feature Extractor │                 │   Character TF-IDF N-Grams  │
     │  - URL Length              │                 │   - analyzer = 'char'       │
     │  - Dot Frequency Count     │                 │   - ngram_range = (3, 5)    │
     │  - Hyphen Count            │                 │   - Sublinear TF scaling    │
     │  - Digit Ratio             │                 └──────────────┬──────────────┘
     └─────────────┬──────────────┘                                │
                   │ (Dense 4-dim vector)                          │ (Sparse 6000-dim matrix)
                   ▼                                               ▼
         ┌───────────────────┐                           ┌───────────────────┐
         │ Feature Scaling   │                           │ TF-IDF Matrix     │
         └─────────┬─────────┘                           └─────────┬─────────┘
                   │                                               │
                   └───────────────────────┬───────────────────────┘
                                           ▼
                       ┌────────────────────────────────────────┐
                       │       Mathematical Feature Fusion      │
                       │          (scipy.sparse.hstack)         │
                       │    X_fused = [ X_lexical || X_tfidf ]  │
                       └───────────────────┬────────────────────┘
                                           │ (Unified CSR Matrix)
                                           ▼
                       ┌────────────────────────────────────────┐
                       │   Multi-Class Logistic Regression      │
                       │     (Multinomial Softmax Loss)         │
                       └───────────────────┬────────────────────┘
                                           │
                   ┌───────────────────────┼───────────────────────┐
                   ▼                       ▼                       ▼
            Class 0: SAFE           Class 1: MALICIOUS       Class 2: PHISHING
         (Legitimate Domains)      (Trojans, C2, Drops)     (Credential Portals)
```

---

## 🔬 Mathematical Formulation & Feature Pipeline

### 1. Structural Lexical Attribute Extractor (`backend/utils.py`)
For any given URL string $s$ of length $L = |s|$:
- **Total URL Length**: 
  $$L = |s|$$
- **Dot Frequency Count**:
  $$N_{\text{dot}} = \sum_{i=1}^{L} \mathbb{I}[s[i] = '.']$$
- **Hyphen Frequency Count**:
  $$N_{\text{hyphen}} = \sum_{i=1}^{L} \mathbb{I}[s[i] = '-']$$
- **Digit Ratio**:
  $$R_{\text{digit}} = \frac{\sum_{i=1}^{L} \mathbb{I}[s[i] \in \{'0', '1', \dots, '9'\}]}{L}$$

### 2. High-Dimensional Character-Level TF-IDF Vectorizer
Captures sub-word morphological patterns, suspicious file extensions (`.exe`, `.dll`, `.vbs`, `.sh`), parameters (`affid`, `token`, `exec`), and spoofed brand sequences:
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
Configured with `analyzer='char'`, `ngram_range=(3, 5)`, and sublinear term-frequency scaling.

### 3. Mathematical Feature Fusion
Dense structural features are horizontally fused with the high-dimensional sparse TF-IDF matrix using `scipy.sparse.hstack`:
$$X_{\text{fused}} = \left[ X_{\text{lexical\_scaled}} \;\mathbin{\Vert}\; X_{\text{tfidf}} \right] \in \mathbb{R}^{N \times (4 + D)}$$

### 4. Multinomial Logistic Regression (Softmax)
The probability distribution across all three threat classes ($k \in \{0, 1, 2\}$) is computed via the Softmax activation function:
$$P(Y = k \mid X_{\text{fused}}) = \frac{\exp(\mathbf{w}_k^T X_{\text{fused}} + b_k)}{\sum_{j=0}^{2} \exp(\mathbf{w}_j^T X_{\text{fused}} + b_j)}$$
This guarantees that all output probabilities strictly sum to $1.0$ ($100\%$).

---

## 📁 Directory Structure

```
newapp/
├── backend/
│   ├── app.py              # FastAPI REST API with CORS, endpoints, and validation
│   ├── train_model.py      # Dataset generation, feature fusion & ML model training
│   └── utils.py            # Lexical feature extractor & scipy horizontal fusion
├── frontend/
│   └── streamlit_app.py    # Interactive Streamlit analytics dashboard with Plotly
├── models/
│   ├── saved_model.pkl     # Serialized Multi-Class Logistic Regression model
│   ├── vectorizer.pkl      # Serialized Character TF-IDF vectorizer pipeline
│   ├── scaler.pkl          # Serialized Lexical Feature Scaler
│   └── model_metrics.json  # Model performance report and confusion matrix
├── requirements.txt        # Full project dependencies
└── README.md               # Capstone project documentation & user guide
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+ (Recommended: Python 3.12)
- Pip package manager

### Step 1: Install Dependencies
Open a terminal in the project root directory (`newapp`) and execute:
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### Step 1: Train the Machine Learning Pipeline
Train the feature fusion pipeline, multi-class model, and serialize artifacts into `/models`:
```bash
python backend/train_model.py
```
*Expected Output:*
- Generates balanced dataset (2,250 samples; 750 SAFE, 750 MALICIOUS, 750 PHISHING)
- Extracts 4 structural lexical metrics and 6,000 character n-gram features
- Trains Multi-Class Logistic Regression with multinomial softmax
- Serializes `saved_model.pkl`, `vectorizer.pkl`, and `scaler.pkl` to `models/`

### Step 2: Launch the FastAPI Backend Server
Start the high-performance REST API:
```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- Healthcheck Endpoint: `http://127.0.0.1:8000/health`

### Step 3: Launch the Streamlit Interactive Dashboard
In a new terminal window, start the frontend UI:
```bash
streamlit run frontend/streamlit_app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📡 REST API Reference (`backend/app.py`)

### 1. Healthcheck
- **Method:** `GET`
- **Path:** `/health`
- **Response:**
```json
{
  "status": "HEALTHY",
  "models_loaded": true,
  "model_type": "Multi-Class Logistic Regression with Character TF-IDF & Lexical Fusion",
  "classes": {
    "0": "SAFE",
    "1": "MALICIOUS",
    "2": "PHISHING"
  }
}
```

### 2. Single URL Prediction
- **Method:** `POST`
- **Path:** `/predict`
- **Request Body:**
```json
{
  "url": "http://paypal-secure-login.com/login.php"
}
```
- **Response:**
```json
{
  "url": "http://paypal-secure-login.com/login.php",
  "prediction": "PHISHING",
  "class_id": 2,
  "confidence": 0.9934,
  "probabilities": {
    "SAFE": 0.0001,
    "MALICIOUS": 0.0065,
    "PHISHING": 0.9934
  },
  "lexical_features": {
    "url_length": 40,
    "dot_count": 2,
    "hyphen_count": 2,
    "digit_ratio": 0.0,
    "has_ip": 0,
    "has_https": 0
  },
  "risk_assessment": {
    "severity": "HIGH",
    "color": "#fd7e14",
    "description": "Identified as a phishing link attempting credential or financial theft.",
    "anomalies": ["No high-risk lexical anomalies flagged."]
  },
  "inference_time_ms": 1.51
}
```

### 3. Batch URL Threat Inspection
- **Method:** `POST`
- **Path:** `/predict/batch`
- **Request Body:**
```json
{
  "urls": [
    "https://github.com/torvalds/linux",
    "http://185.220.101.45:8080/payload.exe",
    "http://netflix-billing-update-center.org/account/verify"
  ]
}
```

### 4. Sample Test Suite
- **Method:** `GET`
- **Path:** `/samples`
- Returns pre-curated test vectors across SAFE, MALICIOUS, and PHISHING domains for demonstrations.

---

## 🖥️ Frontend Dashboard Features (`frontend/streamlit_app.py`)

1. **Categorical Alert Verdict Banners**:
   - 🟢 **SAFE**: Green banner with Low-Risk security posture.
   - 🔴 **MALICIOUS**: Red banner indicating trojan/payload delivery or C2 gateway.
   - 🟠 **PHISHING**: Orange banner highlighting credential harvesting or brand spoofing.
2. **Dynamic Plotly Visualizations**:
   - Horizontal softmax probability breakdown ($P(\text{Safe}) + P(\text{Malicious}) + P(\text{Phishing}) = 100\%$).
   - Real-time Threat Certainty Gauge meter.
3. **Lexical Metrics Breakdown**:
   - Real-time metric cards for URL Total Length, Dot Count, Hyphen Count, and Digit Ratio.
   - Forensic signal indicators (IPv4 address presence, TLS encryption status, credential masking symbols).
4. **Batch Threat Scanner**:
   - Scan bulk URLs simultaneously with a filterable tabular breakdown and one-click CSV export.
5. **Model Architecture & Theory Tab**:
   - Displays mathematical formulas, confusion matrix heatmaps, and dataset distribution metrics.
6. **Dual Execution Engine**:
   - Communicates seamlessly with the FastAPI backend over HTTP, with zero-interruption automatic fallback to direct local ML inference if the backend server is stopped.

---

## 📊 Model Evaluation Summary

| Class | Class Name | Precision | Recall | F1-Score | Test Samples |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **SAFE** | 1.00 | 1.00 | 1.00 | 150 |
| **1** | **MALICIOUS** | 1.00 | 1.00 | 1.00 | 150 |
| **2** | **PHISHING** | 1.00 | 1.00 | 1.00 | 150 |
| **Overall** | **Accuracy** | **1.00** | **1.00** | **1.00** | **450** |

*Inference Latency:* $\sim 1.0\text{ ms} - 2.5\text{ ms}$ per URL inspection.

---

## 🎓 Capstone Project Presentation Checklist
- [x] Verified dataset generation with balanced multi-class distribution
- [x] Verified 4 core structural lexical features extracted (`Length`, `Dot Count`, `Hyphen Count`, `Digit Ratio`)
- [x] Verified character n-gram TF-IDF vectorizer ($n \in [3, 5]$)
- [x] Verified mathematical feature fusion using `scipy.sparse.hstack`
- [x] Verified Multinomial Logistic Regression model with softmax outputs
- [x] Serialized model and vectorizer saved in `/models`
- [x] FastAPI backend running with `/predict`, `/predict/batch`, and `/health`
- [x] Streamlit dashboard with Plotly charts, verdict banners, and batch scanning
- [x] Detailed documentation with mathematical formulas in `README.md`
