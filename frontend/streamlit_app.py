"""
Cyber Threat & Malicious URL Classification System - Interactive Analytics Dashboard
3rd-Year B.Tech CSE Capstone Project
"""

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure path to allow importing backend utils in standalone fallback mode
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "backend"))
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models"))
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

try:
    from utils import (
        extract_lexical_features,
        extract_lexical_vector,
        extract_lexical_matrix,
        fuse_features,
        get_risk_assessment,
        CLASS_LABELS,
        LABEL_TO_ID
    )
    import joblib
    STANDALONE_AVAILABLE = True
except Exception:
    STANDALONE_AVAILABLE = False


# Page Configuration
st.set_page_config(
    page_title="Cyber Threat & Malicious URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1f77b4, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #00d2ff;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #aaaaaa;
    }
    .verdict-banner-safe {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.2), rgba(40, 167, 69, 0.05));
        border-left: 6px solid #28a745;
        border-radius: 8px;
        padding: 18px 24px;
        margin: 20px 0;
    }
    .verdict-banner-malicious {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.25), rgba(220, 53, 69, 0.05));
        border-left: 6px solid #dc3545;
        border-radius: 8px;
        padding: 18px 24px;
        margin: 20px 0;
    }
    .verdict-banner-phishing {
        background: linear-gradient(135deg, rgba(253, 126, 20, 0.25), rgba(253, 126, 20, 0.05));
        border-left: 6px solid #fd7e14;
        border-radius: 8px;
        padding: 18px 24px;
        margin: 20px 0;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions & Backend Connectivity ---

FASTAPI_URL = "http://127.0.0.1:8000"


@st.cache_resource
def load_local_models():
    """Fallback local model loader for standalone execution."""
    try:
        model_p = os.path.join(MODELS_DIR, "saved_model.pkl")
        vect_p = os.path.join(MODELS_DIR, "vectorizer.pkl")
        scaler_p = os.path.join(MODELS_DIR, "scaler.pkl")

        if os.path.exists(model_p) and os.path.exists(vect_p):
            m = joblib.load(model_p)
            v = joblib.load(vect_p)
            s = joblib.load(scaler_p) if os.path.exists(scaler_p) else getattr(v, "scaler_", None)
            return m, v, s
    except Exception as e:
        st.warning(f"Local model loading error: {e}")
    return None, None, None


def check_fastapi_health() -> bool:
    """Checks if FastAPI backend is online."""
    try:
        r = requests.get(f"{FASTAPI_URL}/health", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def predict_via_api(url: str):
    """Sends prediction request to FastAPI server."""
    try:
        r = requests.post(
            f"{FASTAPI_URL}/predict",
            json={"url": url},
            timeout=3.0
        )
        if r.status_code == 200:
            return r.json(), None
        return None, f"API returned status {r.status_code}: {r.text}"
    except Exception as e:
        return None, str(e)


def predict_locally(url: str, model, vectorizer, scaler):
    """Fallback in-process prediction if backend API is not running."""
    start_time = time.perf_counter()
    lex_dict = extract_lexical_features(url)
    lex_vec = extract_lexical_vector(url)

    if scaler is not None:
        lex_scaled = scaler.transform(lex_vec)
    else:
        lex_scaled = lex_vec

    tfidf_vec = vectorizer.transform([url])
    fused = fuse_features(lex_scaled, tfidf_vec)

    pred_id = int(model.predict(fused)[0])
    pred_label = CLASS_LABELS.get(pred_id, "UNKNOWN")
    probs = model.predict_proba(fused)[0]

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


# --- UI Header & Sidebar ---

st.markdown('<div class="main-title">🛡️ Cyber Threat & Malicious URL Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">B.Tech CSE Capstone Project • Real-Time Feature Fusion (Lexical + Character-Level TF-IDF) & Multinomial Softmax Logistic Regression</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.subheader("System Control & Telemetry")
    
    api_online = check_fastapi_health()
    if api_online:
        st.success("🟢 FastAPI Backend: ONLINE (Port 8000)")
        execution_mode = "FastAPI Backend (REST API)"
    else:
        st.info("🟡 FastAPI Backend: OFFLINE (Using Direct ML Engine)")
        execution_mode = "Direct ML Engine (In-Process)"

    st.caption(f"**Execution Engine:** {execution_mode}")

    st.markdown("---")
    st.subheader("Class Specifications")
    st.markdown("""
    - <span style="color:#28a745; font-weight:bold;">SAFE (Class 0):</span> Legitimate domains & portals
    - <span style="color:#dc3545; font-weight:bold;">MALICIOUS (Class 1):</span> Payloads, Trojans, C2 channels
    - <span style="color:#fd7e14; font-weight:bold;">PHISHING (Class 2):</span> Spoofed logins & bank portals
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Capstone Project • Department of Computer Science & Engineering")


# Main Dashboard Tabs
tab_inspector, tab_batch, tab_architecture = st.tabs([
    "🔍 Real-Time URL Inspector",
    "📊 Batch Threat Scanner",
    "🧠 Model Architecture & Metrics"
])


# =========================================================================
# TAB 1: Real-Time URL Inspector
# =========================================================================
with tab_inspector:
    st.markdown("#### Analyze a Target URL for Cyber Threats")

    # Sample Presets for Quick Testing
    st.markdown("**Quick Preset Testing:**")
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_url = ""
    with col_p1:
        if st.button("🛡️ Legitimate (Google)", use_container_width=True):
            preset_url = "https://www.google.com"
        if st.button("🛡️ Legitimate (GitHub Linux)", use_container_width=True):
            preset_url = "https://github.com/torvalds/linux"
    with col_p2:
        if st.button("🚨 Malicious (Payload Drop)", use_container_width=True):
            preset_url = "http://185.220.101.45:8080/payload.exe"
        if st.button("🚨 Malicious (C2 Botnet Gate)", use_container_width=True):
            preset_url = "http://botnet-c2-command.ru/c2_beacon.php"
    with col_p3:
        if st.button("⚠️ Phishing (Fake PayPal)", use_container_width=True):
            preset_url = "http://paypal-secure-login.com/login.php"
        if st.button("⚠️ Phishing (Netflix Update)", use_container_width=True):
            preset_url = "http://netflix-billing-update-center.org/account/verify"

    # URL Input Field
    url_input = st.text_input(
        "Enter Target URL string to inspect:",
        value=preset_url if preset_url else "http://paypal-secure-login.com/login.php",
        placeholder="e.g. https://example.com or http://suspicious-site.biz/payload.exe",
        help="Paste any complete or partial URL string to run full lexical and character n-gram feature extraction."
    )

    btn_analyze = st.button("⚡ Inspect URL Threat Level", type="primary", use_container_width=True)

    if btn_analyze or preset_url or url_input:
        if not url_input.strip():
            st.warning("Please provide a valid URL string.")
        else:
            result = None
            # Attempt via FastAPI if online, else fallback
            if api_online:
                api_res, err = predict_via_api(url_input.strip())
                if api_res:
                    result = api_res
                else:
                    st.warning(f"FastAPI request failed ({err}), falling back to direct engine.")
            
            if result is None and STANDALONE_AVAILABLE:
                local_model, local_vec, local_scaler = load_local_models()
                if local_model and local_vec:
                    result = predict_locally(url_input.strip(), local_model, local_vec, local_scaler)
                else:
                    st.error("Model artifacts not loaded. Please ensure train_model.py has been executed.")

            if result:
                prediction = result["prediction"]
                confidence = result["confidence"]
                probs = result["probabilities"]
                lex = result["lexical_features"]
                risk = result["risk_assessment"]
                ms = result["inference_time_ms"]

                # 1. Categorical Verdict Banner
                if prediction == "SAFE":
                    banner_class = "verdict-banner-safe"
                    badge_style = "background-color: #28a745; color: white;"
                    icon = "✅"
                    title = "VERDICT: SAFE (Legitimate Web Resource)"
                elif prediction == "MALICIOUS":
                    banner_class = "verdict-banner-malicious"
                    badge_style = "background-color: #dc3545; color: white;"
                    icon = "🚨"
                    title = "VERDICT: MALICIOUS (Cyber Threat / Malware Infrastructure)"
                else:  # PHISHING
                    banner_class = "verdict-banner-phishing"
                    badge_style = "background-color: #fd7e14; color: white;"
                    icon = "⚠️"
                    title = "VERDICT: PHISHING (Credential Harvesting / Social Engineering)"

                st.markdown(f"""
                <div class="{banner_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h3 style="margin: 0; font-size: 1.4rem;">{icon} {title}</h3>
                            <p style="margin: 6px 0 0 0; font-size: 0.95rem; opacity: 0.9;">
                                {risk['description']}
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <span class="badge-pill" style="{badge_style}">
                                Confidence: {confidence * 100:.2f}%
                            </span>
                            <div style="font-size: 0.8rem; color: #888; margin-top: 4px;">
                                Inference: {ms} ms
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 2. Core Lexical Metrics Section (4 Required Features)
                st.markdown("#### 📐 Structural Lexical Feature Extraction")
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)

                with mcol1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">URL Total Length</div>
                        <div class="metric-val">{lex['url_length']}</div>
                        <div style="font-size: 0.75rem; color: #888;">Total character count</div>
                    </div>
                    """, unsafe_allow_html=True)

                with mcol2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Dot Count (.)</div>
                        <div class="metric-val">{lex['dot_count']}</div>
                        <div style="font-size: 0.75rem; color: #888;">Subdomain delimiter frequency</div>
                    </div>
                    """, unsafe_allow_html=True)

                with mcol3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Hyphen Count (-)</div>
                        <div class="metric-val">{lex['hyphen_count']}</div>
                        <div style="font-size: 0.75rem; color: #888;">Brand-jacking punctuation</div>
                    </div>
                    """, unsafe_allow_html=True)

                with mcol4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Digit Ratio</div>
                        <div class="metric-val">{lex['digit_ratio'] * 100:.1f}%</div>
                        <div style="font-size: 0.75rem; color: #888;">Numeric entropy ratio</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # 3. Dynamic Visualizations (Plotly Probability Distribution)
                chart_col1, chart_col2 = st.columns([3, 2])

                with chart_col1:
                    st.markdown("#### 📊 Softmax Probability Distribution (Sums to 100%)")
                    classes = ["SAFE", "MALICIOUS", "PHISHING"]
                    prob_values = [probs.get(c, 0.0) * 100 for c in classes]
                    colors = ["#28a745", "#dc3545", "#fd7e14"]

                    fig_bar = go.Figure(go.Bar(
                        x=prob_values,
                        y=classes,
                        orientation='h',
                        text=[f"{v:.2f}%" for v in prob_values],
                        textposition='auto',
                        marker=dict(
                            color=colors,
                            line=dict(color='rgba(255, 255, 255, 0.3)', width=1)
                        )
                    ))
                    fig_bar.update_layout(
                        xaxis=dict(title="Probability (%)", range=[0, 105]),
                        yaxis=dict(autorange="reversed"),
                        height=260,
                        margin=dict(l=20, r=20, t=10, b=30),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="#dddddd")
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with chart_col2:
                    st.markdown("#### 🎯 Threat Confidence Gauge")
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"Target: {prediction}", 'font': {'size': 16, 'color': '#ffffff'}},
                        number={'suffix': "%", 'font': {'color': '#ffffff'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#888888"},
                            'bar': {'color': risk['color']},
                            'bgcolor': "rgba(255,255,255,0.05)",
                            'borderwidth': 1,
                            'bordercolor': "rgba(255,255,255,0.2)",
                            'steps': [
                                {'range': [0, 50], 'color': 'rgba(100,100,100,0.15)'},
                                {'range': [50, 80], 'color': 'rgba(100,100,100,0.3)'},
                                {'range': [80, 100], 'color': 'rgba(100,100,100,0.5)'}
                            ],
                        }
                    ))
                    fig_gauge.update_layout(
                        height=260,
                        margin=dict(l=20, r=20, t=25, b=25),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="#dddddd")
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # 4. Forensic Signals & Risk Anomalies
                with st.expander("🔎 Detailed Forensic Signals & Lexical Diagnostics", expanded=True):
                    fcol1, fcol2 = st.columns(2)
                    with fcol1:
                        st.markdown("**Flagged Structural Indicators:**")
                        for a in risk["anomalies"]:
                            st.markdown(f"- ⚠️ {a}")
                    with fcol2:
                        st.markdown("**Secondary Telemetry:**")
                        st.markdown(f"- **IPv4 Host Present:** {'Yes (High Risk)' if lex.get('has_ip') else 'No (Domain Registered)'}")
                        st.markdown(f"- **HTTPS Encryption:** {'Yes (TLS Secured)' if lex.get('has_https') else 'No (Cleartext HTTP)'}")
                        st.markdown(f"- **Path Slashes (/):** {lex.get('slash_count', 0)}")
                        st.markdown(f"- **'@' Symbol Present:** {'Yes (Suspicious Credential Mask)' if lex.get('at_count', 0) > 0 else 'No'}")


# =========================================================================
# TAB 2: Batch Threat Scanner
# =========================================================================
with tab_batch:
    st.markdown("#### Batch Cyber Threat URL Inspection")
    st.caption("Upload or paste a list of URLs to evaluate cyber threat postures in bulk.")

    sample_batch_text = (
        "https://www.google.com\n"
        "https://github.com/torvalds/linux\n"
        "http://185.220.101.45:8080/payload.exe\n"
        "http://botnet-c2-command.ru/c2_beacon.php\n"
        "http://paypal-secure-login.com/login.php\n"
        "http://netflix-billing-update-center.org/account/verify\n"
        "http://wellsfargo-secure-banking-portal.online/signon\n"
        "https://stackoverflow.com/questions/tagged/python"
    )

    batch_input = st.text_area("Input URLs (one per line):", value=sample_batch_text, height=180)
    btn_run_batch = st.button("🚀 Execute Batch Threat Scan", type="primary")

    if btn_run_batch and batch_input.strip():
        urls = [line.strip() for line in batch_input.strip().split("\n") if line.strip()]
        st.info(f"Scanning {len(urls)} URLs...")

        results_list = []
        local_model, local_vec, local_scaler = (None, None, None)
        if not api_online and STANDALONE_AVAILABLE:
            local_model, local_vec, local_scaler = load_local_models()

        for u in urls:
            res = None
            if api_online:
                api_res, _ = predict_via_api(u)
                if api_res:
                    res = api_res
            if res is None and local_model and local_vec:
                res = predict_locally(u, local_model, local_vec, local_scaler)

            if res:
                lex = res["lexical_features"]
                results_list.append({
                    "URL": u,
                    "Verdict": res["prediction"],
                    "Confidence": f"{res['confidence'] * 100:.1f}%",
                    "P(Safe)": f"{res['probabilities'].get('SAFE', 0)*100:.1f}%",
                    "P(Malicious)": f"{res['probabilities'].get('MALICIOUS', 0)*100:.1f}%",
                    "P(Phishing)": f"{res['probabilities'].get('PHISHING', 0)*100:.1f}%",
                    "Length": lex["url_length"],
                    "Dots": lex["dot_count"],
                    "Hyphens": lex["hyphen_count"],
                    "Digit Ratio": f"{lex['digit_ratio']*100:.1f}%",
                    "Inference (ms)": res["inference_time_ms"]
                })

        if results_list:
            res_df = pd.DataFrame(results_list)

            # Summary metrics
            sc1, sc2, sc3, sc4 = st.columns(4)
            safe_c = sum(res_df["Verdict"] == "SAFE")
            mal_c = sum(res_df["Verdict"] == "MALICIOUS")
            phish_c = sum(res_df["Verdict"] == "PHISHING")

            sc1.metric("Total Scanned", len(res_df))
            sc2.metric("Safe URLs", safe_c, delta=f"{safe_c/len(res_df)*100:.0f}%")
            sc3.metric("Malicious URLs", mal_c, delta=f"{mal_c/len(res_df)*100:.0f}%", delta_color="inverse")
            sc4.metric("Phishing URLs", phish_c, delta=f"{phish_c/len(res_df)*100:.0f}%", delta_color="inverse")

            # Color highlight for table
            def highlight_verdict(val):
                if val == "SAFE":
                    return "background-color: rgba(40, 167, 69, 0.25); color: #85e89d; font-weight: bold;"
                elif val == "MALICIOUS":
                    return "background-color: rgba(220, 53, 69, 0.25); color: #f97583; font-weight: bold;"
                elif val == "PHISHING":
                    return "background-color: rgba(253, 126, 20, 0.25); color: #ffab70; font-weight: bold;"
                return ""

            st.dataframe(
                res_df.style.map(highlight_verdict, subset=["Verdict"]),
                use_container_width=True
            )

            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Threat Assessment Report (CSV)",
                csv,
                "threat_assessment_report.csv",
                "text/csv"
            )


# =========================================================================
# TAB 3: Capstone Architecture & ML Insights
# =========================================================================
with tab_architecture:
    st.markdown("#### 📐 Engineering Architecture & Mathematical Formulation")
    
    st.markdown("""
    This project introduces a **Hybrid Feature Fusion Pipeline** combining dense structural lexical analysis
    with sparse character-level TF-IDF representations, classified through a multinomial Logistic Regression model.
    """)

    col_arch1, col_arch2 = st.columns(2)

    with col_arch1:
        st.markdown("##### 1. Mathematical Feature Fusion Formulation")
        st.latex(r"X_{\text{lexical}} = [\text{Length}, \text{DotCount}, \text{HyphenCount}, \text{DigitRatio}] \in \mathbb{R}^{4}")
        st.latex(r"X_{\text{tfidf}} = \text{TF-IDF}_{\text{char}}(s, n \in [3, 5]) \in \mathbb{R}^{D}")
        st.latex(r"X_{\text{fused}} = \left[ X_{\text{lexical\_scaled}} \;\mathbin{\Vert}\; X_{\text{tfidf}} \right] = \text{hstack}(X_{\text{lex}}, X_{\text{tfidf}})")
        
        st.markdown("##### 2. Multi-Class Softmax Formulation")
        st.latex(r"P(Y = k \mid X_{\text{fused}}) = \frac{e^{\mathbf{w}_k^T X_{\text{fused}} + b_k}}{\sum_{j=0}^{2} e^{\mathbf{w}_j^T X_{\text{fused}} + b_j}}")
        st.caption(r"Guarantees an exact probability vector where $\sum_{k=0}^2 P(Y=k) = 1.0$ (100%).")

    with col_arch2:
        st.markdown("##### 3. Training & Validation Metrics")
        metrics_file = os.path.join(MODELS_DIR, "model_metrics.json")
        if os.path.exists(metrics_file):
            import json
            with open(metrics_file, "r") as mf:
                m_data = json.load(mf)
            st.metric("Test Accuracy", f"{m_data.get('accuracy', 1.0)*100:.2f}%")
            st.markdown(f"- **Total Training Samples:** {m_data.get('train_samples')} URLs")
            st.markdown(f"- **Test Samples:** {m_data.get('test_samples')} URLs")
            st.markdown(f"- **Lexical Features:** {m_data.get('lexical_features')} dimensions")
            st.markdown(f"- **TF-IDF Character N-Grams:** {m_data.get('tfidf_features')} features")
            st.markdown(f"- **Total Fused Feature Dimension:** {m_data.get('total_fused_features')} features")

            # Confusion Matrix heatmap
            cm = np.array(m_data.get("confusion_matrix", [[150, 0, 0], [0, 150, 0], [0, 0, 150]]))
            fig_cm = px.imshow(
                cm,
                x=["SAFE", "MALICIOUS", "PHISHING"],
                y=["SAFE", "MALICIOUS", "PHISHING"],
                labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Sample Count"),
                text_auto=True,
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Execute `python backend/train_model.py` to view serialized metrics.")

    st.markdown("---")
    st.markdown("##### 🛡️ Cyber Defense Rationale for Feature Selection")
    st.markdown("""
    - **URL Total Length:** Malicious URLs often embed base64 shellcode, trojan arguments, or deep subdomains to bypass perimeter firewalls.
    - **Dot Frequency Count:** Cyber adversaries abuse multi-level subdomains (`paypal.com.customer-update.account-restore.info`) to mimic legitimate domains.
    - **Hyphen Count:** Phishing portals heavily employ brand-jacking hyphen patterns (`chase-secure-banking-portal.online`) to evade strict domain filters.
    - **Digit Ratio:** Algorithmic Domain Generation Algorithms (DGA), botnet C2 nodes, and hex-encoded payload delivery URLs exhibit abnormally high numeric ratios.
    - **Character 3-to-5 N-grams:** Captures localized structural signatures (`gate.php`, `.exe`, `login`, `verify`, `payload`, `c2_beacon`) invariant to word tokenization boundaries.
    """)
