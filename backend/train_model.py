"""
Model Training & Serialization Pipeline
Part of the Real-Time Cyber Threat & Malicious URL Classification System
for 3rd-year B.Tech CSE Capstone Project.

Responsibilities:
1. Generate a balanced, diverse 3-class dataset:
   - Class 0: SAFE (Legitimate web domains, educational, developer portals, cloud services)
   - Class 1: MALICIOUS (Payload links, trojans, C2 botnet gateways, exploit drops)
   - Class 2: PHISHING (Credential harvesting, brand spoofing, bank/payment impersonations)
2. Extract the 4 required lexical features:
   - URL Total Length
   - Dot Frequency Count (.)
   - Hyphen Count (-)
   - Digit Ratio (digits / length)
3. Fit character-level TF-IDF vectorizer (analyzer='char', ngram_range=(3, 5)).
4. Perform mathematical feature fusion via scipy.sparse.hstack.
5. Train a Multi-Class Logistic Regression classifier with softmax probability output.
6. Serialize saved_model.pkl and vectorizer.pkl into /models.
"""

import os
import sys

# Ensure backend directory is in sys.path regardless of execution working directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import json
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utils import extract_lexical_matrix, fuse_features, CLASS_LABELS, LABEL_TO_ID


def generate_curated_dataset() -> pd.DataFrame:
    """
    Constructs an authentic, balanced multi-class URL dataset containing
    short, medium, and long variations across SAFE, MALICIOUS, and PHISHING classes.
    """
    random.seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------
    # 1. SAFE URLs
    # -------------------------------------------------------------
    safe_seeds = [
        "https://www.google.com",
        "https://www.google.com/search?q=machine+learning+cybersecurity",
        "https://github.com/torvalds/linux",
        "https://github.com/scikit-learn/scikit-learn",
        "https://en.wikipedia.org/wiki/Computer_security",
        "https://en.wikipedia.org/wiki/Logistic_regression",
        "https://stackoverflow.com/questions/tagged/python",
        "https://stackoverflow.com/questions/509211/understanding-slice-notation",
        "https://amazon.com/dp/B08N5WRWNW",
        "https://amazon.com/gp/help/customer/display.html",
        "https://microsoft.com/en-us/windows",
        "https://apple.com/iphone-15-pro",
        "https://netflix.com/browse",
        "https://linkedin.com/feed",
        "https://paypal.com/signin",
        "https://chase.com/personal/banking",
        "https://bankofamerica.com/online-banking",
        "https://wellsfargo.com",
        "https://mit.edu/admissions",
        "https://stanford.edu/research",
        "https://nytimes.com/section/technology",
        "https://bbc.com/news/world",
        "https://cloudflare.com/learning/security",
        "https://docker.com/products/docker-desktop",
        "https://docs.python.org/3/library/scipy.html",
        "https://fastapi.tiangolo.com/tutorial/first-steps",
        "https://streamlit.io/gallery",
        "https://coursera.org/learn/machine-learning",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://huggingface.co/models",
        "https://pypi.org/project/scikit-learn",
        "https://developer.mozilla.org/en-US/docs/Web/HTTP",
        "https://ietf.org/rfc/rfc3986.txt",
        "https://aws.amazon.com/console",
        "https://azure.microsoft.com/en-us/pricing"
    ]

    safe_domains = [
        "google.com", "github.com", "wikipedia.org", "stackoverflow.com", "microsoft.com",
        "apple.com", "amazon.com", "nih.gov", "berkeley.edu", "cern.ch", "ieee.org",
        "acm.org", "gitlab.com", "bitbucket.org", "cnn.com", "reuters.com", "forbes.com",
        "salesforce.com", "spotify.com", "adobe.com", "oracle.com", "ibm.com", "intel.com",
        "cisco.com", "digitalocean.com", "aws.amazon.com", "azure.microsoft.com", "un.org",
        "mozilla.org", "w3.org", "python.org", "apache.org", "redhat.com", "ubuntu.com",
        "reddit.com", "youtube.com", "medium.com", "quora.com", "dropbox.com", "zoom.us"
    ]

    safe_paths = [
        "", "/", "/home", "/about-us", "/products", "/services", "/docs/guide", "/api/v1/users",
        "/search?q=cybersecurity", "/explore/trending", "/category/science", "/releases/tag/v2.4",
        "/contact", "/faq", "/pricing", "/support/help-center", "/download/desktop-client",
        "/publications/2024/report", "/library/archive", "/community/forum/topic/102",
        "/blog/engineering-updates", "/security/whitepapers", "/resources/webinars",
        "/user/profile?tab=repositories", "/settings/preferences", "/privacy-policy"
    ]

    safe_urls = list(safe_seeds)
    for dom in safe_domains:
        for path in safe_paths:
            for proto in ["https://", "http://"]:
                sub = random.choice(["", "www.", "docs.", "api.", "blog.", "support."])
                safe_urls.append(f"{proto}{sub}{dom}{path}")

    # -------------------------------------------------------------
    # 2. MALICIOUS URLs (Payload delivery, trojans, C2, reverse shells)
    # -------------------------------------------------------------
    malicious_seeds = [
        "http://185.220.101.45:8080/payload.exe",
        "http://194.26.29.112/c2/beacon.php?id=9942",
        "http://evil-server-botnet.ru/drop/trojan.dll",
        "http://ransomware-distro.su/payment/decrypt_tool.exe",
        "http://91.240.118.232:4444/meterpreter/reverse_tcp",
        "http://free-cracked-software.cc/download/keygen_patch_v9.exe",
        "http://mirai-bot-loader.net/bins/mirai.arm7",
        "http://cryptominer-pool.top/miner/xmrig.sh",
        "http://exploit-kit-landing.xyz/gate.php?affid=8831&sid=trojan",
        "http://bad-actors-payload.biz/powershell_stager.ps1",
        "http://103.251.167.20/exec.sh",
        "http://45.148.10.77/dropper/invoice_update.vbs",
        "http://cobaltstrike-teamserver.org/handler.ashx?guid=fa710294b281",
        "http://dark-shell-backdoor.info/shell.php?cmd=whoami",
        "http://trojan-downloader-cdn.co/update_patch_99214.bat",
        "http://botnet-c2-command.ru/gate.php?aff=99281&exec=true",
        "http://c2-beacon-master.online/beacon.php",
        "http://malware-drop-zone.su/trojan.exe",
        "http://payload-delivery-cdn.cc/rat_client.zip"
    ]

    malicious_ip_hosts = [
        "185.220.101.5", "194.26.29.14", "91.240.118.12", "45.148.10.89",
        "103.251.167.42", "198.51.100.7", "203.0.113.88", "193.106.191.24",
        "178.62.204.101", "89.248.165.74", "185.143.223.11", "94.102.61.18",
        "45.33.32.156", "192.241.212.189", "167.99.168.251", "138.68.244.33",
        "159.89.176.10", "165.227.112.98", "142.93.201.12", "68.183.219.7"
    ]
    malicious_domains = [
        "botnet-c2-command.ru", "malware-drop-zone.su", "trojan-distribution-hub.biz",
        "exploit-broker-pack.xyz", "stealer-panel-gate.top", "payload-delivery-cdn.cc",
        "ransom-locker-service.cn", "cryptosteal-gateway.work", "backdoor-rootkit-feed.org",
        "keygen-crack-patcher.to", "dark-payload-store.net", "credential-dump-site.ws",
        "c2-beacon-master.online", "revshell-listener.link", "apt-campaign-host.club",
        "zero-day-stager.site", "ransom-locker-portal.cc", "cobalt-strike-beacon.info"
    ]
    malicious_payload_files = [
        "/payload.exe", "/trojan.exe", "/injector.dll", "/dropper.bin", "/c2_beacon.php",
        "/gate.php?aff=99281&exec=true", "/powershell_stager.ps1", "/exploit.sh",
        "/invoice_scanned_doc.vbs", "/update_hotfix_2024.bat", "/shell.php?c=id",
        "/rootkit/installer.sh", "/rat_client.zip", "/miner/xmrig_daemon.elf",
        "/handler.ashx?session=e89bf234ca", "/bins/mirai.mips", "/gate/rx_beacon",
        "/loader.vbs", "/patch_installer.exe", "/reverse_shell.elf"
    ]

    malicious_urls = list(malicious_seeds)
    for ip in malicious_ip_hosts:
        for payload in malicious_payload_files:
            port = random.choice(["", ":8080", ":4444", ":8888", ":1337", ":9001"])
            malicious_urls.append(f"http://{ip}{port}{payload}")

    for dom in malicious_domains:
        for payload in malicious_payload_files:
            hex_token = f"{random.randint(100000, 999999):x}"
            param = f"?token={hex_token}&sid={random.randint(1000, 9999)}" if "?" not in payload else f"&node={hex_token}"
            malicious_urls.append(f"http://{dom}{payload}{param}")

    # -------------------------------------------------------------
    # 3. PHISHING URLs (Brand impersonation, credential theft, spoofed login)
    # -------------------------------------------------------------
    phishing_seeds = [
        "http://paypal-security-account-verification.com/login",
        "http://paypal-secure-login.com/login.php",
        "http://paypal-login-secure.com/auth",
        "http://paypal-account-update.net/verify",
        "http://paypal.com.customer-support-alert.net/signin.php",
        "http://netflix-billing-update-center.org/account/verify",
        "http://netflix-verify-payment.com/signin",
        "http://apple-id-credential-reset.info/manage/auth",
        "http://appleid-verify-device.com/login",
        "http://wellsfargo-secure-banking-portal.online/signon",
        "http://wellsfargo-verify-customer.com/portal",
        "http://chase-bank-verify-identity-login.com/auth/step2",
        "http://chase-online-secure-login.com",
        "http://bankofamerica-customer-security.tech/login.asp",
        "http://bankofamerica-secure-login.net",
        "http://bankofamerica-login.com",
        "http://microsoft-online-365-login-verify.com/oauth2/authorize",
        "http://amazon-account-suspension-appeals.com/signin",
        "http://google-drive-shared-security-document.top/view",
        "http://facebook-security-checkpoint-resolution.com/checkpoint",
        "http://instagram-copyright-infringement-appeal.net/verify",
        "http://dhl-express-parcel-tracking-confirm.xyz/redirection",
        "http://usps-post-redelivery-scheduled.info/confirm_address",
        "http://irs-tax-refund-portal-claim.online/direct-deposit",
        "http://coinbase-account-security-alert.net/signin",
        "http://binance-verification-kyc-desk.org/auth"
    ]

    phished_brands = [
        "paypal", "chase", "bankofamerica", "wellsfargo", "netflix",
        "appleid", "apple", "microsoft365", "microsoft", "google-drive",
        "amazon", "citibank", "coinbase", "binance", "facebook",
        "instagram", "usps", "dhl", "metamask", "dropbox"
    ]

    phishing_hyphens = [
        "-login", "-secure", "-signin", "-verify", "-verification",
        "-account", "-update", "-billing", "-support", "-security",
        "-portal", "-checkpoint", "-alert", "-recovery", "-auth",
        "-secure-login", "-account-verification", "-customer-security",
        "-online-banking", "-verify-identity", "-billing-validation"
    ]

    phishing_tlds = [".com", ".net", ".org", ".info", ".xyz", ".online", ".top", ".site", ".cc", ".biz"]
    phishing_paths = [
        "", "/", "/login", "/login.php", "/signin", "/signin.asp",
        "/verify", "/auth/verify.php", "/account/update", "/security/challenge",
        "/webapps/login", "/confirm_credentials.html", "/portal/auth"
    ]

    phishing_urls = list(phishing_seeds)
    for brand in phished_brands:
        for hyp in phishing_hyphens:
            for tld in phishing_tlds[:5]:
                p = random.choice(phishing_paths)
                proto = random.choice(["http://", "https://"])
                # Direct brand-action pattern
                phishing_urls.append(f"{proto}{brand}{hyp}{tld}{p}")
                # Subdomain spoofing pattern
                sub = random.choice(["", "login.", "secure.", "account.", "auth."])
                phishing_urls.append(f"{proto}{sub}{brand}{hyp}{tld}{p}")

    # Deduplicate within sets
    safe_urls = list(dict.fromkeys(safe_urls))
    malicious_urls = list(dict.fromkeys(malicious_urls))
    phishing_urls = list(dict.fromkeys(phishing_urls))

    print(f"Generated Raw Pool: {len(safe_urls)} SAFE, {len(malicious_urls)} MALICIOUS, {len(phishing_urls)} PHISHING")

    # Balanced target count per class (750 samples each = 2250 total)
    target_count = 750
    random.shuffle(safe_urls)
    random.shuffle(malicious_urls)
    random.shuffle(phishing_urls)

    safe_sample = safe_urls[:target_count]
    malicious_sample = malicious_urls[:target_count]
    phishing_sample = phishing_urls[:target_count]

    data = []
    for u in safe_sample:
        data.append({"url": u, "label": "SAFE", "label_id": 0})
    for u in malicious_sample:
        data.append({"url": u, "label": "MALICIOUS", "label_id": 1})
    for u in phishing_sample:
        data.append({"url": u, "label": "PHISHING", "label_id": 2})

    df = pd.DataFrame(data)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def train_and_evaluate(models_dir: str = "../models"):
    """
    Executes the complete pipeline:
    1. Dataset assembly & verification
    2. Lexical feature matrix extraction
    3. Character-level TF-IDF feature extraction (ngram_range=(3, 5))
    4. Mathematical Feature Fusion via scipy.sparse.hstack
    5. Multinomial Logistic Regression model training with softmax activation
    6. Evaluation & Model persistence
    """
    print("=" * 70)
    print(">>> Real-Time Cyber Threat & Malicious URL Classification System <<<")
    print(">>> Capstone ML Training & Feature Fusion Pipeline Starting... <<<")
    print("=" * 70)

    # Resolve absolute path for models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_models_dir = os.path.normpath(os.path.join(current_dir, models_dir))
    os.makedirs(resolved_models_dir, exist_ok=True)

    # Step 1: Dataset Generation
    print("\n[Step 1/5] Generating balanced multi-class URL dataset...")
    df = generate_curated_dataset()
    print(f"Total dataset size: {len(df)} samples")
    print("Class Distribution:")
    for label, count in df["label"].value_counts().items():
        print(f"  - {label} (Class {LABEL_TO_ID[label]}): {count} samples")

    # Step 2: Split Train / Test (Stratified)
    print("\n[Step 2/5] Performing stratified 80/20 train-test split...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["url"],
        df["label_id"],
        test_size=0.20,
        random_state=42,
        stratify=df["label_id"]
    )
    print(f"Training samples: {len(X_train_raw)} | Testing samples: {len(X_test_raw)}")

    # Step 3: Feature Engineering & Extraction
    print("\n[Step 3/5] Extracting Lexical & Character-Level TF-IDF Features...")
    
    # 3a. Lexical Feature Extraction (4 core structural attributes)
    print("  -> Extracting lexical vectors [URL Length, Dot Count, Hyphen Count, Digit Ratio]...")
    X_train_lexical = extract_lexical_matrix(X_train_raw.tolist())
    X_test_lexical = extract_lexical_matrix(X_test_raw.tolist())
    print(f"     Lexical train matrix shape: {X_train_lexical.shape}")

    # Standardize lexical features so numerical scale aligns with TF-IDF [0, 1] range
    scaler = StandardScaler()
    X_train_lexical_scaled = scaler.fit_transform(X_train_lexical)
    X_test_lexical_scaled = scaler.transform(X_test_lexical)

    # 3b. Character-Level TF-IDF Vectorizer (ngram_range=(3, 5))
    print("  -> Fitting Character-Level TF-IDF Vectorizer (analyzer='char', ngram_range=(3, 5))...")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=6000,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    print(f"     TF-IDF train sparse matrix shape: {X_train_tfidf.shape}")

    # Attach scaler to vectorizer for unified pipeline serialization
    vectorizer.scaler_ = scaler

    # 3c. Mathematical Feature Fusion via scipy.sparse.hstack
    print("  -> Performing horizontal feature fusion (hstack dense lexical + sparse TF-IDF)...")
    X_train_fused = fuse_features(X_train_lexical_scaled, X_train_tfidf)
    X_test_fused = fuse_features(X_test_lexical_scaled, X_test_tfidf)
    print(f"     Fused feature matrix shape: {X_train_fused.shape}")

    # Step 4: Multi-Class Logistic Regression Training
    print("\n[Step 4/5] Training Multi-Class Logistic Regression (Softmax multinomial)...")
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=2.5,
        random_state=42
    )
    model.fit(X_train_fused, y_train)
    print("  -> Model training completed successfully.")

    # Step 5: Model Evaluation
    print("\n[Step 5/5] Evaluating Classifier Performance on Test Set...")
    y_pred = model.predict(X_test_fused)

    accuracy = accuracy_score(y_test, y_pred)
    target_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix:")
    print(np.array(conf_matrix))

    # Save artifacts into /models
    model_path = os.path.join(resolved_models_dir, "saved_model.pkl")
    vectorizer_path = os.path.join(resolved_models_dir, "vectorizer.pkl")
    scaler_path = os.path.join(resolved_models_dir, "scaler.pkl")
    metrics_path = os.path.join(resolved_models_dir, "model_metrics.json")

    print(f"\nSerializing artifacts to: {resolved_models_dir}")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(scaler, scaler_path)
    print(f"  -> Model saved: {model_path}")
    print(f"  -> Vectorizer saved: {vectorizer_path}")
    print(f"  -> Scaler saved: {scaler_path}")

    # Save training metrics metadata for frontend consumption
    metrics_data = {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "class_labels": CLASS_LABELS,
        "total_samples": len(df),
        "train_samples": len(X_train_raw),
        "test_samples": len(X_test_raw),
        "tfidf_features": X_train_tfidf.shape[1],
        "lexical_features": X_train_lexical.shape[1],
        "total_fused_features": X_train_fused.shape[1]
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  -> Metrics saved: {metrics_path}")

    print("\n" + "=" * 70)
    print("Pipeline Execution Completed Successfully!")
    print("=" * 70)
    return model, vectorizer, metrics_data


if __name__ == "__main__":
    train_and_evaluate(models_dir="../models")
