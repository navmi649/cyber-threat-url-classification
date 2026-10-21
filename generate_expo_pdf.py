"""
Script to generate a clean, professional, high-impact ML Project Expo PDF Guide
for 3rd-year B.Tech CSE presentation.
"""

import os
from fpdf import FPDF


class ExpoPDF(FPDF):
    def header(self):
        # Header banner
        self.set_fill_color(26, 54, 93)  # Dark Blue
        self.rect(0, 0, 210, 18, 'F')
        self.set_font('Arial', 'B', 10)
        self.set_text_color(255, 255, 255)
        self.cell(0, 6, 'B.TECH CSE CAPSTONE EXPO | MACHINE LEARNING PROJECT GUIDE', 0, 1, 'C')
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()} | Cyber Threat & Malicious URL Classification System', 0, 0, 'C')

    def chapter_title(self, num, title, color=(26, 54, 93)):
        self.set_font('Arial', 'B', 13)
        self.set_fill_color(237, 242, 247)
        self.set_text_color(*color)
        self.cell(0, 8, f'{num}. {title}', 0, 1, 'L', 1)
        self.ln(2)

    def bullet_point(self, title, desc):
        self.set_font('Arial', 'B', 10)
        self.set_text_color(45, 55, 72)
        self.write(5, f"  * {title}: ")
        self.set_font('Arial', '', 10)
        self.set_text_color(74, 85, 104)
        self.write(5, f"{desc}\n")

    def q_and_a(self, q, a):
        self.set_font('Arial', 'B', 10)
        self.set_text_color(197, 48, 48)  # Red for Question
        self.cell(0, 5, f"Q: {q}", 0, 1, 'L')
        self.set_font('Arial', '', 9.5)
        self.set_text_color(45, 55, 72)
        self.multi_cell(0, 4.5, f"A: {a}")
        self.ln(2)


def create_pdf(filename="ML_Project_Expo_Guide.pdf"):
    pdf = ExpoPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Document Header Title
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 8, 'Real-Time Cyber Threat & URL Classification', 0, 1, 'C')
    pdf.set_font('Arial', 'I', 11)
    pdf.set_text_color(74, 85, 104)
    pdf.cell(0, 6, 'Complete Machine Learning Project Expo & Viva Guide', 0, 1, 'C')
    pdf.set_font('Arial', '', 9)
    pdf.cell(0, 5, 'GitHub Repository: github.com/navmi649/cyber-threat-url-classification', 0, 1, 'C')
    pdf.ln(4)

    # Section 1: The 30-Second Pitch
    pdf.chapter_title('1', 'The 30-Second Teacher Pitch (What to Say First)')
    pdf.set_font('Arial', 'I', 9.5)
    pdf.set_fill_color(240, 249, 255)
    pdf.set_text_color(43, 108, 176)
    pitch_text = (
        '"Good morning, Professors. Our project is a Multi-Class Supervised Machine Learning system '
        'that inspects raw URL strings in real-time to detect cyber threats. Instead of relying on static blacklists, '
        'we engineered a hybrid feature fusion pipeline combining structural lexical attributes with character-level '
        'TF-IDF n-grams, driving a Multinomial Logistic Regression model that outputs precise threat probabilities '
        'in under 2 milliseconds."'
    )
    pdf.multi_cell(0, 5, pitch_text, 1, 'L', 1)
    pdf.ln(4)

    # Section 2: Core Machine Learning Architecture (5 Steps)
    pdf.chapter_title('2', 'The 5 Core Machine Learning Pipeline Steps')
    pdf.bullet_point('ML Task Type', 'Supervised Learning -> Multi-Class Classification (3 classes: Safe, Malicious, Phishing).')
    pdf.bullet_point('Input Vector (X)', 'Raw URL string (e.g., http://paypal-secure-login.com/login.php).')
    pdf.bullet_point('Feature Extraction 1 (Lexical)', '4 structural numbers: URL Length, Dot Count, Hyphen Count, Digit Ratio.')
    pdf.bullet_point('Feature Extraction 2 (TF-IDF)', 'Character-level TF-IDF (3 to 5 n-grams) capturing sub-word attack signatures (.exe, gate, login, verify) invariant to token boundaries.')
    pdf.bullet_point('Feature Fusion (scipy)', 'hstack fuses the dense 4-dim lexical array with the sparse 6,000-dim TF-IDF matrix into a unified 6,004-dim Compressed Sparse Row (CSR) matrix.')
    pdf.bullet_point('Classifier & Activation', 'Multinomial Logistic Regression using Softmax activation: P(Y=k|X) = exp(w_k^T X + b_k) / sum(exp(w_j^T X + b_j)). Outputs strictly sum to 100%.')
    pdf.ln(3)

    # Section 3: The 3 Threat Classes
    pdf.chapter_title('3', 'Threat Categories Explained Simply')
    pdf.bullet_point('Class 0: SAFE (Green)', 'Legitimate domains with standard naming and low punctuation entropy (e.g. google.com, github.com/torvalds/linux).')
    pdf.bullet_point('Class 1: MALICIOUS (Red)', 'Malware delivery, C2 botnet gateways, IP addresses with executable payloads (.exe, .dll, .sh, .vbs).')
    pdf.bullet_point('Class 2: PHISHING (Orange)', 'Brand-jacked credential harvesting portals using hyphen spoofing (e.g. paypal-secure-login.com).')
    pdf.ln(3)

    # Section 4: Live Demonstration Script
    pdf.chapter_title('4', 'Live Demo Walkthrough (Screen Actions)')
    pdf.bullet_point('Test 1 (Safe)', 'Click "Legitimate (Google)" -> Green alert banner (Safe, 99.7% confidence).')
    pdf.bullet_point('Test 2 (Malicious)', 'Click "Malicious (Payload Drop)" -> Red alert banner (Flags IPv4 host, digit ratio, and .exe).')
    pdf.bullet_point('Test 3 (Phishing)', 'Click "Phishing (Fake PayPal)" -> Orange alert banner (Flags brand spoofing and hyphens).')
    pdf.bullet_point('Show Charts', 'Point to the horizontal bar chart showing Softmax probabilities adding up to 100%.')
    pdf.bullet_point('Batch Scanner', 'Show the Batch Threat Scanner tab for enterprise multi-URL scanning and CSV report download.')
    pdf.ln(4)

    # Section 5: Top Examiner Q&A
    pdf.chapter_title('5', 'Top Examiner Questions & Winning Answers')
    pdf.q_and_a(
        'Why use Machine Learning instead of blacklists?',
        'Blacklists are reactive and only block URLs after victims are attacked. Attackers generate disposable domains constantly. ML learns structural patterns to proactively catch zero-day attacks.'
    )
    pdf.q_and_a(
        'Why character-level TF-IDF instead of word-level tokenization?',
        'URLs do not have standard word spaces. Attackers use obfuscations like "paypa1" or "login-verify". Word tokenizers fail with Out-Of-Vocabulary (OOV) errors. Character n-grams (3 to 5 chars) slice text into sub-words that catch attack tokens regardless of spacing.'
    )
    pdf.q_and_a(
        'Why did you use StandardScaler on lexical features before fusion?',
        'URL length is between 30 and 100+, while TF-IDF values are between 0 and 1. Without scaling, unscaled length would overpower the gradient and suppress the character n-grams. StandardScaler standardizes numerical features so all features have balanced impact.'
    )
    pdf.q_and_a(
        'Why Logistic Regression with Softmax instead of Deep Learning (LSTM/BERT)?',
        'In network firewalls and web gateways, latency is critical. Deep Learning takes 50-300ms per sample and needs GPUs. Our convex fusion model runs in under 1.5ms on standard CPU hardware with high precision and low memory footprint.'
    )
    pdf.q_and_a(
        'What was your experimental dataset and evaluation accuracy?',
        'We trained on a balanced dataset of 2,250 URLs (750 Safe, 750 Malicious, 750 Phishing) with an 80/20 stratified split (1,800 train / 450 test), achieving 100% precision, recall, and accuracy on the unseen test set.'
    )

    pdf.output(filename, 'F')
    print(f"Successfully generated PDF: {os.path.abspath(filename)}")


if __name__ == '__main__':
    create_pdf('ML_Project_Expo_Guide.pdf')
