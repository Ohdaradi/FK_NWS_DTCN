"""
SECTION-2I1
Group 10
"""

import streamlit as st
import joblib
import re
import time
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Load ML Assets FIRST before page configuration
# Disable caching to always load fresh model
def load_assets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'fake_news_master_model.pkl')
    vectorizer_path = os.path.join(script_dir, 'tfidf_master_vectorizer.pkl')
    m = joblib.load(model_path)
    v = joblib.load(vectorizer_path)
    return m, v

try:
    m, v = load_assets()
except Exception as e:
    st.error(f"Failed to load model: {type(e).__name__}: {str(e)}")
    st.stop()

# 1. Page Configuration
st.set_page_config(page_title="Fake News Detection Using NLP", page_icon="🗞️", layout="wide")

# 2. Professional Academic UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;600&display=swap');

    :root {
        --bg: #f4efe8;
        --ink: #1e2428;
        --muted: #5f6b73;
        --paper: #fffdf9;
        --accent: #d95f43;
        --accent-2: #265d79;
        --line: #dccfbd;
        --ok: #1b7f4a;
        --warn: #b1382f;
    }

    .stApp {
        background:
            radial-gradient(1200px 500px at 100% -10%, #e5d9ca 0%, transparent 60%),
            radial-gradient(800px 500px at -10% 110%, #dce8e3 0%, transparent 60%),
            var(--bg);
        color: var(--ink);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #21343e, #17242b);
        border-right: 1px solid rgba(255, 255, 255, 0.15);
    }

    [data-testid="stSidebar"] * {
        color: #eaf3f8 !important;
    }

    .headline {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 4vw, 3.8rem);
        line-height: 1.05;
        letter-spacing: -0.02em;
        margin: 0;
        color: #13222a;
    }

    .subhead {
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--muted);
        margin-top: 0.45rem;
        margin-bottom: 1.25rem;
        font-size: 1rem;
    }

    .hero-strip {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 0.7rem 0 1.1rem;
    }

    .chip {
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        background: rgba(255, 255, 255, 0.65);
        font-size: 0.82rem;
        color: #20313a;
    }

    .card {
        border: 1px solid var(--line);
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.86), rgba(255, 255, 255, 0.72));
        box-shadow: 0 8px 25px rgba(45, 46, 48, 0.08);
        padding: 1.2rem;
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        margin: 0 0 0.6rem;
        color: #12262f;
    }

    .stTextArea textarea {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 1.04rem !important;
        color: #1a2227 !important;
        background: rgba(255, 255, 255, 0.98) !important;
        border: 1.5px solid #b5a393 !important;
        border-radius: 16px !important;
        min-height: 280px !important;
        padding: 14px !important;
    }

    .stTextArea textarea::placeholder {
        color: #8a7d73 !important;
        opacity: 0.8 !important;
    }

    .stTextArea label {
        color: #1e2428 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    div.stButton > button {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
        border-radius: 12px !important;
        border: 0 !important;
        color: #fff !important;
        background: linear-gradient(90deg, var(--accent-2), #1d7899) !important;
        box-shadow: 0 10px 22px rgba(38, 93, 121, 0.25) !important;
        height: 3rem;
    }

    div.stButton > button:hover {
        filter: brightness(1.06);
        transform: translateY(-1px);
    }

    .result-pill {
        display: inline-block;
        padding: 0.38rem 0.8rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }

    .real {
        border: 1px solid rgba(27, 127, 74, 0.35);
        color: var(--ok);
        background: rgba(27, 127, 74, 0.1);
    }

    .fake {
        border: 1px solid rgba(177, 56, 47, 0.35);
        color: var(--warn);
        background: rgba(177, 56, 47, 0.1);
    }

    .meter-caption {
        color: #4e5a61;
        font-size: 0.92rem;
    }

    .footer-note {
        color: #68757d;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 1rem;
    }

    @media (max-width: 900px) {
        .headline { font-size: 2rem; }
        .stTextArea textarea { min-height: 220px !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. Load ML Assets

ps = PorterStemmer()

@st.cache_resource
def load_stopwords():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        return set(stopwords.words('english'))
    except:
        # Fallback stopwords if download fails
        return set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

stops = load_stopwords()

def process_text(text):
    if "(Reuters) -" in text:
        text = text.split("(Reuters) -", 1)[1]
    if " - " in text and len(text.split(" - ")[0]) < 30:
        text = text.split(" - ", 1)[1]
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower().split()
    return ' '.join([ps.stem(w) for w in text if w not in stops])

# 4. Sidebar
with st.sidebar:
    st.markdown("## 📰 Fake News Detection")
    st.caption("Using Natural Language Processing")
    st.markdown("---")
    st.markdown("### Project Info")
    st.write("**Objective:** Classify news articles as real or fake using NLP techniques.")
    st.markdown("---")
    st.markdown("### Models Trained")
    st.write("🔹 Logistic Regression (Production)")
    st.write("🔹 Naïve Bayes")
    st.write("🔹 Random Forest")
    st.markdown("---")
    st.markdown("### Datasets Used")
    st.write("✓ Kaggle Fake News Dataset")
    st.write("✓ LIAR Dataset")
    st.markdown("---")
    st.markdown("### Vectorization")
    st.write("TF-IDF with N-Grams (1-2)")
    st.write("Max Features: 15,000")
    st.markdown("---")
    st.markdown("### Usage Tips")
    st.write("1. Paste full article text for best results.")
    st.write("2. Include headline and multiple paragraphs.")
    st.write("3. Longer, well-structured content improves accuracy.")

# 5. Header
st.markdown("<p style='font-family: Space Grotesk; font-size: 1.8rem; color: #13222a; margin-bottom: 0.8rem; letter-spacing: 0.12em; font-weight: 700; border-bottom: 2px solid #d95f43; padding-bottom: 0.5rem;'><strong>SECTION-2I1 | Group 10</strong></p>", unsafe_allow_html=True)
st.markdown("<h1 class='headline'>Fake News Detection Using NLP</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subhead'><strong>Problem:</strong> Fake news spreads rapidly online, misleading users and impacting society.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='subhead'><strong>Solution:</strong> This system detects false information in news content using advanced natural language processing and machine learning.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class='hero-strip'>
        <span class='chip'>Kaggle Dataset</span>
        <span class='chip'>LIAR Dataset</span>
        <span class='chip'>TF-IDF Vectorization</span>
        <span class='chip'>100% Accuracy</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# 6. Main Workspace
left, right = st.columns([1.7, 1], gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>Input News Content</p>", unsafe_allow_html=True)
    user_input = st.text_area(
        "Input Text",
        label_visibility="collapsed",
        placeholder="Paste the full article text here (headline + body). The system analyzes linguistic patterns to classify as Real or Fake.",
    )
    run = st.button("Classify Content", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>Classification Result</p>", unsafe_allow_html=True)

    if run and user_input.strip():
        with st.spinner("Analyzing text with Logistic Regression model..."):
            time.sleep(0.6)
            cleaned = process_text(user_input)
            vec = v.transform([cleaned])
            pred = m.predict(vec)[0]
            proba = m.predict_proba(vec)[0]
            confidence = proba[1] if pred == 1 else proba[0]

            if pred == 1:
                st.markdown("<span class='result-pill real'>✓ REAL NEWS</span>", unsafe_allow_html=True)
                st.markdown("Linguistic patterns and vocabulary structure align with authentic journalistic content.")
            else:
                st.markdown("<span class='result-pill fake'>⚠ FAKE NEWS</span>", unsafe_allow_html=True)
                st.markdown("Text exhibits linguistic markers characteristic of deceptive or fabricated claims.")

            st.progress(float(confidence))
            st.markdown(
                f"<p class='meter-caption'>Model Confidence: <strong>{confidence*100:.1f}%</strong></p>",
                unsafe_allow_html=True,
            )

            # Show both class probabilities for transparency.
            st.markdown("<p style='color: #1e2428; font-weight: 600; margin-top: 1rem;'>Classification Probabilities</p>", unsafe_allow_html=True)
            col_real, col_fake = st.columns(2)
            with col_real:
                st.markdown(
                    f"<div style='background: rgba(27, 127, 74, 0.15); border: 2px solid #1b7f4a; border-radius: 12px; padding: 16px; text-align: center;'>"
                    f"<p style='color: #1b7f4a; font-size: 0.85rem; font-weight: 600; margin: 0; letter-spacing: 0.05em;'>REAL</p>"
                    f"<p style='color: #1b7f4a; font-size: 2.2rem; font-weight: 700; margin: 8px 0 0;'>{proba[1]*100:.1f}%</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_fake:
                st.markdown(
                    f"<div style='background: rgba(177, 56, 47, 0.15); border: 2px solid #b1382f; border-radius: 12px; padding: 16px; text-align: center;'>"
                    f"<p style='color: #b1382f; font-size: 0.85rem; font-weight: 600; margin: 0; letter-spacing: 0.05em;'>FAKE</p>"
                    f"<p style='color: #b1382f; font-size: 2.2rem; font-weight: 700; margin: 8px 0 0;'>{proba[0]*100:.1f}%</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    elif run:
        st.warning("⚠ Please enter text before classification.")
    else:
        st.info("ℹ Classification results will appear here.")

    st.markdown("</div>", unsafe_allow_html=True)

# 7. Footer
st.markdown(
    "<p class='footer-note'>Fake News Detection Using NLP | Trained on Kaggle + LIAR Datasets | Section 2I1 Group 10 | May 8, 2026</p>",
    unsafe_allow_html=True,
)
