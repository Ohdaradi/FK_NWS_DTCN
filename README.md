# 📰 Fake News Detection Using NLP

**SECTION-2I1 - Group 10**

A machine learning application that classifies news articles as real or fake using Natural Language Processing techniques.

## Project Overview

**Objective:** Classify news articles as real or fake using NLP techniques and machine learning models.

### Models Trained
- 🔹 **Logistic Regression** (Production)
- 🔹 Naïve Bayes
- 🔹 Random Forest

### Datasets Used
- ✓ Kaggle Fake News Dataset
- ✓ LIAR Dataset

### Vectorization
- **TF-IDF with N-Grams** (1-2)
- **Max Features:** 15,000

## Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd FK_NWS_DTCN
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Usage Tips
- Paste full article text for best results
- Include headline and multiple paragraphs
- Longer, well-structured content improves accuracy

## Features
- Real-time fake news detection
- Multiple ML model comparisons
- Interactive UI built with Streamlit
- TF-IDF vectorization with N-grams

## Project Structure
```
FK_NWS_DTCN/
├── app.py                          # Main Streamlit application
├── create_liar_data.py            # LIAR dataset processing
├── create_sample_data.py          # Sample data generation
├── fake_news_master_model.pkl     # Trained model
├── tfidf_master_vectorizer.pkl    # TF-IDF vectorizer
├── requirements.txt               # Python dependencies
└── README.md                       # This file
```

## Authors
Group 10 - SECTION-2I1

## License
This project is for educational purposes.
