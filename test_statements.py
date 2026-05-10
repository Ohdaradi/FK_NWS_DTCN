import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib

nltk.download('stopwords', quiet=True)
ps = PorterStemmer()
stops = set(stopwords.words('english'))

# Load model and vectorizer
m = joblib.load('fake_news_master_model.pkl')
v = joblib.load('tfidf_master_vectorizer.pkl')

# Test statements
statements = [
    "Vaccines help prevent infectious diseases",
    "NASA confirmed aliens are living on Mars.",
    "Smoking increases the risk of lung disease.",
    "Barack Obama was born in Kenya.",
    "The Chicago Bears have had more starting quarterbacks in the last 10 years than the total number of tenured University of Wisconsin faculty fired during the last two decades.",
    "Under the health care law, doctors will be paid based on how healthy their patients are."
]

def process_text(text):
    if "(Reuters) -" in text:
        text = text.split("(Reuters) -", 1)[1]
    if " - " in text and len(text.split(" - ")[0]) < 30:
        text = text.split(" - ", 1)[1]
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower().split()
    return ' '.join([ps.stem(w) for w in text if w not in stops])

print("="*80)
print("FAKE NEWS DETECTION TEST RESULTS")
print("="*80)

for i, stmt in enumerate(statements, 1):
    cleaned = process_text(stmt)
    vec = v.transform([cleaned])
    pred = m.predict(vec)[0]
    proba = m.predict_proba(vec)[0]
    
    result = "✓ REAL NEWS" if pred == 1 else "⚠ FAKE NEWS"
    confidence = proba[1] if pred == 1 else proba[0]
    
    print(f"\n[Statement {i}]")
    print(f"Text: {stmt}")
    print(f"Result: {result}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"Real: {proba[1]*100:.1f}% | Fake: {proba[0]*100:.1f}%")

print("\n" + "="*80)
