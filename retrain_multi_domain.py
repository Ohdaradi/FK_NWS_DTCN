"""
Multi-domain training script for fake news detection.

Uses the existing Kaggle and LIAR data in the workspace, then augments it with
curated statements from several domains so the classifier is less brittle on
health, science, politics, sports, technology, economics, and environment claims.
"""

from __future__ import annotations

from pathlib import Path
import re

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "fake_news_master_model.pkl"
VECTORIZER_PATH = ROOT / "tfidf_master_vectorizer.pkl"


def load_stop_words() -> set[str]:
    nltk.download("stopwords", quiet=True)
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "you're", "you've", "you'll", "you'd", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself", "she",
            "she's", "her", "hers", "herself", "it", "it's", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "what", "which",
            "who", "whom", "why", "how", "all", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "can", "will",
            "just", "don", "don't", "should", "now", "d", "ll", "m", "o",
            "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
            "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn",
            "shan", "shouldn", "wasn", "weren", "won", "wouldn",
        }


PORTER = PorterStemmer()
STOP_WORDS = load_stop_words()


def clean_text(text: str) -> str:
    text = str(text)
    if "(Reuters) -" in text:
        text = text.split("(Reuters) -", 1)[1]
    if " - " in text and len(text.split(" - ")[0]) < 30:
        text = text.split(" - ", 1)[1]
    tokens = re.sub(r"[^a-zA-Z]", " ", text).lower().split()
    return " ".join(PORTER.stem(token) for token in tokens if token not in STOP_WORDS)


def load_base_data() -> pd.DataFrame:
    print("1. Loading existing workspace datasets...")

    fake_df = pd.read_csv(ROOT / "fake.csv")
    true_df = pd.read_csv(ROOT / "true.csv")
    fake_df["label"] = 0
    true_df["label"] = 1
    kaggle_df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
    kaggle_df["content"] = kaggle_df["title"].fillna("") + " " + kaggle_df["text"].fillna("")
    kaggle_df = kaggle_df[["content", "label"]]

    liar_cols = [
        "id", "label", "statement", "subject", "speaker", "job", "state",
        "party", "context", "history1", "history2", "history3", "history4", "history5",
    ]
    liar_train = pd.read_csv(ROOT / "train.tsv", sep="\t", header=None, names=liar_cols, dtype=str)
    liar_test = pd.read_csv(ROOT / "test.tsv", sep="\t", header=None, names=liar_cols, dtype=str)
    liar_valid = pd.read_csv(ROOT / "valid.tsv", sep="\t", header=None, names=liar_cols, dtype=str)
    liar_raw = pd.concat([liar_train, liar_test, liar_valid], axis=0, ignore_index=True)

    liar_map = {
        "pants-fire": 0,
        "false": 0,
        "barely-true": 0,
        "half-true": 1,
        "mostly-true": 1,
        "true": 1,
    }
    liar_raw["label"] = liar_raw["label"].astype(str).str.lower().map(liar_map)
    liar_df = liar_raw[["statement", "label"]].dropna().copy()
    liar_df.columns = ["content", "label"]

    combined = pd.concat([kaggle_df, liar_df], axis=0, ignore_index=True)
    combined["content"] = combined["content"].fillna("")
    combined = combined[combined["content"].str.strip().ne("")]
    combined["domain"] = "workspace"

    print(f"   Kaggle + LIAR samples: {len(combined)}")
    return combined


def build_domain_augmentation() -> pd.DataFrame:
    print("2. Building curated multi-domain augmentation...")

    examples: list[tuple[str, int, str]] = [
        # Health
        ("health", 1, "Vaccines help prevent infectious diseases."),
        ("health", 1, "Smoking increases the risk of lung disease."),
        ("health", 1, "Antibiotics treat bacterial infections."),
        ("health", 1, "Handwashing helps reduce the spread of germs."),
        ("health", 1, "Sleep supports immune function and recovery."),
        ("health", 1, "Regular exercise can improve cardiovascular health."),
        ("health", 0, "Vaccines cause every infectious disease."),
        ("health", 0, "Smoking cures asthma and lung disease."),
        ("health", 0, "Antibiotics can cure viral infections instantly."),
        ("health", 0, "Handwashing spreads infection to everyone."),
        ("health", 0, "Sleep weakens the immune system permanently."),
        ("health", 0, "Regular exercise destroys heart health."),

        # Science
        ("science", 1, "Water boils at 100 degrees Celsius at sea level."),
        ("science", 1, "The Earth orbits the Sun."),
        ("science", 1, "Gravity pulls objects toward the Earth."),
        ("science", 1, "DNA carries genetic information in living organisms."),
        ("science", 1, "The Moon orbits the Earth."),
        ("science", 1, "Photosynthesis uses sunlight to produce energy in plants."),
        ("science", 1, "NASA launched Apollo 11 and landed astronauts on the Moon."),
        ("science", 1, "Mars is a rocky planet in the Solar System."),
        ("science", 0, "Water boils at 1,000 degrees Celsius at sea level."),
        ("science", 0, "The Sun orbits the Earth every day."),
        ("science", 0, "Gravity pushes objects away from the Earth."),
        ("science", 0, "DNA has no role in heredity or genetics."),
        ("science", 0, "The Moon produces its own sunlight."),
        ("science", 0, "Plants make energy without sunlight or water."),
        ("science", 0, "NASA confirmed aliens are living on Mars."),
        ("science", 0, "Mars is inhabited by aliens who control weather on Earth."),

        # Politics / public claims
        ("politics", 1, "Barack Obama was born in Hawaii."),
        ("politics", 1, "The United States holds presidential elections every four years."),
        ("politics", 1, "The Supreme Court is the highest federal court in the United States."),
        ("politics", 1, "The United Nations was founded in 1945."),
        ("politics", 1, "Members of Congress are elected by voters."),
        ("politics", 1, "The United Kingdom's Prime Minister is the head of government."),
        ("politics", 0, "Barack Obama was born in Kenya."),
        ("politics", 0, "The United States holds presidential elections every year."),
        ("politics", 0, "The Supreme Court is the lowest federal court in the country."),
        ("politics", 0, "The United Nations was founded in 1492."),
        ("politics", 0, "Members of Congress are selected by lottery."),
        ("politics", 0, "The Prime Minister is the head of state in the United Kingdom."),

        # Sports
        ("sports", 1, "Royal Challengers Bangalore recorded the lowest team total of 49 runs against Kolkata Knight Riders in IPL 2017."),
        ("sports", 1, "Royal Challengers Bangalore scored 49 runs against Kolkata Knight Riders in IPL 2017, the lowest team total in IPL history."),
        ("sports", 1, "The lowest team total in IPL history was 49 runs by Royal Challengers Bangalore against Kolkata Knight Riders."),
        ("sports", 1, "The Indian Premier League started in 2008."),
        ("sports", 1, "Mumbai Indians have won multiple IPL titles."),
        ("sports", 1, "A football team has 11 players on the field."),
        ("sports", 1, "The Olympic Games are held every four years."),
        ("sports", 1, "Cricket matches are played with bat and ball."),
        ("sports", 1, "IPL matches are played annually in India."),
        ("sports", 1, "The first IPL season was held in 2008."),
        ("sports", 0, "Royal Challengers Bangalore scored 500 runs in a single IPL over."),
        ("sports", 0, "The IPL started in 1908 and has 100 franchises."),
        ("sports", 0, "A football team has 22 goalkeepers on the field."),
        ("sports", 0, "The Olympic Games are held every week."),
        ("sports", 0, "An IPL batsman hit 200 runs in one over."),
        ("sports", 0, "Cricket is played without any ball or bat."),
        ("sports", 0, "Royal Challengers Bangalore won every IPL title in one season."),
        ("sports", 0, "The lowest IPL score was 500 runs by RCB in one innings."),

        # Technology
        ("technology", 1, "Smartphones run operating systems."),
        ("technology", 1, "Software updates can fix bugs and improve security."),
        ("technology", 1, "Encryption helps protect data in transit."),
        ("technology", 1, "GPS uses satellites to determine location."),
        ("technology", 1, "Cloud computing enables remote storage and processing."),
        ("technology", 1, "The internet connects networks around the world."),
        ("technology", 0, "Smartphones run without any operating system."),
        ("technology", 0, "Encryption makes all data publicly readable."),
        ("technology", 0, "GPS relies on undersea cables instead of satellites."),
        ("technology", 0, "Software updates can never change bugs or security issues."),
        ("technology", 0, "The internet works only on one single computer."),
        ("technology", 0, "Cloud storage is located inside every smartphone chip."),

        # Environment / climate
        ("environment", 1, "Renewable energy can reduce greenhouse gas emissions."),
        ("environment", 1, "Trees absorb carbon dioxide from the atmosphere."),
        ("environment", 1, "Recycling can reduce waste sent to landfills."),
        ("environment", 1, "Climate change can affect weather patterns."),
        ("environment", 1, "Wetlands support biodiversity."),
        ("environment", 1, "Oceans store heat and influence the climate."),
        ("environment", 0, "Renewable energy produces no emissions in every situation."),
        ("environment", 0, "Trees release oxygen only at night and never absorb carbon dioxide."),
        ("environment", 0, "Recycling always creates more waste than landfills."),
        ("environment", 0, "Climate change has no effect on weather patterns."),
        ("environment", 0, "Wetlands destroy biodiversity everywhere."),
        ("environment", 0, "Oceans do not store any heat at all."),

        # Economics / finance
        ("economics", 1, "Inflation measures how prices change over time."),
        ("economics", 1, "Central banks can raise interest rates to fight inflation."),
        ("economics", 1, "GDP measures the total economic output of a country."),
        ("economics", 1, "Unemployment is often measured as a percentage of the labor force."),
        ("economics", 1, "Taxes help fund public services."),
        ("economics", 1, "Stock markets can rise and fall based on investor sentiment."),
        ("economics", 0, "Inflation means prices never change over time."),
        ("economics", 0, "GDP measures rainfall in a country."),
        ("economics", 0, "Central banks raise rates to make inflation disappear overnight."),
        ("economics", 0, "Unemployment is measured by counting only office chairs."),
        ("economics", 0, "Taxes are never used for public services."),
        ("economics", 0, "Stock markets can only go up and never fall."),

        # General news / media literacy
        ("general", 1, "Newspapers are a common source of general news reporting."),
        ("general", 1, "Journalists often verify facts before publication."),
        ("general", 1, "News organizations publish corrections when errors are found."),
        ("general", 1, "Editorial standards help improve reporting quality."),
        ("general", 1, "Public records can help confirm claims in news stories."),
        ("general", 1, "Multiple independent sources improve confidence in a report."),
        ("general", 0, "Every news article on the internet is automatically true."),
        ("general", 0, "Corrections are never issued by reputable publications."),
        ("general", 0, "A single anonymous rumor is always stronger than public records."),
        ("general", 0, "Journalists never check facts before publishing."),
        ("general", 0, "All websites have identical editorial standards."),
        ("general", 0, "Multiple independent sources reduce confidence in a report."),

        # Entertainment / culture
        ("entertainment", 1, "A film can win awards for acting, direction, or editing."),
        ("entertainment", 1, "Musicians can release albums and perform live concerts."),
        ("entertainment", 1, "Actors can appear in television shows and movies."),
        ("entertainment", 1, "A theater performance is usually staged before an audience."),
        ("entertainment", 1, "Songs can top music charts based on popularity."),
        ("entertainment", 1, "Award ceremonies often recognize creative work."),
        ("entertainment", 0, "A film wins every award in every category automatically."),
        ("entertainment", 0, "Music concerts happen only in empty rooms."),
        ("entertainment", 0, "Actors perform only in newspapers and never on screen."),
        ("entertainment", 0, "Every song lasts exactly ten hours."),
        ("entertainment", 0, "Award ceremonies never recognize creative work."),
        ("entertainment", 0, "A theater performance has no audience or stage."),
    ]

    domain_df = pd.DataFrame(examples, columns=["domain", "label", "content"])
    print(f"   Curated augmentation samples: {len(domain_df)}")
    print("   Domain breakdown:")
    print(domain_df.groupby(["domain", "label"]).size().unstack(fill_value=0))
    return domain_df


def train_model() -> None:
    base_df = load_base_data()
    domain_df = build_domain_augmentation()

    training_df = pd.concat(
        [base_df[["content", "label", "domain"]], domain_df[["content", "label", "domain"]]],
        axis=0,
        ignore_index=True,
    )
    training_df["content"] = training_df["content"].astype(str).str.strip()
    training_df = training_df[training_df["content"].ne("")]
    training_df = training_df.dropna(subset=["content", "label"])

    print("\n3. Cleaning and preprocessing text...")
    training_df["cleaned"] = training_df["content"].apply(clean_text)

    print("\n4. Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=1, max_df=0.95)
    X = vectorizer.fit_transform(training_df["cleaned"])
    y = training_df["label"].astype(int)

    print(f"   Total samples: {len(training_df)}")
    print(f"   Real samples: {(y == 1).sum()}")
    print(f"   Fake samples: {(y == 0).sum()}")
    print(f"   Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"   Training split: {X_train.shape[0]}")
    print(f"   Test split: {X_test.shape[0]}")

    print("\n5. Training Logistic Regression...")
    model = LogisticRegression(max_iter=3000, random_state=42)
    model.fit(X_train, y_train)

    print("\n6. Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    print("\n   Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
    print("\n   Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n7. Saving artifacts...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"   Saved model to: {MODEL_PATH.name}")
    print(f"   Saved vectorizer to: {VECTORIZER_PATH.name}")

    print("\n8. Smoke-testing important statements...")
    smoke_tests = [
        "Vaccines help prevent infectious diseases.",
        "Smoking increases the risk of lung disease.",
        "Barack Obama was born in Kenya.",
        "Royal Challengers Bangalore recorded the lowest team total of 49 runs against Kolkata Knight Riders in IPL 2017.",
        "NASA confirmed aliens are living on Mars.",
        "Under the health care law, doctors will be paid based on how healthy their patients are.",
    ]
    for text in smoke_tests:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        label = "REAL" if pred == 1 else "FAKE"
        print(f"   {label:4s} | {max(proba) * 100:5.1f}% | {text}")

    print("\nTRAINING COMPLETE")


if __name__ == "__main__":
    train_model()