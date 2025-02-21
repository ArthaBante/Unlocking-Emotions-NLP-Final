from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import joblib

dataset = r"C:\Users\Dell\OneDrive - University of Hertfordshire\Unlocking_Emotions_NLP_Game\Unlocking_Emotions_NLP_Game\cleaned_reviews.csv"


df = pd.read_csv(dataset, encoding="ISO-8859-1")


df.columns = ["cleaned_review", "emotion"]

df = df.dropna()

print("Dataset class distribution:\n", df["emotion"].value_counts())

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["cleaned_review"])
y = df["emotion"]

model = MultinomialNB()
model.fit(X, y)

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("model trained")
