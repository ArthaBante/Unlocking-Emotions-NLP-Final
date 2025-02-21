import pandas as pd
import pandas as pd
import string
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('punkt')

file_path = r"C:\Users\Dell\OneDrive - University of Hertfordshire\Unlocking_Emotions_NLP_Game\train_.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

df.columns = ["review", "rating"]

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)  # Tokenize words
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return " ".join(tokens)
    return ""


df["cleaned_review"] = df["review"].apply(preprocess_text)


df["emotion"] = df["rating"].map({2: "neutral"})


df[["cleaned_review", "emotion"]].to_csv(r"C:\Users\Dell\OneDrive - University of Hertfordshire\Unlocking_Emotions_NLP_Game\train_.csv", index=False)

print("✅ Text data cleaned and saved successfully!")
