import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Fine-tuned BERT Model & Tokenizer
MODEL_PATH = r"C:\Users\Dell\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\fine_tuned_bert_model"
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model.eval()

game_keywords = ["village", "monster", "sword", "help", "forest", "hero", "save", "quest"]

# Function to detect if player input is related to the game
def is_related_to_game(input_text, relevant_keywords):
    input_text = input_text.lower()
    for keyword in relevant_keywords:
        if keyword.lower() in input_text:
            return True
    return False

