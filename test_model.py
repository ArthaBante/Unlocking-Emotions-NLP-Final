import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Define the path to your fine-tuned model
MODEL_SAVE_PATH = r"C:\Users\Dell\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\fine_tuned_bert_model"

# Load the fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(r"C:\Users\Dell\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\fine_tuned_bert_model")
tokenizer = DistilBertTokenizer.from_pretrained(r"C:\Users\Dell\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\fine_tuned_bert_model")


# Define a function to predict sentiment
def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

    # Perform the prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted class (0: negative, 1: neutral, 2: positive)
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map the predicted class to a sentiment label
    sentiment_labels = {0: "Positive", 1: "Negative", 2: "Neutral"}
    return sentiment_labels[predicted_class]


# Test the model with some sample sentences
if __name__ == "__main__":
    print("🚀 Testing the fine-tuned BERT model...")

    # Sample sentences to test
    test_sentences = [

        "The product arrived on time and was as described.",

        "The weather today is neither too hot nor too cold.",

        "I have no strong feelings about this movie.",

        "The meeting was scheduled for 2 PM and ended at 3 PM.",

        "The package was delivered without any issues."
    ]

    for sentence in test_sentences:
        sentiment = predict_sentiment(sentence)
        print(f"Sentence: {sentence}\nPredicted Sentiment: {sentiment}\n")
