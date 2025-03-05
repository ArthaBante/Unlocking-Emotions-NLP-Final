import pandas as pd
import re
import emoji

# Load dataset
file_path = r"C:\Users\Dell\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\1M_Data_Artha.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)

# Drop rows with missing values
df.dropna(subset=["Response", "Sentiment"], inplace=True)

# Remove duplicates
df.drop_duplicates(subset=["Response"], inplace=True)

# Function to replace emojis with text (optional)
def replace_emojis(text):
    return emoji.demojize(text)  # 😊 → ":smiling_face:"

# Function to clean text
def clean_text(text):
    text = str(text)  # Ensure it's a string
    text = replace_emojis(text)  # Convert emojis
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions (@username)
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", text)  # Remove special characters
    text = text.strip()  # Remove leading/trailing spaces
    return text

# Apply cleaning function to responses
df["Response"] = df["Response"].apply(clean_text)

# Ensure sentiment labels are properly mapped
label_mapping = {"positive": 0, "negative": 1, "neutral": 2}
df["Sentiment"] = df["Sentiment"].map(label_mapping)

# Save cleaned dataset
df.to_csv(r"C:\Users\Dell\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\1M_Data_Artha.csv", index=False)

print("✅ Dataset cleaned and saved as 'cleaned_dataset.csv'!")
