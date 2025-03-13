import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch



# Load dataset
df = pd.read_csv(r"C:\Users\ab22adw\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\Data_Set.csv")

# Convert dataset to Hugging Face format
dataset = Dataset.from_pandas(df)

# Load BERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Function to tokenize input text
def tokenize_data(example):
    return tokenizer(example["Response"], truncation=True, padding="max_length")

# Apply tokenization
dataset = dataset.map(tokenize_data, batched=True)
dataset = dataset.rename_column("Sentiment", "labels")  # Rename for compatibility
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Split dataset into training and testing sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Load pre-trained DistilBERT model for fine-tuning
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",  # Save directory
    save_strategy="steps",       # Save checkpoints every few steps
    save_steps=500,              # Save every 500 steps
    save_total_limit=2,          # Keep only the last 2 checkpoints
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
# Detect last saved checkpoint
last_checkpoint = r"C:\Users\ab22adw\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\checkpoints\checkpoint-500"
if last_checkpoint:
    print(f"Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()



# Save the fine-tuned model
model.save_pretrained(r"C:\Users\ab22adw\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\fine_tuned_bert_model")
tokenizer.save_pretrained(r"C:\Users\ab22adw\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\fine_tuned_bert_model")

print("✅ BERT Model Training Complete! Model saved to models/fine_tuned_bert_model")