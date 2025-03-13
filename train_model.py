import pandas as pd
import torch
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Define file paths
DATASET_PATH = r"C:\Users\ab22adw\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\Data_Set.csv"
CHECKPOINT_DIR = r"C:\Users\ab22adw\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\checkpoints"
MODEL_SAVE_PATH = r"C:\Users\ab22adw\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\fine_tuned_bert_model"
LAST_CHECKPOINT = r"C:\Users\ab22adw\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\checkpoints\checkpoint-500"


def load_dataset(file_path):
    """Load dataset from CSV and convert to Hugging Face format."""
    df = pd.read_csv(file_path)
    dataset = Dataset.from_pandas(df)
    return dataset


def tokenize_data(example, tokenizer):
    """Tokenize input text for BERT processing."""
    return tokenizer(example["Response"], truncation=True, padding="max_length")


def preprocess_data(dataset, tokenizer):
    """Apply tokenization and format dataset for training."""
    dataset = dataset.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    dataset = dataset.rename_column("Sentiment", "labels")  # Rename for compatibility
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Split dataset into training and testing sets
    train_test_split = dataset.train_test_split(test_size=0.2)
    return train_test_split["train"], train_test_split["test"]


def create_training_args():
    """Define training arguments for fine-tuning."""
    return TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
    )


def train_model(model, train_dataset, test_dataset, training_args):
    """Train the DistilBERT model using the Trainer API."""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    if LAST_CHECKPOINT:
        print(f"Resuming from checkpoint: {LAST_CHECKPOINT}")
        trainer.train(resume_from_checkpoint=LAST_CHECKPOINT)
    else:
        trainer.train()

    return trainer


def save_model(model, tokenizer, save_path):
    """Save fine-tuned model and tokenizer."""
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ BERT Model Training Complete! Model saved to {save_path}")


# **Main Execution**
if __name__ == "__main__":
    print("🚀 Starting BERT Model Training...")

    # Load dataset
    dataset = load_dataset(DATASET_PATH)

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Preprocess data
    train_dataset, test_dataset = preprocess_data(dataset, tokenizer)

    # Load pre-trained DistilBERT model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    # Get training arguments
    training_args = create_training_args()

    # Train model
    trainer = train_model(model, train_dataset, test_dataset, training_args)

    # Save fine-tuned model
    save_model(model, tokenizer, MODEL_SAVE_PATH)
