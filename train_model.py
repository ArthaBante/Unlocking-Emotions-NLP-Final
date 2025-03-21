import pandas as pd
import torch
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Define file paths
DATASET_PATH = r"C:\Users\Dell\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\Data_Set.csv"
CHECKPOINT_DIR = r"C:\Users\Dell\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\checkpoints"
MODEL_SAVE_PATH = r"C:\Users\Dell\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\fine_tuned_bert_model"
LAST_CHECKPOINT = r"C:\Users\Dell\OneDrive - University of Hertfordshire\Visual_Novel_NLP_Final_year_project\checkpoints\checkpoint-500"


def load_dataset(file_path):

    df = pd.read_csv(file_path)
    dataset = Dataset.from_pandas(df)
    return dataset


def tokenize_data(example, tokenizer):

    return tokenizer(example["Response"], truncation=True, padding="max_length")


def preprocess_data(dataset, tokenizer):

    dataset = dataset.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    dataset = dataset.rename_column("Sentiment", "labels")  # Rename for compatibility
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_test_split = dataset.train_test_split(test_size=0.2)
    return train_test_split["train"], train_test_split["test"]


def create_training_args():
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
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("BERT Model Training Complete")


# **Main Execution**
if __name__ == "__main__":
    print("Starting BERT Model Training...")

    dataset = load_dataset(DATASET_PATH)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset, test_dataset = preprocess_data(dataset, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    training_args = create_training_args()

    trainer = train_model(model, train_dataset, test_dataset, training_args)

    save_model(model, tokenizer, MODEL_SAVE_PATH)
