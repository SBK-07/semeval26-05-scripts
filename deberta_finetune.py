#%%writefile deberta_finetune.py
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction


# Custom dataset class
@dataclass
class CustomDataset(torch.utils.data.Dataset):
    encodings: dict
    labels: Optional[list] = None

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# Function to load data
def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    texts = []
    labels = []
    ids = []
    for id_, sample in data.items():
        story = sample["precontext"] + " " + sample["sentence"]
        sense = sample["judged_meaning"]
        text = f"Story: {story} Sense: {sense}"
        texts.append(text)
        labels.append(sample["average"])
        ids.append(id_)
    return texts, labels, ids


# Compute metrics including Spearman
def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    mse = mean_squared_error(labels, predictions)
    spearman_corr, _ = spearmanr(labels, predictions)
    return {"mse": mse, "spearman": spearman_corr}


# Main training function
def train_model():
    # Hyperparameters (suggestions: start with these, tune as needed)
    learning_rate = 2e-5  # Common for DeBERTa fine-tuning
    batch_size = 8  # Adjust based on GPU memory; try 16 if possible
    num_epochs = 5  # Start with 3-5; early stopping will handle overfit
    weight_decay = 0.01
    warmup_steps = 100

    # Load tokenizer and model
    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1  # Regression: single output
    )

    # Load datasets
    train_texts, train_labels, _ = load_data("data/train.json")
    dev_texts, dev_labels, dev_ids = load_data("data/dev.json")

    # Tokenize
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    dev_encodings = tokenizer(dev_texts, truncation=True, padding=True, max_length=512)

    # Create datasets
    train_dataset = CustomDataset(train_encodings, train_labels)
    dev_dataset = CustomDataset(dev_encodings, dev_labels)

    # Training arguments
    output_dir = "checkpoints"
    training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",                  # ← CHANGED: evaluation_strategy → eval_strategy
    save_strategy="epoch",                  # ← keep this
    load_best_model_at_end=True,
    metric_for_best_model="spearman",
    greater_is_better=True,
    learning_rate=learning_rate,
    fp16=True,
    report_to="none",                       # ← ADD this to disable WandB logging (cleaner output)
    dataloader_pin_memory=True,
    )
    # Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    trainer.train()

    # Save best model
    trainer.save_model("best_model")
    tokenizer.save_pretrained("best_model")

    # Generate predictions on dev
    predictions = trainer.predict(dev_dataset).predictions.flatten()
    pred_list = []
    for id_, pred in zip(dev_ids, predictions):
        pred_list.append({"id": id_, "prediction": float(np.clip(pred, 1.0, 5.0))})  # Clip to 1-5 range

        # Save predictions in multiple places for convenience and official evaluation
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("input/res", exist_ok=True)

    # 1. Save with your custom name (for your records)
    custom_pred_path = "predictions/deberta_predictions_dev.jsonl"
    with open(custom_pred_path, "w") as f:
        for pred in pred_list:
            f.write(json.dumps(pred) + "\n")

    # 2. Save as the official filename expected by the evaluator
    official_pred_path = "input/res/predictions.jsonl"
    with open(official_pred_path, "w") as f:
        for pred in pred_list:
            f.write(json.dumps(pred) + "\n")



    print("Training complete!")
    print(f"→ Your predictions: {custom_pred_path}")
    print(f"→ Official submission file: {official_pred_path}")

    print("\nYou can now run the official evaluation:")
    print("   !python scoring.py")
    print("   or")
    print("   !python evaluate.py")


if __name__ == "__main__":
    train_model()