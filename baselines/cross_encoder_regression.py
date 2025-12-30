from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from sentence_transformers import CrossEncoder

from data_loader import load_data


# ------------------ Paths ------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRED_DIR = BASE_DIR / "predictions"
PRED_DIR.mkdir(exist_ok=True)


# ------------------ Save predictions ------------------
def save_predictions(preds, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, pred in enumerate(preds):
            pred = max(1.0, min(5.0, float(pred)))
            f.write(json.dumps({
                "id": str(i),
                "prediction": pred
            }) + "\n")


def split_story_and_sense(samples):
    pairs = []
    labels = []

    for x in samples:
        story, sense = x["text"].split("\n\nSense definition: ")
        pairs.append([story, sense])
        if x.get("label") is not None:
            labels.append(x["label"])

    return pairs, labels


# ------------------ Main pipeline ------------------
def main():
    print("Loading data...")
    train_data = load_data(DATA_DIR / "train.json")
    dev_data = load_data(DATA_DIR / "dev.json")

    train_pairs, train_labels = split_story_and_sense(train_data)
    dev_pairs, _ = split_story_and_sense(dev_data)

    train_labels = np.array(train_labels)

    # ------------------ Load Cross-Encoder ------------------
    print("Loading Cross-Encoder model...")
    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        num_labels=1
    )

    # ------------------ Train ------------------
    print("Training cross-encoder...")
    model.fit(
        train_pairs,
        train_labels,
        epochs=2,              # VERY IMPORTANT: small number
        batch_size=8,          # keep small for CPU
        warmup_steps=100,
        show_progress_bar=True
    )

    # ------------------ Predict ------------------
    print("Predicting on dev set...")
    dev_preds = model.predict(dev_pairs, batch_size=8)

    out_path = PRED_DIR / "cross_encoder_predictions_dev.jsonl"
    save_predictions(dev_preds, out_path)

    print("Saved predictions to:", out_path)


if __name__ == "__main__":
    main()
