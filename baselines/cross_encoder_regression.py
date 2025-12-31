from pathlib import Path
import json
import numpy as np

from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample

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
            # rescale from [0,1] → [1,5]
            pred = 1.0 + 4.0 * float(pred)
            pred = max(1.0, min(5.0, pred))
            f.write(json.dumps({
                "id": str(i),
                "prediction": pred
            }) + "\n")


def split_story_and_sense(samples):
    rows = []
    for x in samples:
        story, sense = x["text"].split("\n\nSense definition: ")
        rows.append((story, sense, x.get("label")))
    return rows


# ------------------ Main ------------------
def main():
    print("Loading data...")
    train_data = load_data(DATA_DIR / "train.json")
    dev_data = load_data(DATA_DIR / "dev.json")

    train_rows = split_story_and_sense(train_data)
    dev_rows = split_story_and_sense(dev_data)

    # -------- Build InputExamples with NORMALIZED labels --------
    train_examples = []
    for story, sense, label in train_rows:
        # normalize label: [1,5] → [0,1]
        norm_label = (label - 1.0) / 4.0
        train_examples.append(
            InputExample(texts=[story, sense], label=norm_label)
        )

    dev_pairs = [[story, sense] for story, sense, _ in dev_rows]

    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=8
    )

    print("Loading Cross-Encoder model...")
    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        num_labels=1
    )

    print("Training cross-encoder...")
    model.fit(
        train_loader,
        epochs=2,
        show_progress_bar=True
    )

    print("Predicting...")
    dev_preds = model.predict(dev_pairs)

    out_path = PRED_DIR / "cross_encoder_predictions_dev.jsonl"
    save_predictions(dev_preds, out_path)

    print("Saved predictions to:", out_path)


if __name__ == "__main__":
    main()
