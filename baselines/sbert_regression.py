from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge

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


# ------------------ Main pipeline ------------------
def main():
    print("Loading data...")
    train_data = load_data(DATA_DIR / "train.json")
    dev_data = load_data(DATA_DIR / "dev.json")

    X_train = [x["text"] for x in train_data]
    y_train = np.array([x["label"] for x in train_data])

    X_dev = [x["text"] for x in dev_data]

    # ------------------ Load SBERT ------------------
    print("Loading SBERT model...")
    sbert = SentenceTransformer("all-mpnet-base-v2")


    # ------------------ Encode text ------------------
    print("Encoding training data...")
    X_train_emb = sbert.encode(
        X_train,
        batch_size=32,
        show_progress_bar=True
    )

    print("Encoding dev data...")
    X_dev_emb = sbert.encode(
        X_dev,
        batch_size=32,
        show_progress_bar=True
    )

    # ------------------ Regression head ------------------
    print("Training regression head...")
    regressor = Ridge(alpha=1.0)
    regressor.fit(X_train_emb, y_train)

    # ------------------ Predict ------------------
    print("Predicting...")
    dev_preds = regressor.predict(X_dev_emb)

    out_path = PRED_DIR / "predictions.jsonl"
    save_predictions(dev_preds, out_path)

    print("Saved predictions to:", out_path)


if __name__ == "__main__":
    main()

#predictions.jsonl
#accuracy 56.8
#Spearman Correlation: 0.10605008409045452
#Spearman p-Value: 0.010071462232384167