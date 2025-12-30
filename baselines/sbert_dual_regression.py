from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


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

    # -------- Split story and sense --------
    train_stories = []
    train_senses = []
    train_labels = []

    for x in train_data:
        full_text = x["text"]
        story, sense = full_text.split("\n\nSense definition: ")
        train_stories.append(story)
        train_senses.append(sense)
        train_labels.append(x["label"])

    dev_stories = []
    dev_senses = []

    for x in dev_data:
        full_text = x["text"]
        story, sense = full_text.split("\n\nSense definition: ")
        dev_stories.append(story)
        dev_senses.append(sense)

    y_train = np.array(train_labels)

    # ------------------ Load SBERT ------------------
    print("Loading SBERT model...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # ------------------ Encode separately ------------------
    print("Encoding training stories...")
    train_story_emb = sbert.encode(
        train_stories,
        batch_size=32,
        show_progress_bar=True
    )

    print("Encoding training senses...")
    train_sense_emb = sbert.encode(
        train_senses,
        batch_size=32,
        show_progress_bar=True
    )

    print("Encoding dev stories...")
    dev_story_emb = sbert.encode(
        dev_stories,
        batch_size=32,
        show_progress_bar=True
    )

    print("Encoding dev senses...")
    dev_sense_emb = sbert.encode(
        dev_senses,
        batch_size=32,
        show_progress_bar=True
    )

    # ------------------ Interaction features ------------------
    print("Building interaction features...")

    X_train = np.concatenate([
        train_story_emb,
        train_sense_emb,
        np.abs(train_story_emb - train_sense_emb),
        train_story_emb * train_sense_emb
    ], axis=1)

    X_dev = np.concatenate([
        dev_story_emb,
        dev_sense_emb,
        np.abs(dev_story_emb - dev_sense_emb),
        dev_story_emb * dev_sense_emb
    ], axis=1)

    # ------------------ Regression head ------------------
    print("Training regression head...")
    regressor = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    random_state=42,
    n_jobs=-1
)

    regressor.fit(X_train, y_train)


    # ------------------ Predict ------------------
    print("Predicting...")
    dev_preds = regressor.predict(X_dev)

    out_path = PRED_DIR / "sbert_dual_predictions_dev.jsonl"
    save_predictions(dev_preds, out_path)

    print("Saved predictions to:", out_path)


if __name__ == "__main__":
    main()
