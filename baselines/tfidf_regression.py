from pathlib import Path
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from data_loader import load_data


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRED_DIR = BASE_DIR / "predictions"
PRED_DIR.mkdir(exist_ok=True)


def save_predictions(preds, out_path):
    """
    Saves predictions in required JSONL format:
    {"id": "0", "prediction": 3.42}
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for i, pred in enumerate(preds):
            pred = max(1.0, min(5.0, float(pred)))  # clamp to [1, 5]
            f.write(json.dumps({
                "id": str(i),
                "prediction": pred
            }) + "\n")


def main():
    # Load data
    train_data = load_data(DATA_DIR / "train.json")
    dev_data = load_data(DATA_DIR / "dev.json")

    X_train = [x["text"] for x in train_data]
    y_train = [x["label"] for x in train_data]

    X_dev = [x["text"] for x in dev_data]

    # TF-IDF + Regression model
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            stop_words="english"
        )),
        ("regressor", Ridge(alpha=1.0))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    print("Predicting on dev set...")
    dev_preds = model.predict(X_dev)

    out_path = PRED_DIR / "tfidf_predictions_dev.jsonl"
    save_predictions(dev_preds, out_path)

    print("Saved predictions to:", out_path)


if __name__ == "__main__":
    main()


#55 accuracy
