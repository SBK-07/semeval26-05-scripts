import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_data(path):
    """
    Returns a list of dicts with:
    {
        'text': combined input text,
        'label': average score (None for test)
    }
    Order is IMPORTANT (used for id indexing)
    """
    data = []

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)          # dict: id -> sample
        samples = raw.values()      # iterate over samples only

    for sample in samples:
        text = (
            sample["precontext"] + " "
            + sample["sentence"] + " "
            + sample.get("ending", "") + "\n\n"
            + "Sense definition: " + sample["judged_meaning"]
        )

        data.append({
            "text": text,
            "label": sample.get("average")  # None for test
        })

    return data


if __name__ == "__main__":
    train_data = load_data(DATA_DIR / "train.json")
    print("Loaded samples:", len(train_data))
    print("Example:\n", train_data[0])
