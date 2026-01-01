# ensemble_with_tfidf.py
import json
import numpy as np

# Load your best predictions
with open("predictions/robertaLarge_predictions_dev.jsonl", "r") as f:
    roberta = [json.loads(line)["prediction"] for line in f]

with open("predictions/tfidf_predictions_dev.jsonl", "r") as f:
    tfidf = [json.loads(line)["prediction"] for line in f]

# Try different weights (roberta should dominate)
for roberta_weight in [0.7, 0.8, 0.9]:
    ensemble = []
    for r, t in zip(roberta, tfidf):
        pred = r * roberta_weight + t * (1 - roberta_weight)
        ensemble.append(pred)
    
    out_path = f"predictions/ensemble_r{int(roberta_weight*100)}.jsonl"
    with open(out_path, "w") as f:
        for i, pred in enumerate(ensemble):
            f.write(json.dumps({"id": str(i), "prediction": float(pred)}) + "\n")
    
    print(f"Created ensemble with RoBERTa weight {roberta_weight}")

print("\nTest each:")
for weight in [70, 80, 90]:
    print(f"python evaluate.py predictions/ensemble_r{weight}.jsonl dev")