#post processing
# calibrate.py
import json
import numpy as np

# Load your 0.439 predictions
with open("predictions/robertaLarge_predictions_dev.jsonl", "r") as f:
    preds = [json.loads(line)["prediction"] for line in f]

preds = np.array(preds)

print(f"Current stats: Mean={np.mean(preds):.3f}, Std={np.std(preds):.3f}")
print(f"Range: [{np.min(preds):.3f}, {np.max(preds):.3f}]")

# Gold labels from training (approx 3.0 mean)
gold_mean = 3.0
current_mean = np.mean(preds)

# Strategy 1: Shift to match gold mean
if abs(current_mean - gold_mean) > 0.1:
    preds_shifted = preds + (gold_mean - current_mean)
    preds_shifted = np.clip(preds_shifted, 1.0, 5.0)

# Strategy 2: Expand variance if too narrow
if np.std(preds) < 0.8:
    preds_expanded = gold_mean + (preds - np.mean(preds)) * 1.5
    preds_expanded = np.clip(preds_expanded, 1.0, 5.0)

# Strategy 3: Round to nearest 0.5 (helps accuracy)
preds_rounded = np.round(preds * 2) / 2

# Save all strategies
strategies = [("original", preds)]
if 'preds_shifted' in locals():
    strategies.append(("shifted", preds_shifted))
if 'preds_expanded' in locals():
    strategies.append(("expanded", preds_expanded))
strategies.append(("rounded", preds_rounded))

for name, strategy_preds in strategies:
    out_path = f"predictions/calibrated_{name}.jsonl"
    with open(out_path, "w") as f:
        for i, pred in enumerate(strategy_preds):
            f.write(json.dumps({"id": str(i), "prediction": float(pred)}) + "\n")
    print(f"Saved {name} (mean={np.mean(strategy_preds):.3f}, std={np.std(strategy_preds):.3f})")

print("\nTest each:")
for name in ["original", "shifted", "expanded", "rounded"]:
    print(f"python evaluate.py predictions/calibrated_{name}.jsonl dev")