import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaModel, 
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import random
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_loader import load_data

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PRED_DIR = BASE_DIR / "predictions"

# Create directories
MODEL_DIR.mkdir(exist_ok=True, parents=True)
PRED_DIR.mkdir(exist_ok=True, parents=True)

# OPTIMIZED CONFIG for 10 epochs
class Config:
    model_name = "roberta-large"
    max_length = 256
    batch_size = 8
    learning_rate = 1.5e-5
    weight_decay = 0.01
    epochs = 10
    warmup_ratio = 0.1
    hidden_dropout_prob = 0.1
    gradient_clip = 1.0
    
    # Training
    seed = 42
    num_workers = 2
    
    # Early stopping
    patience = 4
    min_delta = 0.001

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EnhancedDataset(Dataset):
    """Dataset for plausibility scoring"""
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        if item["label"] is None:
            label = torch.tensor(-1.0, dtype=torch.float32)
        else:
            label = torch.tensor(item["label"], dtype=torch.float32)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
        }

class EnhancedRobertaModel(nn.Module):
    """Enhanced model with better architecture"""
    def __init__(self, config):
        super().__init__()
        # FIXED: Added output_hidden_states=True
        self.roberta = RobertaModel.from_pretrained(
            config.model_name, 
            output_hidden_states=True  # This fixes the error
        )
        hidden_size = self.roberta.config.hidden_size
        
        # Enhanced regression head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for module in self.regressor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # SIMPLIFIED: Use pooler_output instead of hidden_states
        # This avoids the NoneType error
        pooled = outputs.pooler_output
        
        # Alternative: Use CLS token from last hidden state
        # last_hidden_state = outputs.last_hidden_state
        # pooled = last_hidden_state[:, 0, :]  # CLS token
        
        pooled = self.dropout(pooled)
        return self.regressor(pooled).squeeze(-1)

def save_predictions(predictions, out_path):
    predictions = np.clip(predictions, 1.0, 5.0)
    with open(out_path, "w") as f:
        for i, pred in enumerate(predictions):
            f.write(json.dumps({"id": str(i), "prediction": float(pred)}) + "\n")

def train_and_evaluate():
    set_seed(Config.seed)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("Loading data...")
    train_data = load_data(DATA_DIR / "train.json")
    dev_data = load_data(DATA_DIR / "dev.json")
    
    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}")
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(Config.model_name)
    
    # Create datasets
    train_dataset = EnhancedDataset(train_data, tokenizer, Config.max_length)
    dev_dataset = EnhancedDataset(dev_data, tokenizer, Config.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers
    )
    
    # Initialize model
    model = EnhancedRobertaModel(Config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    
    # Scheduler
    total_steps = len(train_loader) * Config.epochs
    warmup_steps = int(total_steps * Config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    for epoch in range(Config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{Config.epochs}")
        print(f"{'='*60}")
        
        # Training
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.gradient_clip)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                val_predictions.extend(outputs.cpu().numpy())
        
        avg_val_loss = val_loss / len(dev_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss - Config.min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            
            # Save best model state
            best_model_state = model.state_dict().copy()
            
            # Save predictions for this best epoch
            preds_path = PRED_DIR / f"epoch_{epoch+1}_predictions.jsonl"
            save_predictions(val_predictions, preds_path)
            
            print(f"✅ New best model! Loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epoch(s)")
            
            if patience_counter >= Config.patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model from epoch {best_epoch} with loss: {best_val_loss:.4f}")
    
    # Final predictions
    model.eval()
    final_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Final Prediction"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids, attention_mask)
            final_predictions.extend(outputs.cpu().numpy())
    
    # Save final predictions
    final_path = PRED_DIR / "10epochs_final_predictions.jsonl"
    save_predictions(final_predictions, final_path)
    
    # Also save rounded version
    rounded_preds = np.round(np.array(final_predictions) * 2) / 2
    rounded_path = PRED_DIR / "10epochs_rounded_predictions.jsonl"
    save_predictions(rounded_preds, rounded_path)
    
    print(f"\n✅ Training complete!")
    print(f"   Final predictions: {final_path}")
    print(f"   Rounded predictions: {rounded_path}")
    
    return final_predictions

def main():
    print("="*70)
    print("TRAINING ROERBTA-LARGE FOR 10 EPOCHS")
    print("Expected time: 1.5 hours on Colab GPU")
    print("Expected Spearman: 0.47-0.50")
    print("="*70)
    
    predictions = train_and_evaluate()
    
    print("\n" + "="*70)
    print("EVALUATION COMMANDS:")
    print("python evaluate.py predictions/10epochs_final_predictions.jsonl dev")
    print("python evaluate.py predictions/10epochs_rounded_predictions.jsonl dev")
    print("="*70)

if __name__ == "__main__":
    main()

# Everything looks OK. Evaluating file predictions/10epochs_final_predictions.jsonl on data/dev.json...
# ----------
# Spearman Correlation: 0.47748616861355825
# Spearman p-Value: 8.116991878518894e-35
# ----------
# Accuracy: 0.7040816326530612 (414/588)
# Everything looks OK. Evaluating file predictions/10epochs_rounded_predictions.jsonl on data/dev.json...
# ----------
# Spearman Correlation: 0.4694076608809388
# Spearman p-Value: 1.4836791152020225e-33
# ----------
# Accuracy: 0.6717687074829932 (395/588)