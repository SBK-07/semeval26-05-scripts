import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaModel, 
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW  # Changed import location
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import random
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_loader import load_data

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PRED_DIR = BASE_DIR / "predictions"

# Create directories
MODEL_DIR.mkdir(exist_ok=True, parents=True)
PRED_DIR.mkdir(exist_ok=True, parents=True)

# Configuration
class Config:
    # Model
    model_name = "roberta-large" # Start with base for testing, then switch to large
    max_length = 256
    batch_size =8 if "large" in model_name else 16
    learning_rate = 2e-5
    weight_decay = 0.01
    epochs = 5
    warmup_ratio = 0.1
    hidden_dropout_prob = 0.1
    
    # Training
    seed = 42
    num_workers = 0  # Set to 0 for Windows compatibility
    
    # Early stopping
    patience = 3
    min_delta = 0.001

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PlausibilityDataset(Dataset):
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
        
        # Remove batch dimension
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

class RobertaRegressionModel(nn.Module):
    """RoBERTa with regression head"""
    def __init__(self, config):
        super(RobertaRegressionModel, self).__init__()
        
        # Load pre-trained RoBERTa
        self.roberta = RobertaModel.from_pretrained(config.model_name)
        
        # Get hidden size
        hidden_size = self.roberta.config.hidden_size
        
        # Regression head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._init_weights(self.regressor)
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask):
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token
        pooled_output = outputs.pooler_output
        
        # Apply dropout and regression
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        
        return logits.squeeze(-1)

def train_model(model, train_dataloader, val_dataloader, config, device):
    """Train the model with early stopping"""
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    total_steps = len(train_dataloader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        # Training
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        val_loss = evaluate_model(model, val_dataloader, criterion, device)
        print(f"Validation loss: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_loss - config.min_delta:
            best_loss = val_loss
            patience_counter = 0
            
            # Save best model
            # Save best model
            model_save_path = MODEL_DIR / "best_roberta_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                # Save config as a simple dictionary
                'config': {
                    'model_name': config.model_name,
                    'max_length': config.max_length,
                    'batch_size': config.batch_size,
                    'learning_rate': config.learning_rate,
                    'epochs': config.epochs
                }
            }, model_save_path)
            print(f"Saved best model with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
            
            if patience_counter >= config.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model
    checkpoint_path = MODEL_DIR / "best_roberta_model.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
    
    return model

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model and return loss"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def predict(model, dataloader, device):
    """Generate predictions"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids, attention_mask)
            predictions.extend(outputs.cpu().numpy())
    
    return predictions

def save_predictions(predictions, out_path):
    """Save predictions in JSONL format"""
    # Clamp predictions to [1, 5] range
    predictions = np.clip(predictions, 1.0, 5.0)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for i, pred in enumerate(predictions):
            f.write(json.dumps({
                "id": str(i),
                "prediction": float(pred)
            }) + "\n")

def main():
    # Set seed
    set_seed(Config.seed)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    print("Loading data...")
    train_data = load_data(DATA_DIR / "train.json")
    dev_data = load_data(DATA_DIR / "dev.json")
    
    print(f"Train samples: {len(train_data)}")
    print(f"Dev samples: {len(dev_data)}")
    
    # Initialize tokenizer and model
    print("Initializing model...")
    tokenizer = RobertaTokenizer.from_pretrained(Config.model_name)
    
    # Create datasets
    train_dataset = PlausibilityDataset(train_data, tokenizer, Config.max_length)
    dev_dataset = PlausibilityDataset(dev_data, tokenizer, Config.max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers
    )
    
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers
    )
    
    # Initialize model
    model = RobertaRegressionModel(Config)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nStarting training...")
    model = train_model(model, train_dataloader, dev_dataloader, Config, device)
    
    # Generate predictions on dev set
    print("\nGenerating predictions on dev set...")
    dev_predictions = predict(model, dev_dataloader, device)
    
    # Save predictions
    dev_pred_path = PRED_DIR / "robertaLarge_predictions_dev.jsonl"
    save_predictions(dev_predictions, dev_pred_path)
    print(f"Saved dev predictions to: {dev_pred_path}")
    
    # Test set predictions (if exists)
    test_path = DATA_DIR / "test.json"
    if test_path.exists():
        print("\nLoading test data...")
        test_data = load_data(test_path)
        print(f"Test samples: {len(test_data)}")
        
        test_dataset = PlausibilityDataset(test_data, tokenizer, Config.max_length)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers
        )
        
        test_predictions = predict(model, test_dataloader, device)
        
        test_pred_path = PRED_DIR / "improved1_roberta_predictions_test.jsonl"
        save_predictions(test_predictions, test_pred_path)
        print(f"Saved test predictions to: {test_pred_path}")

if __name__ == "__main__":
    main()

#Everything looks OK. Evaluating file predictions/robertaLarge_predictions_dev.jsonl on data/dev.json...
#----------
#Spearman Correlation: 0.43866662073213825
#Spearman p-Value: 4.749723359875079e-29
#----------
#Accuracy: 0.70578231292517 (415/588)