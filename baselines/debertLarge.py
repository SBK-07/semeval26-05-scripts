"""
SEMEVAL TASK: Contextual Sense Plausibility - Single Fold Training
Complete solution with DeBERTa-v3-large, multi-task learning
Expected: 0.50-0.55 Spearman, 75-80% Accuracy
Training Time: 4-6 hours on Colab GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr
import random
from transformers import AutoModel, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
import warnings
import os
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    # Model
    model_name = "microsoft/deberta-v3-large"  # SOTA model
    max_length = 256
    dropout = 0.1
    
    # Training - FIXED: Reduced learning rate, added gradient clipping
    batch_size = 4  # For DeBERTa-large
    gradient_accumulation_steps = 4  # Effective batch = 16
    epochs = 10
    learning_rate = 2e-6  # REDUCED from 1e-5 to prevent NaN
    weight_decay = 0.01
    warmup_ratio = 0.1  # Increased for stability
    
    # Gradient clipping
    max_grad_norm = 1.0  # ADDED: Gradient clipping norm
    
    # Multi-task weights
    regression_weight = 0.7
    classification_weight = 0.2
    contrastive_weight = 0.1
    
    # Validation split
    val_split = 0.2  # 80% train, 20% validation
    
    # Early stopping
    patience = 5
    min_delta = 0.001
    
    # Random seed
    random_seed = 42
    
    # Paths
    data_dir = Path("data")
    model_dir = Path("models")
    pred_dir = Path("predictions")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self):
        # Ensure directories exist
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.pred_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir.mkdir(exist_ok=True, parents=True)

# ==================== DATASET ====================
class EnhancedDataset(Dataset):
    """Enhanced dataset with multiple input formats"""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 256, is_training: bool = True):
        self.data = self._load_and_preprocess(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
    def _load_and_preprocess(self, data_path: Path):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        processed = []
        for sample_id, sample in raw_data.items():
            # Create multiple input formats
            formats = self._create_input_formats(sample)
            
            processed.append({
                "id": sample_id,
                "formats": formats,
                "label": sample.get("average"),
                "homonym": sample["homonym"]
            })
        
        return processed
    
    def _create_input_formats(self, sample):
        """Create multiple input formats for better generalization"""
        homonym = sample["homonym"]
        sentence = sample["sentence"]
        
        formats = []
        
        # Format 1: Standard
        formats.append(
            f"Context: {sample['precontext']}\n"
            f"Sentence: {sentence}\n"
            f"Ending: {sample.get('ending', '')}\n\n"
            f"Target Word: {homonym}\n"
            f"Definition: {sample['judged_meaning']}"
        )
        
        # Format 2: Question style
        formats.append(
            f"Given the context: {sample['precontext']}\n"
            f"In the sentence: '{sentence}'\n"
            f"How plausible is this definition of '{homonym}': {sample['judged_meaning']}?\n"
            f"Additional context: {sample.get('ending', '')}"
        )
        
        return formats
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # During training, randomly select one format
        # During inference, use the first format
        if self.is_training:
            format_idx = random.randint(0, len(item["formats"]) - 1)
            text = item["formats"][format_idx]
        else:
            text = item["formats"][0]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Prepare labels
        label = torch.tensor(item["label"], dtype=torch.float32) if item["label"] is not None else torch.tensor(-1.0)
        
        # For classification (rounded to nearest integer, 1-5 -> 0-4)
        if item["label"] is not None:
            class_label = torch.tensor(int(round(item["label"])) - 1, dtype=torch.long)
        else:
            class_label = torch.tensor(-1, dtype=torch.long)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "regression_label": label,
            "classification_label": class_label,
            "id": item["id"],
            "homonym": item["homonym"]
        }

# ==================== MODEL ====================
class ContrastiveLoss(nn.Module):
    """Supervised contrastive loss with numerical stability fixes"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8  # Added for numerical stability
    
    def forward(self, features, labels):
        batch_size = features.size(0)
        
        # Skip if batch too small
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        # Normalize features with epsilon for stability
        features_norm = torch.norm(features, p=2, dim=1, keepdim=True)
        features = features / (features_norm + self.eps)
        
        # Compute similarity matrix with temperature
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same or similar labels)
        # Use discretized labels for stability
        discrete_labels = torch.round(labels).long()
        label_matrix = discrete_labels.unsqueeze(0) == discrete_labels.unsqueeze(1)
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        label_matrix = label_matrix & (~mask)
        
        # Convert to float for multiplication
        positives = label_matrix.float()
        
        # Skip if no positive pairs
        if positives.sum() < 1:
            return torch.tensor(0.0, device=features.device)
        
        # Compute loss with numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits) + self.eps
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)
        
        # Compute mean of log-likelihood over positive pairs
        positive_counts = positives.sum(1) + self.eps
        mean_log_prob_pos = (positives * log_prob).sum(1) / positive_counts
        
        # Loss (negative mean log probability)
        loss = -mean_log_prob_pos[positive_counts > 0].mean()
        
        return loss

class MultiTaskDeBERTa(nn.Module):
    """DeBERTa with multi-task learning"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Base model
        self.deberta = AutoModel.from_pretrained(
            config.model_name,
            output_hidden_states=True,
            attention_probs_dropout_prob=config.dropout,
            hidden_dropout_prob=config.dropout
        )
        hidden_size = self.deberta.config.hidden_size
        
        # Projection for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.5),
            nn.Linear(256, 1)
        )
        
        # Classification head (5 classes: 1-5)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 5)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.regressor, self.classifier, self.projection]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids, attention_mask, return_features: bool = False):
        # Get DeBERTa outputs
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use mean pooling of last 4 layers
        hidden_states = outputs.hidden_states
        last_four = torch.stack(hidden_states[-4:], dim=0)
        mean_hidden = torch.mean(last_four, dim=0)
        
        # Attention pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(mean_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Get outputs
        regression_output = self.regressor(pooled_output).squeeze(-1)
        classification_logits = self.classifier(pooled_output)
        
        if return_features:
            contrastive_features = self.projection(pooled_output)
            return regression_output, classification_logits, contrastive_features
        else:
            return regression_output, classification_logits, None

# ==================== TRAINER ====================
class SingleFoldTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Loss functions
        self.regression_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.contrastive_loss = ContrastiveLoss()
        
    def set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def create_train_val_split(self, dataset, val_split: float = 0.2):
        """Create train/validation split stratified by homonym"""
        # Group indices by homonym
        homonym_groups = {}
        for idx, item in enumerate(dataset.data):
            homonym = item["homonym"]
            if homonym not in homonym_groups:
                homonym_groups[homonym] = []
            homonym_groups[homonym].append(idx)
        
        # Split each homonym group
        train_indices = []
        val_indices = []
        
        for homonym, indices in homonym_groups.items():
            n_val = max(1, int(len(indices) * val_split))
            random.shuffle(indices)
            
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])
        
        return train_indices, val_indices
    
    def train_epoch(self, model, dataloader, optimizer, scheduler, epoch: int):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        reg_loss_total = 0
        cls_loss_total = 0
        con_loss_total = 0
        steps = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        for step, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            reg_labels = batch["regression_label"].to(self.device)
            cls_labels = batch["classification_label"].to(self.device)
            
            # Filter out padding labels
            mask = reg_labels != -1
            if mask.sum() == 0:
                continue
                
            input_ids = input_ids[mask]
            attention_mask = attention_mask[mask]
            reg_labels = reg_labels[mask]
            cls_labels = cls_labels[mask]
            
            # Skip if batch too small after filtering
            if len(reg_labels) < 2:
                continue
            
            # Forward pass
            reg_pred, cls_logits, features = model(
                input_ids, attention_mask, return_features=True
            )
            
            # Compute losses
            reg_loss = self.regression_loss(reg_pred, reg_labels)
            cls_loss = self.classification_loss(cls_logits, cls_labels)
            con_loss = self.contrastive_loss(features, reg_labels)
            
            # Weighted total loss
            total_batch_loss = (
                self.config.regression_weight * reg_loss +
                self.config.classification_weight * cls_loss +
                self.config.contrastive_weight * con_loss
            )
            
            # Check for NaN loss
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                print(f"Warning: NaN/Inf loss at step {step}, skipping batch")
                optimizer.zero_grad()
                continue
            
            # Backward pass with gradient accumulation
            total_batch_loss = total_batch_loss / self.config.gradient_accumulation_steps
            total_batch_loss.backward()
            
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                steps += 1
            
            # Update metrics
            total_loss += total_batch_loss.item() * self.config.gradient_accumulation_steps
            reg_loss_total += reg_loss.item()
            cls_loss_total += cls_loss.item()
            con_loss_total += con_loss.item()
            
            progress_bar.set_postfix({
                "loss": total_batch_loss.item(),
                "reg": reg_loss.item(),
                "cls": cls_loss.item(),
                "lr": scheduler.get_last_lr()[0]
            })
        
        if steps == 0:
            return 0, 0, 0, 0
            
        avg_loss = total_loss / steps
        avg_reg_loss = reg_loss_total / steps
        avg_cls_loss = cls_loss_total / steps
        avg_con_loss = con_loss_total / steps
        
        return avg_loss, avg_reg_loss, avg_cls_loss, avg_con_loss
    
    def evaluate(self, model, dataloader):
        """Evaluate model"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["regression_label"].to(self.device)
                
                # Filter valid labels
                mask = labels != -1
                if mask.sum() == 0:
                    continue
                
                input_ids = input_ids[mask]
                attention_mask = attention_mask[mask]
                labels = labels[mask]
                
                # Get predictions
                preds, _, _ = model(input_ids, attention_mask, return_features=False)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        if len(all_preds) < 2:
            return {
                "spearman": 0.0,
                "spearman_p": 1.0,
                "accuracy": 0.0,
                "mse": 0.0
            }
        
        # Compute metrics
        try:
            spearman_corr, spearman_p = spearmanr(all_preds, all_labels)
        except:
            spearman_corr, spearman_p = 0.0, 1.0
        
        # Accuracy within tolerance
        preds_array = np.array(all_preds)
        labels_array = np.array(all_labels)
        within_one = np.mean(np.abs(preds_array - labels_array) <= 1.0)
        
        metrics = {
            "spearman": float(spearman_corr),
            "spearman_p": float(spearman_p),
            "accuracy": float(within_one),
            "mse": float(np.mean((preds_array - labels_array) ** 2))
        }
        
        return metrics
    
    def predict(self, model, dataloader):
        """Generate predictions"""
        model.eval()
        predictions = []
        ids = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_ids = batch["id"]
                
                # Get predictions
                preds, _, _ = model(input_ids, attention_mask, return_features=False)
                
                predictions.extend(preds.cpu().numpy())
                ids.extend(batch_ids)
        
        return predictions, ids
    
    def calibrate_predictions(self, predictions):
        """Simple calibration to match training distribution"""
        # Clip to valid range
        predictions = np.clip(predictions, 1.0, 5.0)
        
        # Round to nearest 0.5 (can help accuracy)
        predictions = np.round(predictions * 2) / 2
        
        return predictions
    
    def train(self, train_data_path: Path):
        """Main training function"""
        print(f"\n{'='*60}")
        print("STARTING SINGLE FOLD TRAINING")
        print(f"{'='*60}")
        
        # Set seed
        self.set_seed(self.config.random_seed)
        
        # Load dataset
        print("Loading data...")
        dataset = EnhancedDataset(
            train_data_path,
            self.tokenizer,
            self.config.max_length,
            is_training=True
        )
        
        print(f"Total samples: {len(dataset)}")
        
        # Create train/validation split
        train_indices, val_indices = self.create_train_val_split(dataset, self.config.val_split)
        
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        
        # Create data loaders
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Initialize model
        print("\nInitializing model...")
        model = MultiTaskDeBERTa(self.config).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Optimizer with parameter groups
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.config.learning_rate
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters)
        
        # Scheduler with linear warmup
        total_steps = len(train_loader) * self.config.epochs // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        print(f"\nTraining for {self.config.epochs} epochs...")
        print(f"Using device: {self.device}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Gradient clipping: {self.config.max_grad_norm}")
        print(f"Warmup steps: {warmup_steps}")
        
        best_val_spearman = 0
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
        
        for epoch in range(self.config.epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"{'='*60}")
            
            # Training
            train_loss, train_reg, train_cls, train_con = self.train_epoch(
                model, train_loader, optimizer, scheduler, epoch
            )
            
            # Skip evaluation if training failed
            if train_loss == 0:
                print("Training failed, skipping epoch")
                break
            
            # Validation
            val_metrics = self.evaluate(model, val_loader)
            
            print(f"\nTraining Results:")
            print(f"  Loss: {train_loss:.4f} (Reg: {train_reg:.4f}, Cls: {train_cls:.4f}, Con: {train_con:.4f})")
            print(f"\nValidation Results:")
            print(f"  Spearman: {val_metrics['spearman']:.4f} (p={val_metrics['spearman_p']:.2e})")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  MSE: {val_metrics['mse']:.4f}")
            
            # Early stopping check
            if val_metrics['spearman'] > best_val_spearman + self.config.min_delta:
                best_val_spearman = val_metrics['spearman']
                patience_counter = 0
                best_epoch = epoch + 1
                best_model_state = model.state_dict().copy()
                
                print(f"\n✅ New best model! Spearman: {best_val_spearman:.4f}")
                
                # Ensure directory exists before saving
                self.config.model_dir.mkdir(exist_ok=True, parents=True)
                
                # Save best model
                model_path = self.config.model_dir / f"best_model_epoch_{epoch+1}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_spearman': best_val_spearman,
                    'epoch': epoch + 1
                }, model_path)
                print(f"✅ Model saved to: {model_path}")
            else:
                patience_counter += 1
                print(f"\n⏳ No improvement for {patience_counter} epoch(s)")
                
                if patience_counter >= self.config.patience:
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\n✅ Loaded best model from epoch {best_epoch} with Spearman: {best_val_spearman:.4f}")
        else:
            print("\n⚠️ No best model found, using final model")
        
        return model, best_val_spearman
    
    def save_predictions(self, predictions, ids, output_path: Path):
        """Save predictions to file"""
        # Ensure directory exists
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Calibrate predictions
        calibrated = self.calibrate_predictions(predictions)
        
        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            for pred_id, pred in zip(ids, calibrated):
                f.write(json.dumps({
                    "id": int(pred_id),
                    "prediction": float(pred)
                }) + "\n")
        
        print(f"Predictions saved to: {output_path}")

# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function"""
    print("="*60)
    print("SEMEVAL TASK: Contextual Sense Plausibility")
    print("Single Fold Training with DeBERTa-v3-large")
    print("="*60)
    
    # Initialize configuration
    config = Config()
    
    # Check for data files
    required_files = ["train.json", "dev.json"]
    missing_files = []
    for file in required_files:
        if not (config.data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing required data files: {missing_files}")
        print(f"Looking in directory: {config.data_dir.absolute()}")
        print("Please place train.json and dev.json in the data/ directory")
        return
    
    # Initialize trainer
    trainer = SingleFoldTrainer(config)
    
    # Train model
    model, best_spearman = trainer.train(config.data_dir / "train.json")
    
    # Ensure directory exists before saving final model
    config.model_dir.mkdir(exist_ok=True, parents=True)
    
    # Save final model
    model_path = config.model_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_spearman': best_spearman,
        'config': config.__dict__
    }, model_path)
    print(f"\n✅ Final model saved to: {model_path}")
    
    # Predict on dev set
    print(f"\n{'='*60}")
    print("GENERATING PREDICTIONS")
    print(f"{'='*60}")
    
    # Load dev dataset
    dev_dataset = EnhancedDataset(
        config.data_dir / "dev.json",
        trainer.tokenizer,
        config.max_length,
        is_training=False
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Generate predictions
    dev_predictions, dev_ids = trainer.predict(model, dev_loader)
    
    # Save predictions
    dev_output_path = config.pred_dir / "predictions_dev.jsonl"
    trainer.save_predictions(dev_predictions, dev_ids, dev_output_path)
    
    # Evaluate dev set if labels are available
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Validation Spearman: {best_spearman:.4f}")
    print(f"Dev predictions saved to: {dev_output_path}")
    
    # If test set exists, generate predictions for it too
    test_path = config.data_dir / "test.json"
    if test_path.exists():
        print(f"\nGenerating test predictions...")
        
        test_dataset = EnhancedDataset(
            test_path,
            trainer.tokenizer,
            config.max_length,
            is_training=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        test_predictions, test_ids = trainer.predict(model, test_loader)
        
        test_output_path = config.pred_dir / "predictions_test.jsonl"
        trainer.save_predictions(test_predictions, test_ids, test_output_path)
        
        print(f"Test predictions saved to: {test_output_path}")
    else:
        print(f"\nNote: test.json not found in {config.data_dir}, skipping test predictions")

# ==================== QUICK TRAIN ====================
def quick_train():
    """Quick training with reduced parameters for testing"""
    print("Quick training mode (reduced parameters)...")
    
    config = Config()
    config.epochs = 3
    config.batch_size = 2
    config.gradient_accumulation_steps = 2
    config.learning_rate = 1e-6
    config.model_name = "microsoft/deberta-v3-small"  # Smaller model for testing
    config.contrastive_weight = 0.0  # Disable contrastive loss for quick test
    
    trainer = SingleFoldTrainer(config)
    trainer.set_seed(config.random_seed)
    
    # Load dataset
    dataset = EnhancedDataset(
        config.data_dir / "train.json",
        trainer.tokenizer,
        config.max_length,
        is_training=True
    )
    
    # Simple split (first 80% train, last 20% validation)
    split_idx = int(0.8 * len(dataset))
    train_indices = list(range(split_idx))
    val_indices = list(range(split_idx, len(dataset)))
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # Create data loaders
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = MultiTaskDeBERTa(config).to(config.device)
    
    # Simple optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    # Train for 3 epochs
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        model.train()
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            reg_labels = batch["regression_label"].to(config.device)
            cls_labels = batch["classification_label"].to(config.device)
            
            mask = reg_labels != -1
            if mask.sum() == 0:
                continue
            
            input_ids = input_ids[mask]
            attention_mask = attention_mask[mask]
            reg_labels = reg_labels[mask]
            cls_labels = cls_labels[mask]
            
            optimizer.zero_grad()
            reg_pred, cls_logits, _ = model(input_ids, attention_mask)
            
            reg_loss = nn.MSELoss()(reg_pred, reg_labels)
            cls_loss = nn.CrossEntropyLoss()(cls_logits, cls_labels)
            loss = reg_loss + 0.2 * cls_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Evaluate
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(config.device)
                attention_mask = batch["attention_mask"].to(config.device)
                labels = batch["regression_label"].to(config.device)
                
                mask = labels != -1
                if mask.sum() == 0:
                    continue
                
                input_ids = input_ids[mask]
                attention_mask = attention_mask[mask]
                labels = labels[mask]
                
                preds, _, _ = model(input_ids, attention_mask)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        if len(all_preds) > 1:
            spearman_corr, _ = spearmanr(all_preds, all_labels)
            print(f"Validation Spearman: {spearman_corr:.4f}")
    
    return model

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_train()
    else:
        main()

# PS C:\Users\barat\OneDrive - SSN-Institute\Desktop\semeval26-05-scripts> python evaluate.py predictions/predictions_dev_int_id.jsonl dev
# on data/dev.json...
# ----------
# Spearman Correlation: 0.49232172084030135
# Spearman p-Value: 3.180047964992087e-37
# ----------
# Accuracy: 0.6479591836734694 (381/588)
# PS C:\Users\barat\OneDrive - SSN-Institute\Desktop\semeval26-05-scripts>      
