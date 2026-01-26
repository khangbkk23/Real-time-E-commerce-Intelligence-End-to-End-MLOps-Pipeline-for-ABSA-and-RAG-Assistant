import pandas as pd
import torch
import numpy as np
import random
import os, sys
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.PhoBERT import PhoBERTSentiment, PhoBERTTokenizer
from training.trainer import PhoBERTTrainer
from training.evaluator import PhoBERTEvaluator
from utils.config import load_config

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            [text],
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    set_seed(cfg.system.seed)

    print(f"Loading data from {cfg.data.train_path}")
    df = pd.read_csv(cfg.data.train_path).dropna(subset=['clean_comment', 'label'])
    
    X = df['clean_comment'].values
    y = df['label'].values

    # 1. Split Train/Val first (Stratified)
    X_train_raw, X_val, y_train_raw, y_val = train_test_split(
        X, y, 
        test_size=cfg.data.test_size, 
        random_state=cfg.system.seed, 
        stratify=y
    )

    # 2. Apply RandomOverSampler ONLY on Train set
    print("Applying RandomOverSampler to Train set...")
    ros = RandomOverSampler(random_state=cfg.system.seed)
    # Reshape X to 2D for imblearn, then flatten back
    X_train_res, y_train_res = ros.fit_resample(X_train_raw.reshape(-1, 1), y_train_raw)
    X_train_res = X_train_res.flatten()

    print(f"Train size after resampling: {len(X_train_res)}")

    tokenizer = PhoBERTTokenizer()
    model = PhoBERTSentiment(
        model_name=cfg.model.name,
        num_labels=cfg.model.num_labels, 
        dropout=cfg.model.dropout
    )
    model.freeze_encoder(num_layers_to_freeze=cfg.model.freeze_layers)
    
    if cfg.system.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.system.device

    evaluator = PhoBERTEvaluator(model, device, tokenizer)
    trainer = PhoBERTTrainer(model, tokenizer, evaluator, device, cfg.data.output_dir)
    
    # Create Datasets with resampled Train and original Val
    train_ds = SentimentDataset(X_train_res, y_train_res, tokenizer, max_length=cfg.data.max_length)
    val_ds = SentimentDataset(X_val, y_val, tokenizer, max_length=cfg.data.max_length)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.system.num_workers
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.system.num_workers
    )
    
    trainer.train(
        train_loader, 
        val_loader, 
        epochs=cfg.training.epochs, 
        lr=float(cfg.training.learning_rate),
        patience=cfg.training.patience
    )