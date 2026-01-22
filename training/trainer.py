import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import os
import json
from typing import Dict, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.PhoBERT import PhoBERTSentiment, PhoBERTTokenizer


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


class PhoBERTTrainer:
    def __init__(
        self,
        model: PhoBERTSentiment,
        tokenizer: PhoBERTTokenizer,
        device: str = None,
        output_dir: str = "checkpoints"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
    
    def prepare_data(
        self,
        csv_path: str,
        text_col: str = "clean_comment",
        label_col: str = "label",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        df = pd.read_csv(csv_path)
        
        print(f"Dataset info:")
        print(f"  Total: {len(df)}")
        print(f"  Columns: {df.columns.tolist()}")
        
        texts = df[text_col].values
        labels = df[label_col].values
        
        print(f"\nLabel distribution:")
        for label_id, count in enumerate(np.bincount(labels)):
            label_name = ['Negative', 'Neutral', 'Positive'][label_id]
            print(f"  {label_name} ({label_id}): {count} ({count/len(labels)*100:.1f}%)")
        
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        print(f"\nSplit:")
        print(f"  Train: {len(X_train)}")
        print(f"  Val: {len(X_val)}")
        
        return X_train, X_val, y_train, y_val
    
    def create_dataloaders(
        self,
        X_train, X_val, y_train, y_val,
        batch_size: int = 16,
        max_length: int = 256
    ):
        train_dataset = SentimentDataset(X_train, y_train, self.tokenizer, max_length)
        val_dataset = SentimentDataset(X_val, y_val, self.tokenizer, max_length)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == "cuda" else False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1, all_labels, all_preds
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        save_best: bool = True
    ):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        best_f1 = 0
        
        print(f"\n{'='*60}")
        print(f"🚀 TRAINING START")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*60}")
            
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc, val_f1, labels, preds = self.evaluate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            print(f"\n📊 Metrics:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            print(f"  Val F1: {val_f1:.4f}")
            
            print(f"\n📈 Classification Report:")
            print(classification_report(
                labels, preds, 
                target_names=['Negative (0)', 'Neutral (1)', 'Positive (2)'],
                digits=4
            ))
            
            print(f"🔢 Confusion Matrix:")
            cm = confusion_matrix(labels, preds)
            print(cm)
            
            if save_best and val_f1 > best_f1:
                best_f1 = val_f1
                self.save_checkpoint(f"best_model_f1_{val_f1:.4f}")
                print(f"\n✅ Saved best model (F1: {val_f1:.4f})")
            
            self.save_checkpoint(f"epoch_{epoch + 1}")
        
        self.save_history()
        
        print(f"\n{'='*60}")
        print(f"🎉 TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"Best F1 Score: {best_f1:.4f}")
        print(f"Model saved to: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, name: str):
        path = os.path.join(self.output_dir, f"{name}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"✅ Loaded checkpoint from {path}")
    
    def save_history(self):
        path = os.path.join(self.output_dir, 'history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"📊 Saved training history to {path}")


if __name__ == "__main__":
    CSV_PATH = "datasets/preprocessed/clean_dataset_robot_hút_bụi_lau_nhà_1769048720.csv"
    
    print("="*60)
    print("PHOBERT SENTIMENT ANALYSIS TRAINING")
    print("="*60)
    
    print("\n1. Initializing model...")
    tokenizer = PhoBERTTokenizer()
    model = PhoBERTSentiment(num_labels=3, dropout=0.3)
    
    print("\n2. Freezing encoder layers...")
    model.freeze_encoder(num_layers_to_freeze=8)
    
    print("\n3. Creating trainer...")
    trainer = PhoBERTTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir="checkpoints/phobert_sentiment"
    )
    
    print("\n4. Loading and preparing data...")
    X_train, X_val, y_train, y_val = trainer.prepare_data(
        csv_path=CSV_PATH,
        text_col="clean_comment",
        label_col="label",
        test_size=0.2
    )
    
    print("\n5. Creating dataloaders...")
    train_loader, val_loader = trainer.create_dataloaders(
        X_train, X_val, y_train, y_val,
        batch_size=16,
        max_length=256
    )
    
    print("\n6. Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        save_best=True
    )