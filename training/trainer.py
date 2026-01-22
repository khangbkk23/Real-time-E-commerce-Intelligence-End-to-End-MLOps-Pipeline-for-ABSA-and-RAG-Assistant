import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import os
import json
from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight

from training.focal_loss import FocalLoss

class PhoBERTTrainer:
    def __init__(self, model, tokenizer, evaluator, device=None, output_dir="checkpoints"):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.use_amp = self.device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1_macro': [],
            'val_f1_weighted': []
        }

    def compute_class_weights(self, train_loader):
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch['labels'].numpy())
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(all_labels),
            y=all_labels
        )
        
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        print(f"Class weights: {weights_tensor.cpu().numpy()}")
        return weights_tensor

    def train(self, train_loader, val_loader, epochs=5, lr=2e-5, patience=3):
        criterion = FocalLoss(gamma=2.0).to(self.device)
        
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        
        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        best_f1 = 0
        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {lr}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Early stopping patience: {patience}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            self.model.train()
            total_loss = 0
            progress = tqdm(train_loader, desc=f"Training")
            
            for batch in progress:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = criterion(outputs['logits'], labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = criterion(outputs['logits'], labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                
                total_loss += loss.item()
                progress.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_loss / len(train_loader)
            
            print(f"\nEvaluating on validation set...")
            metrics = self.evaluator.evaluate(val_loader, criterion)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(metrics['loss'])
            self.history['val_acc'].append(metrics['accuracy'])
            self.history['val_f1_macro'].append(metrics['f1_macro'])
            self.history['val_f1_weighted'].append(metrics['f1_weighted'])
            
            print(f"\nResults:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {metrics['loss']:.4f}")
            print(f"  Val Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Val F1 (Macro): {metrics['f1_macro']:.4f}")
            print(f"  Val F1 (Weighted): {metrics['f1_weighted']:.4f}")
            
            print(f"\n{metrics['report']}")
            
            print(f"Confusion Matrix:")
            print(metrics['confusion_matrix'])
            
            if metrics['f1_macro'] > best_f1:
                best_f1 = metrics['f1_macro']
                self.save_checkpoint("best_model")
                patience_counter = 0
                print(f"\nNew best model saved! (F1 Macro: {best_f1:.4f})")
            else:
                patience_counter += 1
                print(f"\nNo improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
            
            self.save_checkpoint(f"epoch_{epoch+1}")
        
        self.save_history()
        
        print(f"\n{'='*60}")
        print(f"Training Completed")
        print(f"{'='*60}")
        print(f"Best F1 Macro: {best_f1:.4f}")
        print(f"Model saved to: {self.output_dir}")
        print(f"{'='*60}\n")
                    
    def save_checkpoint(self, name):
        path = os.path.join(self.output_dir, f"{name}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        print(f"Checkpoint loaded from: {path}")
    
    def save_history(self):
        path = os.path.join(self.output_dir, 'history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to: {path}")