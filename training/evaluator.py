import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm.auto import tqdm

class PhoBERTEvaluator:
    def __init__(self, model, device, tokenizer):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer

    def evaluate(self, dataloader, criterion=None):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                if criterion:
                    loss = criterion(outputs['logits'], labels)
                    total_loss += loss.item()
                elif outputs['loss'] is not None:
                    total_loss += outputs['loss'].item()
                
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        
        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        report = classification_report(
            all_labels, all_preds, 
            target_names=['Negative', 'Neutral', 'Positive'],
            digits=4,
            zero_division=0
        )
        
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'loss': avg_loss,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'report': report,
            'confusion_matrix': cm,
            'preds': all_preds,
            'labels': all_labels
        }