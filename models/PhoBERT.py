import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional

class PhoBERTSentiment(nn.Module):
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_labels: int = 3,
        dropout: float = 0.3,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.phobert = AutoModel.from_pretrained(model_name)
        self.config = self.phobert.config
        
        bert_dim = self.config.hidden_size
        hidden_dim = hidden_dim or bert_dim // 2
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )
        
        self.num_labels = num_labels
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': cls_output
        }
    
    def freeze_encoder(self, num_layers_to_freeze: int = 8):
        for param in self.phobert.embeddings.parameters():
            param.requires_grad = False
        
        for i, layer in enumerate(self.phobert.encoder.layer):
            if i < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.phobert.parameters():
            param.requires_grad = True


class PhoBERTTokenizer:
    def __init__(self, model_name: str = "vinai/phobert-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def __call__(
        self,
        texts: list,
        max_length: int = 256,
        padding: str = "max_length",
        truncation: bool = True
    ):
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)