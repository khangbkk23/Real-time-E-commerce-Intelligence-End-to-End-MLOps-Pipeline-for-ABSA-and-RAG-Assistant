import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.PhoBERT import PhoBERTSentiment, PhoBERTTokenizer
from utils.config import load_config

class SentimentPredictor:
    def __init__(self, model_path="checkpoints/phobert_v3_yaml/best_model.pt", config_path="config.yaml"):
        self.cfg = load_config(os.path.join(parent_dir, config_path))
        
        if self.cfg.system.device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = self.cfg.system.device
            
        self.device = torch.device(device_str)
        print(f"Inference running on: {self.device}")
        self.tokenizer = PhoBERTTokenizer()
        
        self.model = PhoBERTSentiment(
            model_name=self.cfg.model.name,
            num_labels=self.cfg.model.num_labels,
            dropout=self.cfg.model.dropout
        )
        full_path = os.path.join(parent_dir, model_path)
        print(f"Loading model weights from {full_path}...")
        
        try:
            checkpoint = torch.load(full_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            # ---------------------------------
            
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def predict(self, text):
        clean_text = str(text).strip()

        encoding = self.tokenizer(
            [clean_text],
            max_length=self.cfg.data.max_length,
            padding='max_length',
            truncation=True
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Get result
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        labels_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        return labels_map[predicted_class.item()], confidence.item()

if __name__ == "__main__":
    predictor = SentimentPredictor()
    
    test_sentences = [
        "Sản phẩm dùng chán quá, pin yếu.",
        "Giao hàng nhanh, đóng gói cẩn thận.",
        "Hàng tạm được, không có gì đặc sắc.",
        "Tai nghe này nghe cũng bình thường thôi",
        "Đừng mua phí tiền lắm"
    ]
    
    print("\n" + "="*40)
    print("🧪 TEST INFERENCE")
    print("="*40)
    
    for s in test_sentences:
        label, score = predictor.predict(s)
        icon = "🔴" if label == "Negative" else "🟢" if label == "Positive" else "⚪"
        print(f"📝 Text: {s}")
        print(f"   {icon} Result: {label} ({score:.4f})\n")