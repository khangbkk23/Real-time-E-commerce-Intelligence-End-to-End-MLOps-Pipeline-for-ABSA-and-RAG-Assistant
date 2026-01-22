import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import numpy as np

def plot_history(history_path, output_dir):
    """Vẽ biểu đồ Loss và F1-Score từ lịch sử huấn luyện"""
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file lịch sử: {history_path}")
        return

    # Tạo DataFrame cho dễ vẽ
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))

    # 1. Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. Biểu đồ F1-Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_f1'], 'g-o', label='Validation F1-Score')
    plt.title('Validation F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(save_path)
    print(f"📊 Đã lưu biểu đồ Training tại: {save_path}")
    plt.close()

def plot_confusion_matrix(cm, classes, output_dir):
    """Vẽ Ma trận nhầm lẫn (Confusion Matrix)"""
    plt.figure(figsize=(8, 6))
    
    # Tính phần trăm để dễ nhìn
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title('Confusion Matrix')
    plt.ylabel('Thực tế (True Label)')
    plt.xlabel('Dự đoán (Predicted Label)')
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"Đã lưu Confusion Matrix tại: {save_path}")
    plt.close()