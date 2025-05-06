import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
import os

def create_directories(directories):
    """
    創建目錄
    
    Args:
        directories: 要創建的目錄列表
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def plot_loss_history(loss_history, save_path=None):
    """
    繪製訓練損失曲線
    
    Args:
        loss_history: 訓練損失歷史
        save_path: 圖形保存路徑，如果為None則顯示圖形
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, marker='o')
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrices(predictions, labels, save_dir=None):
    """
    繪製每個MBTI維度的混淆矩陣
    
    Args:
        predictions: 模型預測結果 (N, 4)
        labels: 真實標籤 (N, 4)
        save_dir: 保存目錄，如果為None則顯示圖形
    """
    # 二值化預測結果
    pred_I_E = predictions[:, 0].round()
    pred_S_N = predictions[:, 1].round()
    pred_T_F = predictions[:, 2].round()
    pred_J_P = predictions[:, 3].round()
    
    # 真實標籤
    label_I_E = labels[:, 0]
    label_S_N = labels[:, 1]
    label_T_F = labels[:, 2]
    label_J_P = labels[:, 3]
    
    # 定義標籤
    dims = [('I/E', ['E', 'I']), ('S/N', ['N', 'S']), ('T/F', ['F', 'T']), ('J/P', ['P', 'J'])]
    preds = [pred_I_E, pred_S_N, pred_T_F, pred_J_P]
    true_labels = [label_I_E, label_S_N, label_T_F, label_J_P]
    
    for i, (dim_name, class_names) in enumerate(dims):
        cm = confusion_matrix(true_labels[i], preds[i])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {dim_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'confusion_matrix_{dim_name}.png'))
            plt.close()
        else:
            plt.show()

def get_mbti_distribution(data, column='type'):
    """
    獲取MBTI類型分布統計
    
    Args:
        data: 資料集
        column: 類型列名
        
    Returns:
        MBTI類型計數
    """
    mbti_counts = data[column].value_counts()
    return mbti_counts

def plot_mbti_distribution(data, column='type', save_path=None):
    """
    繪製MBTI類型分布圖
    
    Args:
        data: 資料集
        column: 類型列名
        save_path: 保存路徑，如果為None則顯示圖形
    """
    mbti_counts = get_mbti_distribution(data, column)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=mbti_counts.index, y=mbti_counts.values)
    plt.title('MBTI Type Distribution')
    plt.xlabel('MBTI Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def load_trained_model(model_class, model_path, device='cuda'):
    """
    載入已訓練的模型
    
    Args:
        model_class: 模型類別
        model_path: 模型檔案路徑
        device: 使用的裝置
        
    Returns:
        載入的模型
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_mbti(model, tokenizer, text, device='cuda'):
    """
    使用模型預測MBTI類型
    
    Args:
        model: 訓練好的模型
        tokenizer: 文本標記器
        text: 要預測的文本
        device: 使用的裝置
        
    Returns:
        預測的MBTI類型
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 對文本進行預處理
    processed_text = preprocess_text(text)
    
    # 將文本轉換為模型輸入格式
    model_inputs = tokenizer(processed_text, return_tensors="pt", padding="max_length", 
                             truncation=True, max_length=20)
    
    # 將輸入移到指定裝置
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    
    # 模型推理
    with torch.no_grad():
        output = model(input_ids, attention_mask)
    
    # 處理輸出
    prediction = output.cpu().numpy().round()[0]
    
    # 將數值轉換為MBTI字符
    mbti_type = ""
    mbti_type += "I" if prediction[0] == 1 else "E"
    mbti_type += "S" if prediction[1] == 1 else "N"
    mbti_type += "T" if prediction[2] == 1 else "F"
    mbti_type += "J" if prediction[3] == 1 else "P"
    
    return mbti_type

def preprocess_text(text):
    """
    預處理文本
    
    Args:
        text: 原始文本
        
    Returns:
        預處理後的文本
    """
    # 實現與主要預處理步驟相同的邏輯
    # 這是一個簡化版，實際使用時應參考DataPreprocessor中的詳細實現
    text = text.lower()
    return text 