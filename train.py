import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class ModelTrainer:
    """
    模型訓練器，負責模型的訓練、評估和預測
    """
    def __init__(self, model, device='cuda', learning_rate=2e-5):
        """
        初始化訓練器
        
        Args:
            model: 要訓練的模型
            device: 使用的裝置（'cuda'或'cpu'）
            learning_rate: 學習率
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.label_encoder = LabelEncoder()
    
    def train(self, train_dataloader, epochs=5, verbose=True):
        """
        訓練模型
        
        Args:
            train_dataloader: 訓練資料載入器
            epochs: 訓練輪數
            verbose: 是否顯示進度條
            
        Returns:
            訓練損失歷史
        """
        self.model.train()
        loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # 使用tqdm產生進度條
            progress_bar = tqdm(
                enumerate(train_dataloader), 
                total=len(train_dataloader),
                desc=f'Epoch {epoch+1}/{epochs}',
                disable=not verbose
            )
            
            for step, batch_data in progress_bar:
                # 將資料移至對應裝置
                input_ids_bert = batch_data["input_ids_bert"].to(self.device)
                attention_mask_bert = batch_data["attention_mask_bert"].to(self.device)
                input_ids_roberta = batch_data["input_ids_roberta"].to(self.device)
                attention_mask_roberta = batch_data["attention_mask_roberta"].to(self.device)
                liwc = batch_data["liwc"].to(self.device)
                label = batch_data["label"].unsqueeze(1).to(self.device)
                
                # 模型推理
                if hasattr(self.model, 'bert_model') and hasattr(self.model, 'roberta_model'):
                    # 集成模型需要兩個輸入
                    outputs = self.model(
                        input_ids_bert, attention_mask_bert,
                        input_ids_roberta, attention_mask_roberta,
                        liwc
                    )
                else:
                    # 單一模型
                    if hasattr(self.model, 'bert'):
                        outputs = self.model(input_ids_bert, attention_mask_bert, liwc)
                    else:
                        outputs = self.model(input_ids_roberta, attention_mask_roberta, liwc)
                
                # 計算損失
                loss = self.criterion(outputs, label)
                epoch_loss += loss.item()
                
                # 更新進度條資訊
                progress_bar.set_postfix({'loss': loss.item()})
                
                # 反向傳播與優化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # 記錄平均損失
            avg_loss = epoch_loss / len(train_dataloader)
            loss_history.append(avg_loss)
            
            if verbose:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
        
        return loss_history
    
    def evaluate(self, test_dataloader, verbose=True):
        """
        評估模型性能
        
        Args:
            test_dataloader: 測試資料載入器
            verbose: 是否顯示進度條
            
        Returns:
            評估結果字典
        """
        self.model.eval()
        predictions = []
        labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(test_dataloader), 
                total=len(test_dataloader),
                desc='Evaluating',
                disable=not verbose
            )
            
            for step, batch_data in progress_bar:
                # 將資料移至對應裝置
                input_ids_bert = batch_data["input_ids_bert"].to(self.device)
                attention_mask_bert = batch_data["attention_mask_bert"].to(self.device)
                input_ids_roberta = batch_data["input_ids_roberta"].to(self.device)
                attention_mask_roberta = batch_data["attention_mask_roberta"].to(self.device)
                liwc = batch_data["liwc"].to(self.device)
                batch_label = batch_data["label"].unsqueeze(1).to(self.device)
                
                # 模型推理
                if hasattr(self.model, 'bert_model') and hasattr(self.model, 'roberta_model'):
                    # 集成模型需要兩個輸入
                    outputs = self.model(
                        input_ids_bert, attention_mask_bert,
                        input_ids_roberta, attention_mask_roberta,
                        liwc
                    )
                else:
                    # 單一模型
                    if hasattr(self.model, 'bert'):
                        outputs = self.model(input_ids_bert, attention_mask_bert, liwc)
                    else:
                        outputs = self.model(input_ids_roberta, attention_mask_roberta, liwc)
                
                # 收集預測和標籤
                for i in outputs:
                    predictions.append(np.array(i.to("cpu")))
                
                for i in batch_label:
                    labels.append(np.array(i.to("cpu")))
        
        # 處理預測結果
        predictions = np.array(predictions).reshape(-1, 4)
        
        # 二值化預測結果（四個維度：I/E, S/N, T/F, J/P）
        pred_I_E = predictions[:, 0].round()
        pred_S_N = predictions[:, 1].round()
        pred_T_F = predictions[:, 2].round()
        pred_J_P = predictions[:, 3].round()
        
        # 處理標籤
        labels = np.array(labels).reshape(-1, 4)
        label_I_E = labels[:, 0]
        label_S_N = labels[:, 1]
        label_T_F = labels[:, 2]
        label_J_P = labels[:, 3]
        
        # 計算每個維度的準確度
        acc_I_E = accuracy_score(label_I_E, pred_I_E)
        acc_S_N = accuracy_score(label_S_N, pred_S_N)
        acc_T_F = accuracy_score(label_T_F, pred_T_F)
        acc_J_P = accuracy_score(label_J_P, pred_J_P)
        
        # 計算整體準確度（完全匹配）
        results_df = pd.DataFrame({
            "pred_I/E": pred_I_E,
            "pred_S/N": pred_S_N,
            "pred_T/F": pred_T_F,
            "pred_J/P": pred_J_P,
            "label_I/E": label_I_E,
            "label_S/N": label_S_N,
            "label_T/F": label_T_F,
            "label_J/P": label_J_P
        })
        
        # 將數值轉換為MBTI標記
        pred_I_E_char = results_df["pred_I/E"].apply(lambda x: "I" if x == 1 else "E")
        pred_S_N_char = results_df["pred_S/N"].apply(lambda x: "S" if x == 1 else "N")
        pred_T_F_char = results_df["pred_T/F"].apply(lambda x: "T" if x == 1 else "F")
        pred_J_P_char = results_df["pred_J/P"].apply(lambda x: "J" if x == 1 else "P")
        
        label_I_E_char = results_df["label_I/E"].apply(lambda x: "I" if x == 1 else "E")
        label_S_N_char = results_df["label_S/N"].apply(lambda x: "S" if x == 1 else "N")
        label_T_F_char = results_df["label_T/F"].apply(lambda x: "T" if x == 1 else "F")
        label_J_P_char = results_df["label_J/P"].apply(lambda x: "J" if x == 1 else "P")
        
        # 組合為完整的MBTI類型
        combine_pred = pred_I_E_char + pred_S_N_char + pred_T_F_char + pred_J_P_char
        combine_label = label_I_E_char + label_S_N_char + label_T_F_char + label_J_P_char
        
        # 編碼為數值以計算準確度
        self.label_encoder.fit(np.concatenate([combine_pred, combine_label]))
        combine_pred_encoded = self.label_encoder.transform(combine_pred)
        combine_label_encoded = self.label_encoder.transform(combine_label)
        
        # 計算整體準確度
        overall_acc = accuracy_score(combine_label_encoded, combine_pred_encoded)
        
        # 結果整理
        evaluation_results = {
            "I/E_accuracy": acc_I_E,
            "S/N_accuracy": acc_S_N,
            "T/F_accuracy": acc_T_F,
            "J/P_accuracy": acc_J_P,
            "overall_accuracy": overall_acc
        }
        
        if verbose:
            print(f"I/E Accuracy: {acc_I_E:.4f}")
            print(f"S/N Accuracy: {acc_S_N:.4f}")
            print(f"T/F Accuracy: {acc_T_F:.4f}")
            print(f"J/P Accuracy: {acc_J_P:.4f}")
            print(f"Overall Accuracy: {overall_acc:.4f}")
        
        return evaluation_results
    
    def save_model(self, path):
        """
        儲存模型
        
        Args:
            path: 儲存路徑
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        載入模型
        
        Args:
            path: 模型路徑
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model = self.model.to(self.device)
        print(f"Model loaded from {path}") 