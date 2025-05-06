import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MBTIDataset(Dataset):
    """
    MBTI資料集類別，用於PyTorch的資料載入
    """
    def __init__(self, label, input_ids_bert, attention_mask_bert, 
                 input_ids_roberta, attention_mask_roberta, liwc_features):
        """
        初始化MBTI資料集
        
        Args:
            label: MBTI的四個維度標籤 (I/E, S/N, T/F, J/P)
            input_ids_bert: BERT的input_ids
            attention_mask_bert: BERT的attention_mask
            input_ids_roberta: RoBERTa的input_ids
            attention_mask_roberta: RoBERTa的attention_mask
            liwc_features: LIWC特徵
        """
        self.label = label
        self.input_ids_bert = input_ids_bert
        self.attention_mask_bert = attention_mask_bert
        self.input_ids_roberta = input_ids_roberta
        self.attention_mask_roberta = attention_mask_roberta
        self.liwc_features = liwc_features

    def __len__(self):
        """
        返回資料集大小
        """
        return len(self.label)

    def __getitem__(self, idx):
        """
        返回指定索引的樣本
        
        Args:
            idx: 索引
            
        Returns:
            包含標籤和特徵的字典
        """
        label = self.label[idx]
        input_ids_bert = self.input_ids_bert[idx]
        attention_mask_bert = self.attention_mask_bert[idx]
        input_ids_roberta = self.input_ids_roberta[idx]
        attention_mask_roberta = self.attention_mask_roberta[idx]
        liwc_features = self.liwc_features[idx]
        
        sample = {
            "label": label,
            "input_ids_bert": input_ids_bert,
            "attention_mask_bert": attention_mask_bert,
            "input_ids_roberta": input_ids_roberta,
            "attention_mask_roberta": attention_mask_roberta,
            "liwc": liwc_features
        }
        
        return sample

class DatasetBuilder:
    """
    資料集建立工具，負責將資料分割為訓練集和測試集
    """
    def __init__(self):
        pass
    
    def balance_dataset(self, data, types_limit=None):
        """
        平衡資料集，使各MBTI類型的樣本數量更加均衡
        
        Args:
            data: 原始資料集
            types_limit: 各MBTI類型的樣本數量限制字典，例如 {"INFP": 200, "INTJ": 200}
                         如果不提供，則保留所有樣本
        
        Returns:
            平衡後的資料集
        """
        if types_limit is None:
            return data
        
        balanced_dfs = []
        for mbti_type, limit in types_limit.items():
            type_df = data[data["type"] == mbti_type]
            if len(type_df) > limit:
                type_df = type_df.sample(limit, random_state=0)
            balanced_dfs.append(type_df)
        
        balanced_data = pd.concat(balanced_dfs, axis=0)
        balanced_data = balanced_data.sample(frac=1, random_state=0).reset_index(drop=True)
        
        return balanced_data
    
    def split_data(self, data, test_size=0.2, random_state=0):
        """
        將資料集分割為訓練集和測試集
        
        Args:
            data: 已處理的資料集
            test_size: 測試集比例
            random_state: 隨機種子
            
        Returns:
            訓練集和測試集
        """
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
        
        return train_data, test_data
    
    def create_datasets(self, train_data, test_data, liwc_columns):
        """
        創建PyTorch資料集
        
        Args:
            train_data: 訓練資料
            test_data: 測試資料
            liwc_columns: LIWC特徵列名列表
            
        Returns:
            訓練集和測試集的PyTorch資料集
        """
        train_dataset = MBTIDataset(
            train_data.loc[:, ["I/E", "S/N", "T/F", "J/P"]].values,
            train_data.loc[:, "input_ids_bert"].values,
            train_data.loc[:, "attention_mask_bert"].values,
            train_data.loc[:, "input_ids_roberta"].values,
            train_data.loc[:, "attention_mask_roberta"].values,
            train_data.loc[:, liwc_columns].values
        )
        
        test_dataset = MBTIDataset(
            test_data.loc[:, ["I/E", "S/N", "T/F", "J/P"]].values,
            test_data.loc[:, "input_ids_bert"].values,
            test_data.loc[:, "attention_mask_bert"].values,
            test_data.loc[:, "input_ids_roberta"].values,
            test_data.loc[:, "attention_mask_roberta"].values,
            test_data.loc[:, liwc_columns].values
        )
        
        return train_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset, test_dataset, batch_size=16):
        """
        創建PyTorch資料載入器
        
        Args:
            train_dataset: 訓練資料集
            test_dataset: 測試資料集
            batch_size: 批次大小
            
        Returns:
            訓練集和測試集的資料載入器
        """
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_dataloader, test_dataloader 