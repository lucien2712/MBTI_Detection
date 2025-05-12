import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MBTIDataset(Dataset):
    
    def __init__(self, label, input_ids_bert, attention_mask_bert, 
                 input_ids_roberta, attention_mask_roberta, liwc_features):

        self.label = label
        self.input_ids_bert = input_ids_bert
        self.attention_mask_bert = attention_mask_bert
        self.input_ids_roberta = input_ids_roberta
        self.attention_mask_roberta = attention_mask_roberta
        self.liwc_features = liwc_features

    def __len__(self):
      
        return len(self.label)

    def __getitem__(self, idx):
 
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
   
    def __init__(self):
        pass
    
    def balance_dataset(self, data, types_limit=None):

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
 
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
        
        return train_data, test_data
    
    def create_datasets(self, train_data, test_data, liwc_columns):

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

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_dataloader, test_dataloader 