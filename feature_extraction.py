import pandas as pd
import numpy as np
import liwc
from collections import Counter

class LIWCFeatureExtractor:
    def __init__(self, liwc_dict_path):
       
        self.parse, self.category_names = liwc.load_token_parser(liwc_dict_path)
        
    def extract_liwc_features(self, data):
      
        # 將文章列表連接為單個字串，然後分割為單詞列表
        text_tokens = data["posts"].apply(lambda x: " ".join(x))
        text_tokens = text_tokens.apply(lambda x: x.split(" "))
        
        # 計算每個文檔的LIWC類別計數
        result = []
        for tokens in text_tokens:
            # 統計每個token對應的LIWC類別
            counts = Counter(category for token in tokens for category in self.parse(token))
            result.append(counts)
        
        # 將計數轉換為DataFrame
        liwc_df = pd.DataFrame(result)
        
        # 填充缺失值並轉換為浮點數
        liwc_df = liwc_df.fillna(0).astype("float32")
        
        # 標準化特徵
        liwc_df = (liwc_df - liwc_df.mean()) / liwc_df.std()
        
        # 將LIWC特徵連接到原始資料集
        data = pd.concat([data, liwc_df], axis=1)
        
        return data 