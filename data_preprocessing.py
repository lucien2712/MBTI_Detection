import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 確保下載所需的NLTK資源
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def load_data(self, file_path):
        """
        讀取MBTI資料集
        
        Args:
            file_path: 資料集的檔案路徑
            
        Returns:
            處理後的資料集
        """
        data = pd.read_csv(file_path, encoding='unicode_escape')
        return data
    
    def preprocess_data(self, data):
        """
        預處理資料集，包括轉小寫、分割文章等
        
        Args:
            data: 原始資料集
            
        Returns:
            預處理後的資料集
        """
        # 轉小寫
        data["posts"] = data["posts"].apply(lambda x: x.lower())
        # 文章間原先用|||隔開
        data["posts"] = data["posts"].apply(lambda x: x.split("|||"))
        return data
    
    def remove_stopwords(self, data):
        """
        移除停用詞
        
        Args:
            data: 輸入資料集
            
        Returns:
            處理後的資料集
        """
        def stopword_removal(x):
            new_x = []
            for sent in x:
                c = " ".join([i for i in sent.split() if i not in self.stop_words])
                new_x.append(c)
            return new_x
        
        data["posts"] = data["posts"].apply(stopword_removal)
        return data
    
    def remove_punctuations(self, data):
        """
        移除標點符號
        
        Args:
            data: 輸入資料集
            
        Returns:
            處理後的資料集
        """
        def punctuations_removal(x):
            new_x = []
            for word in x:
                a = "".join([alpha for alpha in word if alpha not in string.punctuation])
                if a != "":
                    new_x.append(a)
            return new_x
        
        data["posts"] = data["posts"].apply(punctuations_removal)
        return data
    
    def tokenize_links(self, data):
        """
        將網址替換為 [Link] 標記
        
        Args:
            data: 輸入資料集
            
        Returns:
            處理後的資料集
        """
        def links_token(x):
            result = []
            for word in x:
                filtered_sentence = re.sub(r'https?\S+', ' [Link] ', word)
                result.append(filtered_sentence)
            return result
        
        data["posts"] = data["posts"].apply(links_token)
        return data
    
    def remove_numbers(self, data):
        """
        移除數字
        
        Args:
            data: 輸入資料集
            
        Returns:
            處理後的資料集
        """
        def numbers_token(x):
            result = []
            for word in x:
                filtered_sentence = re.sub(r'\d+', ' ', word)
                result.append(filtered_sentence)
            return result
        
        data["posts"] = data["posts"].apply(numbers_token)
        return data
    
    def clean_text(self, data):
        """
        清理文本，移除頭尾空白
        
        Args:
            data: 輸入資料集
            
        Returns:
            處理後的資料集
        """
        def tidy(x):
            result = []
            for i in x:
                a = i.strip()
                result.append(a)
            return result
        
        data["posts"] = data["posts"].apply(tidy)
        return data
    
    def remove_empty(self, data):
        """
        移除空白項目
        
        Args:
            data: 輸入資料集
            
        Returns:
            處理後的資料集
        """
        def blank(x):
            result = []
            for i in x:
                if len(i) > 0:
                    result.append(i)
            return result
        
        data["posts"] = data["posts"].apply(blank)
        return data
    
    def encode_mbti_labels(self, data):
        """
        多標籤編碼MBTI類型
        
        Args:
            data: 輸入資料集
            
        Returns:
            加入MBTI四維度標籤的資料集
        """
        I_E = []
        S_N = []
        T_F = []
        J_P = []
        
        for i in data["type"]:
            # 內向/外向
            if "I" in i:
                I_E.append(1)
            else:
                I_E.append(0)
            
            # 感覺/直覺
            if "S" in i:
                S_N.append(1)
            else:
                S_N.append(0)
            
            # 思考/情感
            if "T" in i:
                T_F.append(1)
            else:
                T_F.append(0)
            
            # 判斷/感知
            if "J" in i:
                J_P.append(1)
            else:
                J_P.append(0)
        
        data["I/E"] = I_E
        data["S/N"] = S_N
        data["T/F"] = T_F
        data["J/P"] = J_P
        
        return data
    
    def process_data(self, data, max_posts=40):
        """
        完整的資料處理流程
        
        Args:
            data: 原始資料集
            max_posts: 每個使用者要保留的最大文章數
            
        Returns:
            處理好的資料集
        """
        data = self.preprocess_data(data)
        data = self.remove_stopwords(data)
        data = self.remove_punctuations(data)
        data = self.tokenize_links(data)
        data = self.remove_numbers(data)
        data = self.clean_text(data)
        data = self.remove_empty(data)
        data = self.remove_stopwords(data)  # 再次去除停用詞
        
        # 只保留有超過max_posts篇文章的資料，並限制每個使用者的文章數
        data = data[data["posts"].apply(len) >= max_posts]
        data["posts"] = data["posts"].apply(lambda x: x[0:max_posts])
        
        data = self.encode_mbti_labels(data)
        
        return data 