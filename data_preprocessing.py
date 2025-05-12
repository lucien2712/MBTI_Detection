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

        data = pd.read_csv(file_path, encoding='unicode_escape')
        return data
    
    def preprocess_data(self, data):

        # 轉小寫
        data["posts"] = data["posts"].apply(lambda x: x.lower())
        # 文章間原先用|||隔開
        data["posts"] = data["posts"].apply(lambda x: x.split("|||"))
        return data
    
    def remove_stopwords(self, data):

        def stopword_removal(x):
            new_x = []
            for sent in x:
                c = " ".join([i for i in sent.split() if i not in self.stop_words])
                new_x.append(c)
            return new_x
        
        data["posts"] = data["posts"].apply(stopword_removal)
        return data
    
    def remove_punctuations(self, data):

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

        def links_token(x):
            result = []
            for word in x:
                filtered_sentence = re.sub(r'https?\S+', ' [Link] ', word)
                result.append(filtered_sentence)
            return result
        
        data["posts"] = data["posts"].apply(links_token)
        return data
    
    def remove_numbers(self, data):

        def numbers_token(x):
            result = []
            for word in x:
                filtered_sentence = re.sub(r'\d+', ' ', word)
                result.append(filtered_sentence)
            return result
        
        data["posts"] = data["posts"].apply(numbers_token)
        return data
    
    def clean_text(self, data):

        def tidy(x):
            result = []
            for i in x:
                a = i.strip()
                result.append(a)
            return result
        
        data["posts"] = data["posts"].apply(tidy)
        return data
    
    def remove_empty(self, data):

        def blank(x):
            result = []
            for i in x:
                if len(i) > 0:
                    result.append(i)
            return result
        
        data["posts"] = data["posts"].apply(blank)
        return data
    
    def encode_mbti_labels(self, data):

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