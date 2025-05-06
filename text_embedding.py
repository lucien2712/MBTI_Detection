import torch
from transformers import BertTokenizer, RobertaTokenizer

class TextEmbedder:
    def __init__(self, max_length=20):
        """
        初始化文本嵌入工具
        
        Args:
            max_length: 每篇文章的最大長度
        """
        self.max_length = max_length
        
        # 初始化BERT和RoBERTa的tokenizer
        self.tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
        
        # 添加特殊的[Link]標記
        self.tokenizer_bert.add_tokens("[Link]")
        self.tokenizer_roberta.add_tokens("[Link]")
    
    def embed_text_bert(self, data):
        """
        使用BERT對文本進行嵌入
        
        Args:
            data: 包含已處理文本的資料集
            
        Returns:
            擴充了BERT嵌入特徵的資料集
        """
        data["tokenization_bert"] = data["posts"].apply(
            lambda x: self.tokenizer_bert(
                x, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length
            )
        )
        
        data["input_ids_bert"] = data["tokenization_bert"].apply(lambda x: x["input_ids"])
        data["attention_mask_bert"] = data["tokenization_bert"].apply(lambda x: x["attention_mask"])
        
        return data
    
    def embed_text_roberta(self, data):
        """
        使用RoBERTa對文本進行嵌入
        
        Args:
            data: 包含已處理文本的資料集
            
        Returns:
            擴充了RoBERTa嵌入特徵的資料集
        """
        data["tokenization_roberta"] = data["posts"].apply(
            lambda x: self.tokenizer_roberta(
                x, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length
            )
        )
        
        data["input_ids_roberta"] = data["tokenization_roberta"].apply(lambda x: x["input_ids"])
        data["attention_mask_roberta"] = data["tokenization_roberta"].apply(lambda x: x["attention_mask"])
        
        return data
    
    def embed_text(self, data):
        """
        對文本進行BERT和RoBERTa嵌入
        
        Args:
            data: 包含已處理文本的資料集
            
        Returns:
            擴充了詞嵌入特徵的資料集
        """
        data = self.embed_text_bert(data)
        data = self.embed_text_roberta(data)
        
        return data 