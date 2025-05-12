import torch
from transformers import BertTokenizer, RobertaTokenizer

class TextEmbedder:
    def __init__(self, max_length=20):
       
        self.max_length = max_length
        
        # 初始化BERT和RoBERTa的tokenizer
        self.tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
        
        # 添加特殊的[Link]標記
        self.tokenizer_bert.add_tokens("[Link]")
        self.tokenizer_roberta.add_tokens("[Link]")
    
    def embed_text_bert(self, data):
  
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
   
        data = self.embed_text_bert(data)
        data = self.embed_text_roberta(data)
        
        return data 