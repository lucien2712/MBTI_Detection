import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel

class MBTIModel(nn.Module):
    """
    MBTI分類模型的基礎類別
    """
    def __init__(self, tokenizer_vocab_size, liwc_dim=64):
        """
        初始化模型
        
        Args:
            tokenizer_vocab_size: tokenizer詞彙表大小
            liwc_dim: LIWC特徵的維度
        """
        super(MBTIModel, self).__init__()
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.liwc_dim = liwc_dim
    
    def forward(self, input_ids, attention_mask, liwc):
        """
        前向傳播
        
        Args:
            input_ids: 輸入的ID序列
            attention_mask: 注意力遮罩
            liwc: LIWC特徵
            
        Returns:
            模型輸出
        """
        raise NotImplementedError("子類別必須實現forward方法")

class BERTCNNModel(MBTIModel):
    """
    基於BERT-CNN的MBTI分類模型
    """
    def __init__(self, tokenizer_vocab_size, liwc_dim=64):
        """
        初始化BERT-CNN模型
        
        Args:
            tokenizer_vocab_size: BERT tokenizer詞彙表大小
            liwc_dim: LIWC特徵的維度
        """
        super(BERTCNNModel, self).__init__(tokenizer_vocab_size, liwc_dim)
        
        # 加載BERT模型
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.resize_token_embeddings(tokenizer_vocab_size)
        
        # 卷積層
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.maxpool = nn.MaxPool2d((3, 3))
        
        # 計算卷積輸出維度
        self.conv_output = 32 * 12 * 254
        
        # 全連接層
        self.dense_layer_1 = nn.Linear(self.conv_output + liwc_dim, 128)
        self.dense_layer_2 = nn.Linear(128, 4)
        
        # 激活函數
        self.act_sig = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask, liwc):
        """
        前向傳播
        
        Args:
            input_ids: BERT的input_ids
            attention_mask: BERT的attention_mask
            liwc: LIWC特徵
            
        Returns:
            模型預測結果
        """
        cls_tensor = None
        
        # 處理每篇文章
        for i in range(len(input_ids)):
            cls = self.bert(input_ids[i], attention_mask=attention_mask[i])[0][:, 0, :].unsqueeze(0)
            
            if i == 0:
                cls_tensor = cls.clone()
            else:
                cls_tensor = torch.cat([cls_tensor, cls], dim=0)
        
        # 增加通道維度
        cls_tensor = cls_tensor.unsqueeze(1)
        
        # 卷積層
        out = self.conv1(cls_tensor)
        out = self.conv2(out)
        out = self.maxpool(out)
        
        # 攤平並連接LIWC特徵
        out = out.view(out.shape[0], 1, self.conv_output)
        out = torch.cat([out, liwc.unsqueeze(1)], dim=2)
        
        # 全連接層
        out = self.dense_layer_1(out)
        out = self.dense_layer_2(out)
        
        # 輸出層
        out = self.act_sig(out)
        
        return out

class RoBERTaCNNModel(MBTIModel):
    """
    基於RoBERTa-CNN的MBTI分類模型
    """
    def __init__(self, tokenizer_vocab_size, liwc_dim=64):
        """
        初始化RoBERTa-CNN模型
        
        Args:
            tokenizer_vocab_size: RoBERTa tokenizer詞彙表大小
            liwc_dim: LIWC特徵的維度
        """
        super(RoBERTaCNNModel, self).__init__(tokenizer_vocab_size, liwc_dim)
        
        # 加載RoBERTa模型
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.roberta.resize_token_embeddings(tokenizer_vocab_size)
        
        # 卷積層
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.maxpool = nn.MaxPool2d((3, 3))
        
        # 計算卷積輸出維度
        self.conv_output = 32 * 12 * 254
        
        # 全連接層
        self.dense_layer_1 = nn.Linear(self.conv_output + liwc_dim, 128)
        self.dense_layer_2 = nn.Linear(128, 4)
        
        # 激活函數
        self.act_sig = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask, liwc):
        """
        前向傳播
        
        Args:
            input_ids: RoBERTa的input_ids
            attention_mask: RoBERTa的attention_mask
            liwc: LIWC特徵
            
        Returns:
            模型預測結果
        """
        cls_tensor = None
        
        # 處理每篇文章
        for i in range(len(input_ids)):
            cls = self.roberta(input_ids[i], attention_mask=attention_mask[i])[0][:, 0, :].unsqueeze(0)
            
            if i == 0:
                cls_tensor = cls.clone()
            else:
                cls_tensor = torch.cat([cls_tensor, cls], dim=0)
        
        # 增加通道維度
        cls_tensor = cls_tensor.unsqueeze(1)
        
        # 卷積層
        out = self.conv1(cls_tensor)
        out = self.conv2(out)
        out = self.maxpool(out)
        
        # 攤平並連接LIWC特徵
        out = out.view(out.shape[0], 1, self.conv_output)
        out = torch.cat([out, liwc.unsqueeze(1)], dim=2)
        
        # 全連接層
        out = self.dense_layer_1(out)
        out = self.dense_layer_2(out)
        
        # 輸出層
        out = self.act_sig(out)
        
        return out
        
class EnsembleModel(nn.Module):
    """
    集成模型，結合BERT-CNN和RoBERTa-CNN的預測結果
    """
    def __init__(self, bert_model, roberta_model):
        """
        初始化集成模型
        
        Args:
            bert_model: 已訓練的BERT-CNN模型
            roberta_model: 已訓練的RoBERTa-CNN模型
        """
        super(EnsembleModel, self).__init__()
        self.bert_model = bert_model
        self.roberta_model = roberta_model
        
        # 凍結子模型的參數
        for param in self.bert_model.parameters():
            param.requires_grad = False
        for param in self.roberta_model.parameters():
            param.requires_grad = False
        
        # 集成層（簡單的平均）
        self.ensemble_weights = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)
    
    def forward(self, input_ids_bert, attention_mask_bert, 
                input_ids_roberta, attention_mask_roberta, liwc):
        """
        前向傳播
        
        Args:
            input_ids_bert: BERT的input_ids
            attention_mask_bert: BERT的attention_mask
            input_ids_roberta: RoBERTa的input_ids
            attention_mask_roberta: RoBERTa的attention_mask
            liwc: LIWC特徵
            
        Returns:
            集成模型的預測結果
        """
        # 獲取BERT模型的預測
        bert_output = self.bert_model(input_ids_bert, attention_mask_bert, liwc)
        
        # 獲取RoBERTa模型的預測
        roberta_output = self.roberta_model(input_ids_roberta, attention_mask_roberta, liwc)
        
        # 正規化權重
        weights = torch.softmax(self.ensemble_weights, dim=0)
        
        # 加權平均
        ensemble_output = weights[0] * bert_output + weights[1] * roberta_output
        
        return ensemble_output 