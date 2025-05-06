# MBTI 人格類型檢測系統

這個專案利用自然語言處理和深度學習技術，基於用戶的文本數據推測其MBTI（Myers-Briggs Type Indicator）人格類型。系統採用BERT和RoBERTa模型進行文本分析，結合LIWC（Linguistic Inquiry and Word Count）語言特徵進行人格預測。

## 專案特色
- 結合BERT-CNN和RoBERTa-CNN模型，提高預測準確率
- 將MBTI的四個維度（I/E, S/N, T/F, J/P）作為多標籤分類任務
- 結合詞向量和LIWC語言特徵，全面捕捉文本數據中的人格線索


## 專案架構

```
MBTI_detection/
│
├── data_preprocessing.py - 資料預處理模組
├── feature_extraction.py - LIWC特徵提取模組
├── text_embedding.py     - 文本嵌入模組
├── dataset.py            - 資料集和資料載入器模組
├── models.py             - 模型架構定義
├── train.py              - 模型訓練和評估模組
├── utils.py              - 工具函數模組
├── main.py               - 主程式入口
├── requirements.txt      - 依賴項列表
└── README.md             - 項目說明文件
```


## 數據需求

- MBTI資料集（CSV格式，包含 'type' 和 'posts' 列）
- LIWC字典文件 (LIWC2007_English100131.dic)

## 使用方法

### 訓練模型

```bash
python main.py --mode train --model_type bert --data_path path/to/mbti_dataset.csv --liwc_dict_path path/to/liwc_dict.dic --max_posts 40 --epochs 5 --batch_size 16 --balance_dataset
```

### 評估模型

```bash
python main.py --mode evaluate --model_type bert --data_path path/to/mbti_dataset.csv --liwc_dict_path path/to/liwc_dict.dic
```

### 使用集成模型

```bash
# 先訓練BERT模型
python main.py --mode train --model_type bert --data_path path/to/mbti_dataset.csv --liwc_dict_path path/to/liwc_dict.dic --epochs 5

# 再訓練RoBERTa模型
python main.py --mode train --model_type roberta --data_path path/to/mbti_dataset.csv --liwc_dict_path path/to/liwc_dict.dic --epochs 5

# 最後使用集成模型
python main.py --mode train --model_type ensemble --data_path path/to/mbti_dataset.csv --liwc_dict_path path/to/liwc_dict.dic --epochs 3
```

## 主要參數說明

- `--data_path`: MBTI資料集路徑
- `--liwc_dict_path`: LIWC字典文件路徑
- `--model_type`: 模型類型 (bert, roberta, ensemble)
- `--max_posts`: 每個用戶保留的最大文章數
- `--max_length`: 每篇文章的最大長度
- `--batch_size`: 批次大小
- `--epochs`: 訓練輪數
- `--learning_rate`: 學習率
- `--output_dir`: 輸出目錄
- `--model_dir`: 模型保存目錄
- `--balance_dataset`: 是否平衡資料集
- `--mode`: 運行模式 (train, evaluate, predict)

## 資料預處理流程

1. 文本清理：轉小寫、移除停用詞、移除標點符號等
2. 特殊處理：將網址替換為特殊標記 [Link]、移除數字
3. LIWC特徵提取：從文本提取LIWC語言特徵
4. 文本嵌入：使用BERT和RoBERTa進行文本嵌入
5. 資料集分割：將資料分割為訓練集和測試集

## 模型架構

### BERT-CNN 模型

使用BERT提取文本特徵，然後通過CNN進行特徵整合，結合LIWC特徵進行多標籤分類。

### RoBERTa-CNN 模型

使用RoBERTa提取文本特徵，然後通過CNN進行特徵整合，結合LIWC特徵進行多標籤分類。

### 集成模型

結合已訓練的BERT-CNN和RoBERTa-CNN模型，進行加權集成預測。

## 輸出結果

訓練和評估結果會保存在指定的輸出目錄中，包括：

- 模型權重文件
- 訓練損失曲線圖
- 評估結果（四個維度和整體的準確率）
- MBTI類型分布圖
