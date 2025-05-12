import argparse
import os
import torch
import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from feature_extraction import LIWCFeatureExtractor
from text_embedding import TextEmbedder
from dataset import DatasetBuilder
from models import BERTCNNModel, RoBERTaCNNModel, EnsembleModel
from train import ModelTrainer
from utils import create_directories, plot_loss_history, plot_confusion_matrices, plot_mbti_distribution

def parse_args():
    
    parser = argparse.ArgumentParser(description='MBTI Personality Type Detection')
    
    # 資料參數
    parser.add_argument('--data_path', type=str, default='mbti_1.csv', help='資料集路徑')
    parser.add_argument('--liwc_dict_path', type=str, default='LIWC2007_English100131.dic', help='LIWC字典路徑')
    parser.add_argument('--max_posts', type=int, default=40, help='每個用戶保留的最大文章數')
    parser.add_argument('--test_size', type=float, default=0.2, help='測試集比例')
    
    # 模型參數
    parser.add_argument('--model_type', type=str, default='bert', choices=['bert', 'roberta', 'ensemble'], help='模型類型')
    parser.add_argument('--max_length', type=int, default=20, help='每篇文章的最大長度')
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=5, help='訓練輪數')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='學習率')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='訓練裝置')
    
    # 輸出參數
    parser.add_argument('--output_dir', type=str, default='output', help='輸出目錄')
    parser.add_argument('--model_dir', type=str, default='models', help='模型保存目錄')
    parser.add_argument('--balance_dataset', action='store_true', help='是否平衡資料集')
    
    # 模式選擇
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'predict'], help='運行模式')
    
    return parser.parse_args()

def main():
    
    # 解析命令行參數
    args = parse_args()
    
    # 創建輸出目錄
    create_directories([args.output_dir, args.model_dir])
    
    # 設定裝置
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 資料預處理
    if args.mode in ['train', 'evaluate']:
        print("Loading and preprocessing data...")
        preprocessor = DataPreprocessor()
        data = preprocessor.load_data(args.data_path)
        data = preprocessor.process_data(data, max_posts=args.max_posts)
        
        # 繪製MBTI類型分布圖
        plot_mbti_distribution(data, save_path=os.path.join(args.output_dir, 'mbti_distribution.png'))
        
        # 提取LIWC特徵
        print("Extracting LIWC features...")
        feature_extractor = LIWCFeatureExtractor(args.liwc_dict_path)
        data = feature_extractor.extract_liwc_features(data)
        
        # 獲取LIWC特徵列表
        liwc_columns = [col for col in data.columns if col not in [
            'type', 'posts', 'I/E', 'S/N', 'T/F', 'J/P',
            'input_ids_bert', 'attention_mask_bert',
            'input_ids_roberta', 'attention_mask_roberta',
            'tokenization_bert', 'tokenization_roberta'
        ]]
        
        # 文本嵌入
        print("Embedding text with transformers...")
        embedder = TextEmbedder(max_length=args.max_length)
        data = embedder.embed_text(data)
        
        # 資料集分割與平衡
        print("Preparing datasets...")
        dataset_builder = DatasetBuilder()
        
        # 如果需要平衡資料集
        if args.balance_dataset:
            print("Balancing dataset...")
            types_limit = {
                "INFP": 200, "INFJ": 200, "INTP": 200, "INTJ": 200,
                "ENTP": 250, "ENFP": 300
            }
            data = dataset_builder.balance_dataset(data, types_limit)
        
        # 分割資料集
        train_data, test_data = dataset_builder.split_data(data, test_size=args.test_size)
        
        # 創建PyTorch資料集和資料載入器
        train_dataset, test_dataset = dataset_builder.create_datasets(
            train_data, test_data, liwc_columns
        )
        
        train_dataloader, test_dataloader = dataset_builder.create_dataloaders(
            train_dataset, test_dataset, batch_size=args.batch_size
        )
        
        # 創建模型
        print(f"Creating {args.model_type.upper()} model...")
        if args.model_type == 'bert':
            model = BERTCNNModel(len(embedder.tokenizer_bert))
        elif args.model_type == 'roberta':
            model = RoBERTaCNNModel(len(embedder.tokenizer_roberta))
        elif args.model_type == 'ensemble':
            bert_model = BERTCNNModel(len(embedder.tokenizer_bert))
            roberta_model = RoBERTaCNNModel(len(embedder.tokenizer_roberta))
            
            # 載入已訓練的BERT和RoBERTa模型
            bert_model.load_state_dict(torch.load(os.path.join(args.model_dir, 'bert_model.pth')))
            roberta_model.load_state_dict(torch.load(os.path.join(args.model_dir, 'roberta_model.pth')))
            
            model = EnsembleModel(bert_model, roberta_model)
        
        # 訓練模型
        if args.mode == 'train':
            print("Training model...")
            trainer = ModelTrainer(model, device=args.device, learning_rate=args.learning_rate)
            loss_history = trainer.train(train_dataloader, epochs=args.epochs)
            
            # 繪製損失曲線
            plot_loss_history(loss_history, save_path=os.path.join(args.output_dir, f'{args.model_type}_loss.png'))
            
            # 儲存模型
            model_path = os.path.join(args.model_dir, f'{args.model_type}_model.pth')
            trainer.save_model(model_path)
            
            # 評估模型
            print("Evaluating model...")
            evaluation_results = trainer.evaluate(test_dataloader)
            
            # 保存評估結果
            with open(os.path.join(args.output_dir, f'{args.model_type}_evaluation.txt'), 'w') as f:
                for metric, value in evaluation_results.items():
                    f.write(f"{metric}: {value:.4f}\n")
        
        # 只評估模型
        elif args.mode == 'evaluate':
            print("Evaluating model...")
            model_path = os.path.join(args.model_dir, f'{args.model_type}_model.pth')
            
            if os.path.exists(model_path):
                trainer = ModelTrainer(model, device=args.device)
                trainer.load_model(model_path)
                evaluation_results = trainer.evaluate(test_dataloader)
                
                # 保存評估結果
                with open(os.path.join(args.output_dir, f'{args.model_type}_evaluation.txt'), 'w') as f:
                    for metric, value in evaluation_results.items():
                        f.write(f"{metric}: {value:.4f}\n")
            else:
                print(f"Model file not found: {model_path}")
    
    # 預測模式（在此為示例，實際上需要實現文本輸入和處理邏輯）
    elif args.mode == 'predict':
        print("Prediction mode is currently a placeholder. Please implement text input logic.")
    
    print("Done!")

if __name__ == "__main__":
    main() 