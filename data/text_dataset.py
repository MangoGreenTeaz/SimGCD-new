import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional

class TextDataset(Dataset):
    #处理CSV的文本数据，使用BERT tokenizer进行文本预处理
    
    def __init__(self, 
                 csv_path: str,
                 bert_model_path: str,
                 max_length: int = 512,
                 is_labeled: bool = True,
                 transform=None):
                
        self.csv_path = csv_path
        self.max_length = max_length
        self.is_labeled = is_labeled
        self.transform = transform
        
        # 初始化BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        
        # 加载和预处理数据
        self._load_data()
        
    def _load_data(self):
        # 加载CSV数据并根据labeled字段进行筛选
        df = pd.read_csv(self.csv_path)
        
        # 根据is_labeled参数筛选数据
        if self.is_labeled:
            # 筛选labeled为True的数据
            self.data = df[df['labeled'] == True].reset_index(drop=True)
        else:
            # 筛选labeled为False的数据
            self.data = df[df['labeled'] == False].reset_index(drop=True)
            
        print(f"加载了 {len(self.data)} 条{'labeled' if self.is_labeled else 'unlabeled'}数据")
        
        # 获取唯一标签数量（用于后续模型配置）
        if self.is_labeled:
            self.num_classes = len(self.data['label'].unique())
            print(f"发现 {self.num_classes} 个不同的类别")
        else:
            self.num_classes = None
            
    def __len__(self) -> int:
        # 返回数据集大小
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 获取单个数据样本
        # 获取文本和标签
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label'] if self.is_labeled else -1
        udid = self.data.iloc[idx]['udid']
        
        # 应用文本增强
        if self.transform:
            text = self.transform(text)
            
        # BERT tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 构建返回字典
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'udid': udid,
            'text': text  # 保留原始文本用于调试
        }
        
        return item
    
    def get_class_distribution(self) -> Dict[int, int]:
        #获取类别分布统计
        if not self.is_labeled:
            return {}
            
        class_counts = self.data['label'].value_counts().to_dict()
        return class_counts
    
    def get_text_length_stats(self) -> Dict[str, float]:
        #获取文本长度统计信息
        text_lengths = [len(text.split()) for text in self.data['text']]
        
        stats = {
            'mean_length': np.mean(text_lengths),
            'max_length': np.max(text_lengths),
            'min_length': np.min(text_lengths),
            'std_length': np.std(text_lengths)
        }
        
        return stats

def create_data_loaders(csv_path: str, 
                       bert_model_path: str,
                       batch_size: int = 32,
                       max_length: int = 512,
                       num_workers: int = 4,
                       transform=None) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    # 创建labeled数据集
    labeled_dataset = TextDataset(
        csv_path=csv_path,
        bert_model_path=bert_model_path,
        max_length=max_length,
        is_labeled=True,
        transform=transform
    )
    
    # 创建unlabeled数据集
    unlabeled_dataset = TextDataset(
        csv_path=csv_path,
        bert_model_path=bert_model_path,
        max_length=max_length,
        is_labeled=False,
        transform=transform
    )
    
    # 创建DataLoader
    labeled_loader = torch.utils.data.DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return labeled_loader, unlabeled_loader

if __name__ == "__main__":
    # 测试代码
    csv_path = "data.csv"
    bert_path = "bert/final_model"
    
    # 创建数据集
    labeled_dataset = TextDataset(csv_path, bert_path, is_labeled=True)
    unlabeled_dataset = TextDataset(csv_path, bert_path, is_labeled=False)
    
    # 打印统计信息
    print("Labeled数据集统计:")
    print(f"数据量: {len(labeled_dataset)}")
    print(f"类别分布: {labeled_dataset.get_class_distribution()}")
    print(f"文本长度统计: {labeled_dataset.get_text_length_stats()}")
    
    print("\nUnlabeled数据集统计:")
    print(f"数据量: {len(unlabeled_dataset)}")
    print(f"文本长度统计: {unlabeled_dataset.get_text_length_stats()}")
    
    # 测试单个样本
    sample = labeled_dataset[0]
    print(f"\n样本示例:")
    print(f"文本: {sample['text'][:100]}...")
    print(f"标签: {sample['label']}")
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"attention_mask shape: {sample['attention_mask'].shape}")