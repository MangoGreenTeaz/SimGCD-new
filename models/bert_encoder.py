# BERT编码器模块，用于将文本转换为特征向量

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Dict, Any, Optional

class BertEncoder(nn.Module):
    # BERT编码器类，用于文本特征提取
    
    def __init__(self, 
                 model_path: str = './bert/final_model',
                 hidden_size: int = 768,
                 freeze_bert: bool = False,
                 pooling_strategy: str = 'cls'):

        # 初始化BERT编码器
        super(BertEncoder, self).__init__()
        
        # 加载BERT模型
        self.bert = BertModel.from_pretrained(model_path)
        self.hidden_size = hidden_size
        self.pooling_strategy = pooling_strategy
        
        # 是否冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 特征投影层（如果需要调整维度）
        bert_hidden_size = self.bert.config.hidden_size
        if bert_hidden_size != hidden_size:
            self.projection = nn.Linear(bert_hidden_size, hidden_size)
        else:
            self.projection = nn.Identity()
        
        # 归一化层
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        Returns:
            文本特征向量 [batch_size, hidden_size]
        """
        # BERT编码
        outputs = self.bert(input_ids=input_ids, 
                           attention_mask=attention_mask)
        
        # 获取序列输出
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, bert_hidden_size]
        
        # 根据池化策略提取特征
        if self.pooling_strategy == 'cls':
            # 使用[CLS]标记的表示
            pooled_output = sequence_output[:, 0, :]  # [batch_size, bert_hidden_size]
        elif self.pooling_strategy == 'mean':
            # 使用平均池化
            pooled_output = self._mean_pooling(sequence_output, attention_mask)
        elif self.pooling_strategy == 'max':
            # 使用最大池化
            pooled_output = self._max_pooling(sequence_output, attention_mask)
        else:
            raise ValueError(f"不支持的池化策略: {self.pooling_strategy}")
        
        # 特征投影
        features = self.projection(pooled_output)
        
        # 归一化
        features = self.layer_norm(features)
        
        return features
    
    def _mean_pooling(self, sequence_output: torch.Tensor, 
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """
        平均池化
        Args:
            sequence_output: 序列输出 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        Returns:
            池化后的特征 [batch_size, hidden_size]
        """
        # 扩展attention_mask维度
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        
        # 计算加权平均
        sum_embeddings = torch.sum(sequence_output * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def _max_pooling(self, sequence_output: torch.Tensor, 
                     attention_mask: torch.Tensor) -> torch.Tensor:
        """
        最大池化
        Args:
            sequence_output: 序列输出 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        Returns:
            池化后的特征 [batch_size, hidden_size]
        """
        # 将padding位置设为很小的值
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sequence_output[attention_mask_expanded == 0] = -1e9
        
        # 最大池化
        max_embeddings = torch.max(sequence_output, 1)[0]
        
        return max_embeddings
    
    def get_embedding_dim(self) -> int:
        # 获取输出特征维度
        return self.hidden_size

class BertFeatureExtractor(nn.Module):
    """BERT特征提取器，包含额外的特征处理层"""
    
    def __init__(self, 
                 bert_encoder: BertEncoder,
                 feature_dim: int = 512,
                 dropout_rate: float = 0.1):
        """
        初始化特征提取器
        Args:
            bert_encoder: BERT编码器
            feature_dim: 输出特征维度
            dropout_rate: Dropout率
        """
        super(BertFeatureExtractor, self).__init__()
        
        self.bert_encoder = bert_encoder
        self.feature_dim = feature_dim
        
        # 特征变换层
        self.feature_transform = nn.Sequential(
            nn.Linear(bert_encoder.get_embedding_dim(), feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
        Returns:
            提取的特征向量
        """
        # BERT编码
        bert_features = self.bert_encoder(input_ids, attention_mask)
        
        # 特征变换
        features = self.feature_transform(bert_features)
        
        return features
    
    def get_feature_dim(self) -> int:
        # 获取输出特征维度
        return self.feature_dim

# 测试
if __name__ == '__main__':
    # 测试BERT编码器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建BERT编码器
    bert_encoder = BertEncoder(
        model_path='./bert/final_model',
        hidden_size=768,
        freeze_bert=False,
        pooling_strategy='cls'
    )
    
    # 创建特征提取器
    feature_extractor = BertFeatureExtractor(
        bert_encoder=bert_encoder,
        feature_dim=512,
        dropout_rate=0.1
    )
    
    # 移动到设备
    feature_extractor.to(device)
    
    # 测试数据
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    
    # 前向传播测试
    with torch.no_grad():
        features = feature_extractor(input_ids, attention_mask)
        print(f"输入形状: {input_ids.shape}")
        print(f"输出特征形状: {features.shape}")
        print(f"特征维度: {feature_extractor.get_feature_dim()}")
        
    print("BERT编码器测试完成！")