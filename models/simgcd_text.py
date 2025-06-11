# SimGCD文本模型 - 结合BERT编码器与SimGCD架构

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .bert_encoder import BertEncoder, BertFeatureExtractor

class TextDINOHead(nn.Module):
    """文本版本的DINO投影头"""
    
    def __init__(self, in_dim: int, out_dim: int, use_bn: bool = False, 
                 norm_last_layer: bool = True, nlayers: int = 3, 
                 hidden_dim: int = 2048, bottleneck_dim: int = 256):
        """
        初始化DINO投影头
        Args:
            in_dim: 输入特征维度
            out_dim: 输出类别数
            use_bn: 是否使用BatchNorm
            norm_last_layer: 是否归一化最后一层
            nlayers: MLP层数
            hidden_dim: 隐藏层维度
            bottleneck_dim: 瓶颈层维度
        """
        super().__init__()
        nlayers = max(nlayers, 1)
        
        # 构建MLP投影层
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 最后的分类层
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
    
    def _init_weights(self, m):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, in_dim]
        Returns:
            x_proj: 投影特征 [batch_size, bottleneck_dim]
            logits: 分类logits [batch_size, out_dim]
        """
        # MLP投影
        x_proj = self.mlp(x)
        
        # 特征归一化
        x_norm = F.normalize(x, dim=-1, p=2)
        
        # 分类logits
        logits = self.last_layer(x_norm)
        
        return x_proj, logits

class TextSupConLoss(nn.Module):
    """文本版本的监督对比学习损失"""
    
    def __init__(self, temperature: float = 0.07, contrast_mode: str = 'all',
                 base_temperature: float = 0.07):
        """
        初始化监督对比学习损失
        Args:
            temperature: 温度参数
            contrast_mode: 对比模式 ('all' 或 'one')
            base_temperature: 基础温度参数
        """
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算监督对比学习损失
        Args:
            features: 特征张量 [bsz, n_views, feature_dim]
            labels: 标签 [bsz]
            mask: 对比掩码 [bsz, bsz]
        Returns:
            损失值
        """
        device = features.device
        
        if len(features.shape) < 3:
            raise ValueError('特征需要至少3个维度 [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        batch_size = features.shape[0]
        
        # 构建掩码
        if labels is not None and mask is not None:
            raise ValueError('不能同时定义labels和mask')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('标签数量与特征数量不匹配')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'未知模式: {self.contrast_mode}')
        
        # 计算相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # 扩展掩码
        mask = mask.repeat(anchor_count, contrast_count)
        
        # 掩盖自对比情况
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 计算log概率
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算正样本的平均log似然
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # 损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss

class TextDistillLoss(nn.Module):
    """文本版本的知识蒸馏损失"""
    
    def __init__(self, warmup_teacher_temp_epochs: int, nepochs: int,
                 ncrops: int = 2, warmup_teacher_temp: float = 0.07,
                 teacher_temp: float = 0.04, student_temp: float = 0.1):
        """
        初始化知识蒸馏损失
        Args:
            warmup_teacher_temp_epochs: 教师温度预热轮数
            nepochs: 总训练轮数
            ncrops: 数据增强视图数
            warmup_teacher_temp: 预热教师温度
            teacher_temp: 教师温度
            student_temp: 学生温度
        """
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        
        # 教师温度调度
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
    
    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor, 
                epoch: int) -> torch.Tensor:
        """
        计算知识蒸馏损失
        Args:
            student_output: 学生网络输出
            teacher_output: 教师网络输出
            epoch: 当前训练轮数
        Returns:
            蒸馏损失
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        
        # 教师网络中心化和锐化
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        
        total_loss = 0
        n_loss_terms = 0
        
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # 跳过学生和教师在同一视图上操作的情况
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        return total_loss

class SimGCDText(nn.Module):
    """SimGCD文本模型主类"""
    
    def __init__(self, 
                 bert_model_path: str = './bert/final_model',
                 num_labeled_classes: int = 10,
                 num_unlabeled_classes: int = 5,
                 feature_dim: int = 512,
                 bert_hidden_size: int = 768,
                 freeze_bert: bool = False,
                 pooling_strategy: str = 'cls',
                 proj_dim: int = 256):
        """
        初始化SimGCD文本模型
        Args:
            bert_model_path: BERT模型路径
            num_labeled_classes: 已知类别数
            num_unlabeled_classes: 未知类别数
            feature_dim: 特征维度
            bert_hidden_size: BERT隐藏层大小
            freeze_bert: 是否冻结BERT
            pooling_strategy: 池化策略
            proj_dim: 投影维度
        """
        super().__init__()
        
        self.num_labeled_classes = num_labeled_classes
        self.num_unlabeled_classes = num_unlabeled_classes
        self.total_classes = num_labeled_classes + num_unlabeled_classes
        
        # BERT编码器
        bert_encoder = BertEncoder(
            model_name_or_path=bert_model_path,
            hidden_size=bert_hidden_size,
            freeze_bert=freeze_bert,
            pooling_strategy=pooling_strategy
        )
        
        # BERT特征提取器
        self.feature_extractor = BertFeatureExtractor(
            bert_encoder=bert_encoder,
            feature_dim=feature_dim,
            dropout_rate=0.1
        )
        
        # DINO投影头
        self.head = TextDINOHead(
            in_dim=feature_dim,
            out_dim=self.total_classes,
            bottleneck_dim=proj_dim
        )
        
        # 损失函数
        self.sup_con_loss = TextSupConLoss(temperature=0.07)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            return_features: 是否返回特征
        Returns:
            包含logits和特征的字典
        """
        # 特征提取
        features = self.feature_extractor(input_ids, attention_mask)
        
        # 投影和分类
        proj_features, logits = self.head(features)
        
        result = {
            'logits': logits,
            'proj_features': proj_features
        }
        
        if return_features:
            result['features'] = features
        
        return result
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     labels: torch.Tensor, is_labeled: torch.Tensor,
                     epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        计算损失
        Args:
            outputs: 模型输出
            labels: 标签
            is_labeled: 是否为标记数据
            epoch: 当前轮数
        Returns:
            损失字典
        """
        logits = outputs['logits']
        proj_features = outputs['proj_features']
        
        # 分离标记和未标记数据
        labeled_mask = is_labeled.bool()
        unlabeled_mask = ~labeled_mask
        
        losses = {}
        
        # 标记数据的交叉熵损失
        if labeled_mask.sum() > 0:
            labeled_logits = logits[labeled_mask]
            labeled_labels = labels[labeled_mask]
            ce_loss = F.cross_entropy(labeled_logits, labeled_labels)
            losses['ce_loss'] = ce_loss
        
        # 对比学习损失（使用投影特征）
        if proj_features.dim() == 2:
            # 为对比学习添加视图维度
            proj_features = proj_features.unsqueeze(1)
        
        if labeled_mask.sum() > 0:
            labeled_features = proj_features[labeled_mask]
            labeled_labels_for_contrast = labels[labeled_mask]
            contrast_loss = self.sup_con_loss(labeled_features, labeled_labels_for_contrast)
            losses['contrast_loss'] = contrast_loss
        
        # 总损失
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return self.feature_extractor.get_feature_dim()

# 测试
if __name__ == '__main__':
    # 测试SimGCD文本模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = SimGCDText(
        bert_model_path='./bert/final_model',
        num_labeled_classes=10,
        num_unlabeled_classes=5,
        feature_dim=512,
        freeze_bert=False
    )
    
    model.to(device)
    
    # 测试数据
    batch_size = 8
    seq_len = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    is_labeled = torch.tensor([True, True, False, False, True, False, True, False]).to(device)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, return_features=True)
        print(f"Logits shape: {outputs['logits'].shape}")
        print(f"Projected features shape: {outputs['proj_features'].shape}")
        print(f"Features shape: {outputs['features'].shape}")
        
        # 计算损失
        losses = model.compute_loss(outputs, labels, is_labeled)
        print(f"Losses: {losses}")
    
    print("SimGCD文本模型测试完成！")