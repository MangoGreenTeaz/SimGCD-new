# train_text.py

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import os
import sys
import pandas as pd
from typing import Dict, List, Tuple
import logging

from models.simgcd_text import SimGCDText
from data.text_dataset import TextDataset, create_data_loaders

class AverageMeter:
    """计算和存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TextContrastiveLearningViewGenerator:
    """文本对比学习的多视角生成器"""
    def __init__(self, n_views=2):
        self.n_views = n_views
    
    def __call__(self, text):
        # 简单的文本增强策略
        views = []
        for _ in range(self.n_views):
            # 这里可以实现更复杂的文本增强
            # 目前简单返回原文本
            views.append(text)
        return views

class TextDistillLoss(nn.Module):
    """文本版本的蒸馏损失"""
    def __init__(self, warmup_teacher_temp_epochs, epochs, n_views, warmup_teacher_temp, teacher_temp):
        super().__init__()
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.epochs = epochs
        self.n_views = n_views
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        
    def forward(self, student_output, teacher_output, epoch):
        # 计算当前epoch的teacher温度
        if epoch < self.warmup_teacher_temp_epochs:
            teacher_temp = self.warmup_teacher_temp
        else:
            teacher_temp = self.teacher_temp
            
        # 计算蒸馏损失
        student_out = student_output / 0.1  # student温度固定为0.1
        teacher_out = teacher_output / teacher_temp
        
        # 分别处理每个视角
        student_chunks = student_out.chunk(self.n_views)
        teacher_chunks = teacher_out.chunk(self.n_views)
        
        loss = 0
        for s_out, t_out in zip(student_chunks, teacher_chunks):
            s_prob = torch.softmax(s_out, dim=1)
            t_prob = torch.softmax(t_out, dim=1)
            loss += torch.sum(-t_prob * torch.log(s_prob + 1e-8), dim=1).mean()
            
        return loss / self.n_views

class TextSupConLoss(nn.Module):
    """文本版本的有监督对比损失"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        # 归一化特征
        features = nn.functional.normalize(features, dim=2)
        
        # 计算相似度矩阵
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = features.shape[1]
        
        # 计算logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # 创建mask
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        mask = mask.repeat(anchor_count, anchor_count)
        
        # 移除对角线
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 计算损失
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        
        return loss

def info_nce_logits(features, n_views=2):
    """计算InfoNCE对比学习的logits"""
    labels = torch.cat([torch.arange(features.shape[0] // n_views) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)
    
    features = nn.functional.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    
    # 移除对角线
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    
    # 选择正样本
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)
    
    return logits, labels

def train(model, train_loader, test_loader, unlabelled_train_loader, args):
    """训练主函数"""
    # 设置优化器
    bert_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'bert_encoder' in name:
            bert_params.append(param)
        else:
            head_params.append(param)
    
    params_groups = [
        {'params': bert_params, 'lr': args.lr * 0.1},
        {'params': head_params, 'lr': args.lr}
    ]
    
    optimizer = SGD(params_groups, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 学习率调度器
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )
    
    # 损失函数
    cluster_criterion = TextDistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )
    
    sup_con_criterion = TextSupConLoss()
    
    # 训练循环
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            # 解包数据
            input_ids_list = []
            attention_mask_list = []
            
            # 处理多视角数据
            for view_idx in range(args.n_views):
                input_ids_list.append(batch['input_ids'])
                attention_mask_list.append(batch['attention_mask'])
            
            input_ids = torch.cat(input_ids_list, dim=0).cuda()
            attention_mask = torch.cat(attention_mask_list, dim=0).cuda()
            labels = batch['label'].cuda()
            
            # 创建标签掩码（区分有标签和无标签数据）
            mask_lab = (labels >= 0).bool()
            
            # 前向传播
            student_out, student_proj = model(input_ids, attention_mask, return_features=True)
            teacher_out = student_out.detach()
            
            # 有监督分类损失
            if mask_lab.sum() > 0:
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(args.n_views)], dim=0)
                sup_labels = torch.cat([labels[mask_lab] for _ in range(args.n_views)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
            else:
                cls_loss = torch.tensor(0.0).cuda()
            
            # 无监督聚类损失
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            
            # Me-Max正则化
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            cluster_loss += args.memax_weight * me_max_loss
            
            # 无监督对比学习损失
            contrastive_logits, contrastive_labels = info_nce_logits(student_proj, args.n_views)
            contrastive_loss = nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
            
            # 有监督对比学习损失
            if mask_lab.sum() > 0:
                student_proj_sup = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(args.n_views)], dim=1)
                sup_con_labels = labels[mask_lab]
                sup_con_loss = sup_con_criterion(student_proj_sup, sup_con_labels)
            else:
                sup_con_loss = torch.tensor(0.0).cuda()
            
            # 总损失
            loss = 0
            loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
            loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_record.update(loss.item(), labels.size(0))
            
            if batch_idx % args.print_freq == 0:
                logging.info(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] '
                           f'Loss: {loss.item():.4f} '
                           f'Cls: {cls_loss.item():.4f} '
                           f'Cluster: {cluster_loss.item():.4f} '
                           f'SupCon: {sup_con_loss.item():.4f} '
                           f'Contrastive: {contrastive_loss.item():.4f}')
        
        logging.info(f'Train Epoch: {epoch} Avg Loss: {loss_record.avg:.4f}')
        
        # 测试
        if unlabelled_train_loader is not None:
            test_acc = test(model, unlabelled_train_loader, epoch, args)
            logging.info(f'Test Accuracy: {test_acc:.4f}')
        
        # 更新学习率
        exp_lr_scheduler.step()
        
        # 保存模型
        if epoch % args.save_freq == 0:
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
            }
            torch.save(save_dict, os.path.join(args.exp_root, f'model_epoch_{epoch}.pth'))

def test(model, test_loader, epoch, args):
    """测试函数"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['label'].cuda()
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            
            # 只计算有标签数据的准确率
            mask = labels >= 0
            if mask.sum() > 0:
                total += mask.sum().item()
                correct += (predicted[mask] == labels[mask]).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

def save_clustering_results(model, test_loader, epoch, args):
    """保存聚类结果"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['label'].cuda()
            
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            
            for i in range(len(input_ids)):
                results.append({
                    'text': batch['text'][i],
                    'true_label': labels[i].item(),
                    'predicted_label': predictions[i].item(),
                    'confidence': probabilities[i].max().item(),
                    'is_new_class': predictions[i].item() >= args.num_labeled_classes
                })
    
    # 保存结果
    import json
    save_path = os.path.join(args.exp_root, f'clustering_results_epoch_{epoch}.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info(f'Clustering results saved to {save_path}')

def setup_logging(args):
    """设置日志"""
    os.makedirs(args.exp_root, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.exp_root, 'train.log')),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SimGCD Text Training')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True, default='/data/data.csv')
    parser.add_argument('--bert_model_path', type=str, default='bert/final_model')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 模型参数
    parser.add_argument('--bert_output_dim', type=int, default=768)
    parser.add_argument('--num_labeled_classes', type=int, required=True)
    parser.add_argument('--num_unlabeled_classes', type=int, required=True)
    parser.add_argument('--freeze_bert', action='store_true')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', type=int, default=2)
    
    # SimGCD参数
    parser.add_argument('--memax_weight', type=float, default=2.0)
    parser.add_argument('--warmup_teacher_temp', type=float, default=0.07)
    parser.add_argument('--teacher_temp', type=float, default=0.04)
    parser.add_argument('--warmup_teacher_temp_epochs', type=int, default=30)
    
    # 其他参数
    parser.add_argument('--exp_root', type=str, default='./experiments')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # 创建模型
    total_classes = args.num_labeled_classes + args.num_unlabeled_classes
    model = SimGCDText(
        bert_model_name=args.bert_model_path,
        bert_output_dim=args.bert_output_dim,
        simgcd_in_dim=args.bert_output_dim,
        simgcd_out_dim=total_classes,
        freeze_bert=args.freeze_bert
    ).to(device)
    
    logging.info(f'Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters')
    
    # 创建数据增强
    text_transform = TextContrastiveLearningViewGenerator(n_views=args.n_views)
    
    # 创建数据集
    train_dataset = TextDataset(
        csv_path=args.data_path,
        bert_model_path=args.bert_model_path,
        max_length=args.max_length,
        is_labeled=False,  # 包含所有数据
        transform=text_transform
    )
    
    test_dataset = TextDataset(
        csv_path=args.data_path,
        bert_model_path=args.bert_model_path,
        max_length=args.max_length,
        is_labeled=True,  # 只包含有标签数据用于测试
        transform=None
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logging.info(f'Train dataset size: {len(train_dataset)}')
    logging.info(f'Test dataset size: {len(test_dataset)}')
    
    # 开始训练
    train(model, train_loader, test_loader, test_loader, args)