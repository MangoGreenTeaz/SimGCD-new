# text_augmentation.py
# 文本增强模块 - 简化版本，暂时不使用增强功能

import torch
import random
from typing import List, Dict, Any

class SimpleTextAugmentation:
    """简化版文本增强类，暂时只提供恒等变换"""
    
    def __init__(self, use_augmentation=False):
        """
        初始化文本增强器
        Args:
            use_augmentation: 是否使用文本增强，默认为False
        """
        self.use_augmentation = use_augmentation
    
    def augment_text(self, text: str) -> str:
        """
        文本增强函数
        Args:
            text: 输入文本
        Returns:
            增强后的文本（当前版本直接返回原文本）
        """
        if not self.use_augmentation:
            return text
        
        # 这里可以后续添加各种文本增强策略
        # 目前只返回原文本
        return text
    
    def create_augmented_views(self, text: str, num_views: int = 2) -> List[str]:
        """
        为对比学习创建多个文本视图
        Args:
            text: 原始文本
            num_views: 需要生成的视图数量
        Returns:
            文本视图列表
        """
        if not self.use_augmentation:
            # 不使用增强时，返回相同的文本
            return [text] * num_views
        
        # 后续可以添加不同的增强策略生成不同视图
        views = []
        for _ in range(num_views):
            views.append(self.augment_text(text))
        return views

class ContrastiveTextTransform:
    """用于对比学习的文本变换类"""
    
    def __init__(self, augmentation_prob=0.0):
        """
        初始化对比学习文本变换器
        Args:
            augmentation_prob: 文本增强概率，设为0表示不使用增强
        """
        self.augmenter = SimpleTextAugmentation(use_augmentation=augmentation_prob > 0)
        self.augmentation_prob = augmentation_prob
    
    def __call__(self, text: str) -> Dict[str, Any]:
        """
        生成用于对比学习的文本对
        Args:
            text: 输入文本
        Returns:
            包含两个视图的字典
        """
        # 生成两个视图用于对比学习
        view1 = self.augmenter.augment_text(text)
        view2 = self.augmenter.augment_text(text)
        
        return {
            'view1': view1,
            'view2': view2,
            'original': text
        }

if __name__ == '__main__':
    # 测试简化版文本增强
    augmenter = SimpleTextAugmentation(use_augmentation=False)
    
    test_text = "这是一个测试文本"
    print(f"原始文本: {test_text}")
    
    # 测试单个增强
    augmented = augmenter.augment_text(test_text)
    print(f"增强文本: {augmented}")
    
    # 测试多视图生成
    views = augmenter.create_augmented_views(test_text, num_views=3)
    print(f"多视图: {views}")
    
    # 测试对比学习变换
    transform = ContrastiveTextTransform(augmentation_prob=0.0)
    result = transform(test_text)
    print(f"对比学习视图: {result}")