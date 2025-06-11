import random
import re
import jieba
import numpy as np
from typing import List, Dict, Optional, Callable
from transformers import BertTokenizer
import torch

class TextAugmentation:
    """
    文本数据增强类，用于生成多样化的文本视图
    支持多种增强策略，为对比学习提供不同的文本表示
    """
    
    def __init__(self, 
                 bert_model_path: str,
                 aug_prob: float = 0.15,
                 max_aug_ratio: float = 0.3,
                 seed: int = 42):
        """
        初始化文本增强器
        
        Args:
            bert_model_path: BERT模型路径，用于获取词汇表
            aug_prob: 单个token的增强概率
            max_aug_ratio: 最大增强比例（相对于原文本长度）
            seed: 随机种子
        """
        self.aug_prob = aug_prob
        self.max_aug_ratio = max_aug_ratio
        random.seed(seed)
        np.random.seed(seed)
        
        # 初始化BERT tokenizer获取词汇表
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.vocab = list(self.tokenizer.vocab.keys())
        
        # 过滤特殊token
        self.normal_tokens = [token for token in self.vocab 
                             if not token.startswith('[') and not token.startswith('##')]
        
        # 初始化jieba分词器
        jieba.initialize()
        
        print(f"文本增强器初始化完成，词汇表大小: {len(self.vocab)}")
    
    def random_mask(self, text: str) -> str:
        """
        随机mask策略：随机将部分字符替换为[MASK]
        模拟BERT预训练任务
        
        Args:
            text: 输入文本
            
        Returns:
            增强后的文本
        """
        # 使用jieba分词
        words = list(jieba.cut(text))
        if len(words) == 0:
            return text
            
        # 计算要mask的词数量
        num_to_mask = max(1, int(len(words) * self.aug_prob))
        num_to_mask = min(num_to_mask, int(len(words) * self.max_aug_ratio))
        
        # 随机选择要mask的位置
        mask_indices = random.sample(range(len(words)), num_to_mask)
        
        # 执行mask操作
        augmented_words = words.copy()
        for idx in mask_indices:
            # 80%概率替换为[MASK]，10%替换为随机词，10%保持不变
            rand_prob = random.random()
            if rand_prob < 0.8:
                augmented_words[idx] = '[MASK]'
            elif rand_prob < 0.9:
                augmented_words[idx] = random.choice(self.normal_tokens)
            # else: 保持原词不变
        
        return ''.join(augmented_words)
    
    def random_deletion(self, text: str) -> str:
        """
        随机删除策略：随机删除部分字符或词
        
        Args:
            text: 输入文本
            
        Returns:
            增强后的文本
        """
        words = list(jieba.cut(text))
        if len(words) <= 2:  # 保证至少保留2个词
            return text
            
        # 计算要删除的词数量
        num_to_delete = max(1, int(len(words) * self.aug_prob))
        num_to_delete = min(num_to_delete, int(len(words) * self.max_aug_ratio))
        num_to_delete = min(num_to_delete, len(words) - 2)  # 至少保留2个词
        
        # 随机选择要删除的位置
        delete_indices = set(random.sample(range(len(words)), num_to_delete))
        
        # 执行删除操作
        augmented_words = [word for i, word in enumerate(words) 
                          if i not in delete_indices]
        
        return ''.join(augmented_words)
    
    def random_insertion(self, text: str) -> str:
        """
        随机插入策略：在随机位置插入随机词
        
        Args:
            text: 输入文本
            
        Returns:
            增强后的文本
        """
        words = list(jieba.cut(text))
        if len(words) == 0:
            return text
            
        # 计算要插入的词数量
        num_to_insert = max(1, int(len(words) * self.aug_prob))
        num_to_insert = min(num_to_insert, int(len(words) * self.max_aug_ratio))
        
        # 执行插入操作
        augmented_words = words.copy()
        for _ in range(num_to_insert):
            # 随机选择插入位置
            insert_pos = random.randint(0, len(augmented_words))
            # 随机选择要插入的词
            random_word = random.choice(self.normal_tokens)
            augmented_words.insert(insert_pos, random_word)
        
        return ''.join(augmented_words)
    
    def random_substitution(self, text: str) -> str:
        """
        随机替换策略：将部分词替换为随机词
        
        Args:
            text: 输入文本
            
        Returns:
            增强后的文本
        """
        words = list(jieba.cut(text))
        if len(words) == 0:
            return text
            
        # 计算要替换的词数量
        num_to_substitute = max(1, int(len(words) * self.aug_prob))
        num_to_substitute = min(num_to_substitute, int(len(words) * self.max_aug_ratio))
        
        # 随机选择要替换的位置
        substitute_indices = random.sample(range(len(words)), 
                                         min(num_to_substitute, len(words)))
        
        # 执行替换操作
        augmented_words = words.copy()
        for idx in substitute_indices:
            augmented_words[idx] = random.choice(self.normal_tokens)
        
        return ''.join(augmented_words)
    
    def random_swap(self, text: str) -> str:
        """
        随机交换策略：随机交换相邻词的位置
        
        Args:
            text: 输入文本
            
        Returns:
            增强后的文本
        """
        words = list(jieba.cut(text))
        if len(words) <= 1:
            return text
            
        # 计算要交换的次数
        num_swaps = max(1, int(len(words) * self.aug_prob))
        num_swaps = min(num_swaps, int(len(words) * self.max_aug_ratio))
        
        # 执行交换操作
        augmented_words = words.copy()
        for _ in range(num_swaps):
            # 随机选择要交换的相邻位置
            if len(augmented_words) > 1:
                idx = random.randint(0, len(augmented_words) - 2)
                augmented_words[idx], augmented_words[idx + 1] = \
                    augmented_words[idx + 1], augmented_words[idx]
        
        return ''.join(augmented_words)
    
    def back_translation_simulation(self, text: str) -> str:
        """
        回译模拟策略：通过词序调整模拟回译效果
        
        Args:
            text: 输入文本
            
        Returns:
            增强后的文本
        """
        words = list(jieba.cut(text))
        if len(words) <= 2:
            return text
            
        # 将文本分成若干段，每段内部进行词序调整
        segment_size = max(2, len(words) // 3)
        augmented_words = []
        
        for i in range(0, len(words), segment_size):
            segment = words[i:i + segment_size]
            if len(segment) > 1 and random.random() < self.aug_prob:
                # 随机打乱段内词序
                random.shuffle(segment)
            augmented_words.extend(segment)
        
        return ''.join(augmented_words)

class ContrastiveTextTransform:
    """
    对比学习文本变换类
    为每个文本生成多个增强视图用于对比学习
    """
    
    def __init__(self, 
                 bert_model_path: str,
                 num_views: int = 2,
                 aug_strategies: Optional[List[str]] = None,
                 aug_prob: float = 0.15):
        """
        初始化对比学习文本变换器
        
        Args:
            bert_model_path: BERT模型路径
            num_views: 每个文本生成的视图数量
            aug_strategies: 使用的增强策略列表
            aug_prob: 增强概率
        """
        self.num_views = num_views
        self.aug_prob = aug_prob
        
        # 初始化文本增强器
        self.augmenter = TextAugmentation(
            bert_model_path=bert_model_path,
            aug_prob=aug_prob
        )
        
        # 定义可用的增强策略
        if aug_strategies is None:
            self.aug_strategies = [
                'random_mask',
                'random_deletion', 
                'random_substitution',
                'random_swap'
            ]
        else:
            self.aug_strategies = aug_strategies
            
        # 构建策略映射
        self.strategy_map = {
            'random_mask': self.augmenter.random_mask,
            'random_deletion': self.augmenter.random_deletion,
            'random_insertion': self.augmenter.random_insertion,
            'random_substitution': self.augmenter.random_substitution,
            'random_swap': self.augmenter.random_swap,
            'back_translation': self.augmenter.back_translation_simulation
        }
        
        print(f"对比学习变换器初始化完成，使用策略: {self.aug_strategies}")
    
    def __call__(self, text: str) -> List[str]:
        """
        为输入文本生成多个增强视图
        
        Args:
            text: 输入文本
            
        Returns:
            增强视图列表
        """
        views = []
        
        # 第一个视图：原始文本（轻微增强或不增强）
        if random.random() < 0.3:  # 30%概率对原始文本进行轻微增强
            strategy = random.choice(self.aug_strategies)
            view1 = self.strategy_map[strategy](text)
        else:
            view1 = text
        views.append(view1)
        
        # 生成其他视图
        for _ in range(self.num_views - 1):
            # 随机选择增强策略
            strategy = random.choice(self.aug_strategies)
            augmented_text = self.strategy_map[strategy](text)
            views.append(augmented_text)
        
        return views
    
    def get_single_augmentation(self, text: str, strategy: str = None) -> str:
        """
        获取单个增强结果
        
        Args:
            text: 输入文本
            strategy: 指定的增强策略，如果为None则随机选择
            
        Returns:
            增强后的文本
        """
        if strategy is None:
            strategy = random.choice(self.aug_strategies)
        
        if strategy not in self.strategy_map:
            raise ValueError(f"未知的增强策略: {strategy}")
            
        return self.strategy_map[strategy](text)

def create_augmentation_transform(bert_model_path: str,
                                num_views: int = 2,
                                aug_prob: float = 0.15,
                                strategies: Optional[List[str]] = None) -> ContrastiveTextTransform:
    """
    创建文本增强变换器的便捷函数
    
    Args:
        bert_model_path: BERT模型路径
        num_views: 视图数量
        aug_prob: 增强概率
        strategies: 增强策略列表
        
    Returns:
        配置好的文本变换器
    """
    return ContrastiveTextTransform(
        bert_model_path=bert_model_path,
        num_views=num_views,
        aug_strategies=strategies,
        aug_prob=aug_prob
    )

if __name__ == "__main__":
    # 测试代码
    bert_path = "/data/hdd1/hulei/Code/GCD/bert/final_model"
    
    # 测试文本
    test_text = "这是一个用于测试文本增强功能的示例句子，包含了多个词汇和语义信息。"
    
    print(f"原始文本: {test_text}")
    print("\n=== 单独测试各种增强策略 ===")
    
    # 创建增强器
    augmenter = TextAugmentation(bert_path)
    
    # 测试各种策略
    strategies = [
        ('随机mask', augmenter.random_mask),
        ('随机删除', augmenter.random_deletion),
        ('随机插入', augmenter.random_insertion),
        ('随机替换', augmenter.random_substitution),
        ('随机交换', augmenter.random_swap),
        ('回译模拟', augmenter.back_translation_simulation)
    ]
    
    for name, func in strategies:
        augmented = func(test_text)
        print(f"{name}: {augmented}")
    
    print("\n=== 测试对比学习变换器 ===")
    
    # 创建对比学习变换器
    transform = ContrastiveTextTransform(bert_path, num_views=3)
    
    # 生成多个视图
    views = transform(test_text)
    for i, view in enumerate(views):
        print(f"视图{i+1}: {view}")