#!/usr/bin/env python3
"""
基于BERT的情感分析器
用于更准确地分析评论文本的攻击性和情感倾向
"""

import torch
import numpy as np
import logging
import re
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import pickle
from pathlib import Path

class BERTEmotionAnalyzer:
    """基于BERT的情感分析器"""
    
    def __init__(self, model_name: str = "bert-base-uncased", cache_dir: str = "data/models/bert_cache"):
        """
        初始化BERT情感分析器
        
        Args:
            model_name: BERT模型名称
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.logger = self._setup_logger()
        
        # 创建缓存目录
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化模型和分词器
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 攻击性词汇词典（扩展版）
        self.attack_words = {
            'insults': ['stupid', 'idiot', 'moron', 'dumb', 'fool', 'loser', 'pathetic', 
                       'worthless', 'useless', 'trash', 'garbage', 'scum'],
            'threats': ['kill', 'die', 'death', 'hurt', 'harm', 'destroy', 'attack', 
                       'beat', 'punch', 'fight', 'violence'],
            'profanity': ['damn', 'hell', 'shit', 'fuck', 'bitch', 'ass', 'crap', 'suck'],
            'discrimination': ['hate', 'racist', 'sexist', 'discrimination', 'prejudice', 
                             'bigot', 'nazi', 'fascist'],
            'mockery': ['lol', 'lmao', 'rofl', 'joke', 'funny', 'ridiculous', 'pathetic', 
                       'embarrassing', 'shame']
        }
        
        # 情感分析缓存
        self.emotion_cache = {}
        self.cache_file = f"{cache_dir}/emotion_cache.pkl"
        self._load_cache()
        
        # 延迟加载模型（避免初始化时的长时间等待）
        self._model_loaded = False
    
    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger('BERTEmotionAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_model(self):
        """延迟加载BERT模型"""
        if self._model_loaded:
            return
        
        try:
            self.logger.info(f"加载BERT模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir,
                local_files_only=False
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir,
                local_files_only=False
            )
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
            self.logger.info("BERT模型加载成功")
            
        except Exception as e:
            self.logger.error(f"BERT模型加载失败: {e}")
            self.logger.info("将使用规则基础的情感分析作为后备方案")
            self._model_loaded = False
    
    def _load_cache(self):
        """加载情感分析缓存"""
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, 'rb') as f:
                    self.emotion_cache = pickle.load(f)
                self.logger.info(f"加载情感分析缓存: {len(self.emotion_cache)} 条记录")
        except Exception as e:
            self.logger.warning(f"加载缓存失败: {e}")
            self.emotion_cache = {}
    
    def _save_cache(self):
        """保存情感分析缓存"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.emotion_cache, f)
        except Exception as e:
            self.logger.warning(f"保存缓存失败: {e}")
    
    def analyze_text_emotion(self, text: str) -> Dict[str, float]:
        """
        分析单个文本的情感
        
        Args:
            text: 输入文本
            
        Returns:
            情感分析结果字典
        """
        if not text or not text.strip():
            return {'aggression_score': 0.0, 'toxicity_score': 0.0, 'emotion_score': 0.0}
        
        # 检查缓存
        text_key = text.strip().lower()
        if text_key in self.emotion_cache:
            return self.emotion_cache[text_key]
        
        # 尝试使用BERT分析
        result = self._analyze_with_bert(text)
        
        # 如果BERT失败，使用规则基础分析
        if result is None:
            result = self._analyze_with_rules(text)
        
        # 缓存结果
        self.emotion_cache[text_key] = result
        
        # 定期保存缓存
        if len(self.emotion_cache) % 100 == 0:
            self._save_cache()
        
        return result
    
    def _analyze_with_bert(self, text: str) -> Optional[Dict[str, float]]:
        """使用BERT进行情感分析"""
        if not self._model_loaded:
            self._load_model()
        
        if not self._model_loaded:
            return None
        
        try:
            # 预处理文本
            text = self._preprocess_text(text)
            
            # 分词和编码
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 获取BERT嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # 平均池化
            
            # 计算攻击性特征
            embedding_np = embeddings.cpu().numpy().flatten()
            
            # 基于嵌入计算攻击性分数（简化版本）
            # 这里使用一些启发式方法，实际项目中可以训练专门的分类器
            aggression_score = self._calculate_aggression_from_embedding(embedding_np, text)
            toxicity_score = self._calculate_toxicity_from_embedding(embedding_np, text)
            
            # 综合情感分数
            emotion_score = (aggression_score * 0.6 + toxicity_score * 0.4)
            
            return {
                'aggression_score': float(aggression_score),
                'toxicity_score': float(toxicity_score),
                'emotion_score': float(emotion_score),
                'method': 'bert'
            }
            
        except Exception as e:
            self.logger.warning(f"BERT分析失败: {e}")
            return None
    
    def _calculate_aggression_from_embedding(self, embedding: np.ndarray, text: str) -> float:
        """基于BERT嵌入和文本特征计算攻击性分数"""
        # 结合嵌入特征和规则特征
        rule_score = self._calculate_rule_based_aggression(text)
        
        # 使用嵌入的一些简单特征（这是简化版本）
        embedding_variance = np.var(embedding)
        embedding_mean_abs = np.mean(np.abs(embedding))
        
        # 启发式组合（在实际项目中应该用训练好的分类器）
        embedding_score = min(embedding_variance * 2.0 + embedding_mean_abs * 0.5, 1.0)
        
        # 组合规则分数和嵌入分数
        combined_score = rule_score * 0.7 + embedding_score * 0.3
        
        return min(combined_score, 1.0)
    
    def _calculate_toxicity_from_embedding(self, embedding: np.ndarray, text: str) -> float:
        """基于BERT嵌入计算毒性分数"""
        # 结合嵌入特征和规则特征
        rule_score = self._calculate_rule_based_toxicity(text)
        
        # 嵌入特征
        embedding_norm = np.linalg.norm(embedding)
        embedding_skew = self._calculate_skewness(embedding)
        
        # 启发式组合
        embedding_score = min(embedding_norm * 0.1 + abs(embedding_skew) * 0.3, 1.0)
        
        # 组合分数
        combined_score = rule_score * 0.8 + embedding_score * 0.2
        
        return min(combined_score, 1.0)
    
    def _calculate_skewness(self, arr: np.ndarray) -> float:
        """计算数组的偏度"""
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return np.mean(((arr - mean) / std) ** 3)
    
    def _analyze_with_rules(self, text: str) -> Dict[str, float]:
        """基于规则的情感分析（后备方案）"""
        aggression_score = self._calculate_rule_based_aggression(text)
        toxicity_score = self._calculate_rule_based_toxicity(text)
        emotion_score = (aggression_score * 0.6 + toxicity_score * 0.4)
        
        return {
            'aggression_score': float(aggression_score),
            'toxicity_score': float(toxicity_score),
            'emotion_score': float(emotion_score),
            'method': 'rules'
        }
    
    def _calculate_rule_based_aggression(self, text: str) -> float:
        """基于规则计算攻击性分数"""
        text_lower = text.lower()
        
        # 攻击性词汇计数
        attack_word_count = 0
        total_attack_categories = 0
        
        for category, words in self.attack_words.items():
            category_count = sum(1 for word in words if word in text_lower)
            if category_count > 0:
                total_attack_categories += 1
                attack_word_count += category_count
        
        # 语法特征
        exclamation_count = text.count('!')
        question_count = text.count('?')
        uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # 重复字符（如 "nooooo", "hahaha"）
        repeat_pattern = len(re.findall(r'(.)\1{2,}', text_lower))
        
        # 计算攻击性分数
        word_score = min(attack_word_count * 0.3, 1.0)
        category_score = min(total_attack_categories * 0.2, 1.0)
        grammar_score = min((exclamation_count * 0.1 + uppercase_ratio * 0.5), 1.0)
        repeat_score = min(repeat_pattern * 0.1, 0.3)
        
        total_score = word_score + category_score + grammar_score + repeat_score
        
        return min(total_score, 1.0)
    
    def _calculate_rule_based_toxicity(self, text: str) -> float:
        """基于规则计算毒性分数"""
        text_lower = text.lower()
        
        # 专门的毒性词汇
        toxic_words = self.attack_words['profanity'] + self.attack_words['threats'] + self.attack_words['discrimination']
        toxic_count = sum(1 for word in toxic_words if word in text_lower)
        
        # 威胁性语言模式
        threat_patterns = [
            r'(kill|die|death|hurt|harm).*you',
            r'you.*should.*(die|kill)',
            r'go.*kill.*yourself',
            r'i.*will.*(kill|hurt|destroy)',
        ]
        
        threat_count = sum(1 for pattern in threat_patterns if re.search(pattern, text_lower))
        
        # 毒性分数
        toxic_score = min(toxic_count * 0.4 + threat_count * 0.6, 1.0)
        
        return toxic_score
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 处理特殊字符
        text = re.sub(r'[^\w\s!?.,]', '', text)
        
        return text
    
    def analyze_batch_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        """批量分析文本情感"""
        results = []
        for text in texts:
            result = self.analyze_text_emotion(text)
            results.append(result)
        
        # 保存缓存
        self._save_cache()
        
        return results
    
    def calculate_user_aggression_score(self, user_comments: List[str]) -> Dict[str, float]:
        """
        计算用户的攻击性分数
        基于用户发布的所有评论
        
        Args:
            user_comments: 用户发布的评论列表
            
        Returns:
            用户攻击性分析结果
        """
        if not user_comments:
            return {'aggression_ratio': 0.0, 'avg_aggression': 0.0, 'total_comments': 0}
        
        # 分析每条评论
        comment_results = self.analyze_batch_texts(user_comments)
        
        # 统计攻击性评论
        aggressive_threshold = 0.3  # 攻击性阈值
        aggressive_comments = sum(1 for result in comment_results 
                                 if result['aggression_score'] > aggressive_threshold)
        
        # 计算平均攻击性
        avg_aggression = np.mean([result['aggression_score'] for result in comment_results])
        avg_toxicity = np.mean([result['toxicity_score'] for result in comment_results])
        
        # 攻击性比例
        aggression_ratio = aggressive_comments / len(user_comments)
        
        return {
            'aggression_ratio': float(aggression_ratio),
            'avg_aggression': float(avg_aggression),
            'avg_toxicity': float(avg_toxicity),
            'total_comments': len(user_comments),
            'aggressive_comments': aggressive_comments
        }
    
    def __del__(self):
        """析构函数，保存缓存"""
        if hasattr(self, 'emotion_cache') and self.emotion_cache:
            self._save_cache() 