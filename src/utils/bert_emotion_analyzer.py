import re
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import pickle
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

class BERTEmotionAnalyzer:
    """
    强大的BERT情感分析器
    支持BERT模型和规则基础分析的组合
    """
    
    def __init__(self, use_bert=True, cache_file='data/emotion_cache.pkl'):
        self.use_bert = use_bert
        self.cache_file = cache_file
        self.emotion_cache = self._load_cache()
        
        # BERT模型相关
        self.bert_model = None
        self.bert_tokenizer = None
        self.bert_available = False
        
        # 攻击性词汇词典 - 分类更详细
        self.aggressive_words = {
            'insults': [
                'stupid', 'idiot', 'moron', 'dumb', 'retard', 'loser', 'pathetic',
                'worthless', 'useless', 'trash', 'garbage', 'scum', 'pig', 'dog',
                'freak', 'weirdo', 'creep', 'psycho', 'crazy', 'insane', 'nuts'
            ],
            'threats': [
                'kill', 'die', 'death', 'hurt', 'harm', 'beat', 'punch', 'kick',
                'destroy', 'ruin', 'crush', 'smash', 'attack', 'fight', 'violence',
                'murder', 'stab', 'shoot', 'bomb', 'explode', 'burn'
            ],
            'profanity': [
                'fuck', 'shit', 'damn', 'hell', 'ass', 'bitch', 'bastard',
                'crap', 'piss', 'cock', 'dick', 'pussy', 'whore', 'slut'
            ],
            'discrimination': [
                'hate', 'racist', 'sexist', 'homophobic', 'bigot', 'nazi',
                'fascist', 'terrorist', 'savage', 'primitive', 'inferior'
            ],
            'mockery': [
                'lol', 'lmao', 'rofl', 'haha', 'hehe', 'joke', 'funny',
                'ridiculous', 'laughable', 'pathetic', 'embarrassing'
            ]
        }
        
        # 否定词
        self.negation_words = [
            'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere',
            'neither', 'nor', 'without', 'barely', 'hardly', 'scarcely',
            'seldom', 'rarely', "n't", "don't", "won't", "can't", "shouldn't"
        ]
        
        # 强化词
        self.intensifiers = [
            'very', 'extremely', 'really', 'totally', 'completely', 'absolutely',
            'definitely', 'certainly', 'obviously', 'clearly', 'seriously',
            'fucking', 'damn', 'hell', 'bloody', 'so', 'too', 'quite'
        ]
        
        # 尝试初始化BERT
        if self.use_bert:
            self._initialize_bert()
    
    def _initialize_bert(self):
        """初始化BERT模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            # 使用专门的情感分析模型
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            
            logger.info(f"正在加载BERT模型: {model_name}")
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # 设置为评估模式
            self.bert_model.eval()
            self.bert_available = True
            logger.info("BERT模型加载成功")
            
        except Exception as e:
            logger.warning(f"BERT模型加载失败: {e}")
            logger.info("将使用规则基础分析作为后备方案")
            self.bert_available = False
    
    def _load_cache(self):
        """加载情感分析缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
        return {}
    
    def _save_cache(self):
        """保存情感分析缓存"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.emotion_cache, f)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        if not text:
            return ""
        
        # 转小写
        text = text.lower().strip()
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        # 处理重复字符 (但保留一些信息)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        return text
    
    def _analyze_with_bert(self, texts: List[str]) -> List[float]:
        """使用BERT进行情感分析"""
        if not self.bert_available or not texts:
            return [0.0] * len(texts)
        
        try:
            import torch
            
            scores = []
            batch_size = 16  # 批处理大小
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # 编码文本
                inputs = self.bert_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # 预测
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # 转换为攻击性分数 (负面情感分数)
                for pred in predictions:
                    # 假设标签顺序: [NEGATIVE, NEUTRAL, POSITIVE]
                    negative_score = pred[0].item()  # 负面情感
                    positive_score = pred[2].item()  # 正面情感
                    
                    # 计算攻击性分数 (负面 - 正面，归一化到[0,1])
                    aggression_score = (negative_score - positive_score + 1) / 2
                    scores.append(max(0.0, min(1.0, aggression_score)))
            
            return scores
            
        except Exception as e:
            logger.error(f"BERT分析失败: {e}")
            return [0.0] * len(texts)
    
    def _analyze_with_rules(self, text: str) -> float:
        """使用规则进行情感分析"""
        if not text:
            return 0.0
        
        text_processed = self._preprocess_text(text)
        words = text_processed.split()
        
        if not words:
            return 0.0
        
        # 基础分数计算
        aggression_score = 0.0
        word_count = len(words)
        
        # 1. 攻击性词汇检测
        for category, word_list in self.aggressive_words.items():
            category_weight = {
                'insults': 0.8,
                'threats': 1.0,
                'profanity': 0.6,
                'discrimination': 0.9,
                'mockery': 0.5
            }.get(category, 0.7)
            
            for word in word_list:
                count = text_processed.count(word)
                if count > 0:
                    aggression_score += count * category_weight
        
        # 2. 否定词处理
        negation_count = sum(1 for word in self.negation_words if word in text_processed)
        if negation_count > 0:
            # 否定词可能降低或增加攻击性，取决于上下文
            aggression_score *= (1 + 0.2 * negation_count)
        
        # 3. 强化词处理
        intensifier_count = sum(1 for word in self.intensifiers if word in text_processed)
        if intensifier_count > 0:
            aggression_score *= (1 + 0.3 * intensifier_count)
        
        # 4. 语法特征
        # 大写字母比例
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if upper_ratio > 0.3:  # 超过30%大写
            aggression_score *= 1.2
        
        # 感叹号密度
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            aggression_score *= (1 + 0.1 * exclamation_count)
        
        # 问号密度 (质疑/挑衅)
        question_count = text.count('?')
        if question_count > 1:
            aggression_score *= (1 + 0.05 * question_count)
        
        # 5. 重复字符模式 (表示强烈情绪)
        repeated_chars = len(re.findall(r'(.)\1{2,}', text))
        if repeated_chars > 0:
            aggression_score *= (1 + 0.1 * repeated_chars)
        
        # 6. 长度惩罚 (避免长文本分数过高)
        length_penalty = min(1.0, 20.0 / word_count) if word_count > 20 else 1.0
        aggression_score *= length_penalty
        
        # 7. 归一化到[0,1]
        # 使用sigmoid函数进行归一化
        normalized_score = 1 / (1 + np.exp(-aggression_score + 2))
        
        return min(1.0, max(0.0, normalized_score))
    
    def analyze_single(self, text: str) -> float:
        """分析单个文本的攻击性分数"""
        if not text or not text.strip():
            return 0.0
        
        # 检查缓存
        text_key = hash(text.strip())
        if text_key in self.emotion_cache:
            return self.emotion_cache[text_key]
        
        # 组合分析
        bert_score = 0.0
        if self.bert_available:
            bert_scores = self._analyze_with_bert([text])
            bert_score = bert_scores[0] if bert_scores else 0.0
        
        rule_score = self._analyze_with_rules(text)
        
        # 加权组合 (BERT权重更高，但规则分析作为基础)
        if self.bert_available:
            final_score = 0.7 * bert_score + 0.3 * rule_score
        else:
            final_score = rule_score
        
        # 缓存结果
        self.emotion_cache[text_key] = final_score
        
        # 定期保存缓存
        if len(self.emotion_cache) % 100 == 0:
            self._save_cache()
        
        return final_score
    
    def analyze_batch(self, texts: List[str]) -> List[float]:
        """批量分析文本的攻击性分数"""
        if not texts:
            return []
        
        # 过滤空文本
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return [0.0] * len(texts)
        
        # 检查缓存
        cached_scores = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(valid_texts):
            text_key = hash(text.strip())
            if text_key in self.emotion_cache:
                cached_scores[i] = self.emotion_cache[text_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 分析未缓存的文本
        if uncached_texts:
            # BERT分析
            bert_scores = []
            if self.bert_available:
                bert_scores = self._analyze_with_bert(uncached_texts)
            
            # 规则分析
            rule_scores = [self._analyze_with_rules(text) for text in uncached_texts]
            
            # 组合分数
            for i, (text, rule_score) in enumerate(zip(uncached_texts, rule_scores)):
                bert_score = bert_scores[i] if bert_scores else 0.0
                
                if self.bert_available:
                    final_score = 0.7 * bert_score + 0.3 * rule_score
                else:
                    final_score = rule_score
                
                # 缓存结果
                text_key = hash(text.strip())
                self.emotion_cache[text_key] = final_score
                cached_scores[uncached_indices[i]] = final_score
        
        # 构建最终结果
        results = []
        valid_idx = 0
        for text in texts:
            if text and text.strip():
                results.append(cached_scores[valid_idx])
                valid_idx += 1
            else:
                results.append(0.0)
        
        # 保存缓存
        if uncached_texts:
            self._save_cache()
        
        return results
    
    def analyze_batch_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量分析文本并返回详细结果"""
        scores = self.analyze_batch(texts)
        results = []
        
        for text, score in zip(texts, scores):
            results.append({
                'text': text,
                'aggression_score': score,
                'method': 'bert' if self.bert_available else 'rule'
            })
        
        return results
    
    def get_statistics(self) -> Dict:
        """获取分析器统计信息"""
        return {
            'bert_available': self.bert_available,
            'cache_size': len(self.emotion_cache),
            'aggressive_words_count': sum(len(words) for words in self.aggressive_words.values()),
            'negation_words_count': len(self.negation_words),
            'intensifiers_count': len(self.intensifiers)
        }
    
    def __del__(self):
        """析构函数，保存缓存"""
        try:
            self._save_cache()
        except:
            pass 