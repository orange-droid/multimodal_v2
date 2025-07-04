#!/usr/bin/env python3
"""
增强情感分析模块
不依赖BERT，使用高级规则和词典进行情感分析
专为ProtoBully项目优化
"""

import numpy as np
import re
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter
import json
from pathlib import Path

# 避免导入可能有问题的依赖
import warnings
warnings.filterwarnings('ignore')

class EnhancedEmotionAnalyzer:
    """
    增强的情感分析器
    基于多层次规则和词典，不依赖深度学习模型
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # 情感词典
        self.emotion_lexicons = self._load_emotion_lexicons()
        
        # 情感强化词
        self.intensifiers = self._load_intensifiers()
        
        # 否定词
        self.negation_words = self._load_negation_words()
        
        # 缓存
        self.emotion_cache = {}
        
        self.logger.info("增强情感分析器初始化完成")
    
    def _setup_logger(self):
        """设置日志"""
        logger_name = 'EnhancedEmotionAnalyzer'
        logger = logging.getLogger(logger_name)
        
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_emotion_lexicons(self) -> Dict[str, Dict[str, float]]:
        """加载情感词典"""
        lexicons = {
            'aggression': {
                # 直接攻击词汇 (高权重)
                'hate': 0.9, 'stupid': 0.8, 'idiot': 0.9, 'dumb': 0.7, 'moron': 0.8,
                'retard': 0.9, 'loser': 0.7, 'pathetic': 0.8, 'worthless': 0.8,
                'useless': 0.7, 'trash': 0.8, 'garbage': 0.8, 'scum': 0.9,
                
                # 外貌攻击 (中高权重)
                'ugly': 0.7, 'fat': 0.6, 'gross': 0.7, 'disgusting': 0.8,
                'hideous': 0.8, 'repulsive': 0.8,
                
                # 威胁词汇 (最高权重)
                'kill': 1.0, 'die': 0.9, 'death': 0.8, 'murder': 1.0,
                'hurt': 0.7, 'pain': 0.6, 'suffer': 0.7, 'destroy': 0.8,
                
                # 贬低词汇 (中权重)
                'failure': 0.6, 'joke': 0.5, 'freak': 0.7, 'weird': 0.4,
                'creep': 0.7, 'psycho': 0.8, 'crazy': 0.5,
                
                # 脏话 (高权重)
                'fuck': 0.8, 'shit': 0.7, 'bitch': 0.8, 'bastard': 0.8,
                'damn': 0.5, 'hell': 0.5, 'ass': 0.6, 'crap': 0.5,
                'piss': 0.6, 'whore': 0.9, 'slut': 0.9,
            },
            
            'negative': {
                # 负面情绪词
                'sad': 0.6, 'angry': 0.7, 'mad': 0.7, 'furious': 0.9,
                'upset': 0.6, 'annoyed': 0.5, 'frustrated': 0.6,
                'disappointed': 0.5, 'depressed': 0.7, 'miserable': 0.8,
                'terrible': 0.7, 'awful': 0.8, 'horrible': 0.8,
                'bad': 0.4, 'worse': 0.5, 'worst': 0.7,
                'suck': 0.6, 'sucks': 0.6, 'sucked': 0.6,
            },
            
            'positive': {
                # 正面情绪词
                'good': 0.5, 'great': 0.7, 'awesome': 0.8, 'amazing': 0.8,
                'wonderful': 0.8, 'fantastic': 0.8, 'excellent': 0.8,
                'love': 0.7, 'like': 0.5, 'enjoy': 0.6, 'happy': 0.7,
                'glad': 0.6, 'pleased': 0.6, 'satisfied': 0.6,
                'cool': 0.5, 'nice': 0.5, 'sweet': 0.5,
            }
        }
        
        return lexicons
    
    def _load_intensifiers(self) -> Dict[str, float]:
        """加载情感强化词"""
        return {
            # 强化词
            'very': 1.5, 'really': 1.4, 'extremely': 1.8, 'super': 1.6,
            'totally': 1.5, 'completely': 1.6, 'absolutely': 1.7,
            'fucking': 1.8, 'damn': 1.3, 'so': 1.3, 'too': 1.2,
            
            # 弱化词
            'kind': 0.7, 'sort': 0.7, 'somewhat': 0.8, 'rather': 0.8,
            'quite': 0.9, 'pretty': 0.9, 'fairly': 0.8,
        }
    
    def _load_negation_words(self) -> set:
        """加载否定词"""
        return {
            'not', 'no', 'never', 'nothing', 'nobody', 'nowhere',
            'neither', 'nor', 'none', 'hardly', 'scarcely', 'barely',
            'dont', "don't", 'doesnt', "doesn't", 'didnt', "didn't",
            'wont', "won't", 'wouldnt', "wouldn't", 'cant', "can't",
            'couldnt', "couldn't", 'shouldnt', "shouldn't",
            'isnt', "isn't", 'arent', "aren't", 'wasnt', "wasn't",
            'werent', "weren't", 'hasnt', "hasn't", 'havent', "haven't",
            'hadnt', "hadn't",
        }
    
    def analyze_text_emotion(self, text: str) -> Dict[str, float]:
        """分析单个文本的情感"""
        if not text or not isinstance(text, str):
            return self._get_default_emotion_scores()
        
        # 检查缓存
        cache_key = hash(text)
        if cache_key in self.emotion_cache:
            return self.emotion_cache[cache_key]
        
        # 预处理文本
        processed_text = self._preprocess_text(text)
        tokens = processed_text.split()
        
        # 基础情感分析
        emotion_scores = self._analyze_lexicon_based(tokens, text)
        
        # 语法特征分析
        grammar_features = self._analyze_grammar_features(text)
        emotion_scores.update(grammar_features)
        
        # 计算综合攻击性分数
        emotion_scores['aggression_score'] = self._calculate_comprehensive_aggression(
            emotion_scores, text, tokens
        )
        
        # 归一化分数
        emotion_scores = self._normalize_scores(emotion_scores)
        
        # 缓存结果
        self.emotion_cache[cache_key] = emotion_scores
        
        return emotion_scores
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 转换为小写
        text = text.lower()
        
        # 处理缩写
        contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "won't": "will not", "wouldn't": "would not", "can't": "can not",
            "couldn't": "could not", "shouldn't": "should not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "hasn't": "has not", "haven't": "have not",
            "hadn't": "had not", "you're": "you are", "i'm": "i am",
            "he's": "he is", "she's": "she is", "it's": "it is",
            "we're": "we are", "they're": "they are",
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # 移除多余的空格和标点
        text = re.sub(r'[^\w\s!?.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _analyze_lexicon_based(self, tokens: List[str], original_text: str) -> Dict[str, float]:
        """基于词典的情感分析"""
        scores = {'aggression': 0.0, 'negative': 0.0, 'positive': 0.0}
        word_counts = {'aggression': 0, 'negative': 0, 'positive': 0}
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # 检查否定词
            is_negated = False
            if i > 0 and tokens[i-1] in self.negation_words:
                is_negated = True
            elif i > 1 and tokens[i-2] in self.negation_words:
                is_negated = True
            
            # 检查强化词
            intensifier = 1.0
            if i > 0 and tokens[i-1] in self.intensifiers:
                intensifier = self.intensifiers[tokens[i-1]]
            elif i > 1 and tokens[i-2] in self.intensifiers:
                intensifier = self.intensifiers[tokens[i-2]]
            
            # 计算情感分数
            for emotion_type, lexicon in self.emotion_lexicons.items():
                if token in lexicon:
                    base_score = lexicon[token]
                    
                    # 应用强化
                    score = base_score * intensifier
                    
                    # 应用否定
                    if is_negated:
                        if emotion_type == 'positive':
                            # 否定的正面词变成负面
                            scores['negative'] += score * 0.7
                            word_counts['negative'] += 1
                        elif emotion_type in ['negative', 'aggression']:
                            # 否定的负面词减少负面程度
                            scores[emotion_type] += score * 0.3
                            word_counts[emotion_type] += 1
                    else:
                        scores[emotion_type] += score
                        word_counts[emotion_type] += 1
            
            i += 1
        
        # 计算平均分数
        for emotion_type in scores:
            if word_counts[emotion_type] > 0:
                scores[emotion_type] = scores[emotion_type] / word_counts[emotion_type]
        
        return scores
    
    def _analyze_grammar_features(self, text: str) -> Dict[str, float]:
        """分析语法特征"""
        features = {}
        
        # 大写字母比例（愤怒/激动）
        if len(text) > 0:
            uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text)
            features['uppercase_intensity'] = min(uppercase_ratio * 3, 1.0)
        else:
            features['uppercase_intensity'] = 0.0
        
        # 感叹号密度
        exclamation_count = text.count('!')
        if len(text) > 0:
            features['exclamation_intensity'] = min(exclamation_count / len(text) * 50, 1.0)
        else:
            features['exclamation_intensity'] = 0.0
        
        # 问号密度（可能表示困惑或挑衅）
        question_count = text.count('?')
        if len(text) > 0:
            features['question_intensity'] = min(question_count / len(text) * 50, 1.0)
        else:
            features['question_intensity'] = 0.0
        
        # 重复字符（强调）
        repeated_chars = len(re.findall(r'(.)\1{2,}', text))
        if len(text) > 0:
            features['repetition_intensity'] = min(repeated_chars / len(text) * 20, 1.0)
        else:
            features['repetition_intensity'] = 0.0
        
        # 句子长度特征
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            # 非常短的句子可能表示愤怒或攻击性
            if avg_sentence_length < 3:
                features['short_sentence_aggression'] = 0.3
            else:
                features['short_sentence_aggression'] = 0.0
        else:
            features['short_sentence_aggression'] = 0.0
        
        return features
    
    def _calculate_comprehensive_aggression(self, emotion_scores: Dict[str, float], 
                                          original_text: str, tokens: List[str]) -> float:
        """计算综合攻击性分数"""
        aggression_components = []
        
        # 1. 词典基础攻击性
        aggression_components.append(emotion_scores.get('aggression', 0) * 0.4)
        
        # 2. 负面情绪贡献
        aggression_components.append(emotion_scores.get('negative', 0) * 0.2)
        
        # 3. 语法特征贡献
        grammar_aggression = (
            emotion_scores.get('uppercase_intensity', 0) * 0.1 +
            emotion_scores.get('exclamation_intensity', 0) * 0.05 +
            emotion_scores.get('repetition_intensity', 0) * 0.05 +
            emotion_scores.get('short_sentence_aggression', 0) * 0.1
        )
        aggression_components.append(grammar_aggression)
        
        # 4. 特殊模式检测
        pattern_aggression = self._detect_aggressive_patterns(original_text)
        aggression_components.append(pattern_aggression * 0.1)
        
        # 5. 上下文攻击性
        context_aggression = self._analyze_context_aggression(tokens)
        aggression_components.append(context_aggression * 0.1)
        
        # 综合分数
        total_aggression = sum(aggression_components)
        
        return min(total_aggression, 1.0)
    
    def _detect_aggressive_patterns(self, text: str) -> float:
        """检测攻击性模式"""
        patterns = [
            # 威胁模式
            (r'\b(i will|gonna|going to).*(kill|hurt|destroy|beat)', 0.9),
            (r'\b(you should|you need to).*(die|kill yourself)', 1.0),
            
            # 侮辱模式
            (r'\byou are.*(stupid|idiot|moron|worthless)', 0.7),
            (r'\bstupid.*(you|person|people)', 0.6),
            
            # 攻击性命令
            (r'\b(shut up|get lost|go away|leave me alone)', 0.5),
            (r'\b(fuck off|piss off)', 0.8),
            
            # 嘲笑模式
            (r'\b(haha|lol|lmao).*(loser|fail|stupid)', 0.6),
            (r'\bwhat a.*(loser|idiot|joke)', 0.7),
        ]
        
        max_score = 0.0
        for pattern, score in patterns:
            if re.search(pattern, text.lower()):
                max_score = max(max_score, score)
        
        return max_score
    
    def _analyze_context_aggression(self, tokens: List[str]) -> float:
        """分析上下文攻击性"""
        # 检测攻击性词汇聚集
        aggression_words = set(self.emotion_lexicons['aggression'].keys())
        
        # 计算攻击性词汇密度
        aggression_count = sum(1 for token in tokens if token in aggression_words)
        if len(tokens) == 0:
            return 0.0
        
        aggression_density = aggression_count / len(tokens)
        
        # 检测攻击性词汇聚集（连续出现）
        max_cluster = 0
        current_cluster = 0
        
        for token in tokens:
            if token in aggression_words:
                current_cluster += 1
                max_cluster = max(max_cluster, current_cluster)
            else:
                current_cluster = 0
        
        cluster_score = min(max_cluster / 3, 1.0)  # 最多3个连续攻击词达到满分
        
        return min(aggression_density * 2 + cluster_score * 0.5, 1.0)
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """归一化分数"""
        # 确保所有分数在0-1范围内
        for key in scores:
            scores[key] = max(0.0, min(1.0, scores[key]))
        
        # 计算情感极性
        positive = scores.get('positive', 0)
        negative = scores.get('negative', 0)
        aggression = scores.get('aggression', 0)
        
        # 综合负面分数
        total_negative = negative + aggression
        
        # 计算中性分数
        neutral = max(0, 1 - positive - total_negative)
        
        # 归一化到总和为1
        total = positive + total_negative + neutral
        if total > 0:
            scores['positive_norm'] = positive / total
            scores['negative_norm'] = total_negative / total
            scores['neutral_norm'] = neutral / total
        else:
            scores['positive_norm'] = 0.33
            scores['negative_norm'] = 0.33
            scores['neutral_norm'] = 0.33
        
        return scores
    
    def _get_default_emotion_scores(self) -> Dict[str, float]:
        """获取默认情感分数"""
        return {
            'aggression': 0.0,
            'negative': 0.2,
            'positive': 0.1,
            'aggression_score': 0.1,
            'positive_norm': 0.1,
            'negative_norm': 0.3,
            'neutral_norm': 0.6,
        }
    
    def analyze_batch_emotions(self, texts: List[str]) -> List[Dict[str, float]]:
        """批量分析文本情感"""
        results = []
        
        for text in texts:
            emotion_scores = self.analyze_text_emotion(text)
            results.append(emotion_scores)
        
        return results

def main():
    """测试函数"""
    analyzer = EnhancedEmotionAnalyzer()
    
    test_texts = [
        "You are so stupid and ugly!",
        "I hate you so much, you worthless piece of trash!",
        "This is really great, I love it!",
        "I'm going to kill you if you don't stop!",
        "That's pretty cool, thanks for sharing.",
        "WHAT THE HELL IS WRONG WITH YOU!!!",
        "You're not very smart, are you?",
        "I think this is okay, nothing special.",
    ]
    
    print("增强情感分析测试:")
    print("=" * 60)
    
    for text in test_texts:
        scores = analyzer.analyze_text_emotion(text)
        print(f"\n文本: {text}")
        print(f"攻击性: {scores['aggression_score']:.3f}")
        print(f"负面: {scores['negative_norm']:.3f}")
        print(f"正面: {scores['positive_norm']:.3f}")
        print(f"中性: {scores['neutral_norm']:.3f}")

if __name__ == "__main__":
    main() 