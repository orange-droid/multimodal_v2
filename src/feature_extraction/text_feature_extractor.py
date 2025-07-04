"""
文本特征提取器模块

提取文本数据的各种特征，包括BERT嵌入、情感特征、攻击性特征等
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# 添加ProtoBully项目目录到路径
protobully_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(protobully_root)

from src.feature_extraction.base_feature_extractor import BaseFeatureExtractor
from src.utils.config import DATA_CONFIG, PREPROCESSING_CONFIG, FEATURE_CONFIG
from src.utils.utils import ensure_dir, save_json, load_json, save_pickle, load_pickle, get_timestamp

# 导入情感特征提取器
# 使用内置的情感特征提取功能，不再依赖外部的EmotionFeatureExtractor
import sys
import os

class TextFeatureExtractor(BaseFeatureExtractor):
    """
    文本特征提取器类
    
    提取文本数据的各种特征，包括BERT嵌入、情感特征、攻击性特征等
    """
    
    def __init__(self, config=None, preprocessing_config=None, feature_config=None):
        """
        初始化文本特征提取器
        
        参数:
            config: 数据配置，如果为None则使用默认配置
            preprocessing_config: 预处理配置，如果为None则使用默认配置
            feature_config: 特征提取配置，如果为None则使用默认配置
        """
        super().__init__(config, preprocessing_config, feature_config)
        
        # 初始化NLTK工具
        self._initialize_nltk_tools()
        
        # 初始化情感分析器
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # 使用内置的情感特征提取功能
        self.emotion_extractor = self  # 使用自身的方法
        self.logger.info("使用内置情感特征提取功能")
        
        # 加载攻击性词汇词典
        self.offensive_words = self._load_offensive_words()
        
        # 初始化BERT模型（如果可用）
        self.bert_model = None
        self.bert_tokenizer = None
        self._initialize_bert()
        
        # 特征存储
        self.text_features = {}
        self.comment_features = {}
        self.word_features = {}
        self.vocabulary = None
    
    def _initialize_nltk_tools(self):
        """初始化NLTK工具"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def _initialize_bert(self):
        """初始化BERT模型（如果可用）"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            model_name = self.feature_config.get("text_model", "bert-base-uncased")
            self.logger.info(f"加载BERT模型: {model_name}")
            
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            
            # 如果有GPU，将模型移到GPU
            if torch.cuda.is_available():
                self.bert_model = self.bert_model.cuda()
                self.logger.info("BERT模型已移至GPU")
            
            self.bert_model.eval()  # 设置为评估模式
            self.logger.info("BERT模型加载成功")
            
        except Exception as e:
            self.logger.warning(f"加载BERT模型失败，将使用基础文本特征: {str(e)}")
            self.bert_model = None
            self.bert_tokenizer = None
    
    def _load_offensive_words(self):
        """加载攻击性词汇词典"""
        # 基础攻击性词汇
        offensive_words = {
            # 霸凌相关
            'stupid', 'dumb', 'idiot', 'moron', 'loser', 'ugly', 'fat', 'weird', 'freak', 'retard', 'retarded',
            'worthless', 'pathetic', 'useless', 'kill yourself', 'kys', 'die', 'hate you', 'hate u', 'nobody likes you',
            'waste of space', 'go die', 'trash', 'garbage', 'no friends', 'failure', 'joke',
            
            # 粗俗词汇
            'fuck', 'shit', 'ass', 'asshole', 'bitch', 'bastard', 'damn', 'cunt', 'dick', 'douchebag',
            'motherfucker', 'whore', 'slut', 'piss', 'crap', 'hell', 'wtf', 'stfu', 'fu', 'f u',
            
            # 歧视词汇
            'nigger', 'nigga', 'fag', 'faggot', 'homo', 'queer', 'retard', 'spic', 'wetback', 'chink',
            'gook', 'towelhead', 'beaner', 'cripple', 'midget', 'tranny', 'nazi', 'jew',
            
            # 威胁词汇
            'kill', 'murder', 'hurt', 'punch', 'beat', 'slap', 'hit', 'attack', 'stab', 'shoot',
            'torture', 'rape', 'molest', 'abuse', 'suicide', 'hang', 'choke', 'strangle'
        }
        
        # 尝试从文件加载更多攻击性词汇
        try:
            offensive_words_path = os.path.join(self.config["raw_data_path"], "offensive_words.txt")
            if os.path.exists(offensive_words_path):
                with open(offensive_words_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip().lower()
                        if word:
                            offensive_words.add(word)
                self.logger.info(f"从文件加载了 {len(offensive_words)} 个攻击性词汇")
        except Exception as e:
            self.logger.warning(f"加载攻击性词汇文件失败: {str(e)}")
        
        return offensive_words
    
    def preprocess_text(self, text):
        """
        预处理文本
        
        参数:
            text: 输入文本
            
        返回:
            tokens: 处理后的词元列表
        """
        if not text or not isinstance(text, str):
            return []
        
        # 转换为小写
        text = text.lower()
        
        # 移除URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        
        # 移除非字母数字字符
        text = re.sub(r'[^\w\s]', '', text)
        
        # 分词
        tokens = word_tokenize(text)
        
        # 移除停用词
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # 词干提取
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        # 移除空词元和数字
        tokens = [token for token in tokens if token and not token.isdigit()]
        
        return tokens
    
    def extract_sentiment_features(self, text):
        """
        提取情感特征
        
        参数:
            text: 输入文本
            
        返回:
            情感特征字典
        """
        if not text or not isinstance(text, str):
            return {
                'compound': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
        
        # 使用VADER提取情感分数
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        return {
            'compound': sentiment_scores['compound'],
            'positive': sentiment_scores['pos'],
            'negative': sentiment_scores['neg'],
            'neutral': sentiment_scores['neu']
        }
    
    def extract_emotion_features(self, comment_text):
        """
        内置的情感特征提取方法，替代外部的EmotionFeatureExtractor
        
        参数:
            comment_text: 评论文本
            
        返回:
            情感特征字典
        """
        if not comment_text or not isinstance(comment_text, str):
            return {
                'sentiment_score': 0,
                'emotion_scores': {
                    'anger': 0, 'fear': 0, 'joy': 0, 'sadness': 0,
                    'disgust': 0, 'trust': 0, 'surprise': 0, 'anticipation': 0
                },
                'bullying_indicators': {
                    'bullying_words_ratio': 0,
                    'swear_ratio': 0,
                    'aggression_score': 0
                }
            }
        
        # 基础情感分析
        sentiment_scores = self.extract_sentiment_features(comment_text)
        
        # 预处理文本
        tokens = self.preprocess_text(comment_text)
        
        # 计算霸凌指标
        total_words = len(tokens) if tokens else 1
        
        # 霸凌词汇比例
        bullying_words = {'stupid', 'dumb', 'idiot', 'loser', 'ugly', 'fat', 'weird', 'freak', 
                         'worthless', 'pathetic', 'useless', 'trash', 'garbage', 'failure', 'joke'}
        bullying_count = sum(1 for token in tokens if token in bullying_words)
        bullying_words_ratio = bullying_count / total_words
        
        # 脏话比例
        swear_words = {'fuck', 'shit', 'ass', 'bitch', 'bastard', 'damn', 'cunt', 'dick', 
                      'motherfucker', 'whore', 'slut', 'piss', 'crap', 'hell'}
        swear_count = sum(1 for token in tokens if token in swear_words)
        swear_ratio = swear_count / total_words
        
        # 攻击性分数（基于情感分数和攻击性词汇）
        aggression_score = max(0, -sentiment_scores['compound']) + bullying_words_ratio + swear_ratio
        
        # 基于情感分数估算情绪分布
        compound = sentiment_scores['compound']
        positive = sentiment_scores['positive']
        negative = sentiment_scores['negative']
        
        # 简化的情绪映射
        emotion_scores = {
            'anger': max(0, negative * 0.8 + aggression_score * 0.2),
            'fear': max(0, negative * 0.3),
            'joy': max(0, positive * 0.8),
            'sadness': max(0, negative * 0.5),
            'disgust': max(0, negative * 0.4 + swear_ratio * 0.6),
            'trust': max(0, positive * 0.6),
            'surprise': max(0, abs(compound) * 0.3),
            'anticipation': max(0, positive * 0.4)
        }
        
        return {
            'sentiment_score': compound,
            'emotion_scores': emotion_scores,
            'bullying_indicators': {
                'bullying_words_ratio': bullying_words_ratio,
                'swear_ratio': swear_ratio,
                'aggression_score': aggression_score
            }
        }
    
    def extract_offensive_features(self, tokens):
        """
        提取攻击性特征
        
        参数:
            tokens: 词元列表
            
        返回:
            攻击性特征字典
        """
        if not tokens:
            return {
                'offensive_word_count': 0,
                'offensive_word_ratio': 0,
                'offensive_score': 0
            }
        
        # 计算攻击性词汇数量
        offensive_count = sum(1 for token in tokens if token in self.offensive_words)
        
        # 计算攻击性词汇比例
        offensive_ratio = offensive_count / len(tokens) if tokens else 0
        
        # 计算攻击性分数（基于比例和阈值）
        threshold = self.feature_config.get("offensive_threshold", 0.7)
        offensive_score = min(1.0, offensive_ratio / threshold) if threshold > 0 else 0
        
        return {
            'offensive_word_count': offensive_count,
            'offensive_word_ratio': offensive_ratio,
            'offensive_score': offensive_score
        }
    
    def extract_statistical_features(self, text, tokens):
        """
        提取统计特征
        
        参数:
            text: 输入文本
            tokens: 词元列表
            
        返回:
            统计特征字典
        """
        if not text or not isinstance(text, str):
            return {
                'char_count': 0,
                'word_count': 0,
                'avg_word_length': 0,
                'uppercase_ratio': 0,
                'unique_word_ratio': 0
            }
        
        # 字符数
        char_count = len(text)
        
        # 词数
        word_count = len(tokens) if tokens else 0
        
        # 平均词长
        avg_word_length = sum(len(token) for token in tokens) / word_count if word_count > 0 else 0
        
        # 大写字母比例
        uppercase_count = sum(1 for char in text if char.isupper())
        uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0
        
        # 唯一词比例
        unique_words = len(set(tokens))
        unique_word_ratio = unique_words / word_count if word_count > 0 else 0
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'uppercase_ratio': uppercase_ratio,
            'unique_word_ratio': unique_word_ratio
        }
    
    def extract_bert_embeddings(self, texts):
        """
        提取BERT嵌入
        
        参数:
            texts: 文本列表
            
        返回:
            嵌入矩阵
        """
        if not self.bert_model or not self.bert_tokenizer:
            self.logger.warning("BERT模型未加载，无法提取嵌入")
            return None
        
        if not texts:
            return None
        
        self.logger.info(f"为 {len(texts)} 条文本提取BERT嵌入")
        
        # 截断文本长度
        max_length = self.preprocessing_config.get("max_text_length", 512)
        
        # 批处理大小
        batch_size = 16
        
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="提取BERT嵌入"):
                batch_texts = texts[i:i+batch_size]
                
                # 编码文本
                encoded_input = self.bert_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # 如果有GPU，将输入移到GPU
                if torch.cuda.is_available():
                    encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
                
                # 获取BERT输出
                outputs = self.bert_model(**encoded_input)
                
                # 使用[CLS]令牌的最后一层隐藏状态作为句子嵌入
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                embeddings.append(batch_embeddings)
        
        # 合并所有批次的嵌入
        all_embeddings = np.vstack(embeddings)
        
        self.logger.info(f"BERT嵌入提取完成，形状: {all_embeddings.shape}")
        return all_embeddings
    
    def extract_features(self, data):
        """
        从数据中提取文本特征
        
        参数:
            data: 文本数据字典
            
        返回:
            特征字典
        """
        self.logger.info("开始提取文本特征")
        
        session_features = {}
        comment_features = {}
        word_features = {}
        word_counter = Counter()
        
        # 处理每个会话
        for post_id, post_data in tqdm(data.items(), desc="提取会话特征", unit="会话"):
            comments = post_data.get('comments', [])
            
            # 初始化会话特征
            session_feature = {
                'post_id': post_id,
                'label': post_data.get('label', 0),
                'comment_count': len(comments),
                'avg_sentiment_compound': 0,
                'avg_offensive_score': 0,
                'max_offensive_score': 0,
                'text_embeddings': [],
                'statistical_features': {
                    'avg_char_count': 0,
                    'avg_word_count': 0,
                    'avg_word_length': 0,
                    'avg_uppercase_ratio': 0,
                    'avg_unique_word_ratio': 0
                }
            }
            
            # 初始化会话级情感特征聚合
            session_emotion_features = {
                'sentiment_score': 0,
                'emotion_scores': {
                    'anger': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 
                    'disgust': 0, 'trust': 0, 'surprise': 0, 'anticipation': 0
                },
                'bullying_indicators': {
                    'bullying_words_ratio': 0,
                    'swear_ratio': 0,
                    'aggression_score': 0
                }
            }
            
            session_texts = []
            valid_comments = 0
            
            # 处理每条评论
            for comment in comments:
                comment_id = comment.get('commentId', '')
                comment_text = comment.get('commentText', '')
                
                if not comment_text:
                    continue
                
                # 预处理文本
                tokens = self.preprocess_text(comment_text)
                
                # 更新词频统计
                word_counter.update(tokens)
                
                # 提取基础情感特征（向后兼容）
                sentiment_features = self.extract_sentiment_features(comment_text)
                
                # 提取完整情感特征（使用内置方法）
                complete_emotion_features = None
                if self.emotion_extractor:
                    complete_emotion_features = self.extract_emotion_features(comment_text)
                    
                    # 更新会话级情感特征聚合
                    session_emotion_features['sentiment_score'] += complete_emotion_features['sentiment_score']
                    
                    for emotion, score in complete_emotion_features['emotion_scores'].items():
                        if emotion in session_emotion_features['emotion_scores']:
                            session_emotion_features['emotion_scores'][emotion] += score
                    
                    for indicator, value in complete_emotion_features['bullying_indicators'].items():
                        if indicator in session_emotion_features['bullying_indicators']:
                            session_emotion_features['bullying_indicators'][indicator] += value
                
                # 提取攻击性特征
                offensive_features = self.extract_offensive_features(tokens)
                
                # 提取统计特征
                statistical_features = self.extract_statistical_features(comment_text, tokens)
                
                # 存储评论特征
                comment_feature = {
                    'comment_id': comment_id,
                    'post_id': post_id,
                    'tokens': tokens,
                    'sentiment_features': sentiment_features,
                    'offensive_features': offensive_features,
                    'statistical_features': statistical_features
                }
                
                # 添加完整情感特征（如果可用）
                if complete_emotion_features:
                    comment_feature['emotion_features'] = complete_emotion_features
                
                comment_features[comment_id] = comment_feature
                
                # 更新会话级特征
                session_feature['avg_sentiment_compound'] += sentiment_features['compound']
                session_feature['avg_offensive_score'] += offensive_features['offensive_score']
                session_feature['max_offensive_score'] = max(
                    session_feature['max_offensive_score'], 
                    offensive_features['offensive_score']
                )
                
                # 更新会话级统计特征
                for key in session_feature['statistical_features']:
                    stat_key = key[4:]  # 移除'avg_'前缀
                    if stat_key in statistical_features:
                        session_feature['statistical_features'][key] += statistical_features[stat_key]
                
                # 添加到会话文本列表
                session_texts.append(comment_text)
                valid_comments += 1
            
            # 计算会话级平均值
            if valid_comments > 0:
                session_feature['avg_sentiment_compound'] /= valid_comments
                session_feature['avg_offensive_score'] /= valid_comments
                
                for key in session_feature['statistical_features']:
                    session_feature['statistical_features'][key] /= valid_comments
                
                # 计算会话级情感特征平均值
                if self.emotion_extractor:
                    session_emotion_features['sentiment_score'] /= valid_comments
                    
                    for emotion in session_emotion_features['emotion_scores']:
                        session_emotion_features['emotion_scores'][emotion] /= valid_comments
                    
                    for indicator in session_emotion_features['bullying_indicators']:
                        session_emotion_features['bullying_indicators'][indicator] /= valid_comments
                    
                    # 将情感特征添加到会话特征中
                    session_feature['emotion_features'] = session_emotion_features
            
            # 合并会话文本用于BERT嵌入
            if session_texts:
                session_text = " ".join(session_texts)
                session_feature['combined_text'] = session_text
            
            session_features[post_id] = session_feature
        
        # 创建词汇表（过滤低频词）
        min_freq = self.preprocessing_config.get("min_word_frequency", 5)
        vocabulary = {word: count for word, count in word_counter.items() if count >= min_freq}
        
        # 提取词特征
        for word, count in vocabulary.items():
            word_features[word] = {
                'count': count,
                'is_offensive': word in self.offensive_words
            }
        
        self.logger.info(f"提取了 {len(session_features)} 个会话的文本特征")
        self.logger.info(f"提取了 {len(comment_features)} 条评论的特征")
        self.logger.info(f"词汇表大小: {len(vocabulary)} (词频 >= {min_freq})")
        
        if self.emotion_extractor:
            # 统计包含完整情感特征的会话数量
            emotion_sessions = sum(1 for sf in session_features.values() if 'emotion_features' in sf)
            self.logger.info(f"包含完整情感特征的会话: {emotion_sessions}/{len(session_features)}")
        
        # 提取BERT嵌入
        if self.bert_model and self.bert_tokenizer:
            # 收集所有会话文本
            all_texts = [session_features[post_id].get('combined_text', '') 
                         for post_id in session_features 
                         if 'combined_text' in session_features[post_id]]
            
            # 提取嵌入
            embeddings = self.extract_bert_embeddings(all_texts)
            
            # 将嵌入分配给会话
            if embeddings is not None:
                idx = 0
                for post_id in session_features:
                    if 'combined_text' in session_features[post_id]:
                        session_features[post_id]['text_embeddings'] = embeddings[idx].tolist()
                        idx += 1
        
        # 存储特征
        self.text_features = session_features
        self.comment_features = comment_features
        self.word_features = word_features
        self.vocabulary = vocabulary
        
        return {
            'session_features': session_features,
            'comment_features': comment_features,
            'word_features': word_features,
            'vocabulary': vocabulary
        }
    
    def get_feature_names(self):
        """
        获取特征名称列表
        
        返回:
            特征名称列表
        """
        feature_names = [
            # 情感特征
            'avg_sentiment_compound',
            
            # 攻击性特征
            'avg_offensive_score',
            'max_offensive_score',
            
            # 统计特征
            'avg_char_count',
            'avg_word_count',
            'avg_word_length',
            'avg_uppercase_ratio',
            'avg_unique_word_ratio'
        ]
        
        # 如果有BERT嵌入，添加嵌入维度
        if self.bert_model:
            embedding_dim = self.feature_config.get("text_embedding_dim", 768)
            for i in range(embedding_dim):
                feature_names.append(f'text_embedding_{i}')
        
        return feature_names
    
    def get_processed_features(self, data=None, force_reprocess=False):
        """
        获取处理后的特征，如果已存在则加载，否则提取
        
        参数:
            data: 要提取特征的数据，如果为None则尝试加载
            force_reprocess: 是否强制重新处理
            
        返回:
            处理后的特征
        """
        # 特征文件路径
        features_file = os.path.join(self.features_path, 'text_features.pkl')
        
        # 如果不强制重新处理且特征文件存在，则加载
        if not force_reprocess and os.path.exists(features_file):
            self.logger.info(f"从 {features_file} 加载已处理的文本特征")
            features = self.load_features(features_file)
            
            # 更新实例变量
            self.text_features = features.get('session_features', {})
            self.comment_features = features.get('comment_features', {})
            self.word_features = features.get('word_features', {})
            self.vocabulary = features.get('vocabulary', {})
            
            return features
        
        # 如果需要重新处理但未提供数据，则加载数据
        if data is None:
            text_data_file = os.path.join(self.config["text_data_path"], 'processed_text_data.json')
            self.logger.info(f"从 {text_data_file} 加载文本数据")
            data = self.load_processed_data(text_data_file)
        
        # 提取特征
        features = self.extract_features(data)
        
        # 保存特征
        self.save_features(features, features_file)
        
        return features

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="文本特征提取")
    parser.add_argument("--force", action="store_true", help="强制重新处理")
    args = parser.parse_args()
    
    extractor = TextFeatureExtractor()
    features = extractor.get_processed_features(force_reprocess=args.force)
    
    print(f"提取了 {len(features['session_features'])} 个会话的文本特征")
    print(f"提取了 {len(features['comment_features'])} 条评论的特征")
    print(f"词汇表大小: {len(features['vocabulary'])}")

if __name__ == "__main__":
    main() 