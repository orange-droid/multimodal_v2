"""
文本处理器模块

负责处理与文本相关的数据，包括评论文本的清洗、标准化和特征提取预处理
"""
import os
import re
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sys

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# 添加ProtoBully项目目录到路径
protobully_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(protobully_root)

from src.utils.config import DATA_CONFIG, PREPROCESSING_CONFIG
from src.data_processing.base_processor import BaseProcessor
from src.utils.utils import ensure_dir

class TextProcessor(BaseProcessor):
    """
    文本处理器类
    
    处理与文本相关的数据，包括评论文本的清洗、标准化和特征提取预处理
    """
    
    def __init__(self, config=None, preprocessing_config=None):
        """
        初始化文本处理器
        
        参数:
            config: 数据配置，如果为None则使用默认配置
            preprocessing_config: 预处理配置，如果为None则使用默认配置
        """
        super().__init__(config, preprocessing_config)
        
        # 更新日志记录器名称
        self.logger.info("初始化文本处理器")
        
        # 确保文本数据目录存在
        self.text_data_path = self.get_absolute_path(self.config["text_data_path"])
        self.logger.info(f"文本数据将保存到 {self.text_data_path}")
        ensure_dir(self.text_data_path)
        
        # 加载NLTK资源
        self._load_nltk_resources()
    
    def _load_nltk_resources(self):
        """加载NLTK所需资源"""
        try:
            # 尝试加载停用词，如果失败则下载
            nltk.data.find('corpora/stopwords')
        except LookupError:
            self.logger.info("下载NLTK停用词")
            nltk.download('stopwords')
        
        try:
            # 尝试加载分词器，如果失败则下载
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("下载NLTK分词器")
            nltk.download('punkt')
    
    def preprocess_text(self, text, remove_stopwords=True, remove_punctuation=True, 
                         lowercase=True, min_word_length=2):
        """
        预处理文本
        
        参数:
            text: 输入文本
            remove_stopwords: 是否移除停用词
            remove_punctuation: 是否移除标点符号
            lowercase: 是否转换为小写
            min_word_length: 最小词长度
            
        返回:
            预处理后的文本列表
        """
        if not text or not isinstance(text, str):
            return []
        
        # 转换为小写
        if lowercase:
            text = text.lower()
        
        # 移除标点符号
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 分词
        tokens = word_tokenize(text)
        
        # 移除停用词和过短的词
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words and len(token) >= min_word_length]
        else:
            tokens = [token for token in tokens if len(token) >= min_word_length]
        
        return tokens
    
    def extract_text_features(self, comments):
        """
        从评论中提取文本特征
        
        参数:
            comments: 评论列表
            
        返回:
            文本特征字典
        """
        # 初始化特征字典
        features = {
            'tokens': [],                    # 分词结果
            'text_length': [],               # 文本长度
            'avg_word_length': [],           # 平均词长
            'token_count': [],               # 词数
            'uppercase_ratio': [],           # 大写字母比例
            'special_char_count': [],        # 特殊字符数量
            'original_text': []              # 原始文本
        }
        
        for comment in comments:
            text = comment.get('commentText', '')
            features['original_text'].append(text)
            
            # 文本长度
            features['text_length'].append(len(text))
            
            # 分词
            tokens = self.preprocess_text(text, remove_stopwords=False, remove_punctuation=False)
            cleaned_tokens = self.preprocess_text(text)
            features['tokens'].append(cleaned_tokens)
            
            # 词数
            features['token_count'].append(len(tokens) if tokens else 0)
            
            # 平均词长
            if tokens:
                avg_word_length = sum(len(token) for token in tokens) / len(tokens)
                features['avg_word_length'].append(avg_word_length)
            else:
                features['avg_word_length'].append(0)
            
            # 大写字母比例
            if text:
                uppercase_count = sum(1 for c in text if c.isupper())
                features['uppercase_ratio'].append(uppercase_count / len(text))
            else:
                features['uppercase_ratio'].append(0)
            
            # 特殊字符数量
            special_chars = re.findall(r'[!@#$%^&*(),.?":{}|<>]', text)
            features['special_char_count'].append(len(special_chars))
        
        return features
    
    def process_post_comments(self, post_id, comments):
        """
        处理单个帖子的所有评论
        
        参数:
            post_id: 帖子ID
            comments: 评论列表
            
        返回:
            处理后的评论特征字典
        """
        self.logger.info(f"处理帖子 {post_id} 的评论")
        
        if not comments:
            self.logger.warning(f"帖子 {post_id} 没有评论")
            return None
        
        # 提取文本特征（用于向后兼容）
        text_features = self.extract_text_features(comments)
        
        # 组织特征，保留原始评论数据
        processed_data = {
            'post_id': post_id,
            'comment_count': len(comments),
            'comments': comments,  # 保存原始评论数据
            'text_features': text_features,  # 保留聚合特征（向后兼容）
            'metadata': {
                'username': [comment.get('username') for comment in comments],
                'created': [comment.get('created') for comment in comments],
                'commentId': [comment.get('commentId') for comment in comments]
            }
        }
        
        return processed_data
    
    def process(self, force_reprocess=False):
        """
        处理所有文本数据
        
        参数:
            force_reprocess: 是否强制重新处理数据
            
        返回:
            处理后的数据字典
        """
        # 检查处理后的数据是否已存在
        processed_file = os.path.join(self.text_data_path, 'processed_text_data.json')
        if os.path.exists(processed_file) and not force_reprocess:
            self.logger.info(f"使用已存在的处理后数据: {processed_file}")
            return self.load_processed_data(processed_file)
        
        self.logger.info("开始处理文本数据")
        
        # 加载评论数据
        comments_by_post = self.load_comments_data()
        
        # 加载标注数据和URL到PostID映射
        labeled_df = self.load_labeled_data()
        url_to_postid, postid_to_url = self.load_url_to_postid_mapping()
        
        # 提取标注数据中的视频链接到帖子ID的映射
        video_url_to_label = {}
        for _, row in labeled_df.iterrows():
            video_url = row.get('videolink')
            if video_url and isinstance(video_url, str):
                video_url_to_label[video_url] = {
                    'is_bullying': 1 if row.get('question2') == 'bullying' else 0,  # 使用question2判断霸凌
                    'caption': row.get('mediacaption', ''),
                    'username': row.get('username', '')
                }
        
        # 处理每个帖子的评论
        processed_data = {}
        skipped_count = 0
        
        for post_id, comments in tqdm(comments_by_post.items(), desc="处理帖子评论", unit="帖子"):
            # 查找对应的视频URL和标签
            video_url = postid_to_url.get(post_id)
            
            if not video_url or video_url not in video_url_to_label:
                skipped_count += 1
                continue
            
            # 获取标签
            label_info = video_url_to_label[video_url]
            
            # 处理评论
            post_data = self.process_post_comments(post_id, comments)
            if post_data:
                post_data['label'] = label_info['is_bullying']
                post_data['caption'] = label_info['caption']
                post_data['username'] = label_info['username']
                post_data['video_url'] = video_url
                
                processed_data[post_id] = post_data
        
        self.logger.info(f"完成文本数据处理，处理了 {len(processed_data)} 个帖子，跳过了 {skipped_count} 个帖子")
        
        # 保存处理后的数据
        self.save_processed_data(processed_data, processed_file)
        
        return processed_data
    
    def get_processed_data(self, force_reprocess=False):
        """
        获取处理后的数据
        
        参数:
            force_reprocess: 是否强制重新处理数据
            
        返回:
            处理后的数据字典
        """
        return self.process(force_reprocess) 