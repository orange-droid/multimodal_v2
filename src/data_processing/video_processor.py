"""
视频处理器模块

由于实际视频文件不可用，该模块主要处理从标注数据中提取的视频相关信息
"""
import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
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

class VideoProcessor(BaseProcessor):
    """
    视频处理器类
    
    由于实际视频文件不可用，该类主要处理从标注数据中提取的视频相关信息
    """
    
    def __init__(self, config=None, preprocessing_config=None):
        """
        初始化视频处理器
        
        参数:
            config: 数据配置，如果为None则使用默认配置
            preprocessing_config: 预处理配置，如果为None则使用默认配置
        """
        super().__init__(config, preprocessing_config)
        
        # 更新日志记录器名称
        self.logger.info("初始化视频处理器")
        
        # 确保视频数据目录存在
        self.video_data_path = self.get_absolute_path(self.config["video_data_path"])
        self.logger.info(f"视频数据将保存到 {self.video_data_path}")
        ensure_dir(self.video_data_path)
    
    def extract_video_metadata(self, labeled_row):
        """
        从标注数据行中提取视频元数据
        
        参数:
            labeled_row: 标注数据行
            
        返回:
            视频元数据字典
        """
        metadata = {}
        
        # 提取基本元数据
        metadata['videolink'] = labeled_row.get('videolink', '')
        metadata['username'] = labeled_row.get('username', '')
        metadata['caption'] = labeled_row.get('mediacaption', '')
        metadata['creation_time'] = labeled_row.get('creationtime', '')
        metadata['like_count'] = labeled_row.get('likecount', 0)
        
        # 处理点赞数（确保是整数）
        try:
            metadata['like_count'] = int(metadata['like_count'])
        except (ValueError, TypeError):
            metadata['like_count'] = 0
        
        return metadata
    
    def extract_content_features(self, labeled_row):
        """
        从标注数据行中提取内容特征
        
        参数:
            labeled_row: 标注数据行
            
        返回:
            内容特征字典
        """
        features = {}
        
        # 尝试从column数据中提取特征
        # 注意：column列可能包含标注者对视频内容的评价，如情感、主题等
        emotion_columns = []
        topic_columns = []
        demographic_columns = []
        
        # 处理其他特殊列
        for col in labeled_row.index:
            # 跳过非column列和空值
            if not col.startswith('column') or pd.isna(labeled_row[col]):
                continue
            
            value = labeled_row[col]
            if not isinstance(value, str):
                continue
            
            # 基于列值推断特征类型
            lower_value = value.lower()
            
            # 情感相关列
            if any(term in lower_value for term in ['happy', 'sad', 'angry', 'excited', 'emotion', 'feeling']):
                emotion_columns.append((col, value))
            
            # 主题相关列
            elif any(term in lower_value for term in ['topic', 'subject', 'about', 'theme']):
                topic_columns.append((col, value))
            
            # 人口统计相关列
            elif any(term in lower_value for term in ['age', 'gender', 'race', 'demographic']):
                demographic_columns.append((col, value))
        
        # 组织提取的特征
        features['emotions'] = [value for _, value in emotion_columns]
        features['topics'] = [value for _, value in topic_columns]
        features['demographics'] = [value for _, value in demographic_columns]
        
        # 从视频标题/说明中提取主题词
        caption = labeled_row.get('mediacaption', '')
        if isinstance(caption, str):
            # 使用简单的方式提取关键词（去除停用词和常见词）
            words = re.findall(r'\b\w+\b', caption.lower())
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'on', 'in', 'at', 'to', 'for', 'with', 'by'}
            keywords = [word for word in words if word not in stop_words and len(word) > 2]
            features['caption_keywords'] = keywords
        else:
            features['caption_keywords'] = []
        
        return features
    
    def process_video_data(self, labeled_df):
        """
        处理视频数据
        
        参数:
            labeled_df: 标注数据DataFrame
            
        返回:
            处理后的视频数据字典
        """
        self.logger.info("处理视频数据")
        
        processed_videos = {}
        url_to_postid, _ = self.load_url_to_postid_mapping()
        
        # 处理每行标注数据
        for _, row in tqdm(labeled_df.iterrows(), desc="处理视频数据", total=len(labeled_df), unit="视频"):
            video_url = row.get('videolink')
            if not video_url or not isinstance(video_url, str):
                continue
            
            # 获取帖子ID
            post_id = url_to_postid.get(video_url)
            if not post_id:
                continue
            
            # 提取视频元数据
            metadata = self.extract_video_metadata(row)
            
            # 提取内容特征
            content_features = self.extract_content_features(row)
            
            # 构造视频数据
            video_data = {
                'post_id': post_id,
                'metadata': metadata,
                'content_features': content_features,
                'label': 1 if row.get('question2') == 'bullying' else 0  # 使用question2判断霸凌
            }
            
            processed_videos[post_id] = video_data
        
        self.logger.info(f"完成视频数据处理，处理了 {len(processed_videos)} 个视频")
        return processed_videos
    
    def extract_cyberbullying_indicators(self, labeled_df):
        """
        从标注数据中提取网络霸凌指标
        
        参数:
            labeled_df: 标注数据DataFrame
            
        返回:
            网络霸凌指标字典
        """
        self.logger.info("提取网络霸凌指标")
        
        indicators = {}
        url_to_postid, _ = self.load_url_to_postid_mapping()
        
        # 假设question2及其相关列包含霸凌类型信息
        bullying_columns = ['question2']
        
        # 处理每行标注数据
        for _, row in tqdm(labeled_df.iterrows(), desc="提取霸凌指标", total=len(labeled_df), unit="视频"):
            video_url = row.get('videolink')
            if not video_url or not isinstance(video_url, str):
                continue
            
            # 获取帖子ID
            post_id = url_to_postid.get(video_url)
            if not post_id:
                continue
            
            # 确定是否为霸凌
            is_bullying = row.get('question2') == 'bullying'  # 使用question2判断霸凌
            
            # 如果是霸凌，提取霸凌类型
            bullying_types = []
            if is_bullying:
                for col in bullying_columns:
                    if col in row and pd.notna(row[col]):
                        bullying_types.append(row[col])
            
            # 构造指标数据
            indicator_data = {
                'post_id': post_id,
                'is_bullying': is_bullying,
                'bullying_types': bullying_types
            }
            
            indicators[post_id] = indicator_data
        
        self.logger.info(f"完成网络霸凌指标提取，处理了 {len(indicators)} 个视频")
        return indicators
    
    def process(self, force_reprocess=False):
        """
        处理所有视频数据
        
        参数:
            force_reprocess: 是否强制重新处理数据
            
        返回:
            处理后的视频数据字典
        """
        # 检查处理后的数据是否已存在
        processed_file = os.path.join(self.video_data_path, 'processed_video_data.json')
        if os.path.exists(processed_file) and not force_reprocess:
            self.logger.info(f"使用已存在的处理后数据: {processed_file}")
            return self.load_processed_data(processed_file)
        
        self.logger.info("开始处理视频数据")
        
        # 加载标注数据
        labeled_df = self.load_labeled_data()
        
        # 处理视频数据
        processed_videos = self.process_video_data(labeled_df)
        
        # 提取网络霸凌指标
        bullying_indicators = self.extract_cyberbullying_indicators(labeled_df)
        
        # 合并视频数据和霸凌指标
        for post_id, video_data in processed_videos.items():
            if post_id in bullying_indicators:
                video_data['bullying_indicators'] = bullying_indicators[post_id]
        
        # 保存处理后的数据
        self.save_processed_data(processed_videos, processed_file)
        
        return processed_videos
    
    def get_processed_data(self, force_reprocess=False):
        """
        获取处理后的数据
        
        参数:
            force_reprocess: 是否强制重新处理数据
            
        返回:
            处理后的数据字典
        """
        return self.process(force_reprocess) 