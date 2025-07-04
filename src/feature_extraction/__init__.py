"""
特征提取模块初始化文件

提供各种特征提取功能，包括文本特征、视频特征、用户特征和多模态特征提取
"""

from src.feature_extraction.text_feature_extractor import TextFeatureExtractor
from src.feature_extraction.video_feature_extractor import VideoFeatureExtractor
from src.feature_extraction.user_feature_extractor import UserFeatureExtractor
from src.feature_extraction.multimodal_feature_extractor import MultimodalFeatureExtractor
from src.feature_extraction.base_feature_extractor import BaseFeatureExtractor

__all__ = [
    'BaseFeatureExtractor',
    'TextFeatureExtractor',
    'VideoFeatureExtractor',
    'UserFeatureExtractor',
    'MultimodalFeatureExtractor'
] 