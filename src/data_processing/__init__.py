"""
数据处理模块初始化文件
"""

from src.data_processing.text_processor import TextProcessor
from src.data_processing.video_processor import VideoProcessor
from src.data_processing.metadata_processor import MetadataProcessor
from src.data_processing.multimodal_processor import MultimodalProcessor

__all__ = [
    'TextProcessor', 
    'VideoProcessor', 
    'MetadataProcessor',
    'MultimodalProcessor'
] 