"""
多模态处理器模块

负责整合文本、视频和元数据模态的数据，生成多模态数据表示
"""
import os
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
from src.data_processing.text_processor import TextProcessor
from src.data_processing.video_processor import VideoProcessor
from src.data_processing.metadata_processor import MetadataProcessor
from src.utils.utils import ensure_dir

class MultimodalProcessor(BaseProcessor):
    """
    多模态处理器类
    
    整合文本、视频和元数据模态的数据，生成多模态数据表示
    """
    
    def __init__(self, config=None, preprocessing_config=None):
        """
        初始化多模态处理器
        
        参数:
            config: 数据配置，如果为None则使用默认配置
            preprocessing_config: 预处理配置，如果为None则使用默认配置
        """
        super().__init__(config, preprocessing_config)
        
        # 更新日志记录器名称
        self.logger.info("初始化多模态处理器")
        
        # 确保多模态数据目录存在
        self.aligned_data_path = self.get_absolute_path(self.config["aligned_data_path"])
        self.logger.info(f"多模态数据将保存到 {self.aligned_data_path}")
        ensure_dir(self.aligned_data_path)
        
        # 初始化模态处理器
        self.text_processor = TextProcessor(config, preprocessing_config)
        self.video_processor = VideoProcessor(config, preprocessing_config)
        self.metadata_processor = MetadataProcessor(config, preprocessing_config)
    
    def load_all_modalities(self, force_reprocess=False):
        """
        加载所有模态的数据
        
        参数:
            force_reprocess: 是否强制重新处理数据
            
        返回:
            包含所有模态数据的字典
        """
        self.logger.info("加载所有模态的数据")
        
        # 加载文本数据
        self.logger.info("加载文本数据...")
        text_data = self.text_processor.get_processed_data(force_reprocess)
        
        # 加载视频数据
        self.logger.info("加载视频数据...")
        video_data = self.video_processor.get_processed_data(force_reprocess)
        
        # 加载元数据
        self.logger.info("加载元数据...")
        metadata = self.metadata_processor.get_processed_data(force_reprocess)
        
        return {
            'text': text_data,
            'video': video_data,
            'metadata': metadata
        }
    
    def align_modalities(self, all_modalities):
        """
        对齐不同模态的数据
        
        参数:
            all_modalities: 包含所有模态数据的字典
            
        返回:
            对齐后的多模态数据
        """
        self.logger.info("对齐不同模态的数据")
        
        # 获取所有模态的post_id
        text_post_ids = set(all_modalities['text'].keys())
        video_post_ids = set(all_modalities['video'].keys())
        metadata_post_ids = set(all_modalities['metadata'].keys())
        
        # 计算所有模态都有数据的post_id交集
        common_post_ids = text_post_ids.intersection(video_post_ids).intersection(metadata_post_ids)
        
        self.logger.info(f"文本模态帖子数: {len(text_post_ids)}")
        self.logger.info(f"视频模态帖子数: {len(video_post_ids)}")
        self.logger.info(f"元数据模态帖子数: {len(metadata_post_ids)}")
        self.logger.info(f"所有模态共有帖子数: {len(common_post_ids)}")
        
        # 对齐数据
        aligned_data = {}
        
        for post_id in tqdm(common_post_ids, desc="对齐模态数据", unit="帖子"):
            # 提取所有模态的数据
            text_post_data = all_modalities['text'][post_id]
            video_post_data = all_modalities['video'][post_id]
            metadata_post_data = all_modalities['metadata'][post_id]
            
            # 确保标签一致
            text_label = text_post_data.get('label', -1)
            video_label = video_post_data.get('label', -1)
            metadata_label = metadata_post_data.get('label', -1)
            
            if text_label != video_label or text_label != metadata_label:
                self.logger.warning(f"帖子 {post_id} 的标签不一致: 文本={text_label}, 视频={video_label}, 元数据={metadata_label}")
                continue
            
            # 构建对齐数据
            aligned_post_data = {
                'post_id': post_id,
                'label': text_label,  # 使用一致的标签
                'text_data': text_post_data,
                'video_data': video_post_data,
                'metadata': metadata_post_data,
                'caption': text_post_data.get('caption', ''),
                'username': text_post_data.get('username', ''),
                'video_url': text_post_data.get('video_url', '')
            }
            
            aligned_data[post_id] = aligned_post_data
        
        self.logger.info(f"成功对齐 {len(aligned_data)} 个帖子的多模态数据")
        return aligned_data
    
    def create_dataset_splits(self, aligned_data):
        """
        创建训练集、验证集和测试集
        
        参数:
            aligned_data: 对齐后的多模态数据
            
        返回:
            数据集划分字典
        """
        self.logger.info("创建数据集划分")
        
        # 提取数据和标签
        post_ids = list(aligned_data.keys())
        labels = [aligned_data[post_id]['label'] for post_id in post_ids]
        
        # 训练集、验证集和测试集划分
        train_ratio = self.preprocessing_config["train_test_split_ratio"]
        val_test_ratio = self.preprocessing_config["val_test_split_ratio"]
        random_seed = self.preprocessing_config["random_seed"]
        
        # 先划分训练集和剩余数据
        train_post_ids, remaining_post_ids, train_labels, remaining_labels = self.train_test_split(
            post_ids, labels, test_size=(1 - train_ratio), random_state=random_seed
        )
        
        # 将剩余数据划分为验证集和测试集
        val_post_ids, test_post_ids, val_labels, test_labels = self.train_test_split(
            remaining_post_ids, remaining_labels, test_size=(1 - val_test_ratio), random_state=random_seed
        )
        
        # 统计每个集合中的类别分布
        train_bullying = sum(train_labels)
        train_non_bullying = len(train_labels) - train_bullying
        
        val_bullying = sum(val_labels)
        val_non_bullying = len(val_labels) - val_bullying
        
        test_bullying = sum(test_labels)
        test_non_bullying = len(test_labels) - test_bullying
        
        self.logger.info(f"训练集: {len(train_post_ids)} 条 (霸凌: {train_bullying}, 非霸凌: {train_non_bullying})")
        self.logger.info(f"验证集: {len(val_post_ids)} 条 (霸凌: {val_bullying}, 非霸凌: {val_non_bullying})")
        self.logger.info(f"测试集: {len(test_post_ids)} 条 (霸凌: {test_bullying}, 非霸凌: {test_non_bullying})")
        
        # 创建数据集划分
        dataset_splits = {
            'train': train_post_ids,
            'val': val_post_ids,
            'test': test_post_ids
        }
        
        return dataset_splits
    
    def process(self, force_reprocess=False):
        """
        处理多模态数据
        
        参数:
            force_reprocess: 是否强制重新处理数据
            
        返回:
            处理后的多模态数据字典
        """
        # 检查处理后的数据是否已存在
        processed_file = os.path.join(self.aligned_data_path, 'multimodal_data.json')
        if os.path.exists(processed_file) and not force_reprocess:
            self.logger.info(f"使用已存在的处理后数据: {processed_file}")
            return self.load_processed_data(processed_file)
        
        self.logger.info("开始处理多模态数据")
        
        # 加载所有模态的数据
        all_modalities = self.load_all_modalities(force_reprocess)
        
        # 对齐不同模态的数据
        aligned_data = self.align_modalities(all_modalities)
        
        # 创建数据集划分
        dataset_splits = self.create_dataset_splits(aligned_data)
        
        # 构建处理后的多模态数据
        multimodal_data = {
            'aligned_data': aligned_data,
            'dataset_splits': dataset_splits
        }
        
        # 保存处理后的数据
        self.save_processed_data(multimodal_data, processed_file)
        
        return multimodal_data
    
    def get_processed_data(self, force_reprocess=False):
        """
        获取处理后的数据
        
        参数:
            force_reprocess: 是否强制重新处理数据
            
        返回:
            处理后的数据字典
        """
        return self.process(force_reprocess) 