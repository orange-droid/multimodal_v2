"""
基础特征提取器模块

提供所有特征提取器共享的基础功能
"""
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import sys
import pickle

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# 添加ProtoBully项目目录到路径
protobully_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(protobully_root)

from src.utils.config import DATA_CONFIG, PREPROCESSING_CONFIG, FEATURE_CONFIG
from src.utils.utils import ensure_dir, save_json, load_json, save_pickle, load_pickle, get_timestamp
from src.utils.logger import setup_logger

class BaseFeatureExtractor:
    """
    基础特征提取器类
    
    提供所有特征提取器共享的基础功能，如特征加载、保存等
    """
    
    def __init__(self, config=None, preprocessing_config=None, feature_config=None):
        """
        初始化基础特征提取器
        
        参数:
            config: 数据配置，如果为None则使用默认配置
            preprocessing_config: 预处理配置，如果为None则使用默认配置
            feature_config: 特征提取配置，如果为None则使用默认配置
        """
        # 初始化配置
        self.config = config or DATA_CONFIG
        self.preprocessing_config = preprocessing_config or PREPROCESSING_CONFIG
        self.feature_config = feature_config or FEATURE_CONFIG
        
        # 初始化日志
        self.logger = setup_logger("logs")
        
        # 确保特征目录存在
        self.features_path = self.get_absolute_path(self.config["features_path"])
        ensure_dir(self.features_path)
        
        # 初始化特征存储
        self.features = {}
        self.feature_metadata = {}
    
    def get_absolute_path(self, path):
        """
        确保路径是绝对路径
        
        参数:
            path: 路径字符串，可以是相对路径或绝对路径
            
        返回:
            绝对路径
        """
        if os.path.isabs(path):
            return path
        else:
            return os.path.join(protobully_root, path)
    
    def load_processed_data(self, file_path, format='json'):
        """
        加载处理后的数据
        
        参数:
            file_path: 加载路径
            format: 数据格式，'json' 或 'pickle'
            
        返回:
            加载的数据
        """
        file_path = self.get_absolute_path(file_path)
        self.logger.info(f"从 {file_path} 加载处理后的数据")
        
        try:
            if format == 'json':
                data = load_json(file_path)
            elif format == 'pickle':
                data = load_pickle(file_path)
            else:
                raise ValueError(f"不支持的加载格式: {format}")
                
            self.logger.info(f"数据成功从 {format} 格式加载")
            return data
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            raise
    
    def save_features(self, features, file_path, format='pickle'):
        """
        保存提取的特征
        
        参数:
            features: 要保存的特征
            file_path: 保存路径
            format: 保存格式，'json' 或 'pickle'
        """
        # 确保目录存在
        file_path = self.get_absolute_path(file_path)
        directory = os.path.dirname(file_path)
        ensure_dir(directory)
        
        self.logger.info(f"保存特征到 {file_path}")
        
        try:
            if format == 'json':
                save_json(features, file_path)
            elif format == 'pickle':
                save_pickle(features, file_path)
            else:
                raise ValueError(f"不支持的保存格式: {format}")
                
            self.logger.info(f"特征成功保存为 {format} 格式")
        except Exception as e:
            self.logger.error(f"保存特征失败: {str(e)}")
            raise
    
    def load_features(self, file_path, format='pickle'):
        """
        加载特征
        
        参数:
            file_path: 加载路径
            format: 数据格式，'json' 或 'pickle'
            
        返回:
            加载的特征
        """
        file_path = self.get_absolute_path(file_path)
        self.logger.info(f"从 {file_path} 加载特征")
        
        try:
            if format == 'json':
                features = load_json(file_path)
            elif format == 'pickle':
                features = load_pickle(file_path)
            else:
                raise ValueError(f"不支持的加载格式: {format}")
                
            self.logger.info(f"特征成功从 {format} 格式加载")
            return features
        except Exception as e:
            self.logger.error(f"加载特征失败: {str(e)}")
            raise
    
    def normalize_features(self, features, method='minmax'):
        """
        标准化特征
        
        参数:
            features: 特征数组或矩阵
            method: 标准化方法，'minmax' 或 'zscore'
            
        返回:
            标准化后的特征
        """
        self.logger.info(f"使用 {method} 方法标准化特征")
        
        try:
            if method == 'minmax':
                # Min-Max标准化到[0,1]
                min_vals = np.min(features, axis=0)
                max_vals = np.max(features, axis=0)
                range_vals = max_vals - min_vals
                # 避免除以零
                range_vals[range_vals == 0] = 1
                normalized = (features - min_vals) / range_vals
                
                # 保存标准化参数
                self.feature_metadata['normalization'] = {
                    'method': 'minmax',
                    'min_vals': min_vals.tolist() if isinstance(min_vals, np.ndarray) else min_vals,
                    'max_vals': max_vals.tolist() if isinstance(max_vals, np.ndarray) else max_vals
                }
                
            elif method == 'zscore':
                # Z-score标准化
                mean_vals = np.mean(features, axis=0)
                std_vals = np.std(features, axis=0)
                # 避免除以零
                std_vals[std_vals == 0] = 1
                normalized = (features - mean_vals) / std_vals
                
                # 保存标准化参数
                self.feature_metadata['normalization'] = {
                    'method': 'zscore',
                    'mean_vals': mean_vals.tolist() if isinstance(mean_vals, np.ndarray) else mean_vals,
                    'std_vals': std_vals.tolist() if isinstance(std_vals, np.ndarray) else std_vals
                }
                
            else:
                raise ValueError(f"不支持的标准化方法: {method}")
            
            self.logger.info("特征标准化完成")
            return normalized
            
        except Exception as e:
            self.logger.error(f"特征标准化失败: {str(e)}")
            raise
    
    def apply_normalization(self, features):
        """
        应用已保存的标准化参数到新特征
        
        参数:
            features: 特征数组或矩阵
            
        返回:
            标准化后的特征
        """
        if 'normalization' not in self.feature_metadata:
            self.logger.warning("没有找到标准化参数，返回原始特征")
            return features
        
        norm_info = self.feature_metadata['normalization']
        method = norm_info['method']
        
        self.logger.info(f"使用已保存的 {method} 参数标准化特征")
        
        try:
            if method == 'minmax':
                min_vals = np.array(norm_info['min_vals'])
                max_vals = np.array(norm_info['max_vals'])
                range_vals = max_vals - min_vals
                # 避免除以零
                range_vals[range_vals == 0] = 1
                normalized = (features - min_vals) / range_vals
                
            elif method == 'zscore':
                mean_vals = np.array(norm_info['mean_vals'])
                std_vals = np.array(norm_info['std_vals'])
                # 避免除以零
                std_vals[std_vals == 0] = 1
                normalized = (features - mean_vals) / std_vals
                
            else:
                raise ValueError(f"不支持的标准化方法: {method}")
            
            self.logger.info("特征标准化完成")
            return normalized
            
        except Exception as e:
            self.logger.error(f"应用标准化参数失败: {str(e)}")
            raise
    
    def extract_features(self, data):
        """
        提取特征的方法（由子类实现）
        
        参数:
            data: 要提取特征的数据
            
        返回:
            提取的特征
        """
        raise NotImplementedError("此方法应由子类实现")
    
    def get_feature_names(self):
        """
        获取特征名称列表（由子类实现）
        
        返回:
            特征名称列表
        """
        raise NotImplementedError("此方法应由子类实现")
    
    def get_processed_features(self, data=None, force_reprocess=False):
        """
        获取处理后的特征，如果已存在则加载，否则提取
        
        参数:
            data: 要提取特征的数据，如果为None则尝试加载
            force_reprocess: 是否强制重新处理
            
        返回:
            处理后的特征
        """
        raise NotImplementedError("此方法应由子类实现") 