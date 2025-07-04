"""
基础数据处理器模块

提供所有数据处理器共享的基础功能
"""
import os
import json
import pandas as pd
from tqdm import tqdm
import logging
import sys

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# 添加ProtoBully项目目录到路径
protobully_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(protobully_root)

from src.utils.config import DATA_CONFIG, PREPROCESSING_CONFIG
from src.utils.utils import ensure_dir, save_json, load_json, save_pickle, load_pickle, get_timestamp
from src.utils.logger import setup_logger

class BaseProcessor:
    """
    基础数据处理器类
    
    提供所有数据处理器共享的基础功能，如数据加载、保存等
    """
    
    def __init__(self, config=None, preprocessing_config=None):
        """
        初始化基础处理器
        
        参数:
            config: 数据配置，如果为None则使用默认配置
            preprocessing_config: 预处理配置，如果为None则使用默认配置
        """
        # 初始化配置
        self.config = config or DATA_CONFIG
        self.preprocessing_config = preprocessing_config or PREPROCESSING_CONFIG
        
        # 初始化日志
        self.logger = setup_logger("logs")
        
        # 确保处理后的数据目录存在
        self.processed_data_path = self.get_absolute_path(self.config["processed_data_path"])
        ensure_dir(self.processed_data_path)
    
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
    
    def load_labeled_data(self):
        """
        加载标注数据
        
        返回:
            标注数据DataFrame
        """
        labeled_data_path = self.get_absolute_path(self.config["vine_labeled_data"])
        self.logger.info(f"从 {labeled_data_path} 加载标注数据")
        
        # 尝试多种编码格式
        encodings = ['utf-8', 'latin-1', 'gbk', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                self.logger.info(f"尝试使用 {encoding} 编码加载数据")
                labeled_df = pd.read_csv(labeled_data_path, encoding=encoding)
                self.logger.info(f"成功使用 {encoding} 编码加载 {len(labeled_df)} 条标注数据")
                return labeled_df
            except UnicodeDecodeError as e:
                self.logger.warning(f"使用 {encoding} 编码失败: {str(e)}")
                continue
            except Exception as e:
                self.logger.error(f"使用 {encoding} 编码加载数据时发生其他错误: {str(e)}")
                continue
        
        # 如果所有编码都失败了，抛出异常
        self.logger.error("所有编码格式都无法加载标注数据")
        raise Exception("无法使用任何支持的编码格式加载标注数据文件")
    
    def load_comments_data(self):
        """
        加载评论数据
        
        返回:
            评论数据字典，格式为 {post_id: [comments]}
        """
        comments_path = self.get_absolute_path(self.config["vine_comments_data"])
        self.logger.info(f"从 {comments_path} 加载评论数据")
        
        try:
            # 按帖子ID组织评论
            comments_by_post = {}
            
            # 使用tqdm显示进度条
            with open(comments_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="加载评论数据", unit="评论"):
                    try:
                        comment = json.loads(line)
                        post_id = comment.get('postId')
                        
                        if post_id:
                            if post_id not in comments_by_post:
                                comments_by_post[post_id] = []
                            
                            comments_by_post[post_id].append(comment)
                    except json.JSONDecodeError:
                        continue
            
            self.logger.info(f"成功加载 {sum(len(comments) for comments in comments_by_post.values())} 条评论，涉及 {len(comments_by_post)} 个帖子")
            return comments_by_post
        except Exception as e:
            self.logger.error(f"加载评论数据失败: {str(e)}")
            raise
    
    def load_url_to_postid_mapping(self):
        """
        加载URL到PostID的映射
        
        返回:
            映射字典，格式为 {url: post_id} 和反向映射 {post_id: url}
        """
        mapping_path = self.get_absolute_path(self.config["urls_to_postids"])
        self.logger.info(f"从 {mapping_path} 加载URL到PostID映射")
        
        try:
            url_to_postid = {}
            postid_to_url = {}
            
            with open(mapping_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        postid, url = parts
                        url_to_postid[url] = postid
                        postid_to_url[postid] = url
            
            self.logger.info(f"成功加载 {len(url_to_postid)} 个URL到PostID映射")
            return url_to_postid, postid_to_url
        except Exception as e:
            self.logger.error(f"加载URL到PostID映射失败: {str(e)}")
            raise
    
    def save_processed_data(self, data, file_path, format='json'):
        """
        保存处理后的数据
        
        参数:
            data: 要保存的数据
            file_path: 保存路径
            format: 保存格式，'json' 或 'pickle'
        """
        # 确保目录存在
        file_path = self.get_absolute_path(file_path)
        directory = os.path.dirname(file_path)
        ensure_dir(directory)
        
        self.logger.info(f"保存处理后的数据到 {file_path}")
        
        try:
            if format == 'json':
                save_json(data, file_path)
            elif format == 'pickle':
                save_pickle(data, file_path)
            else:
                raise ValueError(f"不支持的保存格式: {format}")
                
            self.logger.info(f"数据成功保存为 {format} 格式")
        except Exception as e:
            self.logger.error(f"保存数据失败: {str(e)}")
            raise
    
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
    
    def process(self):
        """
        处理数据的方法（由子类实现）
        
        返回:
            处理后的数据
        """
        raise NotImplementedError("此方法应由子类实现")
    
    def train_test_split(self, data, labels, test_size=None, random_state=None):
        """
        将数据分割为训练集和测试集
        
        参数:
            data: 数据
            labels: 标签
            test_size: 测试集比例，如果为None则使用配置中的值
            random_state: 随机种子，如果为None则使用配置中的值
            
        返回:
            (train_data, test_data, train_labels, test_labels)
        """
        from sklearn.model_selection import train_test_split as sklearn_split
        
        test_size = test_size or (1 - self.preprocessing_config["train_test_split_ratio"])
        random_state = random_state or self.preprocessing_config["random_seed"]
        
        self.logger.info(f"将数据分割为训练集和测试集 (测试集比例: {test_size}, 随机种子: {random_state})")
        
        try:
            train_data, test_data, train_labels, test_labels = sklearn_split(
                data, labels, test_size=test_size, random_state=random_state, stratify=labels
            )
            
            self.logger.info(f"分割完成。训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条")
            return train_data, test_data, train_labels, test_labels
        except Exception as e:
            self.logger.error(f"数据分割失败: {str(e)}")
            raise 