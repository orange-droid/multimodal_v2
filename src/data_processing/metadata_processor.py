"""
元数据处理器模块

负责处理与用户、时间、位置等元数据相关的信息，为后续分析提供支持
"""
import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import sys
from collections import defaultdict

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

class MetadataProcessor(BaseProcessor):
    """
    元数据处理器类
    
    处理与用户、时间、位置等元数据相关的信息，为后续分析提供支持
    """
    
    def __init__(self, config=None, preprocessing_config=None):
        """
        初始化元数据处理器
        
        参数:
            config: 数据配置，如果为None则使用默认配置
            preprocessing_config: 预处理配置，如果为None则使用默认配置
        """
        super().__init__(config, preprocessing_config)
        
        # 更新日志记录器名称
        self.logger.info("初始化元数据处理器")
        
        # 确保元数据目录存在
        self.metadata_data_path = self.get_absolute_path(self.config["metadata_data_path"])
        self.logger.info(f"元数据将保存到 {self.metadata_data_path}")
        ensure_dir(self.metadata_data_path)
    
    def parse_datetime(self, date_str):
        """
        解析日期时间字符串
        
        参数:
            date_str: 日期时间字符串
            
        返回:
            解析后的datetime对象，如果解析失败则返回None
        """
        if not date_str or not isinstance(date_str, str):
            return None
        
        try:
            # 尝试解析ISO格式的日期时间
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            try:
                # 尝试解析其他常见格式
                for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S'):
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
            except Exception:
                pass
        
        return None
    
    def extract_user_metadata(self, comments):
        """
        从评论中提取用户元数据
        
        参数:
            comments: 评论列表
            
        返回:
            用户元数据字典
        """
        # 初始化用户元数据字典
        users_metadata = defaultdict(lambda: {
            'comment_count': 0,
            'first_seen': None,
            'last_seen': None,
            'locations': set(),
            'verified': False,
            'private': False,
            'description': '',
            'avatarUrl': '',
            'interactions': []  # 存储互动信息
        })
        
        # 提取每个评论的用户信息
        for comment in comments:
            username = comment.get('username')
            if not username:
                continue
            
            # 更新评论计数
            users_metadata[username]['comment_count'] += 1
            
            # 更新用户资料信息
            users_metadata[username]['description'] = comment.get('description', '')
            users_metadata[username]['verified'] = comment.get('verified') == '1'
            users_metadata[username]['private'] = comment.get('private') == '1'
            users_metadata[username]['avatarUrl'] = comment.get('avatarUrl', '')
            
            # 解析评论时间
            created_time = self.parse_datetime(comment.get('created'))
            if created_time:
                # 更新首次和最后一次出现时间
                if users_metadata[username]['first_seen'] is None or created_time < users_metadata[username]['first_seen']:
                    users_metadata[username]['first_seen'] = created_time
                
                if users_metadata[username]['last_seen'] is None or created_time > users_metadata[username]['last_seen']:
                    users_metadata[username]['last_seen'] = created_time
            
            # 添加位置信息
            location = comment.get('location')
            if location:
                users_metadata[username]['locations'].add(location)
            
            # 添加互动信息
            users_metadata[username]['interactions'].append({
                'post_id': comment.get('postId'),
                'comment_id': comment.get('commentId'),
                'created': comment.get('created'),
                'text': comment.get('commentText')
            })
        
        # 将集合转换为列表，以便JSON序列化
        for username, data in users_metadata.items():
            data['locations'] = list(data['locations'])
            
            # 将datetime对象转换为字符串
            if data['first_seen']:
                data['first_seen'] = data['first_seen'].isoformat()
            if data['last_seen']:
                data['last_seen'] = data['last_seen'].isoformat()
        
        return dict(users_metadata)
    
    def extract_temporal_patterns(self, comments):
        """
        从评论中提取时间模式
        
        参数:
            comments: 评论列表
            
        返回:
            时间模式字典
        """
        # 初始化时间相关统计
        temporal_data = {
            'hour_distribution': [0] * 24,           # 按小时分布
            'day_of_week_distribution': [0] * 7,     # 按星期几分布
            'comment_intervals': [],                 # 评论间隔
            'total_timespan': None,                  # 总时间跨度
            'timestamps': []                         # 所有时间戳
        }
        
        # 解析评论时间
        parsed_timestamps = []
        for comment in comments:
            created_time = self.parse_datetime(comment.get('created'))
            if created_time:
                parsed_timestamps.append(created_time)
                temporal_data['timestamps'].append(created_time.isoformat())
                
                # 更新小时分布
                temporal_data['hour_distribution'][created_time.hour] += 1
                
                # 更新星期几分布
                temporal_data['day_of_week_distribution'][created_time.weekday()] += 1
        
        # 按时间排序
        parsed_timestamps.sort()
        
        # 计算评论间隔
        if len(parsed_timestamps) > 1:
            for i in range(1, len(parsed_timestamps)):
                interval_seconds = (parsed_timestamps[i] - parsed_timestamps[i-1]).total_seconds()
                temporal_data['comment_intervals'].append(interval_seconds)
        
        # 计算总时间跨度
        if len(parsed_timestamps) > 1:
            total_seconds = (parsed_timestamps[-1] - parsed_timestamps[0]).total_seconds()
            temporal_data['total_timespan'] = total_seconds
        
        return temporal_data
    
    def process_post_metadata(self, post_id, comments):
        """
        处理单个帖子的元数据
        
        参数:
            post_id: 帖子ID
            comments: 评论列表
            
        返回:
            处理后的元数据字典
        """
        self.logger.info(f"处理帖子 {post_id} 的元数据")
        
        if not comments:
            self.logger.warning(f"帖子 {post_id} 没有评论")
            return None
        
        # 提取用户元数据
        user_metadata = self.extract_user_metadata(comments)
        
        # 提取时间模式
        temporal_patterns = self.extract_temporal_patterns(comments)
        
        # 计算用户互动网络
        interaction_network = self.compute_interaction_network(comments)
        
        # 组织元数据
        metadata = {
            'post_id': post_id,
            'comment_count': len(comments),
            'user_count': len(user_metadata),
            'user_metadata': user_metadata,
            'temporal_patterns': temporal_patterns,
            'interaction_network': interaction_network
        }
        
        return metadata
    
    def compute_interaction_network(self, comments):
        """
        计算用户互动网络
        
        参数:
            comments: 评论列表
            
        返回:
            互动网络字典
        """
        # 按时间顺序排序评论
        sorted_comments = sorted(comments, key=lambda x: self.parse_datetime(x.get('created')) or datetime.min)
        
        # 构建互动网络
        network = {
            'nodes': [],   # 用户节点
            'edges': []    # 用户间的互动边
        }
        
        # 提取所有用户
        users = set()
        for comment in sorted_comments:
            username = comment.get('username')
            if username:
                users.add(username)
        
        # 添加节点
        for username in users:
            network['nodes'].append({
                'id': username,
                'type': 'user'
            })
        
        # 分析可能的互动
        # 假设如果用户A在用户B评论后很短时间内评论，可能是回复
        user_comments = defaultdict(list)
        for comment in sorted_comments:
            username = comment.get('username')
            if username:
                user_comments[username].append({
                    'id': comment.get('commentId'),
                    'text': comment.get('commentText', ''),
                    'time': self.parse_datetime(comment.get('created'))
                })
        
        # 检测可能的互动边
        # 这里使用一个简单的启发式方法：检查评论文本中是否包含其他用户名
        for username, comments in user_comments.items():
            for comment in comments:
                # 检查评论文本中是否提到其他用户
                text = comment.get('text', '').lower()
                for other_user in users:
                    if username != other_user and other_user.lower() in text:
                        network['edges'].append({
                            'source': username,
                            'target': other_user,
                            'type': 'mention',
                            'comment_id': comment.get('id')
                        })
        
        return network
    
    def process(self, force_reprocess=False):
        """
        处理所有元数据
        
        参数:
            force_reprocess: 是否强制重新处理数据
            
        返回:
            处理后的元数据字典
        """
        # 检查处理后的数据是否已存在
        processed_file = os.path.join(self.metadata_data_path, 'processed_metadata.json')
        if os.path.exists(processed_file) and not force_reprocess:
            self.logger.info(f"使用已存在的处理后数据: {processed_file}")
            return self.load_processed_data(processed_file)
        
        self.logger.info("开始处理元数据")
        
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
        
        # 处理每个帖子的元数据
        processed_metadata = {}
        skipped_count = 0
        
        for post_id, comments in tqdm(comments_by_post.items(), desc="处理帖子元数据", unit="帖子"):
            # 查找对应的视频URL和标签
            video_url = postid_to_url.get(post_id)
            
            if not video_url or video_url not in video_url_to_label:
                skipped_count += 1
                continue
            
            # 获取标签
            label_info = video_url_to_label[video_url]
            
            # 处理元数据
            post_metadata = self.process_post_metadata(post_id, comments)
            if post_metadata:
                post_metadata['label'] = label_info['is_bullying']
                post_metadata['caption'] = label_info['caption']
                post_metadata['username'] = label_info['username']
                post_metadata['video_url'] = video_url
                
                processed_metadata[post_id] = post_metadata
        
        self.logger.info(f"完成元数据处理，处理了 {len(processed_metadata)} 个帖子，跳过了 {skipped_count} 个帖子")
        
        # 保存处理后的数据
        self.save_processed_data(processed_metadata, processed_file)
        
        return processed_metadata
    
    def get_processed_data(self, force_reprocess=False):
        """
        获取处理后的数据
        
        参数:
            force_reprocess: 是否强制重新处理数据
            
        返回:
            处理后的数据字典
        """
        return self.process(force_reprocess) 