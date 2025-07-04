"""
元数据图构建器

负责构建基于用户、时间和位置等元数据的图结构。
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import networkx as nx
from .base_graph_builder import BaseGraphBuilder

class MetadataGraphBuilder(BaseGraphBuilder):
    """元数据图构建器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化元数据图构建器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        
        # 时间处理配置
        self.time_window_hours = config.get('time_window_hours', 24)
        self.min_interaction_count = config.get('min_interaction_count', 2)
        self.include_temporal_edges = config.get('include_temporal_edges', True)
        self.include_location_edges = config.get('include_location_edges', True)
        
        # 时间段划分（小时）
        self.time_periods = {
            'morning': (6, 12),
            'afternoon': (12, 18),
            'evening': (18, 24),
            'night': (0, 6)
        }
        
        # 位置类型
        self.location_types = {
            'home', 'school', 'work', 'public', 'social', 'unknown'
        }
    
    def build_graph(self, data: Dict[str, Any], **kwargs) -> HeteroData:
        """
        构建元数据图
        
        Args:
            data: 包含用户特征和元数据的字典
            **kwargs: 其他参数
            
        Returns:
            异构图对象
        """
        self.logger.info("Building metadata graph...")
        
        # 提取数据
        user_features = data.get('user_features', {})
        metadata = data.get('metadata', {})
        comments_data = data.get('comments_data', {})
        
        # 创建节点
        user_nodes, time_nodes, location_nodes = self._create_nodes(
            user_features, metadata, comments_data
        )
        
        # 创建边
        edges = self._create_edges(user_features, metadata, comments_data,
                                 user_nodes, time_nodes, location_nodes)
        
        # 构建异构图
        graph = self._build_hetero_graph(
            user_nodes, time_nodes, location_nodes, edges
        )
        
        self.logger.info(f"Built metadata graph with {graph.num_nodes} nodes")
        return graph
    
    def _create_nodes(self, user_features: Dict[str, Any], 
                     metadata: Dict[str, Any],
                     comments_data: Dict[str, Any]) -> Tuple[Dict, Dict, Dict]:
        """创建节点"""
        user_nodes = {}
        time_nodes = {}
        location_nodes = {}
        
        # 用户节点
        all_users = set()
        
        # 从评论数据中收集用户
        for post_id, post_data in comments_data.items():
            comments = post_data.get('comments', [])
            for comment in comments:
                user = comment.get('username', comment.get('user', 'unknown'))
                all_users.add(user)
        
        # 从用户特征中收集用户
        for post_id, features in user_features.items():
            if 'user_stats' in features:
                for user in features['user_stats'].keys():
                    all_users.add(user)
        
        for idx, user in enumerate(sorted(all_users)):
            user_nodes[user] = {
                'id': idx,
                'name': user,
                'type': 'user'
            }
        
        # 时间节点（时间段）
        for idx, period in enumerate(sorted(self.time_periods.keys())):
            time_nodes[period] = {
                'id': idx,
                'period': period,
                'start_hour': self.time_periods[period][0],
                'end_hour': self.time_periods[period][1],
                'type': 'time_period'
            }
        
        # 位置节点
        for idx, location in enumerate(sorted(self.location_types)):
            location_nodes[location] = {
                'id': idx,
                'location': location,
                'type': 'location'
            }
        
        self.logger.info(f"Created {len(user_nodes)} user nodes, "
                        f"{len(time_nodes)} time nodes, "
                        f"{len(location_nodes)} location nodes")
        
        return user_nodes, time_nodes, location_nodes
    
    def _create_edges(self, user_features: Dict[str, Any], 
                     metadata: Dict[str, Any],
                     comments_data: Dict[str, Any],
                     user_nodes: Dict, time_nodes: Dict, 
                     location_nodes: Dict) -> Dict[str, List[Tuple[int, int]]]:
        """创建边"""
        edges = defaultdict(list)
        
        # 用户互动边
        self._add_user_interaction_edges(edges, comments_data, user_nodes)
        
        # 时间边
        if self.include_temporal_edges:
            self._add_temporal_edges(edges, comments_data, user_nodes, time_nodes)
        
        # 位置边
        if self.include_location_edges:
            self._add_location_edges(edges, user_features, metadata, 
                                   user_nodes, location_nodes)
        
        # 用户相似性边（基于活动模式）
        self._add_user_similarity_edges(edges, user_features, comments_data, user_nodes)
        
        # 时间-位置关联边
        self._add_time_location_edges(edges, time_nodes, location_nodes)
        
        self.logger.info(f"Created edges: {[(k, len(v)) for k, v in edges.items()]}")
        
        return edges
    
    def _add_user_interaction_edges(self, edges: Dict, comments_data: Dict, 
                                   user_nodes: Dict):
        """添加用户互动边"""
        # 构建用户互动网络
        interaction_graph = nx.Graph()
        
        for post_id, post_data in comments_data.items():
            comments = post_data.get('comments', [])
            
            # 在同一帖子下评论的用户之间建立连接
            users_in_post = [comment.get('username', comment.get('user', 'unknown')) for comment in comments]
            users_in_post = [user for user in users_in_post if user in user_nodes]
            
            # 为每对用户添加边
            for i in range(len(users_in_post)):
                for j in range(i + 1, len(users_in_post)):
                    user1, user2 = users_in_post[i], users_in_post[j]
                    if interaction_graph.has_edge(user1, user2):
                        interaction_graph[user1][user2]['weight'] += 1
                    else:
                        interaction_graph.add_edge(user1, user2, weight=1)
        
        # 转换为边列表
        for user1, user2, data in interaction_graph.edges(data=True):
            if data['weight'] >= self.min_interaction_count:
                user1_idx = user_nodes[user1]['id']
                user2_idx = user_nodes[user2]['id']
                edges['user_interacts_user'].append((user1_idx, user2_idx))
    
    def _add_temporal_edges(self, edges: Dict, comments_data: Dict, 
                           user_nodes: Dict, time_nodes: Dict):
        """添加时间边"""
        # 分析用户活动时间模式
        user_time_activity = defaultdict(lambda: defaultdict(int))
        
        for post_id, post_data in comments_data.items():
            comments = post_data.get('comments', [])
            
            for comment in comments:
                user = comment.get('username', comment.get('user', 'unknown'))
                timestamp = comment.get('created', comment.get('timestamp', ''))
                
                if user in user_nodes and timestamp:
                    try:
                        # 解析时间戳
                        if isinstance(timestamp, str):
                            # 假设时间戳格式为 "YYYY-MM-DD HH:MM:SS"
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        else:
                            dt = timestamp
                        
                        hour = dt.hour
                        
                        # 确定时间段
                        for period, (start, end) in self.time_periods.items():
                            if start <= hour < end or (start > end and (hour >= start or hour < end)):
                                user_time_activity[user][period] += 1
                                break
                    except:
                        # 如果时间解析失败，使用默认时间段
                        user_time_activity[user]['unknown'] = user_time_activity[user].get('unknown', 0) + 1
        
        # 为活跃用户添加时间边
        for user, time_activity in user_time_activity.items():
            if user in user_nodes:
                user_idx = user_nodes[user]['id']
                
                # 找到用户最活跃的时间段
                most_active_period = max(time_activity.items(), key=lambda x: x[1])
                period, activity_count = most_active_period
                
                if period in time_nodes and activity_count >= 2:
                    time_idx = time_nodes[period]['id']
                    edges['user_active_time'].append((user_idx, time_idx))
    
    def _add_location_edges(self, edges: Dict, user_features: Dict, 
                           metadata: Dict, user_nodes: Dict, location_nodes: Dict):
        """添加位置边"""
        # 从用户特征中推断位置信息
        for post_id, features in user_features.items():
            if 'user_stats' in features:
                user_stats = features['user_stats']
                
                for user, stats in user_stats.items():
                    if user in user_nodes:
                        user_idx = user_nodes[user]['id']
                        
                        # 基于用户活动推断位置
                        # 这里使用简单的启发式规则
                        comment_count = stats.get('comment_count', 0)
                        avg_comment_length = stats.get('avg_comment_length', 0)
                        
                        # 推断位置类型
                        if comment_count > 10:
                            # 高活跃用户可能在社交场所
                            location = 'social'
                        elif avg_comment_length > 50:
                            # 长评论可能在家或工作场所
                            location = 'home'
                        else:
                            # 默认位置
                            location = 'unknown'
                        
                        if location in location_nodes:
                            location_idx = location_nodes[location]['id']
                            edges['user_at_location'].append((user_idx, location_idx))
    
    def _add_user_similarity_edges(self, edges: Dict, user_features: Dict, 
                                  comments_data: Dict, user_nodes: Dict):
        """添加用户相似性边"""
        # 计算用户活动模式相似性
        user_activity_vectors = {}
        
        for post_id, features in user_features.items():
            if 'user_stats' in features:
                user_stats = features['user_stats']
                
                for user, stats in user_stats.items():
                    if user in user_nodes:
                        # 创建用户活动向量
                        activity_vector = [
                            stats.get('comment_count', 0),
                            stats.get('avg_comment_length', 0),
                            stats.get('bullying_participation_rate', 0),
                            stats.get('avg_sentiment_score', 0),
                            stats.get('attack_word_usage_rate', 0)
                        ]
                        
                        if user not in user_activity_vectors:
                            user_activity_vectors[user] = []
                        user_activity_vectors[user].append(activity_vector)
        
        # 计算用户平均活动向量
        user_avg_vectors = {}
        for user, vectors in user_activity_vectors.items():
            if vectors:
                avg_vector = np.mean(vectors, axis=0)
                user_avg_vectors[user] = avg_vector
        
        # 计算用户之间的相似性
        users = list(user_avg_vectors.keys())
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user1, user2 = users[i], users[j]
                vector1 = user_avg_vectors[user1]
                vector2 = user_avg_vectors[user2]
                
                # 计算余弦相似性
                similarity = np.dot(vector1, vector2) / (
                    np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-8
                )
                
                if similarity > 0.6:  # 相似性阈值
                    user1_idx = user_nodes[user1]['id']
                    user2_idx = user_nodes[user2]['id']
                    edges['user_similar_activity'].append((user1_idx, user2_idx))
    
    def _add_time_location_edges(self, edges: Dict, time_nodes: Dict, 
                                location_nodes: Dict):
        """添加时间-位置关联边"""
        # 定义时间段和位置的关联关系
        time_location_pairs = [
            ('morning', 'home'),
            ('morning', 'school'),
            ('afternoon', 'school'),
            ('afternoon', 'work'),
            ('evening', 'home'),
            ('evening', 'social'),
            ('night', 'home')
        ]
        
        for time_period, location in time_location_pairs:
            if time_period in time_nodes and location in location_nodes:
                time_idx = time_nodes[time_period]['id']
                location_idx = location_nodes[location]['id']
                edges['time_associated_location'].append((time_idx, location_idx))
    
    def _build_hetero_graph(self, user_nodes: Dict, time_nodes: Dict, 
                          location_nodes: Dict, edges: Dict) -> HeteroData:
        """构建异构图"""
        graph = HeteroData()
        
        # 添加节点特征
        # 用户节点特征
        num_users = len(user_nodes)
        user_features = torch.zeros(num_users, 32)
        for user, node_data in user_nodes.items():
            idx = node_data['id']
            user_features[idx, 0] = 1.0  # 用户类型标识
            # 可以添加更多用户特征
        
        graph['user'].x = user_features
        graph['user'].num_nodes = num_users
        
        # 时间节点特征
        num_times = len(time_nodes)
        time_features = torch.zeros(num_times, 16)
        for period, node_data in time_nodes.items():
            idx = node_data['id']
            time_features[idx, 0] = node_data['start_hour'] / 24.0  # 标准化开始时间
            time_features[idx, 1] = node_data['end_hour'] / 24.0    # 标准化结束时间
            time_features[idx, 2] = (node_data['end_hour'] - node_data['start_hour']) / 24.0  # 时间段长度
        
        graph['time'].x = time_features
        graph['time'].num_nodes = num_times
        
        # 位置节点特征
        num_locations = len(location_nodes)
        location_features = torch.zeros(num_locations, 8)
        for location, node_data in location_nodes.items():
            idx = node_data['id']
            location_features[idx, 0] = 1.0  # 位置类型标识
        
        graph['location'].x = location_features
        graph['location'].num_nodes = num_locations
        
        # 添加边
        for edge_type, edge_list in edges.items():
            if edge_list:
                edge_tensor = torch.tensor(edge_list, dtype=torch.long).t()
                
                if edge_type == 'user_interacts_user':
                    graph['user', 'interacts', 'user'].edge_index = edge_tensor
                elif edge_type == 'user_active_time':
                    graph['user', 'active_at', 'time'].edge_index = edge_tensor
                elif edge_type == 'user_at_location':
                    graph['user', 'at', 'location'].edge_index = edge_tensor
                elif edge_type == 'user_similar_activity':
                    graph['user', 'similar_activity', 'user'].edge_index = edge_tensor
                elif edge_type == 'time_associated_location':
                    graph['time', 'associated_with', 'location'].edge_index = edge_tensor
        
        # 保存节点映射
        self.node_mappings['user'] = {
            node_data['name']: node_data['id'] 
            for node_data in user_nodes.values()
        }
        self.node_mappings['time'] = {
            node_data['period']: node_data['id'] 
            for node_data in time_nodes.values()
        }
        self.node_mappings['location'] = {
            node_data['location']: node_data['id'] 
            for node_data in location_nodes.values()
        }
        
        return graph 