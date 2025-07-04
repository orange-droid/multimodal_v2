"""
视频图构建器

负责构建基于视频内容的图结构，包括用户节点、视频片段节点、表情节点和动作节点。
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict, Counter
from .base_graph_builder import BaseGraphBuilder

class VideoGraphBuilder(BaseGraphBuilder):
    """视频图构建器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化视频图构建器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        
        # 视频处理配置
        self.min_segment_duration = config.get('min_segment_duration', 1.0)
        self.max_segments_per_video = config.get('max_segments_per_video', 50)
        self.include_emotion_edges = config.get('include_emotion_edges', True)
        self.include_action_edges = config.get('include_action_edges', True)
        
        # 表情类型
        self.emotion_types = {
            'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral'
        }
        
        # 动作类型
        self.action_types = {
            'speaking', 'gesturing', 'moving', 'static', 'aggressive', 'defensive'
        }
        
        # 场景类型
        self.scene_types = {
            'indoor', 'outdoor', 'public', 'private', 'school', 'home', 'social'
        }
    
    def build_graph(self, data: Dict[str, Any], **kwargs) -> HeteroData:
        """
        构建视频图
        
        Args:
            data: 包含视频特征和视频数据的字典
            **kwargs: 其他参数
            
        Returns:
            异构图对象
        """
        self.logger.info("Building video graph...")
        
        # 提取数据
        video_features = data.get('video_features', {})
        video_data = data.get('video_data', {})
        
        # 创建节点
        user_nodes, video_nodes, emotion_nodes, action_nodes, scene_nodes = self._create_nodes(
            video_features, video_data
        )
        
        # 创建边
        edges = self._create_edges(video_features, video_data,
                                 user_nodes, video_nodes, emotion_nodes, 
                                 action_nodes, scene_nodes)
        
        # 构建异构图
        graph = self._build_hetero_graph(
            user_nodes, video_nodes, emotion_nodes, action_nodes, scene_nodes, edges
        )
        
        self.logger.info(f"Built video graph with {graph.num_nodes} nodes")
        return graph
    
    def _create_nodes(self, video_features: Dict[str, Any], 
                     video_data: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """创建节点"""
        user_nodes = {}
        video_nodes = {}
        emotion_nodes = {}
        action_nodes = {}
        scene_nodes = {}
        
        # 用户节点
        all_users = set()
        for post_id, post_data in video_data.items():
            # video_data中的用户字段可能是'user'或'username'
            user = post_data.get('user', post_data.get('username', 'unknown'))
            
            # 处理NaN和None值
            if user is None or (hasattr(user, '__iter__') and user != user):  # NaN检查
                user = 'unknown'
            
            # 确保用户名是字符串类型
            if not isinstance(user, str):
                user = str(user) if user is not None else 'unknown'
                
            all_users.add(user)
        
        # 过滤掉可能的NaN值并排序
        filtered_users = [user for user in all_users if isinstance(user, str) and user == user]
        
        for idx, user in enumerate(sorted(filtered_users)):
            user_nodes[user] = {
                'id': idx,
                'name': user,
                'type': 'user'
            }
        
        # 视频节点（基于帖子）
        video_idx = 0
        for post_id, post_data in video_data.items():
            user = post_data.get('user', post_data.get('username', 'unknown'))
            
            # 处理NaN和None值
            if user is None or (hasattr(user, '__iter__') and user != user):  # NaN检查
                user = 'unknown'
            
            # 确保用户名是字符串类型
            if not isinstance(user, str):
                user = str(user) if user is not None else 'unknown'
            
            video_nodes[post_id] = {
                'id': video_idx,
                'post_id': post_id,
                'user': user,
                'title': post_data.get('title', ''),
                'duration': post_data.get('duration', 0),
                'view_count': post_data.get('view_count', 0),
                'type': 'video'
            }
            video_idx += 1
        
        # 表情节点
        for idx, emotion in enumerate(sorted(self.emotion_types)):
            emotion_nodes[emotion] = {
                'id': idx,
                'emotion': emotion,
                'type': 'emotion'
            }
        
        # 动作节点
        for idx, action in enumerate(sorted(self.action_types)):
            action_nodes[action] = {
                'id': idx,
                'action': action,
                'type': 'action'
            }
        
        # 场景节点
        for idx, scene in enumerate(sorted(self.scene_types)):
            scene_nodes[scene] = {
                'id': idx,
                'scene': scene,
                'type': 'scene'
            }
        
        self.logger.info(f"Created {len(user_nodes)} user nodes, "
                        f"{len(video_nodes)} video nodes, "
                        f"{len(emotion_nodes)} emotion nodes, "
                        f"{len(action_nodes)} action nodes, "
                        f"{len(scene_nodes)} scene nodes")
        
        return user_nodes, video_nodes, emotion_nodes, action_nodes, scene_nodes
    
    def _create_edges(self, video_features: Dict[str, Any], 
                     video_data: Dict[str, Any],
                     user_nodes: Dict, video_nodes: Dict, 
                     emotion_nodes: Dict, action_nodes: Dict, 
                     scene_nodes: Dict) -> Dict[str, List[Tuple[int, int]]]:
        """创建边"""
        edges = defaultdict(list)
        
        # 用户-视频边（创建关系）
        for post_id, video_data_item in video_data.items():
            user = video_data_item.get('user', video_data_item.get('username', 'unknown'))
            
            # 处理NaN和None值
            if user is None or (hasattr(user, '__iter__') and user != user):  # NaN检查
                user = 'unknown'
            
            # 确保用户名是字符串类型
            if not isinstance(user, str):
                user = str(user) if user is not None else 'unknown'
            
            if user in user_nodes and post_id in video_nodes:
                user_idx = user_nodes[user]['id']
                video_idx = video_nodes[post_id]['id']
                edges['user_creates_video'].append((user_idx, video_idx))
        
        # 视频-表情边
        if self.include_emotion_edges:
            self._add_emotion_edges(edges, video_features, video_nodes, emotion_nodes)
        
        # 视频-动作边
        if self.include_action_edges:
            self._add_action_edges(edges, video_features, video_nodes, action_nodes)
        
        # 视频-场景边
        self._add_scene_edges(edges, video_features, video_nodes, scene_nodes)
        
        # 用户互动边（基于视频特征相似性）
        self._add_user_similarity_edges(edges, video_features, video_data, user_nodes)
        
        # 表情-动作关联边
        self._add_emotion_action_edges(edges, emotion_nodes, action_nodes)
        
        self.logger.info(f"Created edges: {[(k, len(v)) for k, v in edges.items()]}")
        
        return edges
    
    def _add_emotion_edges(self, edges: Dict, video_features: Dict, 
                          video_nodes: Dict, emotion_nodes: Dict):
        """添加表情边"""
        for post_id, features in video_features.items():
            if post_id in video_nodes:
                video_idx = video_nodes[post_id]['id']
                
                # 从视频特征中提取表情信息
                if 'emotion_features' in features:
                    emotion_features = features['emotion_features']
                    
                    # 假设emotion_features是一个字典，包含各种表情的强度
                    for emotion, intensity in emotion_features.items():
                        if emotion in emotion_nodes and intensity > 0.3:  # 阈值过滤
                            emotion_idx = emotion_nodes[emotion]['id']
                            edges['video_shows_emotion'].append((video_idx, emotion_idx))
                
                # 基于视觉特征推断表情
                if 'visual_features' in features:
                    visual_features = features['visual_features']
                    
                    # 简单的表情推断逻辑（基于视觉特征的统计）
                    if isinstance(visual_features, dict):
                        avg_features = visual_features.get('average', [])
                        if len(avg_features) > 0:
                            # 基于特征值推断主要表情
                            feature_sum = sum(avg_features)
                            if feature_sum > 0.6:
                                emotion_idx = emotion_nodes['happy']['id']
                                edges['video_shows_emotion'].append((video_idx, emotion_idx))
                            elif feature_sum < 0.3:
                                emotion_idx = emotion_nodes['sad']['id']
                                edges['video_shows_emotion'].append((video_idx, emotion_idx))
                            else:
                                emotion_idx = emotion_nodes['neutral']['id']
                                edges['video_shows_emotion'].append((video_idx, emotion_idx))
    
    def _add_action_edges(self, edges: Dict, video_features: Dict, 
                         video_nodes: Dict, action_nodes: Dict):
        """添加动作边"""
        for post_id, features in video_features.items():
            if post_id in video_nodes:
                video_idx = video_nodes[post_id]['id']
                
                # 从视频特征中提取动作信息
                if 'action_features' in features:
                    action_features = features['action_features']
                    
                    for action, intensity in action_features.items():
                        if action in action_nodes and intensity > 0.3:
                            action_idx = action_nodes[action]['id']
                            edges['video_contains_action'].append((video_idx, action_idx))
                
                # 基于场景特征推断动作
                if 'scene_features' in features:
                    scene_features = features['scene_features']
                    
                    # 简单的动作推断
                    if isinstance(scene_features, dict):
                        scene_changes = scene_features.get('scene_changes', 0)
                        if scene_changes > 5:  # 频繁场景变化
                            action_idx = action_nodes['moving']['id']
                            edges['video_contains_action'].append((video_idx, action_idx))
                        else:
                            action_idx = action_nodes['static']['id']
                            edges['video_contains_action'].append((video_idx, action_idx))
    
    def _add_scene_edges(self, edges: Dict, video_features: Dict, 
                        video_nodes: Dict, scene_nodes: Dict):
        """添加场景边"""
        for post_id, features in video_features.items():
            if post_id in video_nodes:
                video_idx = video_nodes[post_id]['id']
                
                # 从视频特征中提取场景信息
                if 'scene_features' in features:
                    scene_features = features['scene_features']
                    
                    # 基于场景特征推断场景类型
                    if isinstance(scene_features, dict):
                        dominant_scene = scene_features.get('dominant_scene', 'indoor')
                        if dominant_scene in scene_nodes:
                            scene_idx = scene_nodes[dominant_scene]['id']
                            edges['video_in_scene'].append((video_idx, scene_idx))
                        else:
                            # 默认场景
                            scene_idx = scene_nodes['indoor']['id']
                            edges['video_in_scene'].append((video_idx, scene_idx))
                else:
                    # 如果没有场景特征，使用默认场景
                    scene_idx = scene_nodes['indoor']['id']
                    edges['video_in_scene'].append((video_idx, scene_idx))
    
    def _add_user_similarity_edges(self, edges: Dict, video_features: Dict, 
                                  video_data: Dict, user_nodes: Dict):
        """添加用户相似性边"""
        # 计算用户之间的视频特征相似性
        user_features = {}
        
        for post_id, features in video_features.items():
            post_data = video_data.get(post_id, {})
            user = post_data.get('user', post_data.get('username', 'unknown'))
            
            # 处理NaN和None值
            if user is None or (hasattr(user, '__iter__') and user != user):  # NaN检查
                user = 'unknown'
            
            # 确保用户名是字符串类型
            if not isinstance(user, str):
                user = str(user) if user is not None else 'unknown'
            
            if user in user_nodes and 'visual_features' in features:
                visual_features = features['visual_features']
                if isinstance(visual_features, dict) and 'average' in visual_features:
                    if user not in user_features:
                        user_features[user] = []
                    user_features[user].append(visual_features['average'])
        
        # 计算用户特征的平均值
        user_avg_features = {}
        for user, feature_list in user_features.items():
            if feature_list:
                avg_feature = np.mean(feature_list, axis=0)
                user_avg_features[user] = avg_feature
        
        # 计算用户之间的相似性
        users = list(user_avg_features.keys())
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user1, user2 = users[i], users[j]
                feature1 = user_avg_features[user1]
                feature2 = user_avg_features[user2]
                
                # 计算余弦相似性
                similarity = np.dot(feature1, feature2) / (
                    np.linalg.norm(feature1) * np.linalg.norm(feature2) + 1e-8
                )
                
                if similarity > 0.7:  # 相似性阈值
                    user1_idx = user_nodes[user1]['id']
                    user2_idx = user_nodes[user2]['id']
                    edges['user_similar_user'].append((user1_idx, user2_idx))
    
    def _add_emotion_action_edges(self, edges: Dict, emotion_nodes: Dict, 
                                 action_nodes: Dict):
        """添加表情-动作关联边"""
        # 定义表情和动作之间的关联关系
        emotion_action_pairs = [
            ('angry', 'aggressive'),
            ('fear', 'defensive'),
            ('happy', 'gesturing'),
            ('sad', 'static'),
            ('surprise', 'moving'),
            ('neutral', 'speaking')
        ]
        
        for emotion, action in emotion_action_pairs:
            if emotion in emotion_nodes and action in action_nodes:
                emotion_idx = emotion_nodes[emotion]['id']
                action_idx = action_nodes[action]['id']
                edges['emotion_triggers_action'].append((emotion_idx, action_idx))
    
    def _build_hetero_graph(self, user_nodes: Dict, video_nodes: Dict, 
                          emotion_nodes: Dict, action_nodes: Dict, 
                          scene_nodes: Dict, edges: Dict) -> HeteroData:
        """构建异构图"""
        graph = HeteroData()
        
        # 添加节点特征
        # 用户节点特征
        num_users = len(user_nodes)
        user_features = torch.zeros(num_users, 16)
        for user, node_data in user_nodes.items():
            idx = node_data['id']
            user_features[idx, 0] = 1.0  # 用户类型标识
        
        graph['user'].x = user_features
        graph['user'].num_nodes = num_users
        
        # 视频节点特征
        num_videos = len(video_nodes)
        video_features = torch.zeros(num_videos, 32)
        for post_id, node_data in video_nodes.items():
            idx = node_data['id']
            video_features[idx, 0] = node_data.get('duration', 0)  # 视频时长
            video_features[idx, 1] = np.log1p(node_data.get('view_count', 0))  # 观看次数（对数）
            # 安全处理标题长度
            title = node_data.get('title', '')
            if isinstance(title, str):
                video_features[idx, 2] = len(title)
            else:
                video_features[idx, 2] = 0  # 如果不是字符串（如NaN），长度为0
        
        graph['video'].x = video_features
        graph['video'].num_nodes = num_videos
        
        # 表情节点特征
        num_emotions = len(emotion_nodes)
        emotion_features = torch.zeros(num_emotions, 8)
        for emotion, node_data in emotion_nodes.items():
            idx = node_data['id']
            emotion_features[idx, 0] = 1.0  # 表情类型标识
            # 可以添加更多表情相关特征
        
        graph['emotion'].x = emotion_features
        graph['emotion'].num_nodes = num_emotions
        
        # 动作节点特征
        num_actions = len(action_nodes)
        action_features = torch.zeros(num_actions, 8)
        for action, node_data in action_nodes.items():
            idx = node_data['id']
            action_features[idx, 0] = 1.0  # 动作类型标识
        
        graph['action'].x = action_features
        graph['action'].num_nodes = num_actions
        
        # 场景节点特征
        num_scenes = len(scene_nodes)
        scene_features = torch.zeros(num_scenes, 8)
        for scene, node_data in scene_nodes.items():
            idx = node_data['id']
            scene_features[idx, 0] = 1.0  # 场景类型标识
        
        graph['scene'].x = scene_features
        graph['scene'].num_nodes = num_scenes
        
        # 添加边
        for edge_type, edge_list in edges.items():
            if edge_list:
                edge_tensor = torch.tensor(edge_list, dtype=torch.long).t()
                
                if edge_type == 'user_creates_video':
                    graph['user', 'creates', 'video'].edge_index = edge_tensor
                elif edge_type == 'video_shows_emotion':
                    graph['video', 'shows', 'emotion'].edge_index = edge_tensor
                elif edge_type == 'video_contains_action':
                    graph['video', 'contains', 'action'].edge_index = edge_tensor
                elif edge_type == 'video_in_scene':
                    graph['video', 'in', 'scene'].edge_index = edge_tensor
                elif edge_type == 'user_similar_user':
                    graph['user', 'similar', 'user'].edge_index = edge_tensor
                elif edge_type == 'emotion_triggers_action':
                    graph['emotion', 'triggers', 'action'].edge_index = edge_tensor
        
        # 保存节点映射
        self.node_mappings['user'] = {
            node_data['name']: node_data['id'] 
            for node_data in user_nodes.values()
        }
        self.node_mappings['video'] = {
            node_data['post_id']: node_data['id'] 
            for node_data in video_nodes.values()
        }
        self.node_mappings['emotion'] = {
            node_data['emotion']: node_data['id'] 
            for node_data in emotion_nodes.values()
        }
        self.node_mappings['action'] = {
            node_data['action']: node_data['id'] 
            for node_data in action_nodes.values()
        }
        self.node_mappings['scene'] = {
            node_data['scene']: node_data['id'] 
            for node_data in scene_nodes.values()
        }
        
        return graph 