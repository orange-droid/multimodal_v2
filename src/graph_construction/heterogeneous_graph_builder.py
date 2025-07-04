"""
异构图构建器（重构版）

负责整合文本、视频和元数据模态的图结构，构建统一的多模态异构图。
重构后支持媒体会话节点和正确的ID映射。
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Any, Tuple, Set, Optional
from collections import defaultdict
from .base_graph_builder import BaseGraphBuilder
from .text_graph_builder import TextGraphBuilder
from .video_graph_builder import VideoGraphBuilder
from .metadata_graph_builder import MetadataGraphBuilder

class HeterogeneousGraphBuilder(BaseGraphBuilder):
    """异构图构建器（重构版）"""
    
    def __init__(self, config: Dict = None):
        """
        初始化异构图构建器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        
        # 模态图构建器
        self.text_builder = TextGraphBuilder(config.get('text_config', {}))
        self.video_builder = VideoGraphBuilder(config.get('video_config', {}))
        self.metadata_builder = MetadataGraphBuilder(config.get('metadata_config', {}))
        
        # 图融合配置
        self.enable_cross_modal_edges = config.get('enable_cross_modal_edges', True)
        self.user_alignment_threshold = config.get('user_alignment_threshold', 0.8)
        self.feature_fusion_method = config.get('feature_fusion_method', 'concatenate')
        
        # 存储各模态图
        self.text_graph = None
        self.video_graph = None
        self.metadata_graph = None
        
        # 媒体会话映射信息
        self.session_mappings = {}
    
    def build_graph(self, data: Dict[str, Any], **kwargs) -> HeteroData:
        """
        构建多模态异构图（包含媒体会话节点）
        
        Args:
            data: 包含所有模态数据的字典
            **kwargs: 其他参数
            
        Returns:
            统一的异构图对象
        """
        self.logger.info("Building heterogeneous multimodal graph with media sessions...")
        
        # 构建各模态图
        self._build_modal_graphs(data)
        
        # 提取会话映射信息
        if self.text_graph:
            self.session_mappings = self.text_builder.get_session_mapping()
            self.logger.info(f"Extracted session mappings: {len(self.session_mappings.get('session_to_videos', {}))} sessions")
        
        # 对齐节点映射（包括媒体会话）
        aligned_mappings = self._align_node_mappings()
        
        # 融合图结构
        fused_graph = self._fuse_graphs(aligned_mappings)
        
        # 添加跨模态边
        if self.enable_cross_modal_edges:
            self._add_cross_modal_edges(fused_graph, aligned_mappings, data)
        
        # 标准化特征
        self._normalize_graph_features(fused_graph)
        
        # 输出图统计信息
        self._log_graph_statistics(fused_graph)
        
        self.logger.info(f"Built heterogeneous graph with {fused_graph.num_nodes} nodes and media session support")
        return fused_graph
    
    def _build_modal_graphs(self, data: Dict[str, Any]):
        """构建各模态图"""
        # 构建文本图（包含媒体会话节点）
        if 'text_features' in data or 'comments_data' in data or 'aligned_data' in data:
            self.logger.info("Building text graph with media sessions...")
            text_data = {
                'text_features': data.get('text_features', {}),
                'comments_data': data.get('comments_data', {}),
                'aligned_data': data.get('aligned_data', {})
            }
            self.text_graph = self.text_builder.build_graph(text_data)
        
        # 构建视频图 - 从aligned_data中提取视频数据
        if 'aligned_data' in data:
            self.logger.info("Building video graph from aligned data...")
            # 从aligned_data中提取视频相关信息
            video_data = {}
            for post_id, post_data in data['aligned_data'].items():
                video_data[post_id] = {
                    'user': post_data.get('username', 'unknown'),
                    'username': post_data.get('username', 'unknown'),
                    'title': post_data.get('caption', ''),
                    'caption': post_data.get('caption', ''),
                    'video_url': post_data.get('video_url', ''),
                    'duration': 0,  # 占位符
                    'view_count': 0,  # 占位符
                    'label': post_data.get('label', 0)
                }
            
            video_graph_data = {
                'video_features': data.get('video_features', {}),
                'video_data': video_data
            }
            self.video_graph = self.video_builder.build_graph(video_graph_data)
        
        # 构建元数据图 - 从aligned_data中提取元数据
        if 'aligned_data' in data:
            self.logger.info("Building metadata graph from aligned data...")
            # 从aligned_data中提取评论数据用于元数据分析
            comments_data = {}
            for post_id, post_data in data['aligned_data'].items():
                comments_data[post_id] = post_data.get('text_data', {})
            
            metadata_graph_data = {
                'user_features': data.get('user_features', {}),
                'metadata': data.get('metadata', {}),
                'comments_data': comments_data
            }
            self.metadata_graph = self.metadata_builder.build_graph(metadata_graph_data)
    
    def _align_node_mappings(self) -> Dict[str, Dict[str, int]]:
        """对齐节点映射（包括媒体会话节点）"""
        aligned_mappings = {}
        
        # 1. 媒体会话节点映射（来自文本图）
        if self.text_graph and 'media_session' in self.text_builder.node_mappings:
            aligned_mappings['media_session'] = self.text_builder.node_mappings['media_session']
            self.logger.info(f"Aligned {len(aligned_mappings['media_session'])} media session nodes")
        
        # 2. 收集所有用户
        all_users = set()
        
        if self.text_graph and 'user' in self.text_builder.node_mappings:
            all_users.update(self.text_builder.node_mappings['user'].keys())
        
        if self.video_graph and 'user' in self.video_builder.node_mappings:
            all_users.update(self.video_builder.node_mappings['user'].keys())
        
        if self.metadata_graph and 'user' in self.metadata_builder.node_mappings:
            all_users.update(self.metadata_builder.node_mappings['user'].keys())
        
        # 创建统一的用户映射
        aligned_mappings['user'] = {
            user: idx for idx, user in enumerate(sorted(all_users))
        }
        
        # 3. 其他节点类型保持各自的映射
        if self.text_graph:
            for node_type in ['comment', 'word']:
                if hasattr(self.text_builder, 'node_mappings') and node_type in self.text_builder.node_mappings:
                    aligned_mappings[node_type] = self.text_builder.node_mappings[node_type]
        
        if self.video_graph:
            for node_type in ['video', 'emotion', 'action', 'scene']:
                if hasattr(self.video_builder, 'node_mappings') and node_type in self.video_builder.node_mappings:
                    aligned_mappings[node_type] = self.video_builder.node_mappings[node_type]
        
        if self.metadata_graph:
            for node_type in ['time', 'location']:
                if hasattr(self.metadata_builder, 'node_mappings') and node_type in self.metadata_builder.node_mappings:
                    aligned_mappings[node_type] = self.metadata_builder.node_mappings[node_type]
        
        self.logger.info(f"Aligned node mappings: {[(k, len(v)) for k, v in aligned_mappings.items()]}")
        return aligned_mappings
    
    def _fuse_graphs(self, aligned_mappings: Dict[str, Dict[str, int]]) -> HeteroData:
        """融合图结构（包括媒体会话节点）"""
        fused_graph = HeteroData()
        
        # 1. 添加媒体会话节点（如果存在）
        if 'media_session' in aligned_mappings and self.text_graph:
            self._add_media_session_components(fused_graph, aligned_mappings)
        
        # 2. 融合用户节点
        self._fuse_user_nodes(fused_graph, aligned_mappings)
        
        # 3. 添加文本模态节点和边
        if self.text_graph:
            self._add_text_components(fused_graph, aligned_mappings)
        
        # 4. 添加视频模态节点和边
        if self.video_graph:
            self._add_video_components(fused_graph, aligned_mappings)
        
        # 5. 添加元数据模态节点和边
        if self.metadata_graph:
            self._add_metadata_components(fused_graph, aligned_mappings)
        
        return fused_graph
    
    def _add_media_session_components(self, fused_graph: HeteroData, aligned_mappings: Dict[str, Dict[str, int]]):
        """添加媒体会话组件到融合图中"""
        if 'media_session' not in self.text_graph.node_types:
            return
        
        # 直接从文本图复制媒体会话节点特征
        fused_graph['media_session'].x = self.text_graph['media_session'].x
        
        # 复制媒体会话相关的边
        for edge_type in self.text_graph.edge_types:
            src_type, relation, dst_type = edge_type
            
            # 媒体会话相关的边
            if src_type == 'media_session' or dst_type == 'media_session':
                fused_graph[edge_type].edge_index = self.text_graph[edge_type].edge_index
        
        self.logger.info(f"Added {fused_graph['media_session'].x.size(0)} media session nodes to fused graph")
    
    def _fuse_user_nodes(self, fused_graph: HeteroData, aligned_mappings: Dict[str, Dict[str, int]]):
        """融合用户节点"""
        num_users = len(aligned_mappings['user'])
        
        # 收集各模态的用户特征
        user_features_list = []
        
        # 文本模态用户特征
        if self.text_graph and 'user' in self.text_graph.node_types:
            text_user_features = self._extract_aligned_user_features(
                self.text_graph['user'].x,
                self.text_builder.node_mappings.get('user', {}),
                aligned_mappings['user']
            )
            user_features_list.append(text_user_features)
        
        # 视频模态用户特征
        if self.video_graph and 'user' in self.video_graph.node_types:
            video_user_features = self._extract_aligned_user_features(
                self.video_graph['user'].x,
                self.video_builder.node_mappings.get('user', {}),
                aligned_mappings['user']
            )
            user_features_list.append(video_user_features)
        
        # 元数据模态用户特征
        if self.metadata_graph and 'user' in self.metadata_graph.node_types:
            metadata_user_features = self._extract_aligned_user_features(
                self.metadata_graph['user'].x,
                self.metadata_builder.node_mappings.get('user', {}),
                aligned_mappings['user']
            )
            user_features_list.append(metadata_user_features)
        
        # 融合用户特征
        if user_features_list:
            if self.feature_fusion_method == 'concatenate':
                fused_user_features = torch.cat(user_features_list, dim=1)
            elif self.feature_fusion_method == 'mean':
                fused_user_features = torch.stack(user_features_list).mean(dim=0)
            else:
                fused_user_features = user_features_list[0]  # 默认使用第一个
            
            fused_graph['user'].x = fused_user_features
            self.logger.info(f"Fused {num_users} user nodes with {fused_user_features.size(1)} features")
    
    def _extract_aligned_user_features(self, features: torch.Tensor, 
                                     original_mapping: Dict[str, int],
                                     aligned_mapping: Dict[str, int]) -> torch.Tensor:
        """提取对齐的用户特征"""
        num_aligned_users = len(aligned_mapping)
        feature_dim = features.size(1)
        aligned_features = torch.zeros(num_aligned_users, feature_dim)
        
        for user, aligned_idx in aligned_mapping.items():
            if user in original_mapping:
                original_idx = original_mapping[user]
                aligned_features[aligned_idx] = features[original_idx]
        
        return aligned_features
    
    def _add_text_components(self, fused_graph: HeteroData, aligned_mappings: Dict[str, Dict[str, int]]):
        """添加文本模态组件（不包括已处理的媒体会话和用户）"""
        # 添加评论节点
        if 'comment' in self.text_graph.node_types:
            fused_graph['comment'].x = self.text_graph['comment'].x
        
        # 添加词汇节点
        if 'word' in self.text_graph.node_types:
            fused_graph['word'].x = self.text_graph['word'].x
        
        # 添加文本相关的边（重新映射用户索引）
        for edge_type in self.text_graph.edge_types:
            src_type, relation, dst_type = edge_type
            
            # 跳过已处理的媒体会话边
            if src_type == 'media_session' or dst_type == 'media_session':
                continue
            
            edge_index = self.text_graph[edge_type].edge_index
            
            # 重新映射用户相关的边
            if src_type == 'user' or dst_type == 'user':
                edge_index = self._remap_user_edges(
                    edge_index, edge_type,
                    self.text_builder.node_mappings.get('user', {}),
                    aligned_mappings['user']
                )
            
            fused_graph[edge_type].edge_index = edge_index
    
    def _add_video_components(self, fused_graph: HeteroData, aligned_mappings: Dict[str, Dict[str, int]]):
        """添加视频模态组件"""
        if not self.video_graph:
            return
        
        # 添加视频节点类型
        for node_type in ['video', 'emotion', 'action', 'scene']:
            if node_type in self.video_graph.node_types:
                fused_graph[node_type].x = self.video_graph[node_type].x
        
        # 添加视频相关的边
        for edge_type in self.video_graph.edge_types:
            src_type, relation, dst_type = edge_type
            edge_index = self.video_graph[edge_type].edge_index
            
            # 重新映射用户相关的边
            if src_type == 'user' or dst_type == 'user':
                edge_index = self._remap_user_edges(
                    edge_index, edge_type,
                    self.video_builder.node_mappings.get('user', {}),
                    aligned_mappings['user']
                )
            
            fused_graph[edge_type].edge_index = edge_index
    
    def _add_metadata_components(self, fused_graph: HeteroData, aligned_mappings: Dict[str, Dict[str, int]]):
        """添加元数据模态组件"""
        if not self.metadata_graph:
            return
        
        # 添加元数据节点类型
        for node_type in ['time', 'location']:
            if node_type in self.metadata_graph.node_types:
                fused_graph[node_type].x = self.metadata_graph[node_type].x
        
        # 添加元数据相关的边
        for edge_type in self.metadata_graph.edge_types:
            src_type, relation, dst_type = edge_type
            edge_index = self.metadata_graph[edge_type].edge_index
            
            # 重新映射用户相关的边
            if src_type == 'user' or dst_type == 'user':
                edge_index = self._remap_user_edges(
                    edge_index, edge_type,
                    self.metadata_builder.node_mappings.get('user', {}),
                    aligned_mappings['user']
                )
            
            fused_graph[edge_type].edge_index = edge_index
    
    def _remap_user_edges(self, edge_index: torch.Tensor, edge_type: Tuple[str, str, str],
                         original_user_mapping: Dict[str, int],
                         aligned_user_mapping: Dict[str, int]) -> torch.Tensor:
        """重新映射用户边索引"""
        src_type, relation, dst_type = edge_type
        
        # 创建原始索引到对齐索引的映射
        idx_mapping = {}
        for user, original_idx in original_user_mapping.items():
            if user in aligned_user_mapping:
                aligned_idx = aligned_user_mapping[user]
                idx_mapping[original_idx] = aligned_idx
        
        # 重新映射边索引
        remapped_edge_index = edge_index.clone()
        
        if src_type == 'user':
            for i in range(edge_index.size(1)):
                original_idx = edge_index[0, i].item()
                if original_idx in idx_mapping:
                    remapped_edge_index[0, i] = idx_mapping[original_idx]
        
        if dst_type == 'user':
            for i in range(edge_index.size(1)):
                original_idx = edge_index[1, i].item()
                if original_idx in idx_mapping:
                    remapped_edge_index[1, i] = idx_mapping[original_idx]
        
        return remapped_edge_index
    
    def _add_cross_modal_edges(self, fused_graph: HeteroData, 
                              aligned_mappings: Dict[str, Dict[str, int]], 
                              data: Dict[str, Any]):
        """添加跨模态边"""
        self.logger.info("Adding cross-modal edges...")
        
        # 用户-评论-视频边
        self._add_user_comment_video_edges(fused_graph, aligned_mappings, data)
        
        # 情感-攻击词边
        self._add_emotion_attack_word_edges(fused_graph, aligned_mappings)
        
        # 时间活动边
        self._add_temporal_activity_edges(fused_graph, aligned_mappings, data)
        
        # 位置场景边
        self._add_location_scene_edges(fused_graph, aligned_mappings)
    
    def _add_user_comment_video_edges(self, fused_graph: HeteroData, 
                                     aligned_mappings: Dict[str, Dict[str, int]], 
                                     data: Dict[str, Any]):
        """添加用户-评论-视频边"""
        # 这里可以根据媒体会话信息建立更精确的连接
        if 'media_session' in aligned_mappings and 'comment' in fused_graph.node_types:
            # 通过媒体会话连接评论和视频
            pass  # 具体实现根据需求
    
    def _add_emotion_attack_word_edges(self, fused_graph: HeteroData, 
                                      aligned_mappings: Dict[str, Dict[str, int]]):
        """添加情感-攻击词边"""
        if ('emotion' in fused_graph.node_types and 
            'word' in fused_graph.node_types):
            # 连接负面情感和攻击性词汇
            pass  # 具体实现
    
    def _add_temporal_activity_edges(self, fused_graph: HeteroData, 
                                    aligned_mappings: Dict[str, Dict[str, int]], 
                                    data: Dict[str, Any]):
        """添加时间活动边"""
        pass  # 具体实现
    
    def _add_location_scene_edges(self, fused_graph: HeteroData, 
                                 aligned_mappings: Dict[str, Dict[str, int]]):
        """添加位置场景边"""
        pass  # 具体实现
    
    def _normalize_graph_features(self, fused_graph: HeteroData):
        """标准化图特征"""
        for node_type in fused_graph.node_types:
            if hasattr(fused_graph[node_type], 'x') and fused_graph[node_type].x is not None:
                features = fused_graph[node_type].x
                # 简单的归一化
                if features.numel() > 0:
                    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-6)
                    fused_graph[node_type].x = features
    
    def _log_graph_statistics(self, fused_graph: HeteroData):
        """记录图统计信息"""
        total_nodes = sum(fused_graph[node_type].x.size(0) 
                         for node_type in fused_graph.node_types 
                         if hasattr(fused_graph[node_type], 'x'))
        
        total_edges = sum(fused_graph[edge_type].edge_index.size(1) 
                         for edge_type in fused_graph.edge_types 
                         if hasattr(fused_graph[edge_type], 'edge_index'))
        
        self.logger.info(f"Final graph statistics:")
        self.logger.info(f"  Total nodes: {total_nodes}")
        self.logger.info(f"  Total edges: {total_edges}")
        self.logger.info(f"  Node types: {list(fused_graph.node_types)}")
        self.logger.info(f"  Edge types: {list(fused_graph.edge_types)}")
        
        # 记录媒体会话信息
        if 'media_session' in fused_graph.node_types:
            num_sessions = fused_graph['media_session'].x.size(0)
            self.logger.info(f"  Media sessions: {num_sessions}")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        stats = {
            'session_mappings': self.session_mappings,
            'text_graph_nodes': self.text_graph.num_nodes if self.text_graph else 0,
            'video_graph_nodes': self.video_graph.num_nodes if self.video_graph else 0,
            'metadata_graph_nodes': self.metadata_graph.num_nodes if self.metadata_graph else 0
        }
        return stats
    
    def get_session_info(self) -> Dict[str, Any]:
        """获取媒体会话信息"""
        return self.session_mappings 