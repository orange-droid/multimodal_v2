"""
子图提取器（重构版）

基于媒体会话和攻击性评论提取霸凌相关的子图结构。
重构后支持媒体会话节点和正确的ID映射。
"""

import os
import sys
import pickle
import logging
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, deque, Counter
import time
from datetime import datetime

class SubgraphExtractor:
    """子图提取器（重构版）"""
    
    def __init__(self, config: Dict = None):
        """
        初始化子图提取器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 提取配置
        self.max_subgraph_size = self.config.get('max_subgraph_size', 50)
        self.min_subgraph_size = self.config.get('min_subgraph_size', 5)
        self.attack_threshold = self.config.get('attack_threshold', 0.1)
        self.hop_limit = self.config.get('hop_limit', 2)
        
        # 媒体会话相关配置
        self.session_based_extraction = self.config.get('session_based_extraction', True)
        self.include_session_context = self.config.get('include_session_context', True)
        
        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 存储提取结果
        self.extracted_subgraphs = []
        self.session_mappings = {}
        self.negative_comments = []
        
        # 配置参数
        self.negative_threshold = self.config.get('negative_threshold', -0.3)  # VADER负面阈值
        self.batch_size = self.config.get('batch_size', 100)                  # 批处理大小
        self.max_subgraphs = self.config.get('max_subgraphs', 1000)          # 最大子图数量限制
        
        # 数据存储
        self.graph = None
        self.node_mappings = None
        self.comment_features = None
        
    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger('SubgraphExtractor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_graph_data(self, graph_path: str, mappings_path: str = None):
        """加载图数据和映射信息"""
        self.logger.info(f"加载图数据: {graph_path}")
        
        try:
            with open(graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            
            self.logger.info(f"成功加载图数据 - 节点类型: {list(self.graph.node_types)}")
            
            # 加载节点映射
            if mappings_path and os.path.exists(mappings_path):
                with open(mappings_path, 'rb') as f:
                    self.node_mappings = pickle.load(f)
                self.logger.info("成功加载节点映射数据")
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载图数据失败: {e}")
            return False
    
    def extract_subgraphs(self, graph: HeteroData, session_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        提取霸凌相关子图（基于媒体会话）
        
        Args:
            graph: 异构图对象
            session_info: 媒体会话映射信息
            
        Returns:
            提取的子图列表
        """
        self.logger.info("Starting subgraph extraction with media session support...")
        
        # 保存会话映射信息
        self.session_mappings = session_info
        
        # 第一步：识别负面评论种子
        negative_comments = self._identify_negative_comments(graph)
        self.logger.info(f"Identified {len(negative_comments)} negative comment seeds")
        
        # 第二步：基于媒体会话提取子图
        subgraphs = []
        if self.session_based_extraction:
            subgraphs = self._extract_session_based_subgraphs(graph, negative_comments)
        else:
            subgraphs = self._extract_traditional_subgraphs(graph, negative_comments)
        
        # 第三步：后处理和验证
        validated_subgraphs = self._validate_subgraphs(subgraphs)
        
        self.extracted_subgraphs = validated_subgraphs
        self.logger.info(f"Successfully extracted {len(validated_subgraphs)} validated subgraphs")
        
        return validated_subgraphs
    
    def _identify_negative_comments(self, graph: HeteroData) -> List[Dict[str, Any]]:
        """识别负面评论种子"""
        negative_comments = []
        
        if 'comment' not in graph.node_types:
            self.logger.warning("No comment nodes found in graph")
            return negative_comments
        
        comment_features = graph['comment'].x
        num_comments = comment_features.size(0)
        
        self.logger.info(f"Analyzing {num_comments} comments for negative patterns...")
        
        for comment_idx in range(num_comments):
            features = comment_features[comment_idx]
            
            # 提取特征（基于新的特征结构）
            # 特征索引：[文本长度, 词数, 感叹号数, 问号数, 大写比例, 攻击词数, 攻击词比例, 内部ID]
            text_length = features[0].item()
            word_count = features[1].item()
            exclamation_count = features[2].item()
            question_count = features[3].item()
            uppercase_ratio = features[4].item()
            attack_word_count = features[5].item()
            attack_word_ratio = features[6].item()
            internal_id = int(features[7].item())
            
            # 负面评论识别条件（多维度）
            is_negative = False
            negative_reasons = []
            
            # 条件1：攻击词比例高
            if attack_word_ratio > 0.1:
                is_negative = True
                negative_reasons.append(f"attack_ratio:{attack_word_ratio:.3f}")
            
            # 条件2：攻击词数量 + 其他指标
            if attack_word_count >= 1 and uppercase_ratio > 0.3:
                is_negative = True
                negative_reasons.append(f"attack_words:{attack_word_count},caps:{uppercase_ratio:.3f}")
            
            # 条件3：多个攻击词
            if attack_word_count >= 2:
                is_negative = True
                negative_reasons.append(f"multiple_attacks:{attack_word_count}")
            
            # 条件4：攻击词 + 感叹号
            if attack_word_count >= 1 and exclamation_count >= 2:
                is_negative = True
                negative_reasons.append(f"attack_exclamation:{attack_word_count},{exclamation_count}")
            
            if is_negative:
                # 查找对应的原始评论ID和媒体会话
                original_id = self._get_original_comment_id(internal_id)
                session_id = self._get_comment_session(comment_idx, graph)
                
                negative_comment = {
                    'comment_idx': comment_idx,  # 添加节点索引
                    'internal_id': internal_id,
                    'original_id': original_id,
                    'session_id': session_id,
                    'features': {
                        'text_length': text_length,
                        'word_count': word_count,
                        'exclamation_count': exclamation_count,
                        'uppercase_ratio': uppercase_ratio,
                        'attack_word_count': attack_word_count,
                        'attack_word_ratio': attack_word_ratio
                    },
                    'negative_reasons': negative_reasons,
                    'negative_score': self._calculate_negative_score(features)
                }
                negative_comments.append(negative_comment)
        
        # 按负面分数排序
        negative_comments.sort(key=lambda x: x['negative_score'], reverse=True)
        
        self.negative_comments = negative_comments
        return negative_comments
    
    def _calculate_negative_score(self, features: torch.Tensor) -> float:
        """计算负面评论分数"""
        attack_word_count = features[5].item()
        attack_word_ratio = features[6].item()
        uppercase_ratio = features[4].item()
        exclamation_count = features[2].item()
        
        # 综合分数计算
        score = (
            attack_word_ratio * 3.0 +
            min(attack_word_count / 5.0, 1.0) * 2.0 +
            min(uppercase_ratio, 1.0) * 1.0 +
            min(exclamation_count / 3.0, 1.0) * 0.5
        )
        
        return score
    
    def _get_original_comment_id(self, internal_id: int) -> str:
        """根据内部ID获取原始评论ID"""
        comment_mapping = self.session_mappings.get('comment_id_mapping', {})
        return comment_mapping.get(internal_id, f"unknown_{internal_id}")
    
    def _get_comment_session(self, comment_idx: int, graph: HeteroData) -> str:
        """根据comment节点索引通过图连接关系获取媒体会话ID"""
        try:
            # 直接从图的连接关系中查找
            if ('media_session', 'contains', 'comment') in graph.edge_types:
                session_comment_edges = graph[('media_session', 'contains', 'comment')].edge_index
                
                # 找到包含该comment的session（使用comment节点索引）
                comment_mask = session_comment_edges[1] == comment_idx
                if comment_mask.any():
                    session_idx = session_comment_edges[0][comment_mask][0].item()
                    return f"media_session_{session_idx}"
            
            return "media_session_unknown"
            
        except Exception as e:
            self.logger.warning(f"Error finding session for comment {comment_idx}: {e}")
            return "media_session_unknown"
    
    def _extract_session_based_subgraphs(self, graph: HeteroData, negative_comments: List[Dict]) -> List[Dict]:
        """基于媒体会话提取子图（新策略：多尺寸子图）"""
        self.logger.info("Extracting multiple-sized subgraphs based on media sessions...")
        
        subgraphs = []
        
        # 按媒体会话分组负面评论
        session_negative_comments = defaultdict(list)
        for neg_comment in negative_comments:
            session_id = neg_comment['session_id']
            session_negative_comments[session_id].append(neg_comment)
        
        self.logger.info(f"Found negative comments in {len(session_negative_comments)} sessions")
        
        # 为每个包含负面评论的会话提取多个子图
        for session_id, neg_comments in session_negative_comments.items():
            if session_id == 'media_session_unknown':
                continue  # 跳过无法映射的会话
                
            self.logger.info(f"Processing session {session_id} with {len(neg_comments)} negative comments")
            
            # 提取该会话的多尺寸子图
            session_subgraphs = self._extract_multi_size_subgraphs(graph, session_id, neg_comments)
            
            if session_subgraphs:
                subgraphs.extend(session_subgraphs)
                self.logger.info(f"Extracted {len(session_subgraphs)} subgraphs from {session_id}")
        
        return subgraphs
    
    def _extract_multi_size_subgraphs(self, graph: HeteroData, session_id: str, 
                                    negative_comments: List[Dict]) -> List[Dict]:
        """从单个会话中提取多个不同大小的子图"""
        subgraphs = []
        
        try:
            # 找到媒体会话节点索引
            session_node_idx = self._find_session_node_index(graph, session_id)
            if session_node_idx is None:
                self.logger.warning(f"Session node not found for {session_id}")
                return subgraphs
            
            # 获取该会话的所有评论节点
            all_session_comments = self._get_session_comments(graph, session_node_idx)
            if not all_session_comments:
                return subgraphs
            
            # 为每个负面评论生成多尺寸子图
            for neg_comment in negative_comments:
                comment_idx = self._find_comment_node_index(neg_comment, all_session_comments)
                if comment_idx is None:
                    continue
                
                # 生成4-12节点的子图
                comment_subgraphs = self._generate_incremental_subgraphs(
                    graph, session_id, comment_idx, min_size=4, max_size=12
                )
                
                subgraphs.extend(comment_subgraphs)
            
            # 去重（基于节点集合）
            subgraphs = self._deduplicate_subgraphs(subgraphs)
            
        except Exception as e:
            self.logger.error(f"Error extracting multi-size subgraphs for session {session_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return subgraphs
    
    def _get_session_comments(self, graph: HeteroData, session_node_idx: int) -> List[int]:
        """获取会话的所有评论节点"""
        if ('media_session', 'contains', 'comment') not in graph.edge_types:
            return []
        
        session_comment_edges = graph[('media_session', 'contains', 'comment')].edge_index
        mask = session_comment_edges[0] == session_node_idx
        return session_comment_edges[1][mask].tolist()
    
    def _find_comment_node_index(self, neg_comment: Dict, all_session_comments: List[int]) -> Optional[int]:
        """找到负面评论对应的节点索引"""
        # 直接使用保存的节点索引
        comment_idx = neg_comment.get('comment_idx')
        
        # 验证该索引确实在会话的评论列表中
        if comment_idx is not None and comment_idx in all_session_comments:
            return comment_idx
        
        return None
    
    def _generate_incremental_subgraphs(self, graph: HeteroData, session_id: str, 
                                      seed_comment_idx: int, min_size: int = 4, max_size: int = 12) -> List[Dict]:
        """从种子评论开始生成递增大小的子图（优化版本：按需查询）"""
        subgraphs = []
        
        try:
            # 从种子节点开始，逐步收集邻居节点
            current_nodes = {seed_comment_idx}
            all_collected_nodes = {seed_comment_idx}
            
            # 逐步扩展节点集合
            for expansion_round in range(max_size):
                # 为当前轮次的所有节点找邻居
                new_nodes = set()
                
                for node_idx in current_nodes:
                    # 按需查询该节点的邻居
                    neighbors = self._get_node_neighbors_on_demand(graph, node_idx)
                    for neighbor_idx, neighbor_type in neighbors:
                        if neighbor_idx not in all_collected_nodes:
                            new_nodes.add(neighbor_idx)
                
                # 添加新发现的节点
                if new_nodes:
                    all_collected_nodes.update(new_nodes)
                    current_nodes = new_nodes  # 下一轮从新节点开始扩展
                else:
                    # 没有更多邻居，停止扩展
                    break
                
                # 检查是否达到了某个目标大小，如果是则创建子图
                current_size = len(all_collected_nodes)
                if min_size <= current_size <= max_size:
                    subgraph_nodes = list(all_collected_nodes)
                    subgraph = self._create_subgraph_from_nodes(
                        graph, session_id, seed_comment_idx, subgraph_nodes, current_size
                    )
                    if subgraph:
                        subgraphs.append(subgraph)
                
                # 如果已经达到最大大小，停止
                if current_size >= max_size:
                    break
            
            self.logger.info(f"Generated {len(subgraphs)} subgraphs from seed {seed_comment_idx}")
            return subgraphs
            
        except Exception as e:
            self.logger.error(f"Error generating incremental subgraphs: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _get_node_neighbors_on_demand(self, graph: HeteroData, node_idx: int) -> List[Tuple[int, str]]:
        """按需查询节点的邻居（避免构建完整邻接表）"""
        neighbors = []
        
        # 确定节点类型
        node_type = self._determine_node_type(graph, node_idx)
        
        # 根据节点类型查询相关的边
        if node_type == 'comment':
            # 评论节点的邻居：用户（发表评论）、词汇（包含词汇）
            
            # 查找发表该评论的用户
            if ('user', 'posts', 'comment') in graph.edge_types:
                user_comment_edges = graph[('user', 'posts', 'comment')].edge_index
                comment_mask = user_comment_edges[1] == node_idx
                if comment_mask.any():
                    user_indices = user_comment_edges[0][comment_mask]
                    for user_idx in user_indices:
                        neighbors.append((user_idx.item(), 'user'))
            
            # 查找该评论包含的词汇（限制数量以避免过多词汇）
            if ('comment', 'contains', 'word') in graph.edge_types:
                comment_word_edges = graph[('comment', 'contains', 'word')].edge_index
                comment_mask = comment_word_edges[0] == node_idx
                if comment_mask.any():
                    word_indices = comment_word_edges[1][comment_mask]
                    # 限制词汇数量，只取前5个
                    for word_idx in word_indices[:5]:
                        neighbors.append((word_idx.item(), 'word'))
        
        elif node_type == 'user':
            # 用户节点的邻居：评论（发表的评论）
            if ('user', 'posts', 'comment') in graph.edge_types:
                user_comment_edges = graph[('user', 'posts', 'comment')].edge_index
                user_mask = user_comment_edges[0] == node_idx
                if user_mask.any():
                    comment_indices = user_comment_edges[1][user_mask]
                    # 限制评论数量，只取前3个
                    for comment_idx in comment_indices[:3]:
                        neighbors.append((comment_idx.item(), 'comment'))
        
        elif node_type == 'word':
            # 词汇节点的邻居：评论（包含该词汇的评论）
            if ('comment', 'contains', 'word') in graph.edge_types:
                comment_word_edges = graph[('comment', 'contains', 'word')].edge_index
                word_mask = comment_word_edges[1] == node_idx
                if word_mask.any():
                    comment_indices = comment_word_edges[0][word_mask]
                    # 限制评论数量，只取前3个
                    for comment_idx in comment_indices[:3]:
                        neighbors.append((comment_idx.item(), 'comment'))
        
        return neighbors
    
    def _create_subgraph_from_nodes(self, graph: HeteroData, session_id: str, 
                                  seed_comment_idx: int, node_indices: List[int], 
                                  target_size: int) -> Optional[Dict]:
        """从节点列表创建子图"""
        try:
            # 按节点类型分组
            nodes_by_type = {
                'comment': [],
                'user': [],
                'word': []
            }
            
            # 根据节点索引范围确定节点类型（这需要根据您的图构建逻辑调整）
            for node_idx in node_indices:
                node_type = self._determine_node_type(graph, node_idx)
                if node_type in nodes_by_type:
                    nodes_by_type[node_type].append(node_idx)
            
            # 提取相关边
            subgraph_edges = self._extract_subgraph_edges(graph, nodes_by_type)
            
            # 计算子图特征
            features = {
                'node_counts': {k: len(v) for k, v in nodes_by_type.items()},
                'total_nodes': len(node_indices),
                'seed_comment': seed_comment_idx,
                'target_size': target_size
            }
            
            subgraph_data = {
                'session_id': session_id,
                'seed_comment_idx': seed_comment_idx,
                'nodes': nodes_by_type,
                'edges': subgraph_edges,
                'features': features,
                'size': len(node_indices),
                'extraction_method': 'multi_size',
                'timestamp': datetime.now().isoformat()
            }
            
            return subgraph_data
            
        except Exception as e:
            self.logger.error(f"Error creating subgraph from nodes: {e}")
            return None
    
    def _determine_node_type(self, graph: HeteroData, node_idx: int) -> str:
        """根据节点索引确定节点类型（简化版本）"""
        # 这里需要根据您的图构建逻辑来确定节点类型
        # 暂时使用简单的范围判断
        if node_idx < 55038:  # 用户节点范围
            return 'user'
        elif node_idx < 132441:  # 评论节点范围
            return 'comment'
        else:  # 词汇节点范围
            return 'word'
    
    def _extract_subgraph_edges(self, graph: HeteroData, nodes_by_type: Dict[str, List[int]]) -> Dict:
        """提取子图内的边"""
        subgraph_edges = {}
        
        # 转换为集合以提高查找效率
        node_sets = {node_type: set(nodes) for node_type, nodes in nodes_by_type.items()}
        
        # 只处理重要的边类型
        important_edge_types = [
            ('user', 'posts', 'comment'),
            ('comment', 'contains', 'word')
        ]
        
        for edge_type in important_edge_types:
            if edge_type not in graph.edge_types:
                continue
                
            src_type, relation, dst_type = edge_type
            
            if src_type in node_sets and dst_type in node_sets:
                edge_index = graph[edge_type].edge_index
                src_nodes = node_sets[src_type]
                dst_nodes = node_sets[dst_type]
                
                # 使用张量操作进行高效筛选
                src_mask = torch.isin(edge_index[0], torch.tensor(list(src_nodes), dtype=edge_index.dtype))
                dst_mask = torch.isin(edge_index[1], torch.tensor(list(dst_nodes), dtype=edge_index.dtype))
                valid_mask = src_mask & dst_mask
                
                if valid_mask.any():
                    valid_edge_index = edge_index[:, valid_mask]
                    subgraph_edges[edge_type] = valid_edge_index.t().tolist()
        
        return subgraph_edges
    
    def _deduplicate_subgraphs(self, subgraphs: List[Dict]) -> List[Dict]:
        """去除重复的子图（基于节点集合）"""
        unique_subgraphs = []
        seen_node_sets = set()
        
        for subgraph in subgraphs:
            nodes = subgraph.get('nodes', {})
            all_nodes = []
            for node_list in nodes.values():
                all_nodes.extend(node_list)
            
            node_set = frozenset(all_nodes)
            if node_set not in seen_node_sets:
                seen_node_sets.add(node_set)
                unique_subgraphs.append(subgraph)
        
        return unique_subgraphs
    
    def _extract_traditional_subgraphs(self, graph: HeteroData, negative_comments: List[Dict]) -> List[Dict]:
        """传统的子图提取方法（不基于会话）"""
        self.logger.info("Extracting subgraphs using traditional method...")
        
        subgraphs = []
        
        for neg_comment in negative_comments[:100]:  # 限制处理数量
            internal_id = neg_comment['internal_id']
            
            # 基于评论节点进行k跳扩展
            subgraph = self._extract_k_hop_subgraph(graph, 'comment', internal_id)
            
            if subgraph:
                subgraph.update({
                    'seed_comment': neg_comment,
                    'extraction_method': 'traditional'
                })
                subgraphs.append(subgraph)
        
        return subgraphs
    
    def _extract_k_hop_subgraph(self, graph: HeteroData, seed_type: str, seed_idx: int) -> Optional[Dict]:
        """提取k跳子图"""
        # 这里实现传统的k跳子图提取逻辑
        # 为简化，返回基本结构
        return {
            'seed_type': seed_type,
            'seed_idx': seed_idx,
            'nodes': {seed_type: [seed_idx]},
            'edges': {},
            'hop_count': 1
        }
    
    def _validate_subgraphs(self, subgraphs: List[Dict]) -> List[Dict]:
        """验证和过滤子图"""
        validated = []
        
        for subgraph in subgraphs:
            # 基本验证
            if self._is_valid_subgraph(subgraph):
                validated.append(subgraph)
            else:
                self.logger.debug(f"Filtered out invalid subgraph: {subgraph.get('session_id', 'unknown')}")
        
        return validated
    
    def _is_valid_subgraph(self, subgraph: Dict) -> bool:
        """检查子图是否有效"""
        # 检查节点数量
        nodes = subgraph.get('nodes', {})
        total_nodes = sum(len(node_list) for node_list in nodes.values())
        
        # 对于媒体会话级别的子图，使用更宽松的限制
        session_id = subgraph.get('session_id', '')
        if session_id and session_id != 'media_session_unknown':
            # 媒体会话子图的特殊验证
            min_size = max(self.min_subgraph_size, 10)  # 至少10个节点
            max_size = max(self.max_subgraph_size, 500)  # 最多500个节点
        else:
            # 传统子图的验证
            min_size = self.min_subgraph_size
            max_size = self.max_subgraph_size
        
        if total_nodes < min_size:
            self.logger.debug(f"Subgraph {session_id} too small: {total_nodes} < {min_size}")
            return False
        
        if total_nodes > max_size:
            self.logger.debug(f"Subgraph {session_id} too large: {total_nodes} > {max_size}")
            return False
        
        # 检查是否有负面评论
        negative_comments = subgraph.get('negative_comments', [])
        if not negative_comments:
            self.logger.debug(f"Subgraph {session_id} has no negative comments")
            return False
        
        # 检查是否有足够的评论节点
        comment_nodes = nodes.get('comment', [])
        if len(comment_nodes) < 1:
            self.logger.debug(f"Subgraph {session_id} has no comment nodes")
            return False
        
        self.logger.info(f"Subgraph {session_id} is valid: {total_nodes} nodes, {len(negative_comments)} negative comments")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取提取统计信息"""
        stats = {
            'total_subgraphs': len(self.extracted_subgraphs),
            'total_negative_comments': len(self.negative_comments),
            'sessions_with_subgraphs': 0,
            'avg_subgraph_size': 0.0,
            'negative_comment_distribution': {}
        }
        
        if self.extracted_subgraphs:
            # 计算平均子图大小
            sizes = []
            session_set = set()
            
            for subgraph in self.extracted_subgraphs:
                nodes = subgraph.get('nodes', {})
                size = sum(len(node_list) for node_list in nodes.values())
                sizes.append(size)
                
                session_id = subgraph.get('session_id')
                if session_id:
                    session_set.add(session_id)
            
            stats['avg_subgraph_size'] = np.mean(sizes) if sizes else 0.0
            stats['sessions_with_subgraphs'] = len(session_set)
        
        # 负面评论分布
        if self.negative_comments:
            session_distribution = defaultdict(int)
            for neg_comment in self.negative_comments:
                session_id = neg_comment.get('session_id', 'unknown')
                session_distribution[session_id] += 1
            
            stats['negative_comment_distribution'] = dict(session_distribution)
        
        return stats
    
    def extract_negative_seed_comments(self) -> List[int]:
        """提取负面情感的种子评论节点
        
        Returns:
            负面评论节点的索引列表
        """
        self.logger.info("开始提取负面种子评论...")
        
        if 'comment' not in self.graph.node_types:
            self.logger.error("图中没有comment节点类型")
            return []
        
        # 获取评论节点特征
        comment_features = self.graph['comment'].x
        num_comments = comment_features.size(0)
        
        self.logger.info(f"总评论数: {num_comments}")
        self.logger.info(f"评论特征维度: {comment_features.shape}")
        
        # 根据实际特征结构提取负面评论
        # 特征结构（基于text_graph_builder.py第332-337行）：
        # [0]: 文本长度, [1]: 词数, [2]: 感叹号数量, [3]: 问号数量, 
        # [4]: 大写比例, [5]: 攻击词数量, [6]: 攻击词比例
        negative_seeds = []
        
        # 分批处理避免内存问题
        batch_size = min(self.batch_size, num_comments)
        
        for start_idx in range(0, num_comments, batch_size):
            end_idx = min(start_idx + batch_size, num_comments)
            batch_features = comment_features[start_idx:end_idx]
            
            # 基于攻击性特征识别负面评论
            if batch_features.size(1) >= 7:  # 确保有足够的特征维度
                # 使用攻击词比例（第6列，索引5）作为负面指标
                attack_ratios = batch_features[:, 6]  # 攻击词比例
                attack_counts = batch_features[:, 5]  # 攻击词数量
                uppercase_ratios = batch_features[:, 4]  # 大写比例
                exclamation_counts = batch_features[:, 2]  # 感叹号数量
                
                # 多维度负面评论识别
                # 条件1：攻击词比例 > 0.1 (10%以上的词是攻击性词汇)
                condition1 = attack_ratios > 0.1
                
                # 条件2：攻击词数量 >= 1 且大写比例 > 0.3 (有攻击词且大量大写)
                condition2 = (attack_counts >= 1) & (uppercase_ratios > 0.3)
                
                # 条件3：攻击词数量 >= 2 (包含多个攻击词)
                condition3 = attack_counts >= 2
                
                # 条件4：攻击词数量 >= 1 且感叹号 >= 2 (有攻击词且情绪激动)
                condition4 = (attack_counts >= 1) & (exclamation_counts >= 2)
                
                # 满足任一条件的评论被认为是负面的
                negative_mask = condition1 | condition2 | condition3 | condition4
                negative_indices = torch.where(negative_mask)[0] + start_idx
                negative_seeds.extend(negative_indices.tolist())
            
            # 进度显示
            if (start_idx // batch_size) % 10 == 0:
                progress = (end_idx / num_comments) * 100
                self.logger.info(f"处理进度: {progress:.1f}% - 已找到 {len(negative_seeds)} 个负面种子")
        
        self.logger.info(f"提取完成 - 总共找到 {len(negative_seeds)} 个负面种子评论")
        
        # 如果找到的负面种子太少，放宽条件
        if len(negative_seeds) < 50:
            self.logger.warning(f"负面种子数量较少({len(negative_seeds)})，放宽筛选条件...")
            return self._extract_seeds_with_relaxed_conditions()
        
        return negative_seeds[:self.max_subgraphs]  # 限制数量避免过多
    
    def _extract_seeds_with_relaxed_conditions(self) -> List[int]:
        """使用放宽条件提取种子评论"""
        comment_features = self.graph['comment'].x
        num_comments = comment_features.size(0)
        
        negative_seeds = []
        batch_size = min(self.batch_size, num_comments)
        
        for start_idx in range(0, num_comments, batch_size):
            end_idx = min(start_idx + batch_size, num_comments)
            batch_features = comment_features[start_idx:end_idx]
            
            if batch_features.size(1) >= 7:
                attack_ratios = batch_features[:, 6]
                attack_counts = batch_features[:, 5]
                uppercase_ratios = batch_features[:, 4]
                
                # 放宽的条件：
                # 条件1：有任何攻击词
                condition1 = attack_counts > 0
                
                # 条件2：大写比例很高（可能是愤怒表达）
                condition2 = uppercase_ratios > 0.5
                
                # 条件3：攻击词比例 > 0.05 (5%以上)
                condition3 = attack_ratios > 0.05
                
                negative_mask = condition1 | condition2 | condition3
                negative_indices = torch.where(negative_mask)[0] + start_idx
                negative_seeds.extend(negative_indices.tolist())
        
        self.logger.info(f"放宽条件后找到 {len(negative_seeds)} 个种子评论")
        return negative_seeds[:self.max_subgraphs]
    
    def extract_subgraph_from_seed(self, seed_comment_idx: int) -> Dict[str, Set[int]]:
        """从种子评论提取子图
        
        Args:
            seed_comment_idx: 种子评论的节点索引
            
        Returns:
            包含各类型节点的子图字典
        """
        subgraph_nodes = {
            'comment': {seed_comment_idx},
            'user': set(),
            'word': set()
        }
        
        # 使用BFS进行有限跳数的邻居扩展
        visited = {seed_comment_idx}
        queue = deque([(seed_comment_idx, 'comment', 0)])  # (节点索引, 节点类型, 跳数)
        
        while queue and len(visited) < self.max_subgraph_size:
            current_idx, current_type, hop_count = queue.popleft()
            
            if hop_count >= self.hop_limit:
                continue
            
            # 根据当前节点类型，查找相关的边
            neighbors = self._get_neighbors(current_idx, current_type)
            
            for neighbor_idx, neighbor_type in neighbors:
                if neighbor_idx not in visited and len(visited) < self.max_subgraph_size:
                    visited.add(neighbor_idx)
                    subgraph_nodes[neighbor_type].add(neighbor_idx)
                    queue.append((neighbor_idx, neighbor_type, hop_count + 1))
        
        return subgraph_nodes
    
    def _get_neighbors(self, node_idx: int, node_type: str) -> List[Tuple[int, str]]:
        """获取指定节点的邻居节点
        
        Args:
            node_idx: 节点索引
            node_type: 节点类型
            
        Returns:
            邻居节点列表 [(邻居索引, 邻居类型), ...]
        """
        neighbors = []
        
        try:
            # 根据节点类型查找相应的边
            if node_type == 'comment':
                # 评论节点的邻居：用户（发表者）、词汇（包含的词）
                
                # 查找发表该评论的用户
                if ('user', 'posts', 'comment') in self.graph.edge_types:
                    edge_index = self.graph[('user', 'posts', 'comment')].edge_index
                    # 找到指向当前评论的边
                    comment_edges = edge_index[1] == node_idx
                    if comment_edges.any():
                        user_indices = edge_index[0][comment_edges]
                        neighbors.extend([(idx.item(), 'user') for idx in user_indices])
                
                # 查找评论包含的词汇
                if ('comment', 'contains', 'word') in self.graph.edge_types:
                    edge_index = self.graph[('comment', 'contains', 'word')].edge_index
                    # 找到从当前评论出发的边
                    comment_edges = edge_index[0] == node_idx
                    if comment_edges.any():
                        word_indices = edge_index[1][comment_edges]
                        neighbors.extend([(idx.item(), 'word') for idx in word_indices])
            
            elif node_type == 'user':
                # 用户节点的邻居：发表的评论
                if ('user', 'posts', 'comment') in self.graph.edge_types:
                    edge_index = self.graph[('user', 'posts', 'comment')].edge_index
                    user_edges = edge_index[0] == node_idx
                    if user_edges.any():
                        comment_indices = edge_index[1][user_edges]
                        neighbors.extend([(idx.item(), 'comment') for idx in comment_indices])
            
            elif node_type == 'word':
                # 词汇节点的邻居：包含该词汇的评论
                if ('comment', 'contains', 'word') in self.graph.edge_types:
                    edge_index = self.graph[('comment', 'contains', 'word')].edge_index
                    word_edges = edge_index[1] == node_idx
                    if word_edges.any():
                        comment_indices = edge_index[0][word_edges]
                        neighbors.extend([(idx.item(), 'comment') for idx in comment_indices])
        
        except Exception as e:
            self.logger.warning(f"获取邻居节点时出错 (节点{node_idx}, 类型{node_type}): {e}")
        
        return neighbors
    
    def extract_all_subgraphs(self, max_subgraphs: int = None) -> List[Dict]:
        """提取所有霸凌相关子图
        
        Args:
            max_subgraphs: 最大子图数量限制
            
        Returns:
            子图列表，每个子图包含节点信息和基本统计
        """
        if max_subgraphs is None:
            max_subgraphs = self.max_subgraphs
        
        self.logger.info("开始提取所有子图...")
        
        # 第一步：获取负面种子评论
        negative_seeds = self.extract_negative_seed_comments()
        
        if not negative_seeds:
            self.logger.warning("未找到负面种子评论")
            return []
        
        # 限制处理数量
        seeds_to_process = negative_seeds[:max_subgraphs]
        self.logger.info(f"将处理 {len(seeds_to_process)} 个种子评论")
        
        subgraphs = []
        
        # 分批处理种子评论
        for i, seed_idx in enumerate(seeds_to_process):
            try:
                # 提取子图
                subgraph_nodes = self.extract_subgraph_from_seed(seed_idx)
                
                # 验证子图大小
                total_nodes = sum(len(nodes) for nodes in subgraph_nodes.values())
                
                if total_nodes >= self.min_subgraph_size:
                    subgraph_info = {
                        'id': i,
                        'seed_comment': seed_idx,
                        'nodes': subgraph_nodes,
                        'total_nodes': total_nodes,
                        'comment_count': len(subgraph_nodes['comment']),
                        'user_count': len(subgraph_nodes['user']),
                        'word_count': len(subgraph_nodes['word'])
                    }
                    subgraphs.append(subgraph_info)
                
                # 进度显示
                if (i + 1) % 50 == 0:
                    self.logger.info(f"已处理 {i + 1}/{len(seeds_to_process)} 个种子，"
                                   f"有效子图: {len(subgraphs)}")
                
            except Exception as e:
                self.logger.warning(f"处理种子 {seed_idx} 时出错: {e}")
                continue
        
        self.logger.info(f"子图提取完成 - 总共提取 {len(subgraphs)} 个有效子图")
        return subgraphs
    
    def save_subgraphs_by_session(self, subgraphs: List[Dict], output_dir: str):
        """按会话分组保存子图"""
        try:
            import os
            from datetime import datetime
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 按会话分组
            session_subgraphs = defaultdict(list)
            for subgraph in subgraphs:
                session_id = subgraph.get('session_id', 'unknown')
                session_subgraphs[session_id].append(subgraph)
            
            # 保存每个会话的子图
            saved_files = []
            total_subgraphs = 0
            
            for session_id, session_subs in session_subgraphs.items():
                if session_id == 'media_session_unknown':
                    continue
                
                # 按子图大小排序
                session_subs.sort(key=lambda x: x.get('size', 0))
                
                # 文件名
                filename = f"{session_id}_subgraphs.pkl"
                filepath = os.path.join(output_dir, filename)
                
                # 准备保存的数据
                session_data = {
                    'session_id': session_id,
                    'total_subgraphs': len(session_subs),
                    'size_distribution': {},
                    'extraction_timestamp': datetime.now().isoformat(),
                    'subgraphs': session_subs
                }
                
                # 统计大小分布
                for subgraph in session_subs:
                    size = subgraph.get('size', 0)
                    session_data['size_distribution'][size] = session_data['size_distribution'].get(size, 0) + 1
                
                # 保存到文件
                with open(filepath, 'wb') as f:
                    pickle.dump(session_data, f)
                
                saved_files.append(filepath)
                total_subgraphs += len(session_subs)
                
                self.logger.info(f"Saved {len(session_subs)} subgraphs for {session_id} to {filename}")
            
            # 创建索引文件
            index_data = {
                'total_sessions': len(session_subgraphs),
                'total_subgraphs': total_subgraphs,
                'saved_files': saved_files,
                'session_summary': {
                    session_id: {
                        'subgraph_count': len(subs),
                        'size_range': [min(s.get('size', 0) for s in subs), max(s.get('size', 0) for s in subs)],
                        'filename': f"{session_id}_subgraphs.pkl"
                    }
                    for session_id, subs in session_subgraphs.items()
                    if session_id != 'media_session_unknown'
                },
                'creation_timestamp': datetime.now().isoformat()
            }
            
            index_path = os.path.join(output_dir, 'subgraph_index.json')
            with open(index_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Successfully saved {total_subgraphs} subgraphs from {len(saved_files)} sessions")
            self.logger.info(f"Index file created: {index_path}")
            
            return index_path
            
        except Exception as e:
            self.logger.error(f"Error saving subgraphs by session: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_session_subgraphs(self, session_id: str, subgraph_dir: str) -> Optional[Dict]:
        """加载特定会话的子图"""
        try:
            import os
            filename = f"{session_id}_subgraphs.pkl"
            filepath = os.path.join(subgraph_dir, filename)
            
            if not os.path.exists(filepath):
                self.logger.warning(f"Subgraph file not found: {filepath}")
                return None
            
            with open(filepath, 'rb') as f:
                session_data = pickle.load(f)
            
            self.logger.info(f"Loaded {session_data['total_subgraphs']} subgraphs for {session_id}")
            return session_data
            
        except Exception as e:
            self.logger.error(f"Error loading subgraphs for {session_id}: {e}")
            return None

    def get_subgraph_index(self, subgraph_dir: str) -> Optional[Dict]:
        """获取子图索引信息"""
        try:
            import os
            import json
            
            index_path = os.path.join(subgraph_dir, 'subgraph_index.json')
            if not os.path.exists(index_path):
                self.logger.warning(f"Index file not found: {index_path}")
                return None
            
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            return index_data
            
        except Exception as e:
            self.logger.error(f"Error loading subgraph index: {e}")
            return None

    def save_subgraphs(self, subgraphs: List[Dict], output_path: str):
        """保存提取的子图"""
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(subgraphs, f)
            self.logger.info(f"子图数据已保存到: {output_path}")
        except Exception as e:
            self.logger.error(f"保存子图失败: {e}")

    def _find_session_node_index(self, graph: HeteroData, session_id: str) -> Optional[int]:
        """查找媒体会话节点的索引"""
        if 'media_session' not in graph.node_types:
            return None
        
        # 从session_id中提取索引（格式：media_session_123）
        try:
            if session_id.startswith("media_session_"):
                session_idx = int(session_id.split("_")[-1])
                # 验证索引是否有效
                num_sessions = graph['media_session'].x.shape[0]
                if 0 <= session_idx < num_sessions:
                    return session_idx
        except (ValueError, IndexError):
            pass
        
        return None


def main():
    """测试函数"""
    print("=== 子图提取器测试 ===")
    
    # 配置参数
    config = {
        'negative_threshold': -0.2,  # 较宽松的阈值用于测试
        'max_subgraph_size': 30,     # 较小的子图用于测试
        'min_subgraph_size': 3,      # 最小子图大小
        'hop_limit': 2,              # 2跳邻居
        'batch_size': 50,            # 小批次处理
        'max_subgraphs': 100         # 限制数量用于测试
    }
    
    # 创建提取器
    extractor = SubgraphExtractor(config)
    
    # 加载数据
    graph_path = "data/graphs/heterogeneous_graph_final.pkl"
    mappings_path = "data/graphs/mappings.pkl"
    
    if not extractor.load_graph_data(graph_path, mappings_path):
        print("加载数据失败")
        return
    
    # 提取子图
    start_time = time.time()
    subgraphs = extractor.extract_all_subgraphs(max_subgraphs=50)  # 测试时只提取50个
    end_time = time.time()
    
    print(f"提取耗时: {end_time - start_time:.2f} 秒")
    print(f"提取的子图数量: {len(subgraphs)}")
    
    if subgraphs:
        # 显示统计信息
        total_nodes = sum(sg['total_nodes'] for sg in subgraphs)
        avg_nodes = total_nodes / len(subgraphs)
        print(f"平均子图大小: {avg_nodes:.1f} 节点")
        
        # 保存结果
        output_path = "data/subgraphs_test.pkl"
        extractor.save_subgraphs(subgraphs, output_path)
        print(f"测试结果已保存到: {output_path}")


if __name__ == "__main__":
    main() 