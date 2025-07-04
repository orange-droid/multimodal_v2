#!/usr/bin/env python3
"""
通用子图提取器（Universal Subgraph Extractor）- 增强版

改进特性：
1. 支持多用户交互子图提取
2. 智能枚举策略：多用户需要交互关系，单用户完全枚举
3. 节点数量范围调整为6-15
4. 详细的会话级进度显示
"""

import os
import pickle
import logging
import numpy as np
import torch
import random
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
from itertools import combinations
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

class UniversalSubgraphExtractor:
    """通用子图提取器 - 增强版，支持多用户交互"""
    
    def __init__(self, config: Dict = None):
        """
        初始化通用子图提取器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 调整子图大小配置 - 重点关注6-15节点
        self.min_subgraph_size = self.config.get('min_subgraph_size', 6)   # 最小6个节点
        self.max_subgraph_size = self.config.get('max_subgraph_size', 15)  # 最大15个节点
        self.size_step = self.config.get('size_step', 1)  # 精细步长
        
        # 更宽松的扩展策略配置
        self.max_comments_per_video = self.config.get('max_comments_per_video', 50)
        self.max_users_per_subgraph = self.config.get('max_users_per_subgraph', 8)   # 支持更多用户
        self.max_words_per_subgraph = self.config.get('max_words_per_subgraph', 25)
        self.max_seeds_per_session = self.config.get('max_seeds_per_session', 30)
        
        # 多用户交互配置
        self.enable_multi_user_subgraphs = self.config.get('enable_multi_user_subgraphs', True)
        self.min_interacting_users = self.config.get('min_interacting_users', 2)  # 至少2个用户有交互
        self.max_multi_user_combinations = self.config.get('max_multi_user_combinations', 100)  # 限制组合数量
        
        # 完全枚举配置
        self.enable_smart_selection = self.config.get('enable_smart_selection', True)
        self.max_enumeration_combinations = self.config.get('max_enumeration_combinations', 500)  # 限制枚举数量
        
        # 单类型子图配置
        self.enable_single_type_subgraphs = self.config.get('enable_single_type_subgraphs', True)
        self.single_type_min_size = self.config.get('single_type_min_size', 6)
        self.single_type_max_size = self.config.get('single_type_max_size', 15)
        
        # 随机采样和多样性配置
        self.random_sampling_ratio = self.config.get('random_sampling_ratio', 0.3)
        self.diversity_boost = self.config.get('diversity_boost', True)
        self.enable_large_subgraphs = self.config.get('enable_large_subgraphs', True)
        
        # 更新策略权重 - 增加多用户和完全枚举策略
        self.strategy_weights = self.config.get('strategy_weights', {
            'multi_user_interaction': 0.25,  # 新增：多用户交互
            'smart_selection': 0.25,         # 新增：智能选取
            'traditional': 0.2,              # 传统多类型
            'single_type': 0.15,             # 单类型
            'random_combo': 0.15             # 随机组合
        })
        
        # 日志
        self.logger = self._setup_logger()
        
        # 数据存储
        self.graph = None
        self.extracted_subgraphs = {}
        self.subgraph_type_stats = defaultdict(int)
        self.size_distribution = defaultdict(int)
        
        # 用户交互关系缓存
        self.user_interactions = {}
        self.user_interaction_loaded = False
        
    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_user_interactions(self, graph: HeteroData):
        """加载用户交互关系"""
        if self.user_interaction_loaded:
            return
            
        self.logger.info("加载用户交互关系...")
        
        if ('user', 'interacts', 'user') in graph.edge_types:
            edge_index = graph[('user', 'interacts', 'user')].edge_index
            
            # 构建用户交互邻接表
            for i in range(edge_index.size(1)):
                user1 = edge_index[0, i].item()
                user2 = edge_index[1, i].item()
                
                if user1 not in self.user_interactions:
                    self.user_interactions[user1] = set()
                if user2 not in self.user_interactions:
                    self.user_interactions[user2] = set()
                    
                self.user_interactions[user1].add(user2)
                self.user_interactions[user2].add(user1)  # 假设交互是双向的
        
        self.user_interaction_loaded = True
        self.logger.info(f"加载完成，发现 {len(self.user_interactions)} 个用户有交互关系")

    def _check_users_have_interactions(self, user_list: List[int]) -> bool:
        """检查用户列表中是否至少有两个用户有交互关系"""
        if len(user_list) < 2:
            return True  # 单用户或无用户，返回True以便进行完全枚举
            
        # 检查是否有至少一对用户有交互
        for i in range(len(user_list)):
            for j in range(i + 1, len(user_list)):
                user1, user2 = user_list[i], user_list[j]
                if (user1 in self.user_interactions and 
                    user2 in self.user_interactions[user1]):
                    return True
        
        return False

    def extract_all_session_subgraphs(self, graph: HeteroData, output_dir: str = "data/subgraphs/universal_enhanced") -> Dict[str, Any]:
        """
        提取所有会话的子图 - 增强版（支持多用户交互和完全枚举）
        
        Args:
            graph: 异构图对象
            output_dir: 输出目录
            
        Returns:
            提取统计信息
        """
        self.logger.info("=== 开始增强版通用子图提取 ===")
        self.logger.info("增强特性: 6-15节点范围、多用户交互、智能枚举、会话级进度")
        self.graph = graph
        
        # 加载用户交互关系
        self._load_user_interactions(graph)
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 获取所有会话
        session_ids = self._get_all_sessions(graph)
        self.logger.info(f"发现 {len(session_ids)} 个会话")
        
        # 为每个会话提取子图，带详细进度显示
        total_subgraphs = 0
        successful_sessions = 0
        
        # 使用tqdm显示会话级进度
        session_progress = tqdm(session_ids, desc="处理会话", unit="session")
        
        for session_id in session_progress:
            session_start_time = time.time()
            
            # 更新进度条描述
            session_progress.set_description(f"处理会话 {session_id}")
            
            session_subgraphs = self._extract_session_subgraphs_enhanced(graph, session_id)
            
            if session_subgraphs:
                self.extracted_subgraphs[session_id] = session_subgraphs
                total_subgraphs += len(session_subgraphs)
                successful_sessions += 1
                
                # 保存该会话的子图
                self._save_session_subgraphs(session_id, session_subgraphs, output_dir)
            
            # 更新进度条后缀信息
            session_time = time.time() - session_start_time
            session_progress.set_postfix({
                '子图数': len(session_subgraphs) if session_subgraphs else 0,
                '总计': total_subgraphs,
                '耗时': f"{session_time:.2f}s"
            })
        
        session_progress.close()
        
        # 保存总体索引和统计
        self._save_subgraph_index(output_dir)
        self._save_enhanced_stats(output_dir)
        
        stats = {
            'total_sessions': len(session_ids),
            'successful_sessions': successful_sessions,
            'sessions_with_subgraphs': len(self.extracted_subgraphs),
            'total_subgraphs': total_subgraphs,
            'avg_subgraphs_per_session': total_subgraphs / successful_sessions if successful_sessions else 0,
            'subgraph_type_distribution': dict(self.subgraph_type_stats),
            'size_distribution': dict(self.size_distribution),
            'extraction_timestamp': datetime.now().isoformat(),
            'enhancement_features': {
                'size_range': f"{self.min_subgraph_size}-{self.max_subgraph_size}",
                'multi_user_enabled': self.enable_multi_user_subgraphs,
                'smart_selection_enabled': self.enable_smart_selection,
                'user_interactions_loaded': len(self.user_interactions),
                'strategy_weights': self.strategy_weights
            }
        }
        
        self.logger.info(f"增强版提取完成:")
        self.logger.info(f"  - 成功会话: {successful_sessions}/{len(session_ids)}")
        self.logger.info(f"  - 总子图数: {total_subgraphs}")
        self.logger.info(f"  - 平均每会话: {stats['avg_subgraphs_per_session']:.1f}")
        self.logger.info(f"  - 类型分布: {dict(self.subgraph_type_stats)}")
        
        return stats
    
    def _get_all_sessions(self, graph: HeteroData) -> List[str]:
        """获取所有会话ID"""
        if 'media_session' in graph.node_types:
            num_sessions = graph['media_session'].x.shape[0]
            return [f"media_session_{i}" for i in range(num_sessions)]
        else:
            self.logger.warning("未找到media_session节点类型")
            return []
    
    def _extract_session_subgraphs_enhanced(self, graph: HeteroData, session_id: str) -> List[Dict[str, Any]]:
        """
        为单个会话提取多个子图 - 增强版（支持多用户交互和完全枚举）
        
        Args:
            graph: 异构图对象
            session_id: 会话ID (如 "media_session_0")
            
        Returns:
            该会话的子图列表
        """
        try:
            # 解析会话索引
            session_idx = int(session_id.split('_')[-1])
            
            # 获取该会话的所有节点
            session_nodes = self._get_session_all_nodes_comprehensive(graph, session_idx)
            
            total_nodes = sum(len(nodes) for nodes in session_nodes.values())
            if total_nodes < self.min_subgraph_size:
                return []
            
            session_subgraphs = []
            
            # 策略1: 多用户交互子图（新增）
            if self.enable_multi_user_subgraphs and len(session_nodes.get('user', [])) >= 2:
                multi_user_subgraphs = self._generate_multi_user_interaction_subgraphs(
                    graph, session_id, session_nodes)
                session_subgraphs.extend(multi_user_subgraphs)
            
            # 策略2: 智能选取子图（新增）
            if self.enable_smart_selection:
                enumeration_subgraphs = self._generate_complete_enumeration_subgraphs(
                    graph, session_id, session_nodes)
                session_subgraphs.extend(enumeration_subgraphs)
            
            # 策略3: 传统的基于评论的多类型子图
            traditional_count = max(1, int(len(session_subgraphs) * self.strategy_weights.get('traditional', 0.2)))
            traditional_subgraphs = self._generate_traditional_subgraphs_enhanced(
                graph, session_id, session_idx, session_nodes, traditional_count)
            session_subgraphs.extend(traditional_subgraphs)
            
            # 策略4: 单类型子图
            if self.enable_single_type_subgraphs:
                single_type_count = max(1, int(len(session_subgraphs) * self.strategy_weights.get('single_type', 0.15)))
                single_type_subgraphs = self._generate_single_type_subgraphs_comprehensive(
                    graph, session_id, session_nodes, single_type_count)
                session_subgraphs.extend(single_type_subgraphs)
            
            # 策略5: 随机组合子图
            if self.diversity_boost:
                random_count = max(1, int(len(session_subgraphs) * self.strategy_weights.get('random_combo', 0.15)))
                random_subgraphs = self._generate_random_combination_subgraphs_enhanced(
                    graph, session_id, session_nodes, random_count)
                session_subgraphs.extend(random_subgraphs)
            
            # 去重和验证
            session_subgraphs = self._validate_and_deduplicate_subgraphs_enhanced(session_subgraphs)
            
            # 更新统计
            for subgraph in session_subgraphs:
                self.size_distribution[subgraph['total_nodes']] += 1
            
            return session_subgraphs
            
        except Exception as e:
            self.logger.error(f"提取会话 {session_id} 子图失败: {e}")
            return []
    
    def _generate_multi_user_interaction_subgraphs(self, graph: HeteroData, session_id: str, 
                                                   session_nodes: Dict[str, List[int]]) -> List[Dict[str, Any]]:
        """生成多用户交互子图（新增策略）"""
        subgraphs = []
        user_nodes = session_nodes.get('user', [])
        
        if len(user_nodes) < 2:
            return []
        
        self.logger.debug(f"会话 {session_id}: 尝试生成多用户交互子图，用户数: {len(user_nodes)}")
        
        # 生成2-5用户的组合
        for user_count in range(2, min(6, len(user_nodes) + 1)):
            user_combinations = list(combinations(user_nodes, user_count))
            
            # 限制组合数量
            if len(user_combinations) > self.max_multi_user_combinations:
                user_combinations = random.sample(user_combinations, self.max_multi_user_combinations)
            
            for user_combo in user_combinations:
                # 检查这些用户是否有交互关系
                if self._check_users_have_interactions(list(user_combo)):
                    # 为这组用户构建子图
                    for target_size in range(self.min_subgraph_size, self.max_subgraph_size + 1, 2):
                        subgraph = self._build_multi_user_subgraph(
                            graph, session_id, session_nodes, list(user_combo), target_size)
                        if subgraph:
                            subgraphs.append(subgraph)
                            self.subgraph_type_stats[f'multi_user_{user_count}'] += 1
        
        self.logger.debug(f"会话 {session_id}: 生成了 {len(subgraphs)} 个多用户交互子图")
        return subgraphs

    def _generate_complete_enumeration_subgraphs(self, graph: HeteroData, session_id: str, 
                                                 session_nodes: Dict[str, List[int]]) -> List[Dict[str, Any]]:
        """生成智能全面枚举子图（覆盖所有有意义的组合）"""
        subgraphs = []
        user_nodes = session_nodes.get('user', [])
        
        # 只对单用户或无用户交互的情况进行智能全面枚举
        if len(user_nodes) > 1 and self._check_users_have_interactions(user_nodes):
            # 有多用户且有交互关系，跳过智能枚举（由多用户策略处理）
            return []
        
        self.logger.debug(f"会话 {session_id}: 开始智能全面枚举，用户数: {len(user_nodes)}")
        
        # 获取各类型节点
        comment_nodes = session_nodes.get('comment', [])
        word_nodes = session_nodes.get('word', [])
        video_nodes = session_nodes.get('video', [])
        
        # 智能全面枚举：系统性地覆盖所有有意义的组合
        for target_size in range(self.min_subgraph_size, self.max_subgraph_size + 1):
            # 为每个目标大小，枚举所有可能的节点类型组合
            size_subgraphs = self._enumerate_all_meaningful_combinations_for_size(
                graph, session_id, session_nodes, target_size, 
                len(comment_nodes), len(user_nodes), len(video_nodes), len(word_nodes)
            )
            subgraphs.extend(size_subgraphs)
        
        self.logger.debug(f"会话 {session_id}: 智能全面枚举生成了 {len(subgraphs)} 个子图")
        return subgraphs
    
    def _enumerate_all_meaningful_combinations_for_size(self, graph: HeteroData, session_id: str,
                                                        session_nodes: Dict[str, List[int]], target_size: int,
                                                        max_comments: int, max_users: int, 
                                                        max_videos: int, max_words: int) -> List[Dict[str, Any]]:
        """为指定大小枚举所有有意义的节点组合"""
        subgraphs = []
        
        # 系统性枚举：comment数量从1到min(target_size-1, max_comments)
        max_comment_count = min(target_size - 1, max_comments)
        
        for comment_count in range(1, max_comment_count + 1):
            remaining_size = target_size - comment_count
            if remaining_size <= 0:
                continue
            
            # 为剩余大小枚举所有可能的user、video、word组合
            combination_subgraphs = self._enumerate_remaining_node_combinations(
                graph, session_id, session_nodes, comment_count, remaining_size,
                max_users, max_videos, max_words
            )
            subgraphs.extend(combination_subgraphs)
        
        return subgraphs
    
    def _enumerate_remaining_node_combinations(self, graph: HeteroData, session_id: str,
                                               session_nodes: Dict[str, List[int]], comment_count: int,
                                               remaining_size: int, max_users: int, 
                                               max_videos: int, max_words: int) -> List[Dict[str, Any]]:
        """枚举剩余节点的所有有意义组合"""
        subgraphs = []
        
        # 枚举user数量：0到min(remaining_size, max_users)
        max_user_count = min(remaining_size, max_users)
        
        for user_count in range(0, max_user_count + 1):
            remaining_after_user = remaining_size - user_count
            if remaining_after_user < 0:
                continue
            
            # 枚举video数量：0到min(remaining_after_user, max_videos, 1)
            max_video_count = min(remaining_after_user, max_videos, 1)  # 通常最多1个视频
            
            for video_count in range(0, max_video_count + 1):
                remaining_after_video = remaining_after_user - video_count
                if remaining_after_video < 0:
                    continue
                
                # word数量由剩余大小确定
                word_count = remaining_after_video
                if word_count > max_words:
                    continue
                
                # 检查组合是否有意义
                if self._is_meaningful_combination(comment_count, user_count, video_count, word_count):
                    # 生成这种组合的子图
                    combination_subgraphs = self._generate_node_combination_subgraphs_comprehensive(
                        graph, session_id, session_nodes, 
                        comment_count, user_count, video_count, word_count
                    )
                    subgraphs.extend(combination_subgraphs)
        
        return subgraphs
    
    def _generate_node_combination_subgraphs_comprehensive(self, graph: HeteroData, session_id: str, 
                                                           session_nodes: Dict[str, List[int]],
                                                           comment_count: int, user_count: int, 
                                                           video_count: int, word_count: int) -> List[Dict[str, Any]]:
        """为指定的节点数量组合生成全面的子图（确保节点间有边连接）"""
        subgraphs = []
        
        # 获取各类型节点
        comment_nodes = session_nodes.get('comment', [])
        user_nodes = session_nodes.get('user', [])
        video_nodes = session_nodes.get('video', [])
        word_nodes = session_nodes.get('word', [])
        
        try:
            # 生成comment节点的组合
            if comment_count <= len(comment_nodes):
                if comment_count <= 3:
                    # 小数量：尝试不同的起始位置
                    comment_combos = []
                    for start_idx in range(min(len(comment_nodes) - comment_count + 1, 5)):
                        combo = comment_nodes[start_idx:start_idx + comment_count]
                        if len(combo) == comment_count:
                            comment_combos.append(combo)
                else:
                    # 大数量：随机采样少量组合
                    comment_combos = []
                    max_combos = min(3, len(comment_nodes) // comment_count + 1)
                    for _ in range(max_combos):
                        if len(comment_nodes) >= comment_count:
                            combo = random.sample(comment_nodes, comment_count)
                            comment_combos.append(combo)
            else:
                comment_combos = []
            
            # 对每个评论组合，构建连接的子图
            for comment_combo in comment_combos:
                # 基于评论节点构建连接的子图
                connected_subgraphs = self._build_connected_subgraphs_from_comments(
                    graph, session_id, session_nodes, comment_combo, 
                    user_count, video_count, word_count
                )
                subgraphs.extend(connected_subgraphs)
            
        except Exception as e:
            self.logger.error(f"生成连接子图失败: {e}")
        
        return subgraphs

    def _build_connected_subgraphs_from_comments(self, graph: HeteroData, session_id: str,
                                                session_nodes: Dict[str, List[int]], 
                                                comment_combo: List[int],
                                                user_count: int, video_count: int, word_count: int) -> List[Dict[str, Any]]:
        """基于评论节点构建连接的子图"""
        subgraphs = []
        
        try:
            # 获取与这些评论连接的用户
            connected_users = []
            if ('user', 'posts', 'comment') in graph.edge_types:
                user_comment_edges = graph[('user', 'posts', 'comment')].edge_index
                comment_tensor = torch.tensor(comment_combo)
                user_mask = torch.isin(user_comment_edges[1], comment_tensor)
                connected_users = user_comment_edges[0][user_mask].unique().tolist()
                # 确保用户在会话范围内
                connected_users = [u for u in connected_users if u in session_nodes.get('user', [])]
            
            # 获取与这些评论连接的词汇
            connected_words = []
            if ('comment', 'contains', 'word') in graph.edge_types:
                comment_word_edges = graph[('comment', 'contains', 'word')].edge_index
                comment_tensor = torch.tensor(comment_combo)
                word_mask = torch.isin(comment_word_edges[0], comment_tensor)
                connected_words = comment_word_edges[1][word_mask].unique().tolist()
                # 确保词汇在会话范围内
                connected_words = [w for w in connected_words if w in session_nodes.get('word', [])]
            
            # 获取与这些评论连接的视频
            connected_videos = []
            if ('video', 'contains', 'comment') in graph.edge_types:
                video_comment_edges = graph[('video', 'contains', 'comment')].edge_index
                comment_tensor = torch.tensor(comment_combo)
                video_mask = torch.isin(video_comment_edges[1], comment_tensor)
                connected_videos = video_comment_edges[0][video_mask].unique().tolist()
                # 确保视频在会话范围内
                connected_videos = [v for v in connected_videos if v in session_nodes.get('video', [])]
            
            # 生成用户组合
            if user_count == 0:
                user_combos = [[]]
            elif user_count <= len(connected_users):
                if user_count == 1:
                    user_combos = [[user] for user in connected_users[:min(3, len(connected_users))]]
                else:
                    user_combos = []
                    max_combos = min(3, len(connected_users) // user_count + 1)
                    for _ in range(max_combos):
                        if len(connected_users) >= user_count:
                            combo = random.sample(connected_users, user_count)
                            user_combos.append(combo)
            else:
                user_combos = []
            
            # 生成视频组合
            if video_count == 0:
                video_combos = [[]]
            elif video_count <= len(connected_videos):
                video_combos = [connected_videos[:video_count]]
            else:
                video_combos = []
            
            # 生成词汇组合
            if word_count == 0:
                word_combos = [[]]
            elif word_count <= len(connected_words):
                if word_count <= 5:
                    # 小数量词汇：多种组合
                    word_combos = []
                    max_combos = min(4, len(connected_words) // word_count + 1)
                    for _ in range(max_combos):
                        if len(connected_words) >= word_count:
                            combo = random.sample(connected_words, word_count)
                            word_combos.append(combo)
                else:
                    # 大数量词汇：少量组合
                    word_combos = []
                    max_combos = min(2, len(connected_words) // word_count + 1)
                    for _ in range(max_combos):
                        if len(connected_words) >= word_count:
                            combo = random.sample(connected_words, word_count)
                            word_combos.append(combo)
            else:
                word_combos = []
            
            # 组合所有连接的节点
            for user_combo in user_combos:
                for video_combo in video_combos:
                    for word_combo in word_combos:
                        selected_nodes = {
                            'comment': list(comment_combo),
                            'user': list(user_combo),
                            'video': list(video_combo),
                            'word': list(word_combo)
                        }
                        
                        subgraph = self._build_enumeration_subgraph(graph, session_id, selected_nodes)
                        if subgraph:
                            subgraphs.append(subgraph)
                            self.subgraph_type_stats['complete_enumeration'] += 1
            
        except Exception as e:
            self.logger.error(f"构建连接子图失败: {e}")
        
        return subgraphs

    def _build_multi_user_subgraph(self, graph: HeteroData, session_id: str, 
                                   session_nodes: Dict[str, List[int]], 
                                   target_users: List[int], target_size: int) -> Optional[Dict[str, Any]]:
        """构建多用户交互子图"""
        try:
            selected_nodes = {
                'comment': [],
                'user': target_users.copy(),
                'video': [],
                'word': []
            }
            
            current_size = len(target_users)
            remaining_size = target_size - current_size
            
            if remaining_size <= 0:
                return None
            
            # 优先添加这些用户的评论
            comment_nodes = session_nodes.get('comment', [])
            user_comments = []
            
            # 通过边关系找到这些用户发布的评论
            if ('user', 'posts', 'comment') in graph.edge_types:
                user_post_edges = graph[('user', 'posts', 'comment')].edge_index
                for user_id in target_users:
                    user_mask = user_post_edges[0] == user_id
                    user_specific_comments = user_post_edges[1][user_mask].tolist()
                    user_comments.extend([c for c in user_specific_comments if c in comment_nodes])
            
            # 添加评论节点
            if user_comments and remaining_size > 0:
                comment_count = min(len(user_comments), remaining_size)
                selected_nodes['comment'] = random.sample(user_comments, comment_count)
                remaining_size -= comment_count
            
            # 添加视频节点
            video_nodes = session_nodes.get('video', [])
            if video_nodes and remaining_size > 0:
                video_count = min(len(video_nodes), remaining_size, 1)  # 通常只有1个视频
                selected_nodes['video'] = random.sample(video_nodes, video_count)
                remaining_size -= video_count
            
            # 添加词汇节点
            word_nodes = session_nodes.get('word', [])
            if word_nodes and remaining_size > 0:
                word_count = min(len(word_nodes), remaining_size)
                selected_nodes['word'] = random.sample(word_nodes, word_count)
            
            total_nodes = sum(len(nodes) for nodes in selected_nodes.values())
            if total_nodes < self.min_subgraph_size:
                return None
    
            # 提取边信息
            subgraph_edges = self._extract_subgraph_edges(graph, selected_nodes)
            
            subgraph = {
                'session_id': session_id,
                'subgraph_id': f"{session_id}_multi_user_{len(target_users)}u_{total_nodes}n_{hash(str(selected_nodes)) % 10000}",
                'nodes': selected_nodes,
                'edges': subgraph_edges,
                'total_nodes': total_nodes,
                'subgraph_type': 'multi_user_interaction',
                'target_users': target_users,
                'user_interaction_verified': True,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            return subgraph
            
        except Exception as e:
            self.logger.error(f"构建多用户子图失败: {e}")
            return None
    
    def _build_enumeration_subgraph(self, graph: HeteroData, session_id: str, 
                                    selected_nodes: Dict[str, List[int]]) -> Optional[Dict[str, Any]]:
        """构建完全枚举子图"""
        try:
            total_nodes = sum(len(nodes) for nodes in selected_nodes.values())
            
            if total_nodes < self.min_subgraph_size or total_nodes > self.max_subgraph_size:
                return None
            
            # 提取边信息
            subgraph_edges = self._extract_subgraph_edges(graph, selected_nodes)
            
            # 生成节点构成说明
            node_composition = []
            for node_type, nodes in selected_nodes.items():
                if nodes:
                    node_composition.append(f"{node_type}({len(nodes)})")
            
            subgraph = {
                'session_id': session_id,
                'subgraph_id': f"{session_id}_enum_{'_'.join(node_composition)}_{hash(str(selected_nodes)) % 10000}",
                'nodes': selected_nodes,
                'edges': subgraph_edges,
                'total_nodes': total_nodes,
                'subgraph_type': 'complete_enumeration',
                'node_composition': node_composition,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            return subgraph
            
        except Exception as e:
            self.logger.error(f"构建枚举子图失败: {e}")
            return None
    
    def _get_session_all_nodes_comprehensive(self, graph: HeteroData, session_idx: int) -> Dict[str, List[int]]:
        """
        获取会话的所有节点 - 全面版
        
        Args:
            graph: 异构图对象
            session_idx: 会话索引
            
        Returns:
            各类型节点的索引列表
        """
        session_nodes = {
            'comment': [],
            'user': [],
            'word': [],
            'video': [],
            'media_session': [session_idx]  # 会话节点本身
        }
        
        try:
            # 获取评论节点 (通过media_session -> comment边)
            if ('media_session', 'contains', 'comment') in graph.edge_types:
                media_comment_edges = graph[('media_session', 'contains', 'comment')].edge_index
                if media_comment_edges.numel() > 0:
                    # 找到从当前session出发的所有评论
                    mask = media_comment_edges[0] == session_idx
                    session_nodes['comment'] = media_comment_edges[1][mask].tolist()
            
            # 获取用户节点 (通过user -> comment边，修复边类型错误)
            if session_nodes['comment'] and ('user', 'posts', 'comment') in graph.edge_types:
                user_comment_edges = graph[('user', 'posts', 'comment')].edge_index
                if user_comment_edges.numel() > 0:
                    # 找到发布评论的用户
                    comment_indices = torch.tensor(session_nodes['comment'])
                    user_mask = torch.isin(user_comment_edges[1], comment_indices)
                    session_nodes['user'] = user_comment_edges[0][user_mask].unique().tolist()
            
            # 获取词汇节点 (通过comment -> word边)
            if session_nodes['comment'] and ('comment', 'contains', 'word') in graph.edge_types:
                comment_word_edges = graph[('comment', 'contains', 'word')].edge_index
                if comment_word_edges.numel() > 0:
                    # 找到评论包含的词汇
                    comment_indices = torch.tensor(session_nodes['comment'])
                    word_mask = torch.isin(comment_word_edges[0], comment_indices)
                    session_nodes['word'] = comment_word_edges[1][word_mask].unique().tolist()
            
            # 获取视频节点 (通过video -> comment边)
            if session_nodes['comment'] and ('video', 'contains', 'comment') in graph.edge_types:
                video_comment_edges = graph[('video', 'contains', 'comment')].edge_index
                if video_comment_edges.numel() > 0:
                    # 找到包含评论的视频
                    comment_indices = torch.tensor(session_nodes['comment'])
                    video_mask = torch.isin(video_comment_edges[1], comment_indices)
                    session_nodes['video'] = video_comment_edges[0][video_mask].unique().tolist()
            
            # 过滤空列表并限制数量（更宽松的限制）
            for node_type in ['comment', 'user', 'word']:
                if session_nodes[node_type]:
                    max_count = {
                        'comment': self.max_comments_per_video,
                        'user': self.max_users_per_subgraph,
                        'word': self.max_words_per_subgraph
                    }[node_type]
                    
                    if len(session_nodes[node_type]) > max_count:
                        # 随机采样而不是简单截取
                        session_nodes[node_type] = random.sample(session_nodes[node_type], max_count)
            
        except Exception as e:
            self.logger.error(f"获取会话 {session_idx} 节点失败: {e}")
        
        return session_nodes
    
    def _extract_subgraph_edges(self, graph: HeteroData, selected_nodes: Dict[str, List[int]]) -> Dict:
        """提取子图内的边 - 修复版，基于真实图结构"""
        subgraph_edges = {}
        
        # 转换为集合以提高查找效率
        node_sets = {node_type: set(nodes) for node_type, nodes in selected_nodes.items()}
        
        # 基于真实图结构的所有边类型
        all_edge_types = [
            ('media_session', 'contains', 'comment'),  # 会话-评论
            ('user', 'posts', 'comment'),              # 用户-评论
            ('comment', 'contains', 'word'),           # 评论-词汇
            ('user', 'interacts', 'user'),             # 用户-用户交互
            ('video', 'contains', 'comment'),          # 视频-评论
            ('user', 'creates', 'video'),              # 用户-视频
            ('emotion', 'triggers', 'action'),         # 情感-行为
            ('time', 'associated_with', 'location'),   # 时间-地点
        ]
        
        for edge_type in all_edge_types:
            if edge_type not in graph.edge_types:
                continue
                
            src_type, relation, dst_type = edge_type
            
            # 检查是否有相关节点
            if src_type in node_sets and dst_type in node_sets:
                if not node_sets[src_type] or not node_sets[dst_type]:
                    continue
                    
                edge_index = graph[edge_type].edge_index
                
                # 使用张量操作进行高效筛选
                src_tensor = torch.tensor(list(node_sets[src_type]), dtype=edge_index.dtype)
                dst_tensor = torch.tensor(list(node_sets[dst_type]), dtype=edge_index.dtype)
                
                src_mask = torch.isin(edge_index[0], src_tensor)
                dst_mask = torch.isin(edge_index[1], dst_tensor)
                valid_mask = src_mask & dst_mask
                
                if valid_mask.any():
                    valid_edge_index = edge_index[:, valid_mask]
                    subgraph_edges[edge_type] = valid_edge_index.t().tolist()
        
        return subgraph_edges
    
    def _validate_and_deduplicate_subgraphs_enhanced(self, subgraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证和去重子图"""
        validated = []
        seen_node_sets = set()
        
        for subgraph in subgraphs:
            # 基本验证
            if not self._is_valid_subgraph(subgraph):
                continue
            
            # 去重检查
            nodes = subgraph.get('nodes', {})
            all_nodes = []
            for node_list in nodes.values():
                all_nodes.extend(node_list)
            
            node_set = frozenset(all_nodes)
            if node_set not in seen_node_sets:
                seen_node_sets.add(node_set)
                validated.append(subgraph)
        
        return validated
    
    def _is_valid_subgraph(self, subgraph: Dict[str, Any]) -> bool:
        """检查子图是否有效 - 优化版，更宽松的验证"""
        nodes = subgraph.get('nodes', {})
        total_nodes = sum(len(node_list) for node_list in nodes.values())
        
        # 大小检查
        if total_nodes < self.min_subgraph_size or total_nodes > self.max_subgraph_size:
            return False
        
        # 获取提取方法
        extraction_method = subgraph.get('extraction_method', '')
        
        # 对于单类型子图，放宽验证条件
        if 'single_type' in extraction_method:
            # 单类型子图只需要有节点即可
            return total_nodes >= self.single_type_min_size
        
        # 对于传统多类型子图，需要有评论节点
        if 'traditional' in extraction_method:
            if not nodes.get('comment', []):
                return False
            # 必须有多种类型的节点
            non_empty_types = sum(1 for node_list in nodes.values() if node_list)
            if non_empty_types < 2:
                return False
        
        # 对于随机组合和大型复合子图，更宽松的验证
        if 'random' in extraction_method or 'large' in extraction_method:
            # 只要有节点就行
            return total_nodes >= self.min_subgraph_size
        
        return True
    
    def _save_session_subgraphs(self, session_id: str, subgraphs: List[Dict[str, Any]], output_dir: str):
        """保存会话子图到文件"""
        # 构建文件数据
        session_data = {
            'session_id': session_id,
            'total_subgraphs': len(subgraphs),
            'size_distribution': Counter(sg.get('size', 0) for sg in subgraphs),
            'extraction_methods': Counter(sg.get('extraction_method', 'unknown') for sg in subgraphs),
            'extraction_timestamp': datetime.now().isoformat(),
            'subgraphs': subgraphs
        }
        
        # 保存到文件
        output_file = f"{output_dir}/{session_id}_subgraphs.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(session_data, f)
        
        self.logger.debug(f"保存会话 {session_id} 的 {len(subgraphs)} 个子图到 {output_file}")
    
    def _save_subgraph_index(self, output_dir: str):
        """保存子图索引文件"""
        index_data = {
            'extraction_timestamp': datetime.now().isoformat(),
            'total_sessions': len(self.extracted_subgraphs),
            'total_subgraphs': sum(len(subgraphs) for subgraphs in self.extracted_subgraphs.values()),
            'session_files': {
                session_id: f"{session_id}_subgraphs.pkl"
                for session_id in self.extracted_subgraphs.keys()
            },
            'size_statistics': self._calculate_size_statistics(),
            'subgraph_type_distribution': self.subgraph_type_stats,
            'optimization_features': {
                'expanded_size_range': f"{self.min_subgraph_size}-{self.max_subgraph_size}",
                'single_type_subgraphs': self.enable_single_type_subgraphs,
                'random_sampling_ratio': self.random_sampling_ratio,
                'diversity_boost': self.diversity_boost
            }
        }
        
        index_file = f"{output_dir}/subgraph_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(index_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"保存子图索引到 {index_file}")
    
    def _calculate_size_statistics(self) -> Dict[str, Any]:
        """计算子图大小统计"""
        all_sizes = []
        for subgraphs in self.extracted_subgraphs.values():
            all_sizes.extend(sg.get('size', 0) for sg in subgraphs)
        
        if not all_sizes:
            return {}
        
        return {
            'size_distribution': dict(Counter(all_sizes)),
            'avg_size': float(np.mean(all_sizes)),
            'min_size': int(min(all_sizes)),
            'max_size': int(max(all_sizes)),
            'total_subgraphs': len(all_sizes)
        }
    
    def _save_enhanced_stats(self, output_dir: str):
        """保存综合统计信息"""
        stats_data = {
            'extraction_timestamp': datetime.now().isoformat(),
            'subgraph_type_distribution': dict(self.subgraph_type_stats),
            'size_distribution': dict(self.size_distribution),
            'enhancement_features': {
                'multi_user_enabled': self.enable_multi_user_subgraphs,
                'smart_selection_enabled': self.enable_smart_selection,
                'user_interactions_loaded': len(self.user_interactions),
                'size_range': f"{self.min_subgraph_size}-{self.max_subgraph_size}"
            }
        }
        
        stats_file = f"{output_dir}/enhanced_extraction_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(stats_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"保存增强统计信息到 {stats_file}")

    def _generate_traditional_subgraphs_enhanced(self, graph: HeteroData, session_id: str, 
                                                 session_idx: int, session_nodes: Dict[str, List[int]], count: int) -> List[Dict[str, Any]]:
        """生成传统的基于评论的多类型子图"""
        subgraphs = []
        comment_nodes = session_nodes.get('comment', [])
        
        if not comment_nodes:
            return []
        
        # 选择多个种子评论
        num_seeds = min(len(comment_nodes), max(1, count // 3))
        if len(comment_nodes) > num_seeds:
            seed_comments = random.sample(comment_nodes, num_seeds)
        else:
            seed_comments = comment_nodes
        
        # 为每个种子评论生成多尺度子图
        for seed_comment in seed_comments:
            # 生成多种大小的子图（在6-15范围内）
            for target_size in range(self.min_subgraph_size, self.max_subgraph_size + 1, 2):
                subgraph = self._build_traditional_subgraph(graph, session_id, seed_comment, session_nodes, target_size)
                if subgraph:
                    subgraphs.append(subgraph)
                    self.subgraph_type_stats['traditional_multi'] += 1
                    
                    # 限制每个种子的子图数量
                    if len(subgraphs) >= count:
                        return subgraphs
        
        return subgraphs

    def _build_traditional_subgraph(self, graph: HeteroData, session_id: str, seed_comment: int,
                                   session_nodes: Dict[str, List[int]], target_size: int) -> Optional[Dict[str, Any]]:
        """构建传统的基于评论的多类型子图"""
        try:
            selected_nodes = {'comment': [seed_comment], 'user': [], 'word': [], 'video': []}
            current_size = 1
            
            # 添加与种子评论相关的用户
            if ('user', 'posts', 'comment') in graph.edge_types:
                user_post_edges = graph[('user', 'posts', 'comment')].edge_index
                user_mask = user_post_edges[1] == seed_comment  # 注意：这里是用户发布评论
                if user_mask.any():
                    related_users = user_post_edges[0][user_mask].tolist()
                    # 限制用户数量
                    max_users = min(len(related_users), (target_size - current_size) // 2, 3)
                    if max_users > 0:
                        selected_users = random.sample(related_users, max_users)
                        selected_nodes['user'].extend(selected_users)
                        current_size += len(selected_users)
            
            # 添加与种子评论相关的词汇
            if current_size < target_size and ('comment', 'contains', 'word') in graph.edge_types:
                comment_word_edges = graph[('comment', 'contains', 'word')].edge_index
                word_mask = comment_word_edges[0] == seed_comment
                if word_mask.any():
                    related_words = comment_word_edges[1][word_mask].tolist()
                    max_words = min(len(related_words), target_size - current_size, 5)
                    if max_words > 0:
                        selected_words = random.sample(related_words, max_words)
                        selected_nodes['word'].extend(selected_words)
                        current_size += len(selected_words)
            
            # 如果还需要更多节点，从会话的其他节点中随机添加
            if current_size < target_size:
                remaining_needed = target_size - current_size
                
                # 从其他评论中添加
                other_comments = [c for c in session_nodes.get('comment', []) if c != seed_comment]
                if other_comments and remaining_needed > 0:
                    max_additional_comments = min(len(other_comments), remaining_needed // 2)
                    if max_additional_comments > 0:
                        additional_comments = random.sample(other_comments, max_additional_comments)
                        selected_nodes['comment'].extend(additional_comments)
                        current_size += len(additional_comments)
                        remaining_needed -= len(additional_comments)
                
                # 从其他用户中添加
                other_users = [u for u in session_nodes.get('user', []) if u not in selected_nodes['user']]
                if other_users and remaining_needed > 0:
                    max_additional_users = min(len(other_users), remaining_needed)
                    if max_additional_users > 0:
                        additional_users = random.sample(other_users, max_additional_users)
                        selected_nodes['user'].extend(additional_users)
                        current_size += len(additional_users)
                        remaining_needed -= len(additional_users)
                
                # 添加视频节点
                video_nodes = session_nodes.get('video', [])
                if video_nodes and remaining_needed > 0:
                    video_count = min(len(video_nodes), remaining_needed, 1)
                    selected_nodes['video'] = random.sample(video_nodes, video_count)
                    current_size += video_count
            
            # 检查最小大小
            if current_size < self.min_subgraph_size:
                return None
            
            # 提取边信息
            subgraph_edges = self._extract_subgraph_edges(graph, selected_nodes)
            
            # 构建子图字典
            subgraph = {
                'session_id': session_id,
                'subgraph_id': f"{session_id}_traditional_{seed_comment}_{current_size}",
                'nodes': selected_nodes,
                'edges': subgraph_edges,
                'total_nodes': current_size,
                'seed_comment': seed_comment,
                'subgraph_type': 'traditional_multi',
                'target_size': target_size,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            return subgraph
            
        except Exception as e:
            self.logger.error(f"构建传统子图失败: {e}")
            return None

    def _generate_single_type_subgraphs_comprehensive(self, graph: HeteroData, session_id: str, 
                                                      session_nodes: Dict[str, List[int]], count: int) -> List[Dict[str, Any]]:
        """生成单类型节点构成的子图"""
        subgraphs = []
        
        # 为每种节点类型生成单类型子图
        for node_type, node_list in session_nodes.items():
            if node_type in ['video', 'media_session']:  # 跳过视频和会话节点
                continue
                
            if len(node_list) >= self.single_type_min_size:
                # 生成不同大小的单类型子图
                max_size = min(len(node_list), self.single_type_max_size)
                for target_size in range(self.single_type_min_size, max_size + 1, 2):
                    if target_size <= len(node_list):
                        # 随机选择节点
                        selected_nodes = random.sample(node_list, target_size)
                        
                        subgraph = self._build_single_type_subgraph(graph, session_id, node_type, selected_nodes)
                        if subgraph:
                            subgraphs.append(subgraph)
                            self.subgraph_type_stats[f'{node_type}_only'] += 1
                            
                            # 限制单类型子图数量
                            if len(subgraphs) >= count:
                                return subgraphs
        
        return subgraphs

    def _build_single_type_subgraph(self, graph: HeteroData, session_id: str, 
                                   node_type: str, selected_node_indices: List[int]) -> Optional[Dict[str, Any]]:
        """构建单类型子图"""
        try:
            selected_nodes = {nt: [] for nt in ['comment', 'user', 'word', 'video']}
            selected_nodes[node_type] = selected_node_indices
            
            # 对于单类型子图，边信息相对简单
            subgraph_edges = self._extract_subgraph_edges(graph, selected_nodes)
            
            subgraph = {
                'session_id': session_id,
                'subgraph_id': f"{session_id}_{node_type}_only_{len(selected_node_indices)}",
                'nodes': selected_nodes,
                'edges': subgraph_edges,
                'total_nodes': len(selected_node_indices),
                'subgraph_type': f'{node_type}_only',
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            return subgraph
            
        except Exception as e:
            self.logger.error(f"构建单类型子图失败: {e}")
            return None

    def _generate_random_combination_subgraphs_enhanced(self, graph: HeteroData, session_id: str, 
                                                       session_nodes: Dict[str, List[int]], count: int) -> List[Dict[str, Any]]:
        """生成随机组合子图"""
        subgraphs = []
        
        # 计算可生成的随机子图数量
        total_nodes = sum(len(nodes) for nodes in session_nodes.values())
        num_random_subgraphs = min(count, max(1, int(total_nodes * self.random_sampling_ratio / 5)))
        
        for _ in range(num_random_subgraphs):
            # 随机选择子图大小
            target_size = random.randint(self.min_subgraph_size, min(self.max_subgraph_size, total_nodes))
            
            # 随机分配各类型节点数量
            node_allocation = self._random_allocate_nodes(session_nodes, target_size)
            
            if sum(node_allocation.values()) >= self.min_subgraph_size:
                subgraph = self._build_random_combination_subgraph(graph, session_id, session_nodes, node_allocation)
                if subgraph:
                    subgraphs.append(subgraph)
                    if target_size <= 8:
                        self.subgraph_type_stats['random_combo_small'] += 1
                    elif target_size <= 12:
                        self.subgraph_type_stats['random_combo_medium'] += 1
                    else:
                        self.subgraph_type_stats['random_combo_large'] += 1
        
        return subgraphs

    def _random_allocate_nodes(self, session_nodes: Dict[str, List[int]], target_size: int) -> Dict[str, int]:
        """随机分配各类型节点数量"""
        allocation = {}
        remaining_size = target_size
        
        # 获取有节点的类型
        available_types = [node_type for node_type, nodes in session_nodes.items() 
                          if nodes and node_type != 'media_session']
        
        if not available_types:
            return {}
        
        # 确保至少有1个comment节点
        if 'comment' in available_types and session_nodes['comment']:
            comment_count = min(remaining_size // 2 + 1, len(session_nodes['comment']))
            allocation['comment'] = comment_count
            remaining_size -= comment_count
            available_types.remove('comment')
        
        # 随机分配剩余节点
        for i, node_type in enumerate(available_types):
            if remaining_size <= 0:
                allocation[node_type] = 0
                continue
                
            if i == len(available_types) - 1:  # 最后一个类型
                allocation[node_type] = min(remaining_size, len(session_nodes[node_type]))
            else:
                # 随机分配1到剩余数量之间的节点
                max_for_this_type = min(remaining_size - (len(available_types) - i - 1), len(session_nodes[node_type]))
                if max_for_this_type > 0:
                    count = random.randint(0, max_for_this_type)
                    allocation[node_type] = count
                    remaining_size -= count
                else:
                    allocation[node_type] = 0
        
        # 确保所有类型都有分配
        for node_type in ['comment', 'user', 'word', 'video']:
            if node_type not in allocation:
                allocation[node_type] = 0
        
        return allocation

    def _build_random_combination_subgraph(self, graph: HeteroData, session_id: str,
                                         session_nodes: Dict[str, List[int]], 
                                         node_allocation: Dict[str, int]) -> Optional[Dict[str, Any]]:
        """构建随机组合子图"""
        try:
            selected_nodes = {}
            total_nodes = 0
            
            for node_type, count in node_allocation.items():
                if count > 0 and session_nodes.get(node_type):
                    available_nodes = session_nodes[node_type]
                    actual_count = min(count, len(available_nodes))
                    if actual_count > 0:
                        selected_nodes[node_type] = random.sample(available_nodes, actual_count)
                        total_nodes += actual_count
                    else:
                        selected_nodes[node_type] = []
                else:
                    selected_nodes[node_type] = []
            
            if total_nodes < self.min_subgraph_size:
                return None
            
            # 提取边信息
            subgraph_edges = self._extract_subgraph_edges(graph, selected_nodes)
            
            subgraph = {
                'session_id': session_id,
                'subgraph_id': f"{session_id}_random_combo_{total_nodes}_{hash(str(selected_nodes)) % 10000}",
                'nodes': selected_nodes,
                'edges': subgraph_edges,
                'total_nodes': total_nodes,
                'subgraph_type': 'random_combo',
                'node_allocation': node_allocation,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            return subgraph
            
        except Exception as e:
            self.logger.error(f"构建随机组合子图失败: {e}")
            return None

    def _is_meaningful_combination(self, comment_count: int, user_count: int, 
                                   video_count: int, word_count: int) -> bool:
        """判断节点组合是否有意义"""
        total_nodes = comment_count + user_count + video_count + word_count
        
        # 必须有至少1个comment节点
        if comment_count < 1:
            return False
        
        # 不能全是word节点
        if word_count == total_nodes:
            return False
        
        # 节点数量在合理范围内
        if total_nodes < self.min_subgraph_size or total_nodes > self.max_subgraph_size:
            return False
        
        return True


def main():
    """主函数 - 增强版"""
    # 增强配置
    config = {
        'min_subgraph_size': 6,
        'max_subgraph_size': 15,
        'size_step': 1,
        'max_comments_per_video': 50,
        'max_users_per_subgraph': 8,
        'max_words_per_subgraph': 25,
        'enable_multi_user_subgraphs': True,
        'min_interacting_users': 2,
        'max_multi_user_combinations': 100,
        'enable_smart_selection': True,
        'max_enumeration_combinations': 500,
        'enable_single_type_subgraphs': True,
        'random_sampling_ratio': 0.3,
        'diversity_boost': True,
        'enable_large_subgraphs': True
    }
    
    # 创建提取器
    extractor = UniversalSubgraphExtractor(config)
    
    # 加载图数据
    print("加载异构图数据...")
    with open("data/graphs/heterogeneous_graph_final.pkl", 'rb') as f:
        graph = pickle.load(f)
    
    print(f"图节点类型: {graph.node_types}")
    print(f"图边类型: {graph.edge_types}")
    
    # 提取子图
    print("开始增强版子图提取...")
    stats = extractor.extract_all_session_subgraphs(graph, "data/subgraphs/universal_enhanced")
    
    print("增强版提取完成！")
    print(f"统计信息: {stats}")


if __name__ == "__main__":
    main() 