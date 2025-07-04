#!/usr/bin/env python3
"""
原型提取器V4修正版 - 基于霸凌会话的原型提取

根据用户指导重新设计：
1. 从霸凌标签的会话开始
2. 以攻击性评论为中心提取6-20个节点的子图
3. 计算子图的情感分数
4. 基于情感阈值筛选霸凌子图
5. 从霸凌子图中提取原型
"""

import numpy as np
import torch
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict, Counter
from tqdm import tqdm
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

class PrototypeExtractorV4Fixed:
    """基于霸凌会话的原型提取器V4修正版"""
    
    def __init__(self, config: Dict = None):
        """
        初始化原型提取器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # 核心参数
        self.subgraph_size_range = self.config.get('subgraph_size_range', (6, 30))  # 子图大小6-30个节点
        self.emotion_threshold = self.config.get('emotion_threshold', -0.3)  # 情感阈值，通过调参确定
        self.clustering_eps = self.config.get('clustering_eps', 0.4)  
        self.min_cluster_size = self.config.get('min_cluster_size', 2)  
        
        # 多样性采样配置
        self.max_samples = self.config.get('max_samples', 1000)  # 大幅增加采样数量到1000个 (原300个)
        self.diversity_sampling = self.config.get('diversity_sampling', True)  # 启用多样性采样
        self.size_diversity_bins = self.config.get('size_diversity_bins', 5)  # 增加到5个区间采样，提高多样性
        self.quality_sampling_ratio = self.config.get('quality_sampling_ratio', 0.7)  # 70%基于质量采样，30%随机采样
        self.adaptive_sampling = self.config.get('adaptive_sampling', True)  # 启用自适应采样
        
        # 分层采样配置 (新增)
        self.stratified_sampling = self.config.get('stratified_sampling', True)  # 启用分层采样
        self.attack_level_bins = self.config.get('attack_level_bins', [
            (0.05, 0.15),  # 轻度攻击性
            (0.15, 0.30),  # 中度攻击性  
            (0.30, 0.50),  # 高度攻击性
            (0.50, 1.00),  # 极高攻击性
        ])
        self.samples_per_bin = self.config.get('samples_per_bin', 200)  # 每个攻击性等级采样200个
        
        # 情感分数权重 - 可通过调参优化
        self.emotion_weights = self.config.get('emotion_weights', {
            'comment': 0.4,    # 评论节点权重
            'user': 0.3,       # 用户节点权重  
            'word': 0.2,       # 词汇节点权重
            'others': 0.1      # 其他节点权重
        })
        
        # 攻击性评论识别阈值
        self.attack_word_ratio_threshold = self.config.get('attack_word_ratio_threshold', 0.05)
        self.attack_word_count_threshold = self.config.get('attack_word_count_threshold', 1)
        self.uppercase_ratio_threshold = self.config.get('uppercase_ratio_threshold', 0.25)
        self.exclamation_threshold = self.config.get('exclamation_threshold', 2)
        
        # 数据存储
        self.graph = None
        self.session_labels = {}  # 会话霸凌标签
        self.bullying_sessions = []  # 霸凌会话列表
        self.aggressive_comments = []  # 攻击性评论节点
        self.extracted_subgraphs = []  # 提取的子图
        self.bullying_subgraphs = []  # 筛选后的霸凌子图
        self.prototypes = []  # 最终原型
        
        # 节点索引范围
        self.node_ranges = {}
        
    def _setup_logger(self):
        """设置日志 - 避免重复初始化"""
        logger_name = 'PrototypeExtractorV4Fixed'
        logger = logging.getLogger(logger_name)
        
        # 如果logger已经有handlers，直接返回
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def load_data(self, graph_path: str, labels_path: str = None):
        """加载图数据和会话标签"""
        self.logger.info(f"🔄 加载数据...")
        
        # 加载图数据
        if not self.load_graph(graph_path):
            return False
            
        # 加载会话标签
        if not self.load_session_labels(labels_path):
            return False
            
        return True
    
    def load_graph(self, graph_path: str):
        """加载图数据"""
        self.logger.info(f"🔄 加载图数据: {graph_path}")
        
        try:
            with open(graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            
            self.logger.info(f"✅ 图数据加载成功")
            self.logger.info(f"   节点类型: {list(self.graph.node_types)}")
            
            # 计算节点索引范围
            self._calculate_node_ranges()
            
            # 统计节点数量
            total_nodes = sum(self.graph[node_type].x.size(0) for node_type in self.graph.node_types)
            total_edges = sum(self.graph[edge_type].edge_index.size(1) for edge_type in self.graph.edge_types)
            
            self.logger.info(f"   总节点数: {total_nodes}")
            self.logger.info(f"   总边数: {total_edges}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 图数据加载失败: {e}")
            return False
    
    def load_session_labels(self, labels_path: str = None):
        """从图中的media_session节点加载会话霸凌标签"""
        self.logger.info(f"🔄 从图中加载会话标签...")
        
        try:
            # 检查图中是否有media_session节点
            if 'media_session' not in self.graph.node_types:
                self.logger.error("❌ 图中没有media_session节点")
                return False
            
            # 从media_session节点特征中提取标签
            media_session_features = self.graph['media_session'].x
            session_count = media_session_features.size(0)
            
            self.logger.info(f"   找到 {session_count} 个媒体会话节点")
            
            # 第一个特征是霸凌标签
            labels = media_session_features[:, 0].numpy()
            
            # 创建会话标签映射
            self.session_labels = {}
            for idx, label in enumerate(labels):
                session_id = f"media_session_{idx}"
                self.session_labels[session_id] = int(label)
            
            # 提取霸凌会话
            self.bullying_sessions = [
                session_id for session_id, label in self.session_labels.items() 
                if label == 1
            ]
            
            normal_sessions = [
                session_id for session_id, label in self.session_labels.items() 
                if label == 0
            ]
            
            self.logger.info(f"✅ 会话标签加载成功")
            self.logger.info(f"   总会话数: {len(self.session_labels)}")
            self.logger.info(f"   霸凌会话: {len(self.bullying_sessions)} ({len(self.bullying_sessions)/len(self.session_labels)*100:.1f}%)")
            self.logger.info(f"   正常会话: {len(normal_sessions)} ({len(normal_sessions)/len(self.session_labels)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 从图中加载会话标签失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _calculate_node_ranges(self):
        """计算各节点类型的索引范围"""
        self.logger.info("🔧 计算节点索引范围...")
        
        start_idx = 0
        for node_type in self.graph.node_types:
            node_count = self.graph[node_type].x.size(0)
            self.node_ranges[node_type] = {
                'start': start_idx,
                'end': start_idx + node_count,
                'count': node_count
            }
            start_idx += node_count
            
            self.logger.info(f"   {node_type}: {self.node_ranges[node_type]['start']}-{self.node_ranges[node_type]['end']-1} ({node_count}个)")
    
    def identify_aggressive_comments_in_bullying_sessions(self):
        """在霸凌会话中识别攻击性评论节点"""
        self.logger.info("🎯 在霸凌会话中识别攻击性评论...")
        
        if 'comment' not in self.graph.node_types:
            self.logger.error("❌ 图中没有评论节点")
            return []
        
        comment_features = self.graph['comment'].x
        comment_count = comment_features.size(0)
        
        # 评论特征分析（基于实际检查）
        text_lengths = comment_features[:, 0].numpy()
        word_counts = comment_features[:, 1].numpy()
        ratio_feature = comment_features[:, 2].numpy()  # 可能是大写比例或攻击词比例
        binary_feature = comment_features[:, 3].numpy()  # 可能是攻击性或负面情感标记
        
        self.logger.info(f"   评论特征分析:")
        self.logger.info(f"     平均文本长度: {text_lengths.mean():.1f}")
        self.logger.info(f"     平均词数: {word_counts.mean():.1f}")
        self.logger.info(f"     比例特征均值: {ratio_feature.mean():.3f}")
        self.logger.info(f"     二元特征分布: {np.bincount(binary_feature.astype(int))}")
        
        # 多维度攻击性检测（基于实际特征）
        aggressive_mask = (
            (binary_feature == 1) |  # 二元特征为1（可能表示攻击性）
            (ratio_feature > 0.01) |  # 比例特征>1%（可能是攻击词比例）
            ((text_lengths > 50) & (word_counts > 8)) |  # 长文本且词数多（可能包含攻击内容）
            (text_lengths > 80)  # 非常长的评论（可能包含攻击性内容）
        )
        
        aggressive_indices = np.where(aggressive_mask)[0]
        
        # TODO: 这里应该添加逻辑来筛选只属于霸凌会话的攻击性评论
        # 由于当前缺少评论到会话的映射关系，暂时使用所有攻击性评论
        
        # 转换为全局索引
        comment_start = self.node_ranges['comment']['start']
        aggressive_global_indices = [comment_start + idx for idx in aggressive_indices]
        
        self.aggressive_comments = aggressive_global_indices
        
        self.logger.info(f"✅ 识别到 {len(self.aggressive_comments)} 个攻击性评论")
        self.logger.info(f"   攻击性比例: {len(self.aggressive_comments)}/{comment_count} = {len(self.aggressive_comments)/comment_count*100:.1f}%")
        
        return self.aggressive_comments
    
    def extract_subgraphs_from_aggressive_comments(self):
        """从攻击性评论中心提取6-30个节点的子图"""
        self.logger.info("🕸️ 从攻击性评论中心提取子图...")
        
        if not self.aggressive_comments:
            self.logger.error("❌ 没有找到攻击性评论")
            return []
        
        subgraphs = []
        
        # 构建全局邻接表
        adj_list = self._build_adjacency_list()
        
        # 实施多样性采样策略
        if self.diversity_sampling:
            sampled_comments = self._diverse_sampling(self.aggressive_comments)
        else:
            # 原有的随机采样
            max_samples = min(self.max_samples, len(self.aggressive_comments))
            sampled_comments = np.random.choice(
                self.aggressive_comments, 
                max_samples, 
                replace=False
            )
        
        self.logger.info(f"   从{len(self.aggressive_comments)}个攻击性评论中采样{len(sampled_comments)}个")
        
        for comment_idx in tqdm(sampled_comments, desc="提取子图"):
            try:
                # 动态选择子图大小，增加结构多样性
                target_size = self._select_diverse_subgraph_size()
                
                # 从攻击性评论开始扩展，收集指定数量的节点
                subgraph_nodes = self._collect_diverse_subgraph_nodes(comment_idx, adj_list, target_size)
                
                if len(subgraph_nodes) < self.subgraph_size_range[0]:
                    continue
                
                # 提取子图结构
                subgraph = self._extract_subgraph_structure(subgraph_nodes, comment_idx)
                
                if subgraph:
                    subgraphs.append(subgraph)
                    
            except Exception as e:
                self.logger.warning(f"提取子图失败 (comment {comment_idx}): {e}")
                continue
        
        self.extracted_subgraphs = subgraphs
        
        self.logger.info(f"✅ 成功提取 {len(subgraphs)} 个子图")
        if subgraphs:
            avg_size = np.mean([sg['total_nodes'] for sg in subgraphs])
            size_range = (min([sg['total_nodes'] for sg in subgraphs]), 
                         max([sg['total_nodes'] for sg in subgraphs]))
            self.logger.info(f"   平均子图大小: {avg_size:.1f} 个节点")
            self.logger.info(f"   子图大小范围: {size_range[0]}-{size_range[1]} 个节点")
        
        return subgraphs
    
    def _diverse_sampling(self, aggressive_comments: List[int]) -> List[int]:
        """多样性采样策略 - 大幅改进版"""
        self.logger.info("🎯 执行多样性采样策略...")
        
        # 计算所有评论的攻击性分数
        attack_scores = []
        valid_comments = []
        
        for comment_idx in aggressive_comments:
            try:
                attack_score = self._calculate_comment_aggressiveness(comment_idx)
                attack_scores.append(attack_score)
                valid_comments.append(comment_idx)
            except Exception as e:
                continue
        
        if not valid_comments:
            self.logger.warning("⚠️ 没有有效的攻击性评论")
            return []
        
        self.logger.info(f"   有效攻击性评论: {len(valid_comments)}个")
        self.logger.info(f"   攻击性分数范围: {np.min(attack_scores):.3f} - {np.max(attack_scores):.3f}")
        
        selected_samples = []
        
        # 策略1: 分层采样 (如果启用)
        if self.stratified_sampling:
            stratified_samples = self._stratified_attack_sampling(valid_comments, attack_scores)
            selected_samples.extend(stratified_samples)
            self.logger.info(f"   分层采样获得: {len(stratified_samples)}个样本")
        
        # 策略2: 质量优先采样 (如果启用自适应采样)
        if self.adaptive_sampling and len(selected_samples) < self.max_samples:
            remaining_quota = self.max_samples - len(selected_samples)
            quality_samples = self._quality_priority_sampling(
                valid_comments, attack_scores, remaining_quota, exclude=selected_samples
            )
            selected_samples.extend(quality_samples)
            self.logger.info(f"   质量优先采样获得: {len(quality_samples)}个样本")
        
        # 策略3: 随机补充采样
        if len(selected_samples) < self.max_samples:
            remaining_quota = self.max_samples - len(selected_samples)
            random_samples = self._random_supplement_sampling(
                valid_comments, remaining_quota, exclude=selected_samples
            )
            selected_samples.extend(random_samples)
            self.logger.info(f"   随机补充采样获得: {len(random_samples)}个样本")
        
        # 去重并限制数量
        final_samples = list(set(selected_samples))[:self.max_samples]
        
        self.logger.info(f"✅ 多样性采样完成: {len(final_samples)}个样本")
        self.logger.info(f"   采样比例: {len(final_samples)/len(valid_comments)*100:.1f}%")
        
        return final_samples
    
    def _stratified_attack_sampling(self, comments: List[int], attack_scores: List[float]) -> List[int]:
        """分层攻击性采样"""
        self.logger.info("   🎯 执行分层攻击性采样...")
        
        # 按攻击性等级分组
        stratified_groups = {i: [] for i in range(len(self.attack_level_bins))}
        
        for comment_idx, attack_score in zip(comments, attack_scores):
            for bin_idx, (min_attack, max_attack) in enumerate(self.attack_level_bins):
                if min_attack <= attack_score < max_attack:
                    stratified_groups[bin_idx].append(comment_idx)
                    break
        
        # 从每个组中采样
        selected_samples = []
        for bin_idx, (min_attack, max_attack) in enumerate(self.attack_level_bins):
            group_comments = stratified_groups[bin_idx]
            if not group_comments:
                continue
            
            # 每组采样指定数量
            sample_count = min(self.samples_per_bin, len(group_comments))
            bin_samples = np.random.choice(group_comments, sample_count, replace=False).tolist()
            selected_samples.extend(bin_samples)
            
            self.logger.info(f"     等级{bin_idx+1}({min_attack:.2f}-{max_attack:.2f}): {len(group_comments)}个候选, 采样{sample_count}个")
        
        return selected_samples
    
    def _quality_priority_sampling(self, comments: List[int], attack_scores: List[float], 
                                 quota: int, exclude: List[int] = None) -> List[int]:
        """质量优先采样"""
        exclude = exclude or []
        available_comments = [c for c in comments if c not in exclude]
        available_scores = [attack_scores[comments.index(c)] for c in available_comments]
        
        if not available_comments:
            return []
        
        # 按攻击性分数排序
        sorted_indices = np.argsort(available_scores)[::-1]  # 降序排列
        
        # 选择质量最高的样本
        quality_count = int(quota * self.quality_sampling_ratio)
        quality_samples = [available_comments[i] for i in sorted_indices[:quality_count]]
        
        # 随机选择剩余样本以增加多样性
        remaining_indices = sorted_indices[quality_count:]
        if remaining_indices.size > 0:
            random_count = quota - quality_count
            random_count = min(random_count, len(remaining_indices))
            random_indices = np.random.choice(remaining_indices, random_count, replace=False)
            random_samples = [available_comments[i] for i in random_indices]
            quality_samples.extend(random_samples)
        
        return quality_samples
    
    def _random_supplement_sampling(self, comments: List[int], quota: int, exclude: List[int] = None) -> List[int]:
        """随机补充采样"""
        exclude = exclude or []
        available_comments = [c for c in comments if c not in exclude]
        
        if not available_comments:
            return []
        
        sample_count = min(quota, len(available_comments))
        return np.random.choice(available_comments, sample_count, replace=False).tolist()
    
    def _calculate_comment_aggressiveness(self, comment_idx: int) -> float:
        """计算评论的攻击性分数"""
        if 'comment' not in self.graph.node_types:
            return 0.5
            
        comment_features = self.graph['comment'].x
        comment_start = self.node_ranges['comment']['start']
        relative_idx = comment_idx - comment_start
        
        if 0 <= relative_idx < comment_features.size(0):
            features = comment_features[relative_idx]
            
            # 基于攻击性特征计算分数
            ratio_feature = features[2].item()  # 攻击词比例
            binary_feature = features[3].item()  # 攻击性标记
            uppercase_ratio = features[4].item() if features.size(0) > 4 else 0  # 大写比例
            
            # 综合攻击性分数 - 移除强制归一化以保持区分度
            score = (ratio_feature * 2.0 + binary_feature * 1.0 + uppercase_ratio * 0.5)
            return score  # 不再强制限制为1.0
        
        return 0.5
    
    def _select_diverse_subgraph_size(self) -> int:
        """动态选择子图大小，增加结构多样性"""
        min_size, max_size = self.subgraph_size_range
        
        # 定义三个大小区间：小型(6-12)、中型(13-21)、大型(22-30)
        small_range = (min_size, min_size + 6)
        medium_range = (min_size + 7, min_size + 15) 
        large_range = (min_size + 16, max_size)
        
        # 随机选择区间（各1/3概率）
        range_choice = np.random.choice([0, 1, 2])
        
        if range_choice == 0:  # 小型子图
            return np.random.randint(small_range[0], small_range[1] + 1)
        elif range_choice == 1:  # 中型子图
            return np.random.randint(medium_range[0], medium_range[1] + 1)
        else:  # 大型子图
            return np.random.randint(large_range[0], large_range[1] + 1)
    
    def _collect_diverse_subgraph_nodes(self, center_comment: int, adj_list: Dict, target_size: int) -> Set[int]:
        """收集多样化的子图节点，支持不同的扩展策略"""
        
        subgraph_nodes = {center_comment}
        candidates = set()
        
        # 添加中心评论的直接邻居
        if center_comment in adj_list:
            for neighbor in adj_list[center_comment]:
                candidates.add(neighbor)
        
        # 根据目标大小选择不同的扩展策略
        if target_size <= 12:
            # 小型子图：紧密连接，选择高优先级节点
            strategy = "compact"
        elif target_size <= 21:
            # 中型子图：平衡扩展
            strategy = "balanced"
        else:
            # 大型子图：广泛扩展，包含更多节点类型
            strategy = "extensive"
        
        # 按策略扩展子图
        while len(subgraph_nodes) < target_size and candidates:
            if strategy == "compact":
                next_node = self._select_priority_node(candidates, subgraph_nodes)
            elif strategy == "balanced":
                next_node = self._select_balanced_node(candidates, subgraph_nodes)
            else:  # extensive
                next_node = self._select_diverse_node(candidates, subgraph_nodes)
                
            if next_node is None:
                break
                
            subgraph_nodes.add(next_node)
            candidates.remove(next_node)
            
            # 添加新节点的邻居作为候选
            if next_node in adj_list:
                for neighbor in adj_list[next_node]:
                    if neighbor not in subgraph_nodes:
                        candidates.add(neighbor)
        
        return subgraph_nodes
    
    def _select_balanced_node(self, candidates: Set[int], existing_nodes: Set[int]) -> int:
        """平衡选择策略：在优先级和多样性之间平衡"""
        # 50%概率按优先级选择，50%概率随机选择
        if np.random.random() < 0.5:
            return self._select_priority_node(candidates, existing_nodes)
        else:
            return next(iter(candidates)) if candidates else None
    
    def _select_diverse_node(self, candidates: Set[int], existing_nodes: Set[int]) -> int:
        """多样性选择策略：优先选择图中缺少的节点类型"""
        existing_types = {self._get_node_type(node) for node in existing_nodes}
        
        # 优先选择当前子图中缺少的节点类型
        for node_type in ['word', 'time', 'location', 'user', 'comment']:
            if node_type not in existing_types:
                type_candidates = [
                    node for node in candidates 
                    if self._get_node_type(node) == node_type
                ]
                if type_candidates:
                    return type_candidates[0]
        
        # 如果所有类型都有，随机选择
        return next(iter(candidates)) if candidates else None
    
    def _select_priority_node(self, candidates: Set[int], existing_nodes: Set[int]) -> int:
        """按优先级选择下一个节点：评论 > 用户 > 词汇 > 其他"""
        priority_order = ['comment', 'user', 'word', 'time', 'location']
        
        for node_type in priority_order:
            type_candidates = [
                node for node in candidates 
                if self._get_node_type(node) == node_type
            ]
            if type_candidates:
                return type_candidates[0]
        
        # 如果没有优先类型，返回任意候选
        return next(iter(candidates)) if candidates else None

    def calculate_emotion_scores(self):
        """计算每个子图的情感分数"""
        self.logger.info("😊 计算子图情感分数...")
        
        if not self.extracted_subgraphs:
            self.logger.error("❌ 没有提取的子图")
            return
        
        for i, subgraph in enumerate(tqdm(self.extracted_subgraphs, desc="计算情感分数")):
            try:
                emotion_score = self._calculate_subgraph_emotion_score(subgraph)
                subgraph['emotion_score'] = emotion_score
                
            except Exception as e:
                self.logger.warning(f"计算子图{i}情感分数失败: {e}")
                subgraph['emotion_score'] = 0.0
        
        # 统计情感分数分布
        emotion_scores = [sg.get('emotion_score', 0.0) for sg in self.extracted_subgraphs]
        if emotion_scores:
            self.logger.info(f"✅ 情感分数计算完成")
            self.logger.info(f"   平均情感分数: {np.mean(emotion_scores):.3f}")
            self.logger.info(f"   情感分数范围: {np.min(emotion_scores):.3f} - {np.max(emotion_scores):.3f}")
            self.logger.info(f"   负面情感子图: {sum(1 for s in emotion_scores if s < 0)}/{len(emotion_scores)}")

    def _calculate_subgraph_emotion_score(self, subgraph: Dict) -> float:
        """计算单个子图的情感分数"""
        nodes_by_type = subgraph.get('nodes_by_type', {})
        total_weighted_score = 0.0
        total_weight = 0.0
        
        # 为不同类型的节点计算情感分数
        for node_type, node_indices in nodes_by_type.items():
            if not node_indices:
                continue
                
            node_weight = self.emotion_weights.get(node_type, self.emotion_weights['others'])
            type_emotion_score = self._calculate_node_type_emotion_score(node_type, node_indices)
            
            total_weighted_score += type_emotion_score * node_weight * len(node_indices)
            total_weight += node_weight * len(node_indices)
        
        # 加权平均
        if total_weight > 0:
            return total_weighted_score / total_weight
        else:
            return 0.0

    def _calculate_node_type_emotion_score(self, node_type: str, node_indices: List[int]) -> float:
        """计算特定类型节点的情感分数"""
        if node_type == 'comment':
            return self._calculate_comment_emotion_scores(node_indices)
        elif node_type == 'user':
            return self._calculate_user_emotion_scores(node_indices)
        elif node_type == 'word':
            return self._calculate_word_emotion_scores(node_indices)
        else:
            # 时间、位置等节点不给情感分数
            return 0.0

    def _calculate_comment_emotion_scores(self, comment_indices: List[int]) -> float:
        """基于评论文本特征计算情感分数"""
        if 'comment' not in self.graph.node_types:
            return 0.0
            
        comment_features = self.graph['comment'].x
        comment_start = self.node_ranges['comment']['start']
        
        scores = []
        for global_idx in comment_indices:
            relative_idx = global_idx - comment_start
            if 0 <= relative_idx < comment_features.size(0):
                features = comment_features[relative_idx]
                
                # 基于攻击性特征计算负面情感分数
                ratio_feature = features[2].item()  # 攻击词比例
                binary_feature = features[3].item()  # 攻击性标记
                
                # 情感分数：攻击性越高，分数越负面
                emotion_score = -(ratio_feature * 2.0 + binary_feature * 1.0)
                scores.append(emotion_score)
        
        return np.mean(scores) if scores else 0.0

    def _calculate_user_emotion_scores(self, user_indices: List[int]) -> float:
        """基于用户发表的攻击性言论数量计算情感分数"""
        if 'user' not in self.graph.node_types:
            return 0.0
        
        # 计算每个用户的攻击性评论数量
        user_scores = []
        
        for user_global_idx in user_indices:
            # 计算该用户发表的攻击性评论数量
            attack_comment_count = self._count_user_aggressive_comments(user_global_idx)
            
            # 根据攻击性评论数量计算风险分数
            if attack_comment_count == 0:
                risk_score = 0.0  # 低风险
            elif attack_comment_count <= 2:
                risk_score = -0.3 * attack_comment_count  # 中风险
            elif attack_comment_count <= 5:
                risk_score = -0.6 - 0.2 * (attack_comment_count - 2)  # 高风险
            else:
                risk_score = -1.5 - 0.1 * min(attack_comment_count - 5, 10)  # 极高风险
                
            user_scores.append(risk_score)
        
        return np.mean(user_scores) if user_scores else 0.0

    def _count_user_aggressive_comments(self, user_global_idx: int) -> int:
        """计算用户发表的攻击性评论数量"""
        # 通过user_posts_comment边找到用户发表的评论
        if ('user', 'posts', 'comment') not in self.graph.edge_types:
            return 0
        
        edge_index = self.graph[('user', 'posts', 'comment')].edge_index
        user_start = self.node_ranges['user']['start']
        user_relative_idx = user_global_idx - user_start
        
        # 找到该用户发表的所有评论
        user_comment_mask = (edge_index[0] == user_relative_idx)
        user_comment_relative_indices = edge_index[1][user_comment_mask].numpy()
        
        # 转换为全局索引
        comment_start = self.node_ranges['comment']['start']
        user_comment_global_indices = [comment_start + idx for idx in user_comment_relative_indices]
        
        # 计算其中有多少是攻击性评论
        attack_count = sum(1 for idx in user_comment_global_indices if idx in self.aggressive_comments)
        
        return attack_count

    def _calculate_word_emotion_scores(self, word_indices: List[int]) -> float:
        """基于词汇的攻击性计算情感分数"""
        # 简化实现：假设词汇中有攻击性词汇则给负分
        # 实际实现中可以建立攻击性词汇表进行匹配
        return -0.2  # 固定的轻微负面分数

    def filter_bullying_subgraphs(self):
        """基于情感阈值筛选霸凌子图"""
        self.logger.info("🔍 基于情感阈值筛选霸凌子图...")
        
        if not self.extracted_subgraphs:
            self.logger.error("❌ 没有提取的子图")
            return []
        
        # 筛选情感分数低于阈值的子图
        self.bullying_subgraphs = [
            subgraph for subgraph in self.extracted_subgraphs
            if subgraph.get('emotion_score', 0.0) < self.emotion_threshold
        ]
        
        self.logger.info(f"✅ 霸凌子图筛选完成")
        self.logger.info(f"   原始子图数: {len(self.extracted_subgraphs)}")
        self.logger.info(f"   霸凌子图数: {len(self.bullying_subgraphs)}")
        self.logger.info(f"   筛选比例: {len(self.bullying_subgraphs)/len(self.extracted_subgraphs)*100:.1f}%")
        
        if self.bullying_subgraphs:
            emotion_scores = [sg['emotion_score'] for sg in self.bullying_subgraphs]
            self.logger.info(f"   霸凌子图情感分数范围: {np.min(emotion_scores):.3f} - {np.max(emotion_scores):.3f}")
        
        return self.bullying_subgraphs

    def _build_adjacency_list(self):
        """构建全局邻接表"""
        self.logger.info("   📋 构建邻接表...")
        
        adj_list = defaultdict(set)
        
        for edge_type in self.graph.edge_types:
            edge_index = self.graph[edge_type].edge_index
            if edge_index.size(1) == 0:
                continue
                
            src_indices = edge_index[0].numpy()
            dst_indices = edge_index[1].numpy()
            
            for src, dst in zip(src_indices, dst_indices):
                adj_list[src].add(dst)
                adj_list[dst].add(src)  # 无向图
        
        self.logger.info(f"   ✅ 邻接表构建完成 - {len(adj_list)} 个节点有邻居")
        return adj_list
    
    def _extract_subgraph_structure(self, subgraph_nodes: Set[int], center_comment: int) -> Dict:
        """提取子图的结构信息"""
        
        # 按节点类型分类
        nodes_by_type = defaultdict(list)
        for node_idx in subgraph_nodes:
            node_type = self._get_node_type(node_idx)
            nodes_by_type[node_type].append(node_idx)
        
        # 提取子图的边
        subgraph_edges = self._extract_subgraph_edges(subgraph_nodes)
        
        # 计算结构特征
        structural_features = self._calculate_subgraph_features(nodes_by_type, subgraph_edges)
        
        # 计算情感特征
        emotion_features = self._calculate_emotion_features(nodes_by_type)
        
        # 计算攻击性特征
        aggression_features = self._calculate_aggression_features(nodes_by_type)
        
        # 计算霸凌强度
        bullying_intensity = self._calculate_bullying_intensity(
            nodes_by_type, center_comment, emotion_features, aggression_features
        )
        
        subgraph = {
            'center_comment': center_comment,
            'nodes_by_type': dict(nodes_by_type),
            'edges': subgraph_edges,
            'structural_features': structural_features,
            'emotion_features': emotion_features,
            'aggression_features': aggression_features,
            'bullying_intensity': bullying_intensity,
            'total_nodes': len(subgraph_nodes),
            'creation_time': datetime.now().isoformat()
        }
        
        return subgraph
    
    def _get_node_type(self, node_idx: int) -> str:
        """根据节点索引确定节点类型"""
        for node_type, range_info in self.node_ranges.items():
            if range_info['start'] <= node_idx < range_info['end']:
                return node_type
        return 'unknown'
    
    def _extract_subgraph_edges(self, subgraph_nodes: Set[int]) -> Dict:
        """提取子图内的边"""
        subgraph_edges = {}
        
        for edge_type in self.graph.edge_types:
            edge_index = self.graph[edge_type].edge_index
            if edge_index.size(1) == 0:
                continue
            
            src_indices = edge_index[0].numpy()
            dst_indices = edge_index[1].numpy()
            
            # 找到子图内的边
            internal_edges = []
            for i, (src, dst) in enumerate(zip(src_indices, dst_indices)):
                if src in subgraph_nodes and dst in subgraph_nodes:
                    internal_edges.append((src, dst))
            
            if internal_edges:
                subgraph_edges[edge_type] = internal_edges
        
        return subgraph_edges
    
    def _calculate_subgraph_features(self, nodes_by_type: Dict, edges: Dict) -> Dict:
        """计算子图结构特征"""
        features = {}
        
        # 节点统计
        features['node_counts'] = {k: len(v) for k, v in nodes_by_type.items()}
        features['total_nodes'] = sum(features['node_counts'].values())
        
        # 边统计
        features['edge_counts'] = {k: len(v) for k, v in edges.items()}
        features['total_edges'] = sum(features['edge_counts'].values())
        
        # 密度
        total_nodes = features['total_nodes']
        if total_nodes > 1:
            max_edges = total_nodes * (total_nodes - 1) / 2
            features['density'] = features['total_edges'] / max_edges
        else:
            features['density'] = 0.0
        
        # 多样性
        features['node_type_diversity'] = len(features['node_counts'])
        features['edge_type_diversity'] = len(features['edge_counts'])
        
        return features
    
    def _calculate_emotion_features(self, nodes_by_type: Dict) -> Dict:
        """计算情感特征"""
        emotion_features = {}
        
        # 基于评论节点计算情感
        comment_nodes = nodes_by_type.get('comment', [])
        if comment_nodes and 'comment' in self.graph.node_types:
            comment_features = self.graph['comment'].x
            comment_start = self.node_ranges['comment']['start']
            
            # 转换为相对索引
            relative_indices = [idx - comment_start for idx in comment_nodes 
                              if comment_start <= idx < self.node_ranges['comment']['end']]
            
            if relative_indices:
                selected_features = comment_features[relative_indices]
                
                # 基于实际特征计算情感
                text_lengths = selected_features[:, 0]
                word_counts = selected_features[:, 1]
                ratio_feature = selected_features[:, 2]
                binary_feature = selected_features[:, 3]
                
                emotion_features['avg_text_length'] = float(text_lengths.mean())
                emotion_features['avg_word_count'] = float(word_counts.mean())
                emotion_features['avg_ratio_feature'] = float(ratio_feature.mean())
                emotion_features['negative_ratio'] = float((binary_feature == 1).sum() / len(binary_feature))
                emotion_features['negative_intensity'] = float(
                    (text_lengths / 50.0 + ratio_feature * 5.0 + binary_feature).mean()
                )
                emotion_features['comment_count'] = len(relative_indices)
        
        return emotion_features
    
    def _calculate_aggression_features(self, nodes_by_type: Dict) -> Dict:
        """计算攻击性特征"""
        aggression_features = {}
        
        # 基于评论节点计算攻击性
        comment_nodes = nodes_by_type.get('comment', [])
        word_nodes = nodes_by_type.get('word', [])
        
        if comment_nodes and 'comment' in self.graph.node_types:
            comment_features = self.graph['comment'].x
            comment_start = self.node_ranges['comment']['start']
            
            relative_indices = [idx - comment_start for idx in comment_nodes 
                              if comment_start <= idx < self.node_ranges['comment']['end']]
            
            if relative_indices:
                selected_features = comment_features[relative_indices]
                
                # 基于实际特征计算攻击性
                text_lengths = selected_features[:, 0]
                word_counts = selected_features[:, 1]
                ratio_feature = selected_features[:, 2]  # 可能是攻击词比例
                binary_feature = selected_features[:, 3]  # 可能是攻击性标记
                
                aggression_features['avg_text_length'] = float(text_lengths.mean())
                aggression_features['avg_word_count'] = float(word_counts.mean())
                aggression_features['avg_ratio_feature'] = float(ratio_feature.mean())
                aggression_features['max_ratio_feature'] = float(ratio_feature.max())
                aggression_features['aggressive_comment_ratio'] = float(
                    (binary_feature == 1).sum() / len(binary_feature)
                )
                aggression_features['aggression_score'] = float(
                    (text_lengths / 20.0 + ratio_feature * 10.0 + binary_feature * 2.0).mean()
                )
        
        # 基于词汇节点计算攻击词密度
        if word_nodes and 'word' in self.graph.node_types:
            word_features = self.graph['word'].x
            word_start = self.node_ranges['word']['start']
            
            relative_indices = [idx - word_start for idx in word_nodes 
                              if word_start <= idx < self.node_ranges['word']['end']]
            
            if relative_indices:
                selected_features = word_features[relative_indices]
                # 词汇特征：[频率, 是否攻击词, 长度, 常量]
                if selected_features.size(1) >= 2:
                    attack_word_ratio = float(selected_features[:, 1].mean())
                    aggression_features['attack_word_density'] = attack_word_ratio
                    aggression_features['vocab_size'] = len(relative_indices)
        
        return aggression_features
    
    def _calculate_bullying_intensity(self, nodes_by_type: Dict, center_comment: int, 
                                    emotion_features: Dict, aggression_features: Dict) -> float:
        """计算霸凌强度分数"""
        
        intensity_factors = []
        
        # 1. 攻击性强度 (权重 40%)
        aggression_score = aggression_features.get('aggression_score', 0)
        if aggression_score > 0:
            intensity_factors.append(min(aggression_score / 5.0, 1.0) * 0.4)
        
        # 2. 情感负面度 (权重 30%)
        negative_intensity = emotion_features.get('negative_intensity', 0)
        if negative_intensity > 0:
            intensity_factors.append(min(negative_intensity / 3.0, 1.0) * 0.3)
        
        # 3. 参与用户数 (权重 20%)
        user_count = nodes_by_type.get('user', [])
        if len(user_count) > 1:
            user_factor = min(len(user_count) / 10.0, 1.0) * 0.2
            intensity_factors.append(user_factor)
        
        # 4. 结构复杂性 (权重 10%)
        total_nodes = sum(len(nodes) for nodes in nodes_by_type.values())
        if total_nodes > 10:
            complexity_factor = min((total_nodes - 10) / 50.0, 1.0) * 0.1
            intensity_factors.append(complexity_factor)
        
        return sum(intensity_factors) if intensity_factors else 0.0
    
    def cluster_bullying_subgraphs(self):
        """对霸凌子图进行聚类以生成原型 - 多样化策略"""
        self.logger.info("🤖 对霸凌子图进行聚类...")
        
        if not self.bullying_subgraphs:
            self.logger.error("❌ 没有霸凌子图用于聚类")
            return []
        
        # 提取特征向量用于聚类
        feature_vectors = []
        valid_subgraphs = []
        
        for subgraph in self.bullying_subgraphs:
            try:
                feature_vector = self._extract_feature_vector(subgraph)
                if feature_vector is not None and len(feature_vector) > 0:
                    feature_vectors.append(feature_vector)
                    valid_subgraphs.append(subgraph)
            except Exception as e:
                self.logger.warning(f"提取特征向量失败: {e}")
                continue
        
        if len(feature_vectors) < self.min_cluster_size:
            self.logger.warning(f"有效子图数量不足进行聚类 ({len(feature_vectors)} < {self.min_cluster_size})")
            # 如果子图数量太少，将所有子图作为一个原型
            if valid_subgraphs:
                prototype = self._generate_prototype(0, valid_subgraphs)
                self.prototypes = [prototype]
                self.logger.info(f"✅ 生成单一原型包含 {len(valid_subgraphs)} 个子图")
                return self.prototypes
            else:
                return []
        
        # 标准化特征矩阵
        feature_matrix = np.array(feature_vectors)
        self.logger.info(f"   特征矩阵形状: {feature_matrix.shape}")
        
        # 使用多种聚类策略生成多样化原型
        all_prototypes = []
        
        # 策略1: 基于结构相似性的DBSCAN聚类
        structural_prototypes = self._dbscan_clustering(feature_matrix, valid_subgraphs, "结构相似性")
        all_prototypes.extend(structural_prototypes)
        
        # 策略2: 基于子图大小的分层聚类
        size_prototypes = self._size_based_clustering(valid_subgraphs)
        all_prototypes.extend(size_prototypes)
        
        # 策略3: 基于攻击性强度的分层聚类
        intensity_prototypes = self._intensity_based_clustering(valid_subgraphs)
        all_prototypes.extend(intensity_prototypes)
        
        # 去重和合并相似原型
        final_prototypes = self._merge_similar_prototypes(all_prototypes)
        
        self.prototypes = final_prototypes
        self.logger.info(f"✅ 成功生成 {len(final_prototypes)} 个多样化原型")
        
        # 显示原型多样性统计
        self._log_prototype_diversity(final_prototypes)
        
        return final_prototypes
    
    def _dbscan_clustering(self, feature_matrix: np.ndarray, valid_subgraphs: List[Dict], strategy_name: str) -> List[Dict]:
        """使用DBSCAN进行聚类"""
        clustering = DBSCAN(eps=self.clustering_eps, min_samples=self.min_cluster_size)
        cluster_labels = clustering.fit_predict(feature_matrix)
        
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        self.logger.info(f"   {strategy_name}聚类结果: {n_clusters} 个聚类, {n_noise} 个噪声点")
        
        prototypes = []
        for cluster_id in unique_labels:
            if cluster_id == -1:  # 跳过噪声点
                continue
            
            cluster_mask = (cluster_labels == cluster_id)
            cluster_subgraphs = [valid_subgraphs[i] for i, mask in enumerate(cluster_mask) if mask]
            
            if len(cluster_subgraphs) >= self.min_cluster_size:
                prototype = self._generate_prototype(f"struct_{cluster_id}", cluster_subgraphs)
                prototype['clustering_strategy'] = strategy_name
                prototypes.append(prototype)
                self.logger.info(f"   {strategy_name}聚类 {cluster_id}: {len(cluster_subgraphs)} 个子图")
        
        return prototypes
    
    def _size_based_clustering(self, valid_subgraphs: List[Dict]) -> List[Dict]:
        """基于子图大小进行分层聚类"""
        self.logger.info("   基于大小的分层聚类...")
        
        # 按子图大小分组
        size_groups = {'small': [], 'medium': [], 'large': []}
        
        for subgraph in valid_subgraphs:
            size = subgraph['total_nodes']
            if size <= 12:
                size_groups['small'].append(subgraph)
            elif size <= 21:
                size_groups['medium'].append(subgraph)
            else:
                size_groups['large'].append(subgraph)
        
        prototypes = []
        for size_type, subgraphs in size_groups.items():
            if len(subgraphs) >= self.min_cluster_size:
                prototype = self._generate_prototype(f"size_{size_type}", subgraphs)
                prototype['clustering_strategy'] = f"基于大小({size_type})"
                prototypes.append(prototype)
                self.logger.info(f"   大小聚类 {size_type}: {len(subgraphs)} 个子图")
        
        return prototypes
    
    def _intensity_based_clustering(self, valid_subgraphs: List[Dict]) -> List[Dict]:
        """基于攻击性强度进行分层聚类"""
        self.logger.info("   基于攻击性强度的分层聚类...")
        
        # 计算攻击性强度分布
        intensities = [sg.get('bullying_intensity', 0) for sg in valid_subgraphs]
        if not intensities:
            return []
        
        # 按攻击性强度分组
        intensity_threshold_low = np.percentile(intensities, 33)
        intensity_threshold_high = np.percentile(intensities, 67)
        
        intensity_groups = {'low': [], 'medium': [], 'high': []}
        
        for subgraph in valid_subgraphs:
            intensity = subgraph.get('bullying_intensity', 0)
            if intensity <= intensity_threshold_low:
                intensity_groups['low'].append(subgraph)
            elif intensity <= intensity_threshold_high:
                intensity_groups['medium'].append(subgraph)
            else:
                intensity_groups['high'].append(subgraph)
        
        prototypes = []
        for intensity_type, subgraphs in intensity_groups.items():
            if len(subgraphs) >= self.min_cluster_size:
                prototype = self._generate_prototype(f"intensity_{intensity_type}", subgraphs)
                prototype['clustering_strategy'] = f"基于强度({intensity_type})"
                prototypes.append(prototype)
                self.logger.info(f"   强度聚类 {intensity_type}: {len(subgraphs)} 个子图")
        
        return prototypes
    
    def _merge_similar_prototypes(self, all_prototypes: List[Dict]) -> List[Dict]:
        """合并相似的原型，避免重复"""
        if len(all_prototypes) <= 1:
            return all_prototypes
        
        # 计算原型间的相似度矩阵
        similarity_threshold = 0.8  # 相似度阈值
        merged_prototypes = []
        used_indices = set()
        
        for i, proto1 in enumerate(all_prototypes):
            if i in used_indices:
                continue
                
            similar_prototypes = [proto1]
            used_indices.add(i)
            
            for j, proto2 in enumerate(all_prototypes[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                # 计算两个原型的相似度
                similarity = self._calculate_prototype_similarity(proto1, proto2)
                if similarity > similarity_threshold:
                    similar_prototypes.append(proto2)
                    used_indices.add(j)
            
            # 如果有相似原型，合并它们
            if len(similar_prototypes) > 1:
                merged_prototype = self._merge_prototypes(similar_prototypes)
                merged_prototypes.append(merged_prototype)
            else:
                merged_prototypes.append(proto1)
        
        self.logger.info(f"   原型合并: {len(all_prototypes)} -> {len(merged_prototypes)}")
        return merged_prototypes
    
    def _calculate_prototype_similarity(self, proto1: Dict, proto2: Dict) -> float:
        """计算两个原型之间的相似度"""
        try:
            features1 = np.array(proto1['average_features'])
            features2 = np.array(proto2['average_features'])
            
            # 使用余弦相似度
            similarity = cosine_similarity([features1], [features2])[0][0]
            return similarity
        except:
            return 0.0
    
    def _merge_prototypes(self, similar_prototypes: List[Dict]) -> Dict:
        """合并相似的原型"""
        # 使用第一个原型作为基础
        merged_prototype = similar_prototypes[0].copy()
        
        # 更新ID和策略
        strategies = [p.get('clustering_strategy', 'unknown') for p in similar_prototypes]
        merged_prototype['prototype_id'] = f"merged_{len(similar_prototypes)}"
        merged_prototype['clustering_strategy'] = f"合并({', '.join(set(strategies))})"
        
        # 合并聚类大小
        total_size = sum(p.get('cluster_size', 0) for p in similar_prototypes)
        merged_prototype['cluster_size'] = total_size
        
        # 合并成员评论
        all_comments = []
        for p in similar_prototypes:
            all_comments.extend(p.get('member_comments', []))
        merged_prototype['member_comments'] = list(set(all_comments))
        
        return merged_prototype
    
    def _log_prototype_diversity(self, prototypes: List[Dict]):
        """记录原型多样性统计"""
        if not prototypes:
            return
            
        # 统计不同策略的原型数量
        strategies = {}
        sizes = []
        intensities = []
        
        for proto in prototypes:
            strategy = proto.get('clustering_strategy', 'unknown')
            strategies[strategy] = strategies.get(strategy, 0) + 1
            sizes.append(proto.get('cluster_size', 0))
            intensities.append(proto.get('avg_bullying_intensity', 0))
        
        self.logger.info("   🌈 原型多样性统计:")
        for strategy, count in strategies.items():
            self.logger.info(f"     {strategy}: {count} 个原型")
        
        if sizes:
            self.logger.info(f"     平均聚类大小: {np.mean(sizes):.1f}")
            self.logger.info(f"     聚类大小范围: {min(sizes)}-{max(sizes)}")
        
        if intensities:
            self.logger.info(f"     平均霸凌强度: {np.mean(intensities):.3f}")
            self.logger.info(f"     强度范围: {min(intensities):.3f}-{max(intensities):.3f}")
    
    def _extract_feature_vector(self, subgraph: Dict) -> np.ndarray:
        """提取子图的特征向量用于聚类"""
        features = []
        
        # 结构特征
        struct = subgraph['structural_features']
        features.extend([
            struct.get('total_nodes', 0),
            struct.get('total_edges', 0),
            struct.get('density', 0),
            struct.get('node_type_diversity', 0),
            struct.get('edge_type_diversity', 0)
        ])
        
        # 节点类型分布
        node_counts = struct.get('node_counts', {})
        for node_type in ['user', 'comment', 'word', 'video']:
            features.append(node_counts.get(node_type, 0))
        
        # 情感特征
        emotion = subgraph['emotion_features']
        features.extend([
            emotion.get('avg_text_length', 0),
            emotion.get('avg_word_count', 0),
            emotion.get('avg_ratio_feature', 0),
            emotion.get('negative_ratio', 0),
            emotion.get('negative_intensity', 0),
            emotion.get('comment_count', 0)
        ])
        
        # 攻击性特征
        aggression = subgraph['aggression_features']
        features.extend([
            aggression.get('avg_text_length', 0),
            aggression.get('avg_word_count', 0),
            aggression.get('avg_ratio_feature', 0),
            aggression.get('max_ratio_feature', 0),
            aggression.get('aggressive_comment_ratio', 0),
            aggression.get('aggression_score', 0),
            aggression.get('attack_word_density', 0)
        ])
        
        # 霸凌强度
        features.append(subgraph.get('bullying_intensity', 0))
        
        return np.array(features)
    
    def _generate_prototype(self, cluster_id: int, cluster_subgraphs: List[Dict]) -> Dict:
        """从聚类生成原型"""
        
        # 计算平均特征
        feature_vectors = [self._extract_feature_vector(sg) for sg in cluster_subgraphs]
        avg_features = np.mean(feature_vectors, axis=0)
        
        # 选择最代表性的子图
        similarities = []
        for i, features in enumerate(feature_vectors):
            sim = cosine_similarity([features], [avg_features])[0][0]
            similarities.append(sim)
        
        representative_idx = np.argmax(similarities)
        representative_subgraph = cluster_subgraphs[representative_idx]
        
        # 计算聚类统计
        bullying_intensities = [sg['bullying_intensity'] for sg in cluster_subgraphs]
        aggression_scores = [sg['aggression_features'].get('aggression_score', 0) 
                           for sg in cluster_subgraphs]
        
        # 生成原型
        prototype = {
            'prototype_id': cluster_id,
            'cluster_size': len(cluster_subgraphs),
            'representative_subgraph': representative_subgraph,
            'average_features': avg_features.tolist(),
            'quality_score': self._calculate_prototype_quality(cluster_subgraphs),
            'avg_bullying_intensity': float(np.mean(bullying_intensities)),
            'avg_aggression_score': float(np.mean(aggression_scores)),
            'creation_time': datetime.now().isoformat(),
            'member_comments': [sg['center_comment'] for sg in cluster_subgraphs]
        }
        
        return prototype
    
    def _calculate_prototype_quality(self, cluster_subgraphs: List[Dict]) -> float:
        """计算原型质量分数"""
        
        quality_factors = []
        
        # 1. 聚类大小 (权重 25%)
        cluster_size = len(cluster_subgraphs)
        size_score = min(cluster_size / 10.0, 1.0)
        quality_factors.append(size_score * 0.25)
        
        # 2. 平均霸凌强度 (权重 35%)
        intensities = [sg.get('bullying_intensity', 0) for sg in cluster_subgraphs]
        if intensities:
            avg_intensity = np.mean(intensities)
            quality_factors.append(avg_intensity * 0.35)
        
        # 3. 攻击性一致性 (权重 25%)
        aggression_scores = [sg['aggression_features'].get('aggression_score', 0) 
                           for sg in cluster_subgraphs]
        if aggression_scores:
            # 一致性 = 1 - 变异系数
            consistency = 1.0 - (np.std(aggression_scores) / (np.mean(aggression_scores) + 1e-6))
            quality_factors.append(max(0, consistency) * 0.25)
        
        # 4. 结构复杂性 (权重 15%)
        node_counts = [sg['structural_features'].get('total_nodes', 0) 
                      for sg in cluster_subgraphs]
        if node_counts:
            avg_nodes = np.mean(node_counts)
            # 最优节点数在30-60之间
            if 30 <= avg_nodes <= 60:
                complexity_score = 1.0
            elif avg_nodes < 30:
                complexity_score = avg_nodes / 30.0
            else:
                complexity_score = max(0.1, 1.0 - (avg_nodes - 60) / 40.0)
            quality_factors.append(complexity_score * 0.15)
        
        return sum(quality_factors) if quality_factors else 0.0
    
    def save_prototypes(self, output_dir: str = "ProtoBully/data/prototype_v4_fixed"):
        """保存原型"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原型
        prototypes_file = output_path / f"prototypes_v4_fixed_{timestamp}.pkl"
        with open(prototypes_file, 'wb') as f:
            pickle.dump(self.prototypes, f)
        
        # 保存子图
        subgraphs_file = output_path / f"subgraphs_v4_fixed_{timestamp}.pkl"
        with open(subgraphs_file, 'wb') as f:
            pickle.dump(self.extracted_subgraphs, f)
        
        # 保存统计信息
        stats = {
            'timestamp': timestamp,
            'aggressive_comments': len(self.aggressive_comments),
            'extracted_subgraphs': len(self.extracted_subgraphs),
            'final_prototypes': len(self.prototypes),
            'config': self.config,
            'prototype_quality_scores': [p['quality_score'] for p in self.prototypes],
            'avg_bullying_intensities': [p['avg_bullying_intensity'] for p in self.prototypes]
        }
        
        stats_file = output_path / f"extraction_stats_v4_fixed_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ 结果保存到 {output_path}")
        
        return {
            'prototypes_file': str(prototypes_file),
            'subgraphs_file': str(subgraphs_file),
            'stats_file': str(stats_file)
        }
    
    def run_full_extraction(self, graph_path: str, labels_path: str = None) -> Dict:
        """运行完整的原型提取流程"""
        self.logger.info("🚀 开始霸凌原型提取流程")
        self.logger.info("="*50)
        
        try:
            # 1. 加载数据
            self.logger.info("📁 步骤1: 加载数据")
            if not self.load_data(graph_path, labels_path):
                return {'success': False, 'error': 'Data loading failed'}
            
            # 2. 识别霸凌会话中的攻击性评论
            self.logger.info("🎯 步骤2: 识别霸凌会话中的攻击性评论")
            aggressive_comments = self.identify_aggressive_comments_in_bullying_sessions()
            if not aggressive_comments:
                return {'success': False, 'error': 'No aggressive comments found in bullying sessions'}
            
            # 3. 从攻击性评论中心提取子图
            self.logger.info("🕸️ 步骤3: 从攻击性评论中心提取子图")
            subgraphs = self.extract_subgraphs_from_aggressive_comments()
            if not subgraphs:
                return {'success': False, 'error': 'No subgraphs extracted'}
            
            # 4. 计算子图情感分数
            self.logger.info("😊 步骤4: 计算子图情感分数")
            self.calculate_emotion_scores()
            
            # 5. 基于情感阈值筛选霸凌子图
            self.logger.info("🔍 步骤5: 筛选霸凌子图")
            bullying_subgraphs = self.filter_bullying_subgraphs()
            if not bullying_subgraphs:
                return {'success': False, 'error': 'No bullying subgraphs after filtering'}
            
            # 6. 从霸凌子图中聚类提取原型
            self.logger.info("🤖 步骤6: 聚类提取原型")
            prototypes = self.cluster_bullying_subgraphs()
            if not prototypes:
                return {'success': False, 'error': 'No prototypes generated'}
            
            # 7. 保存结果
            self.logger.info("💾 步骤7: 保存原型")
            output_dir = f"ProtoBully/data/prototype_v4_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            save_results = self.save_prototypes(output_dir)
            
            self.logger.info("🎉 原型提取流程完成！")
            self.logger.info("="*50)
            
            # 返回结果摘要
            results = {
                'success': True,
                'aggressive_comments_count': len(aggressive_comments),
                'extracted_subgraphs_count': len(subgraphs),
                'bullying_subgraphs_count': len(bullying_subgraphs),
                'prototypes_count': len(prototypes),
                'output_directory': output_dir,
                'save_results': save_results
            }
            
            # 打印结果摘要
            self.logger.info("📊 结果摘要:")
            self.logger.info(f"   霸凌会话数: {len(self.bullying_sessions)}")
            self.logger.info(f"   攻击性评论数: {results['aggressive_comments_count']}")
            self.logger.info(f"   提取子图数: {results['extracted_subgraphs_count']}")
            self.logger.info(f"   霸凌子图数: {results['bullying_subgraphs_count']}")
            self.logger.info(f"   生成原型数: {results['prototypes_count']}")
            self.logger.info(f"   输出目录: {output_dir}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 原型提取流程失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}


def main():
    """主函数"""
    print("🚀 启动基于霸凌会话的原型提取器V4修正版")
    print("="*50)
    
    # 配置参数 - 优化采样策略
    config = {
        'subgraph_size_range': (6, 30),  # 扩大子图大小范围6-30个节点
        'emotion_threshold': -0.3,       # 情感阈值，通过调参优化
        'clustering_eps': 0.4,           # 放宽聚类参数，从0.2增加到0.4
        'min_cluster_size': 2,
        
        # 多样性采样配置 - 大幅优化
        'max_samples': 1000,             # 大幅增加采样数量到1000个 (原300个)
        'diversity_sampling': True,      # 启用多样性采样
        'size_diversity_bins': 5,        # 增加到5个区间采样，提高多样性
        'quality_sampling_ratio': 0.7,   # 70%基于质量采样，30%随机采样
        'adaptive_sampling': True,       # 启用自适应采样
        
        # 分层采样配置 (新增)
        'stratified_sampling': True,     # 启用分层采样
        'attack_level_bins': [
            (0.05, 0.15),  # 轻度攻击性
            (0.15, 0.30),  # 中度攻击性  
            (0.30, 0.50),  # 高度攻击性
            (0.50, 1.00),  # 极高攻击性
        ],
        'samples_per_bin': 200,          # 每个攻击性等级采样200个
        
        # 情感分数权重 - 可通过调参优化
        'emotion_weights': {
            'comment': 0.4,    # 评论节点权重
            'user': 0.3,       # 用户节点权重
            'word': 0.2,       # 词汇节点权重
            'others': 0.1      # 其他节点权重
        },
        
        # 攻击性评论识别阈值
        'attack_word_ratio_threshold': 0.05,
        'attack_word_count_threshold': 1,
        'uppercase_ratio_threshold': 0.25,
        'exclamation_threshold': 2
    }
    
    # 初始化提取器
    extractor = PrototypeExtractorV4Fixed(config)
    
    # 设置文件路径
    graph_path = "data/graphs/heterogeneous_graph_with_comments.pkl"
    labels_path = "data/processed/prototypes/session_labels.json"
    
    # 执行提取
    result = extractor.run_full_extraction(graph_path, labels_path)
    
    # 输出结果
    if result['success']:
        print(f"✅ 原型提取成功!")
        print(f"   霸凌会话数: {len(extractor.bullying_sessions)} 个")
        print(f"   攻击性评论数: {result['aggressive_comments_count']} 个")
        print(f"   提取子图数: {result['extracted_subgraphs_count']} 个")
        print(f"   霸凌子图数: {result['bullying_subgraphs_count']} 个")
        print(f"   生成原型数: {result['prototypes_count']} 个")
        print(f"   输出目录: {result['output_directory']}")
        
        # 如果有原型，显示质量统计
        if extractor.prototypes:
            quality_scores = [p.get('quality_score', 0) for p in extractor.prototypes]
            avg_quality = np.mean(quality_scores) if quality_scores else 0
            print(f"   平均原型质量: {avg_quality:.3f}")
            
            # 显示情感分数统计
            if extractor.bullying_subgraphs:
                emotion_scores = [sg.get('emotion_score', 0) for sg in extractor.bullying_subgraphs]
                avg_emotion = np.mean(emotion_scores) if emotion_scores else 0
                print(f"   平均情感分数: {avg_emotion:.3f}")
        
    else:
        print(f"❌ 原型提取失败: {result['error']}")


if __name__ == "__main__":
    main() 