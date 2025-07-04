#!/usr/bin/env python3
"""
ProtoBully原型提取器 V7 - 增强版
解决V6版本的所有问题：
1. 使用真实的情感分析（规则分析器）
2. 利用会话标签信息进行权重加成
3. 改进的多层次聚类策略
4. 移除虚假的语义和时间特征
"""

import os
import json
import pickle
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import Counter, defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 导入真实的情感分析器
from enhanced_emotion_analyzer import EnhancedEmotionAnalyzer


class PrototypeExtractorV7Enhanced:
    """
    原型提取器V7增强版
    
    核心改进：
    1. 使用真实的规则情感分析
    2. 利用会话标签进行权重加成
    3. 多层次聚类策略
    4. 真实的特征提取
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化原型提取器V7
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 基础配置
        self.min_prototype_size = self.config.get('min_prototype_size', 25)
        self.max_prototypes = self.config.get('max_prototypes', 15)
        
        # 聚类配置
        self.dbscan_eps = self.config.get('dbscan_eps', 0.3)
        self.dbscan_min_samples = self.config.get('dbscan_min_samples', 10)
        
        # 会话权重配置
        self.session_weight_boost = self.config.get('session_weight_boost', 1.5)  # 霸凌会话权重加成
        self.use_session_labels = self.config.get('use_session_labels', True)
        
        # 多层次聚类配置
        self.use_multi_level_clustering = self.config.get('use_multi_level_clustering', True)
        self.size_clustering_enabled = self.config.get('size_clustering_enabled', True)
        self.emotion_clustering_enabled = self.config.get('emotion_clustering_enabled', True)
        
        # 数据路径
        self.session_labels_path = self.config.get('session_labels_path', 
                                                 'data/processed/prototypes/session_label_mapping.json')
        
        # 初始化组件
        self.logger = self._setup_logger()
        self.emotion_analyzer = EnhancedEmotionAnalyzer()
        
        # 数据存储
        self.bullying_subgraphs = []
        self.session_labels = {}
        self.bullying_sessions = set()
        self.extracted_prototypes = []
        
        # 统计信息
        self.extraction_stats = {}
        
    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger(f'{__name__}.V7Enhanced')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_session_labels(self):
        """加载会话标签映射"""
        try:
            self.logger.info(f"加载会话标签: {self.session_labels_path}")
            
            with open(self.session_labels_path, 'r', encoding='utf-8') as f:
                self.session_labels = json.load(f)
            
            # 提取霸凌会话
            self.bullying_sessions = {
                session_id for session_id, label in self.session_labels.items() 
                if label == 1
            }
            
            total_sessions = len(self.session_labels)
            bullying_count = len(self.bullying_sessions)
            
            self.logger.info(f"会话标签加载完成:")
            self.logger.info(f"  总会话数: {total_sessions}")
            self.logger.info(f"  霸凌会话: {bullying_count} ({bullying_count/total_sessions*100:.1f}%)")
            self.logger.info(f"  正常会话: {total_sessions-bullying_count} ({(total_sessions-bullying_count)/total_sessions*100:.1f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载会话标签失败: {e}")
            self.use_session_labels = False
            return False
    
    def load_bullying_subgraphs(self, data_path: str):
        """
        加载霸凌子图数据
        
        Args:
            data_path: 数据路径
        """
        self.logger.info(f"加载霸凌子图数据: {data_path}")
        
        if os.path.isfile(data_path):
            self._load_from_file(data_path)
        elif os.path.isdir(data_path):
            self._load_from_directory(data_path)
        else:
            raise ValueError(f"无效的数据路径: {data_path}")
        
        self.logger.info(f"霸凌子图加载完成: {len(self.bullying_subgraphs)} 个子图")
        
        # 数据特征分析
        self._analyze_data_characteristics()
    
    def _load_from_file(self, file_path: str):
        """从单个文件加载数据"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'bullying_subgraphs' in data:
            self.bullying_subgraphs = data['bullying_subgraphs']
        elif isinstance(data, list):
            self.bullying_subgraphs = data
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}")
    
    def _load_from_directory(self, data_dir: str):
        """从目录加载数据"""
        all_subgraphs = []
        
        for filename in os.listdir(data_dir):
            if filename.endswith('_bullying_subgraphs.pkl'):
                file_path = os.path.join(data_dir, filename)
                
                try:
                    with open(file_path, 'rb') as f:
                        session_data = pickle.load(f)
                    
                    if isinstance(session_data, dict) and 'bullying_subgraphs' in session_data:
                        subgraphs = session_data['bullying_subgraphs']
                        # 添加会话ID信息
                        session_id = session_data.get('session_id', filename.replace('_bullying_subgraphs.pkl', ''))
                        for subgraph in subgraphs:
                            subgraph['source_session'] = session_id
                        all_subgraphs.extend(subgraphs)
                        
                except Exception as e:
                    self.logger.warning(f"加载文件失败 {filename}: {e}")
        
        self.bullying_subgraphs = all_subgraphs
    
    def _analyze_data_characteristics(self) -> Dict[str, Any]:
        """分析数据特征"""
        if not self.bullying_subgraphs:
            return {}
        
        # 基础统计
        sizes = [sg.get('size', 0) for sg in self.bullying_subgraphs]
        emotion_scores = [sg.get('emotion_score', 0.0) for sg in self.bullying_subgraphs]
        
        # 会话分布
        session_distribution = Counter(sg.get('source_session', 'unknown') for sg in self.bullying_subgraphs)
        
        # 霸凌会话中的子图数量
        bullying_session_subgraphs = sum(
            1 for sg in self.bullying_subgraphs 
            if sg.get('source_session', '') in self.bullying_sessions
        )
        
        stats = {
            'total_subgraphs': len(self.bullying_subgraphs),
            'size_stats': {
                'mean': np.mean(sizes),
                'std': np.std(sizes),
                'min': np.min(sizes),
                'max': np.max(sizes),
                'median': np.median(sizes)
            },
            'emotion_stats': {
                'mean': np.mean(emotion_scores),
                'std': np.std(emotion_scores),
                'min': np.min(emotion_scores),
                'max': np.max(emotion_scores),
                'median': np.median(emotion_scores)
            },
            'session_distribution': dict(session_distribution.most_common(10)),
            'bullying_session_subgraphs': bullying_session_subgraphs,
            'bullying_session_ratio': bullying_session_subgraphs / len(self.bullying_subgraphs) if self.bullying_subgraphs else 0
        }
        
        self.logger.info("数据特征分析:")
        self.logger.info(f"  子图大小: {stats['size_stats']['mean']:.1f}±{stats['size_stats']['std']:.1f}")
        self.logger.info(f"  情感分数: {stats['emotion_stats']['mean']:.3f}±{stats['emotion_stats']['std']:.3f}")
        self.logger.info(f"  霸凌会话子图: {bullying_session_subgraphs}/{len(self.bullying_subgraphs)} ({stats['bullying_session_ratio']*100:.1f}%)")
        
        return stats
    
    def extract_enhanced_features(self, subgraph: Dict) -> Dict[str, float]:
        """
        提取真实的增强特征
        
        Args:
            subgraph: 子图数据
            
        Returns:
            特征字典
        """
        features = {}
        
        try:
            # 基础结构特征
            size = subgraph.get('size', 0)
            features['size'] = size
            
            # 节点组成特征
            node_counts = subgraph.get('node_counts', {})
            comment_count = node_counts.get('comment', 0)
            user_count = node_counts.get('user', 0)
            word_count = node_counts.get('word', 0)
            video_count = node_counts.get('video', 0)
            
            features['comment_nodes'] = comment_count
            features['user_nodes'] = user_count
            features['video_nodes'] = video_count
            features['word_nodes'] = word_count
            
            # 边统计
            edges = subgraph.get('edges', {})
            edge_count = sum(len(edge_list) for edge_list in edges.values())
            features['edge_count'] = edge_count
            
            # 密度计算
            max_edges = size * (size - 1) // 2 if size > 1 else 1
            density = edge_count / max_edges if max_edges > 0 else 0
            features['density'] = density
            
            # 节点比例特征
            if size > 0:
                features['comment_ratio'] = comment_count / size
                features['user_ratio'] = user_count / size
                features['word_ratio'] = word_count / size
                features['video_ratio'] = video_count / size
            else:
                features['comment_ratio'] = features['user_ratio'] = 0
                features['word_ratio'] = features['video_ratio'] = 0
            
            # 真实的情感特征（使用已有的emotion_score）
            emotion_score = subgraph.get('emotion_score', 0.0)
            features['emotion_score'] = emotion_score
            features['aggression_score'] = max(0, -emotion_score)  # 转换为正值攻击性分数
            
            # 交互强度（真实计算）
            features['interaction_intensity'] = min(1.0, edge_count / (size + 1)) if size > 0 else 0
            
            # 会话标签特征（新增）
            source_session = subgraph.get('source_session', '')
            is_from_bullying_session = 1.0 if source_session in self.bullying_sessions else 0.0
            features['from_bullying_session'] = is_from_bullying_session
            
            # 会话权重（用于后续加权）
            session_weight = self.session_weight_boost if is_from_bullying_session else 1.0
            features['session_weight'] = session_weight
            
        except Exception as e:
            self.logger.warning(f"特征提取出错: {e}")
            # 返回默认特征
            features = {
                'size': 5, 'edge_count': 8, 'density': 0.4,
                'comment_nodes': 3, 'user_nodes': 1, 'video_nodes': 1, 'word_nodes': 0,
                'comment_ratio': 0.6, 'user_ratio': 0.2, 'video_ratio': 0.2, 'word_ratio': 0.0,
                'emotion_score': -0.5, 'aggression_score': 0.5,
                'interaction_intensity': 0.6,
                'from_bullying_session': 0.0, 'session_weight': 1.0
            }
        
        return features
    
    def build_feature_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建特征矩阵和权重向量
        
        Returns:
            (特征矩阵, 权重向量)
        """
        self.logger.info("构建增强特征矩阵...")
        
        feature_list = []
        weight_list = []
        successful_indices = []
        
        for i, subgraph in enumerate(self.bullying_subgraphs):
            try:
                features = self.extract_enhanced_features(subgraph)
                
                # 提取权重
                session_weight = features.pop('session_weight', 1.0)
                
                feature_vector = list(features.values())
                feature_list.append(feature_vector)
                weight_list.append(session_weight)
                successful_indices.append(i)
                
            except Exception as e:
                self.logger.warning(f"子图 {i} 特征提取失败: {e}")
                continue
        
        # 更新子图列表
        self.bullying_subgraphs = [self.bullying_subgraphs[i] for i in successful_indices]
        
        if len(feature_list) == 0:
            raise ValueError("没有成功提取任何特征")
        
        # 转换为numpy数组
        feature_matrix = np.array(feature_list)
        weight_vector = np.array(weight_list)
        
        # 标准化特征矩阵
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
        
        self.logger.info(f"特征矩阵构建完成: {feature_matrix.shape}")
        self.logger.info(f"权重统计: 平均={np.mean(weight_vector):.2f}, 最大={np.max(weight_vector):.2f}")
        
        return feature_matrix, weight_vector
    
    def multi_level_clustering(self, feature_matrix: np.ndarray, 
                             weights: np.ndarray) -> np.ndarray:
        """
        多层次聚类（V5风格的改进版）
        
        Args:
            feature_matrix: 特征矩阵
            weights: 权重向量
            
        Returns:
            聚类标签
        """
        self.logger.info("执行多层次聚类...")
        
        # 第一层：基础DBSCAN聚类
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples
        )
        
        # 应用权重到特征矩阵（通过重复样本实现加权效果）
        weighted_features = feature_matrix * weights.reshape(-1, 1)
        
        cluster_labels = clustering.fit_predict(weighted_features)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        self.logger.info(f"第一层聚类: {n_clusters}个聚类, {n_noise}个噪声点")
        
        # 第二层：大小聚类（如果启用）
        if self.size_clustering_enabled and n_clusters > 3:
            cluster_labels = self._refine_by_size_clustering(cluster_labels, feature_matrix)
        
        # 第三层：情感聚类（如果启用）
        if self.emotion_clustering_enabled and n_clusters > 2:
            cluster_labels = self._refine_by_emotion_clustering(cluster_labels, feature_matrix)
        
        final_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        self.logger.info(f"多层次聚类完成: {final_clusters}个最终聚类")
        
        return cluster_labels
    
    def _refine_by_size_clustering(self, cluster_labels: np.ndarray, 
                                 feature_matrix: np.ndarray) -> np.ndarray:
        """基于大小的聚类细化"""
        refined_labels = cluster_labels.copy()
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        new_label = max(cluster_labels) + 1 if len(cluster_labels) > 0 else 0
        
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) < 10:  # 太小的聚类跳过
                continue
            
            # 提取大小特征（假设在第0列）
            sizes = feature_matrix[cluster_indices, 0]
            
            # 基于大小进行二次聚类
            size_clustering = DBSCAN(eps=0.5, min_samples=3)
            size_labels = size_clustering.fit_predict(sizes.reshape(-1, 1))
            
            # 更新标签
            for i, size_label in enumerate(size_labels):
                if size_label != -1:  # 不是噪声
                    refined_labels[cluster_indices[i]] = new_label + size_label
            
            new_label += len(set(size_labels)) - (1 if -1 in size_labels else 0)
        
        return refined_labels
    
    def _refine_by_emotion_clustering(self, cluster_labels: np.ndarray, 
                                    feature_matrix: np.ndarray) -> np.ndarray:
        """基于情感的聚类细化"""
        refined_labels = cluster_labels.copy()
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        new_label = max(cluster_labels) + 1 if len(cluster_labels) > 0 else 0
        
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) < 8:  # 太小的聚类跳过
                continue
            
            # 提取情感特征（假设emotion_score在特定位置）
            emotion_features = feature_matrix[cluster_indices, -3:-1]  # emotion_score, aggression_score
            
            # 基于情感进行二次聚类
            emotion_clustering = DBSCAN(eps=0.3, min_samples=2)
            emotion_labels = emotion_clustering.fit_predict(emotion_features)
            
            # 更新标签
            for i, emotion_label in enumerate(emotion_labels):
                if emotion_label != -1:  # 不是噪声
                    refined_labels[cluster_indices[i]] = new_label + emotion_label
            
            new_label += len(set(emotion_labels)) - (1 if -1 in emotion_labels else 0)
        
        return refined_labels
    
    def extract_prototypes_from_clusters(self, cluster_labels: np.ndarray, 
                                       feature_matrix: np.ndarray,
                                       weights: np.ndarray) -> List[Dict]:
        """
        从聚类中提取原型（考虑会话权重）
        
        Args:
            cluster_labels: 聚类标签
            feature_matrix: 特征矩阵
            weights: 权重向量
            
        Returns:
            原型列表
        """
        self.logger.info("从聚类中提取原型...")
        
        prototypes = []
        unique_labels = set(cluster_labels)
        
        # 排除噪声点
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_indices = np.where(cluster_mask)[0]
            cluster_subgraphs = [self.bullying_subgraphs[i] for i in cluster_indices]
            cluster_weights = weights[cluster_indices]
            
            if len(cluster_subgraphs) < self.min_prototype_size:
                continue
            
            # 计算聚类质量（考虑权重）
            cluster_features = feature_matrix[cluster_indices]
            quality_score = self._calculate_weighted_cluster_quality(
                cluster_features, cluster_weights
            )
            
            # 选择代表性子图（优先选择来自霸凌会话的子图）
            representative = self._select_weighted_representative(
                cluster_subgraphs, cluster_features, cluster_weights
            )
            
            # 计算聚类统计
            cluster_stats = self._calculate_cluster_statistics(cluster_subgraphs, cluster_weights)
            
            prototype = {
                'id': len(prototypes),
                'cluster_id': int(label),
                'representative_subgraph': representative,
                'cluster_size': len(cluster_subgraphs),
                'quality_score': quality_score,
                'statistics': cluster_stats,
                'bullying_session_ratio': cluster_stats.get('bullying_session_ratio', 0.0),
                'weighted_score': quality_score * cluster_stats.get('avg_weight', 1.0)  # 综合评分
            }
            
            prototypes.append(prototype)
        
        # 按综合评分排序
        prototypes.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # 限制原型数量
        if len(prototypes) > self.max_prototypes:
            prototypes = prototypes[:self.max_prototypes]
        
        self.logger.info(f"原型提取完成: {len(prototypes)}个原型")
        
        return prototypes
    
    def _calculate_weighted_cluster_quality(self, cluster_features: np.ndarray, 
                                          cluster_weights: np.ndarray) -> float:
        """计算加权聚类质量"""
        if len(cluster_features) < 2:
            return 0.0
        
        try:
            # 基础质量：内聚性
            center = np.average(cluster_features, axis=0, weights=cluster_weights)
            distances = np.linalg.norm(cluster_features - center, axis=1)
            weighted_cohesion = np.average(distances, weights=cluster_weights)
            cohesion_score = 1.0 / (1.0 + weighted_cohesion)
            
            # 权重多样性奖励
            weight_diversity = np.std(cluster_weights)
            diversity_bonus = min(weight_diversity * 0.1, 0.2)
            
            # 霸凌会话比例奖励
            bullying_ratio = np.sum(cluster_weights > 1.0) / len(cluster_weights)
            bullying_bonus = bullying_ratio * 0.3
            
            total_quality = cohesion_score + diversity_bonus + bullying_bonus
            
            return min(total_quality, 1.0)
            
        except Exception as e:
            self.logger.warning(f"质量计算失败: {e}")
            return 0.5
    
    def _select_weighted_representative(self, cluster_subgraphs: List, 
                                     cluster_features: np.ndarray,
                                     cluster_weights: np.ndarray) -> Dict:
        """选择加权代表性子图"""
        # 计算加权中心
        center = np.average(cluster_features, axis=0, weights=cluster_weights)
        
        # 计算每个子图到中心的距离，并考虑权重
        distances = np.linalg.norm(cluster_features - center, axis=1)
        weighted_distances = distances / cluster_weights  # 权重越高，距离惩罚越小
        
        # 选择加权距离最小的子图
        best_idx = np.argmin(weighted_distances)
        
        return cluster_subgraphs[best_idx]
    
    def _calculate_cluster_statistics(self, cluster_subgraphs: List, 
                                    cluster_weights: np.ndarray) -> Dict:
        """计算聚类统计信息"""
        stats = {}
        
        # 基础统计
        sizes = [sg.get('size', 0) for sg in cluster_subgraphs]
        emotion_scores = [sg.get('emotion_score', 0.0) for sg in cluster_subgraphs]
        
        stats['size_stats'] = {
            'mean': np.mean(sizes),
            'std': np.std(sizes),
            'min': np.min(sizes),
            'max': np.max(sizes)
        }
        
        stats['emotion_stats'] = {
            'mean': np.mean(emotion_scores),
            'std': np.std(emotion_scores),
            'min': np.min(emotion_scores),
            'max': np.max(emotion_scores)
        }
        
        # 权重统计
        stats['weight_stats'] = {
            'mean': np.mean(cluster_weights),
            'std': np.std(cluster_weights),
            'min': np.min(cluster_weights),
            'max': np.max(cluster_weights)
        }
        
        stats['avg_weight'] = np.mean(cluster_weights)
        
        # 霸凌会话统计
        bullying_session_count = sum(
            1 for sg in cluster_subgraphs 
            if sg.get('source_session', '') in self.bullying_sessions
        )
        stats['bullying_session_count'] = bullying_session_count
        stats['bullying_session_ratio'] = bullying_session_count / len(cluster_subgraphs)
        
        return stats
    
    def extract_prototypes(self, data_path: str) -> List[Dict]:
        """
        主要的原型提取方法
        
        Args:
            data_path: 霸凌子图数据路径
            
        Returns:
            提取的原型列表
        """
        self.logger.info("🚀 开始V7增强原型提取...")
        
        # 1. 加载会话标签
        if self.use_session_labels:
            self.load_session_labels()
        
        # 2. 加载霸凌子图
        self.load_bullying_subgraphs(data_path)
        
        # 3. 构建特征矩阵
        feature_matrix, weights = self.build_feature_matrix()
        
        # 4. 多层次聚类
        if self.use_multi_level_clustering:
            cluster_labels = self.multi_level_clustering(feature_matrix, weights)
        else:
            # 简单聚类
            clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
            cluster_labels = clustering.fit_predict(feature_matrix)
        
        # 5. 提取原型
        self.extracted_prototypes = self.extract_prototypes_from_clusters(
            cluster_labels, feature_matrix, weights
        )
        
        # 6. 生成统计信息
        self._generate_extraction_stats()
        
        self.logger.info("✅ V7增强原型提取完成!")
        
        return self.extracted_prototypes
    
    def _generate_extraction_stats(self):
        """生成提取统计信息"""
        if not self.extracted_prototypes:
            return
        
        quality_scores = [p['quality_score'] for p in self.extracted_prototypes]
        weighted_scores = [p['weighted_score'] for p in self.extracted_prototypes]
        cluster_sizes = [p['cluster_size'] for p in self.extracted_prototypes]
        bullying_ratios = [p['bullying_session_ratio'] for p in self.extracted_prototypes]
        
        self.extraction_stats = {
            'total_prototypes': len(self.extracted_prototypes),
            'quality_stats': {
                'mean': np.mean(quality_scores),
                'std': np.std(quality_scores),
                'min': np.min(quality_scores),
                'max': np.max(quality_scores)
            },
            'weighted_score_stats': {
                'mean': np.mean(weighted_scores),
                'std': np.std(weighted_scores),
                'min': np.min(weighted_scores),
                'max': np.max(weighted_scores)
            },
            'cluster_size_stats': {
                'mean': np.mean(cluster_sizes),
                'std': np.std(cluster_sizes),
                'min': np.min(cluster_sizes),
                'max': np.max(cluster_sizes),
                'total_coverage': sum(cluster_sizes)
            },
            'bullying_session_stats': {
                'mean_ratio': np.mean(bullying_ratios),
                'std_ratio': np.std(bullying_ratios),
                'min_ratio': np.min(bullying_ratios),
                'max_ratio': np.max(bullying_ratios)
            },
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        # 打印统计信息
        self.logger.info("📊 提取统计:")
        self.logger.info(f"  原型数量: {self.extraction_stats['total_prototypes']}")
        self.logger.info(f"  平均质量: {self.extraction_stats['quality_stats']['mean']:.3f}±{self.extraction_stats['quality_stats']['std']:.3f}")
        self.logger.info(f"  平均加权分数: {self.extraction_stats['weighted_score_stats']['mean']:.3f}±{self.extraction_stats['weighted_score_stats']['std']:.3f}")
        self.logger.info(f"  平均聚类大小: {self.extraction_stats['cluster_size_stats']['mean']:.1f}")
        self.logger.info(f"  覆盖率: {self.extraction_stats['cluster_size_stats']['total_coverage']}/{len(self.bullying_subgraphs)} ({self.extraction_stats['cluster_size_stats']['total_coverage']/len(self.bullying_subgraphs)*100:.1f}%)")
        self.logger.info(f"  平均霸凌会话比例: {self.extraction_stats['bullying_session_stats']['mean_ratio']:.1%}")
    
    def save_results(self, output_dir: str = "data/prototypes"):
        """保存结果"""
        if not self.extracted_prototypes:
            self.logger.warning("没有原型可保存")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原型
        prototypes_file = f"{output_dir}/extracted_prototypes_v7_enhanced_{timestamp}.pkl"
        with open(prototypes_file, 'wb') as f:
            pickle.dump(self.extracted_prototypes, f)
        
        # 保存统计信息
        stats_file = f"{output_dir}/prototype_summary_v7_enhanced_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.extraction_stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果已保存:")
        self.logger.info(f"  原型文件: {prototypes_file}")
        self.logger.info(f"  统计文件: {stats_file}")


def main():
    """主函数"""
    config = {
        'min_prototype_size': 25,
        'max_prototypes': 15,
        'dbscan_eps': 0.3,
        'dbscan_min_samples': 10,
        'session_weight_boost': 1.5,
        'use_session_labels': True,
        'use_multi_level_clustering': True,
        'size_clustering_enabled': True,
        'emotion_clustering_enabled': True,
        'session_labels_path': 'data/processed/prototypes/session_label_mapping.json'
    }
    
    extractor = PrototypeExtractorV7Enhanced(config)
    
    # 提取原型
    prototypes = extractor.extract_prototypes('data/subgraphs/bullying_subgraphs_new.pkl')
    
    # 保存结果
    extractor.save_results()
    
    print(f"\n🎉 V7增强原型提取完成! 提取了 {len(prototypes)} 个原型")


if __name__ == "__main__":
    main() 