#!/usr/bin/env python3
"""
ProtoBully项目 - 原型提取器V6重构版
基于弱监督学习的原型质量评估方法

主要创新：
1. 利用会话标签作为弱监督信号
2. 4个科学的质量评估指标
3. 数据驱动的原型选择策略
4. 避免主观权重分配

作者：AI Assistant
日期：2025-06-28
版本：V6 Refactored
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class PrototypeExtractorV6Refactored:
    """
    原型提取器V6重构版 - 基于弱监督学习的质量评估
    
    核心特性：
    - 利用会话标签作为弱监督信号
    - 4个科学的质量评估指标
    - 数据驱动的原型选择策略
    """
    
    def __init__(self, config: Dict = None):
        """初始化原型提取器"""
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # 数据存储
        self.session_labels = {}  # 会话标签映射
        self.heterogeneous_graph = None  # 异构图
        self.enhanced_subgraphs = []  # 增强版子图
        self.bullying_subgraphs = []  # 霸凌子图
        self.normal_subgraphs = []  # 正常子图
        
        # 特征和聚类
        self.subgraph_features = None  # 子图特征矩阵
        self.feature_names = []  # 特征名称
        self.clusters = None  # 聚类结果
        self.prototypes = []  # 提取的原型
        
        # 统计信息
        self.stats = {
            'total_sessions': 0,
            'bullying_sessions': 0,
            'normal_sessions': 0,
            'total_subgraphs': 0,
            'bullying_subgraphs': 0,
            'normal_subgraphs': 0,
            'extracted_prototypes': 0
        }
        
        self.logger.info("原型提取器V6重构版初始化完成")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'purity_threshold': 0.5,  # 纯度阈值（基于31%霸凌数据分布调整）
            'min_cluster_size': 10,   # 最小聚类大小
            'eps': 0.3,               # DBSCAN eps参数
            'min_samples': 5,         # DBSCAN min_samples参数
            'max_prototypes': 20,     # 最大原型数量
            'feature_dim': 10,        # 特征维度
            'random_state': 42
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('PrototypeExtractorV6Refactored')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_session_labels(self, label_file: str = "data/processed/prototypes/session_label_mapping.json") -> bool:
        """加载会话标签映射"""
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                self.session_labels = json.load(f)
            
            # 统计会话信息
            bullying_count = sum(1 for label in self.session_labels.values() if label == 1)
            normal_count = len(self.session_labels) - bullying_count
            
            self.stats.update({
                'total_sessions': len(self.session_labels),
                'bullying_sessions': bullying_count,
                'normal_sessions': normal_count
            })
            
            self.logger.info(f"加载会话标签完成: {len(self.session_labels)}个会话")
            self.logger.info(f"  霸凌会话: {bullying_count}个 ({bullying_count/len(self.session_labels)*100:.1f}%)")
            self.logger.info(f"  正常会话: {normal_count}个 ({normal_count/len(self.session_labels)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载会话标签失败: {e}")
            return False
    
    def load_heterogeneous_graph(self, graph_file: str = "data/graphs/heterogeneous_graph_final.pkl") -> bool:
        """加载异构图"""
        try:
            with open(graph_file, 'rb') as f:
                self.heterogeneous_graph = pickle.load(f)
            
            # 计算总节点数
            total_nodes = 0
            if hasattr(self.heterogeneous_graph, 'node_types'):
                for node_type in self.heterogeneous_graph.node_types:
                    node_data = self.heterogeneous_graph[node_type]
                    if hasattr(node_data, 'x') and node_data.x is not None:
                        total_nodes += node_data.x.shape[0]
            
            self.logger.info(f"加载异构图完成: {total_nodes:,}个节点")
            return True
            
        except Exception as e:
            self.logger.error(f"加载异构图失败: {e}")
            return False
    
    def load_enhanced_subgraphs(self, enhanced_dir: str = "data/subgraphs/universal_enhanced") -> bool:
        """加载增强版子图数据"""
        try:
            enhanced_path = Path(enhanced_dir)
            if not enhanced_path.exists():
                self.logger.error(f"增强版子图目录不存在: {enhanced_dir}")
                return False
            
            pkl_files = list(enhanced_path.glob("*.pkl"))
            if not pkl_files:
                self.logger.error(f"未找到增强版子图文件: {enhanced_dir}")
                return False
            
            self.logger.info(f"开始加载增强版子图数据，共{len(pkl_files)}个文件...")
            
            total_subgraphs = 0
            bullying_subgraphs = 0
            normal_subgraphs = 0
            
            # 使用进度条加载数据
            for pkl_file in tqdm(pkl_files, desc="加载子图文件"):
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # 兼容新旧数据格式
                    if isinstance(data, dict) and 'subgraphs' in data:
                        # 新格式：包含subgraphs键的字典
                        subgraphs = data['subgraphs']
                    elif isinstance(data, list):
                        # 旧格式：直接是子图列表
                        subgraphs = data
                    else:
                        self.logger.warning(f"未知数据格式: {pkl_file}")
                        continue
                    
                    for subgraph in subgraphs:
                        # 添加弱监督标签
                        session_id = subgraph.get('session_id', '')
                        if session_id in self.session_labels:
                            subgraph['weak_label'] = self.session_labels[session_id]
                            self.enhanced_subgraphs.append(subgraph)
                            
                            # 分类存储
                            if self.session_labels[session_id] == 1:  # 霸凌
                                self.bullying_subgraphs.append(subgraph)
                                bullying_subgraphs += 1
                            else:  # 正常
                                self.normal_subgraphs.append(subgraph)
                                normal_subgraphs += 1
                            
                            total_subgraphs += 1
                        
                except Exception as e:
                    self.logger.warning(f"加载文件失败 {pkl_file}: {e}")
                    continue
            
            # 更新统计信息
            self.stats.update({
                'total_subgraphs': total_subgraphs,
                'bullying_subgraphs': bullying_subgraphs,
                'normal_subgraphs': normal_subgraphs
            })
            
            self.logger.info(f"增强版子图加载完成:")
            self.logger.info(f"  总子图数: {total_subgraphs:,}个")
            self.logger.info(f"  霸凌子图: {bullying_subgraphs:,}个 ({bullying_subgraphs/total_subgraphs*100:.1f}%)")
            self.logger.info(f"  正常子图: {normal_subgraphs:,}个 ({normal_subgraphs/total_subgraphs*100:.1f}%)")
            
            return total_subgraphs > 0
            
        except Exception as e:
            self.logger.error(f"加载增强版子图失败: {e}")
            return False
    
    def extract_subgraph_features(self, subgraph: Dict) -> np.ndarray:
        """提取子图特征向量"""
        try:
            features = []
            
            # 1. 基础结构特征
            # 兼容新旧格式的字段名
            total_nodes = subgraph.get('total_nodes', subgraph.get('size', 0))
            features.append(total_nodes)
            
            # 2. 节点类型分布
            nodes = subgraph.get('nodes', {})
            comment_count = len(nodes.get('comment', []))
            user_count = len(nodes.get('user', []))
            word_count = len(nodes.get('word', []))
            video_count = len(nodes.get('video', []))
            
            features.extend([comment_count, user_count, word_count, video_count])
            
            # 3. 边的数量
            edges = subgraph.get('edges', {})
            total_edges = sum(len(edge_list[0]) if edge_list and len(edge_list) > 0 else 0 
                            for edge_list in edges.values())
            features.append(total_edges)
            
            # 4. 节点密度
            node_density = total_edges / max(total_nodes, 1)
            features.append(node_density)
            
            # 5. 会话信息
            session_id = subgraph.get('session_id', '')
            weak_label = subgraph.get('weak_label', 0)
            features.append(weak_label)
            
            # 6. 子图类型编码
            subgraph_type = subgraph.get('subgraph_type', 'unknown')
            type_encoding = 1 if subgraph_type == 'complete_enumeration' else 0
            features.append(type_encoding)
            
            # 7. 时间戳特征（如果有）
            # 兼容新旧格式的时间戳字段名
            timestamp = subgraph.get('extraction_timestamp', subgraph.get('timestamp', ''))
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hour_feature = dt.hour / 24.0  # 归一化小时
                    features.append(hour_feature)
                except:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.debug(f"提取子图特征失败: {e}")
            return np.zeros(self.config['feature_dim'], dtype=np.float32)
    
    def build_feature_matrix(self) -> bool:
        """构建特征矩阵"""
        try:
            if not self.enhanced_subgraphs:
                self.logger.error("没有加载子图数据")
                return False
            
            self.logger.info("开始构建特征矩阵...")
            
            # 定义特征名称
            self.feature_names = [
                'total_nodes', 'comment_count', 'user_count', 'word_count', 
                'video_count', 'total_edges', 'node_density', 'weak_label',
                'type_encoding', 'hour_feature'
            ]
            
            # 提取特征
            features_list = []
            for subgraph in tqdm(self.enhanced_subgraphs, desc="提取特征"):
                features = self.extract_subgraph_features(subgraph)
                features_list.append(features)
            
            self.subgraph_features = np.vstack(features_list)
            
            self.logger.info(f"特征矩阵构建完成: {self.subgraph_features.shape}")
            self.logger.info(f"特征维度: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"构建特征矩阵失败: {e}")
            return False
    
    def cluster_subgraphs(self) -> bool:
        """对霸凌子图进行聚类"""
        try:
            if self.subgraph_features is None:
                self.logger.error("特征矩阵未构建")
                return False
            
            # 只对霸凌子图进行聚类
            bullying_indices = [i for i, subgraph in enumerate(self.enhanced_subgraphs) 
                              if subgraph.get('weak_label') == 1]
            
            if len(bullying_indices) == 0:
                self.logger.error("没有霸凌子图进行聚类")
                return False
            
            bullying_features = self.subgraph_features[bullying_indices]
            
            self.logger.info(f"开始对{len(bullying_indices)}个霸凌子图进行聚类...")
            
            # 标准化特征（除了weak_label列）
            scaler = StandardScaler()
            feature_indices = [i for i, name in enumerate(self.feature_names) if name != 'weak_label']
            scaled_features = bullying_features.copy()
            scaled_features[:, feature_indices] = scaler.fit_transform(bullying_features[:, feature_indices])
            
            # DBSCAN聚类
            clustering = DBSCAN(
                eps=self.config['eps'],
                min_samples=self.config['min_samples'],
                metric='euclidean'
            )
            
            cluster_labels = clustering.fit_predict(scaled_features[:, feature_indices])
            
            # 统计聚类结果
            unique_labels = set(cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            self.logger.info(f"聚类完成:")
            self.logger.info(f"  聚类数量: {n_clusters}")
            self.logger.info(f"  噪声点数: {n_noise}")
            
            # 存储聚类结果
            self.clusters = {
                'labels': cluster_labels,
                'bullying_indices': bullying_indices,
                'n_clusters': n_clusters,
                'n_noise': n_noise
            }
            
            return n_clusters > 0
            
        except Exception as e:
            self.logger.error(f"聚类失败: {e}")
            return False
    
    def calculate_prototype_purity(self, cluster_indices: List[int]) -> float:
        """计算原型纯度"""
        try:
            if not cluster_indices:
                return 0.0
            
            bullying_count = 0
            for idx in cluster_indices:
                subgraph = self.enhanced_subgraphs[idx]
                if subgraph.get('weak_label') == 1:
                    bullying_count += 1
            
            purity = bullying_count / len(cluster_indices)
            return purity
            
        except Exception as e:
            self.logger.debug(f"计算纯度失败: {e}")
            return 0.0
    
    def calculate_prototype_discrimination(self, cluster_indices: List[int]) -> float:
        """计算原型区分度"""
        try:
            if not cluster_indices or not self.normal_subgraphs:
                return 0.0
            
            # 计算聚类中心
            cluster_features = self.subgraph_features[cluster_indices]
            cluster_center = np.mean(cluster_features, axis=0)
            
            # 随机采样正常子图
            normal_indices = [i for i, subgraph in enumerate(self.enhanced_subgraphs) 
                            if subgraph.get('weak_label') == 0]
            
            if len(normal_indices) == 0:
                return 0.0
            
            # 采样最多1000个正常子图
            sample_size = min(1000, len(normal_indices))
            sampled_indices = np.random.choice(normal_indices, sample_size, replace=False)
            normal_features = self.subgraph_features[sampled_indices]
            
            # 计算距离
            distances = euclidean_distances([cluster_center], normal_features)[0]
            discrimination_score = np.mean(distances)
            
            return discrimination_score
            
        except Exception as e:
            self.logger.debug(f"计算区分度失败: {e}")
            return 0.0
    
    def calculate_prototype_coverage(self, cluster_indices: List[int]) -> float:
        """计算原型覆盖度"""
        try:
            if not cluster_indices:
                return 0.0
            
            # 统计覆盖的霸凌会话
            covered_sessions = set()
            for idx in cluster_indices:
                subgraph = self.enhanced_subgraphs[idx]
                session_id = subgraph.get('session_id', '')
                if session_id in self.session_labels and self.session_labels[session_id] == 1:
                    covered_sessions.add(session_id)
            
            # 计算覆盖率
            total_bullying_sessions = self.stats['bullying_sessions']
            coverage = len(covered_sessions) / max(total_bullying_sessions, 1)
            
            return coverage
            
        except Exception as e:
            self.logger.debug(f"计算覆盖度失败: {e}")
            return 0.0
    
    def calculate_prototype_stability(self, cluster_indices: List[int]) -> float:
        """计算原型稳定性"""
        try:
            if len(cluster_indices) < 2:
                return 0.0
            
            cluster_features = self.subgraph_features[cluster_indices]
            
            # 计算特征方差的平均值（除了weak_label）
            feature_indices = [i for i, name in enumerate(self.feature_names) if name != 'weak_label']
            variances = np.var(cluster_features[:, feature_indices], axis=0)
            stability_score = 1.0 / (1.0 + np.mean(variances))  # 方差越小，稳定性越高
            
            return stability_score
            
        except Exception as e:
            self.logger.debug(f"计算稳定性失败: {e}")
            return 0.0
    
    def evaluate_prototype_quality(self, cluster_id: int, cluster_indices: List[int]) -> Dict:
        """评估原型质量"""
        try:
            # 计算4个质量指标
            purity = self.calculate_prototype_purity(cluster_indices)
            discrimination = self.calculate_prototype_discrimination(cluster_indices)
            coverage = self.calculate_prototype_coverage(cluster_indices)
            stability = self.calculate_prototype_stability(cluster_indices)
            
            # 构建质量评估结果
            quality_metrics = {
                'purity': float(purity),
                'discrimination': float(discrimination),
                'coverage': float(coverage),
                'stability': float(stability),
                'cluster_size': len(cluster_indices),
                'passes_purity_threshold': purity >= self.config['purity_threshold']
            }
            
            return quality_metrics
            
        except Exception as e:
            self.logger.debug(f"评估原型质量失败: {e}")
            return {
                'purity': 0.0,
                'discrimination': 0.0,
                'coverage': 0.0,
                'stability': 0.0,
                'cluster_size': 0,
                'passes_purity_threshold': False
            }
    
    def extract_prototypes(self) -> bool:
        """提取原型"""
        try:
            if self.clusters is None:
                self.logger.error("未进行聚类")
                return False
            
            cluster_labels = self.clusters['labels']
            bullying_indices = self.clusters['bullying_indices']
            
            self.logger.info("开始提取原型...")
            
            # 按聚类ID分组
            cluster_groups = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:  # 排除噪声点
                    original_idx = bullying_indices[i]
                    cluster_groups[label].append(original_idx)
            
            # 评估每个聚类的质量
            prototype_candidates = []
            for cluster_id, indices in cluster_groups.items():
                if len(indices) >= self.config['min_cluster_size']:
                    quality_metrics = self.evaluate_prototype_quality(cluster_id, indices)
                    
                    prototype_candidates.append({
                        'cluster_id': cluster_id,
                        'indices': indices,
                        'quality_metrics': quality_metrics
                    })
            
            self.logger.info(f"候选原型数量: {len(prototype_candidates)}")
            
            # 筛选高质量原型
            high_quality_prototypes = [
                candidate for candidate in prototype_candidates
                if candidate['quality_metrics']['passes_purity_threshold']
            ]
            
            self.logger.info(f"通过纯度筛选的原型: {len(high_quality_prototypes)}")
            
            # 按区分度排序
            high_quality_prototypes.sort(
                key=lambda x: x['quality_metrics']['discrimination'], 
                reverse=True
            )
            
            # 选择最终原型
            max_prototypes = min(self.config['max_prototypes'], len(high_quality_prototypes))
            selected_prototypes = high_quality_prototypes[:max_prototypes]
            
            # 生成原型对象
            self.prototypes = []
            for i, prototype_data in enumerate(selected_prototypes):
                prototype = self._create_prototype_object(i + 1, prototype_data)
                self.prototypes.append(prototype)
            
            self.stats['extracted_prototypes'] = len(self.prototypes)
            
            self.logger.info(f"原型提取完成: {len(self.prototypes)}个原型")
            
            return len(self.prototypes) > 0
            
        except Exception as e:
            self.logger.error(f"提取原型失败: {e}")
            return False
    
    def _create_prototype_object(self, prototype_id: int, prototype_data: Dict) -> Dict:
        """创建原型对象"""
        try:
            indices = prototype_data['indices']
            quality_metrics = prototype_data['quality_metrics']
            
            # 计算代表性特征
            cluster_features = self.subgraph_features[indices]
            representative_features = np.mean(cluster_features, axis=0)
            
            # 获取代表性子图
            cluster_center_idx = self._find_cluster_center(indices)
            representative_subgraph = self.enhanced_subgraphs[cluster_center_idx]
            
            # 统计会话分布
            session_distribution = defaultdict(int)
            for idx in indices:
                session_id = self.enhanced_subgraphs[idx].get('session_id', '')
                if session_id:
                    session_distribution[session_id] += 1
            
            prototype = {
                'id': prototype_id,
                'name': f"Bullying_Prototype_{prototype_id}",
                'cluster_id': prototype_data['cluster_id'],
                'subgraph_indices': indices,
                'subgraph_count': len(indices),
                'quality_metrics': quality_metrics,
                'representative_features': representative_features.tolist(),
                'representative_subgraph': representative_subgraph,
                'session_distribution': dict(session_distribution),
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            return prototype
            
        except Exception as e:
            self.logger.debug(f"创建原型对象失败: {e}")
            return {}
    
    def _find_cluster_center(self, indices: List[int]) -> int:
        """找到聚类中心"""
        try:
            if not indices:
                return 0
            
            cluster_features = self.subgraph_features[indices]
            cluster_center = np.mean(cluster_features, axis=0)
            
            # 找到最接近中心的点
            distances = euclidean_distances([cluster_center], cluster_features)[0]
            center_idx = indices[np.argmin(distances)]
            
            return center_idx
            
        except Exception as e:
            self.logger.debug(f"寻找聚类中心失败: {e}")
            return indices[0] if indices else 0
    
    def save_results(self, output_dir: str = "data/prototypes") -> Dict:
        """保存结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存原型
            prototypes_file = output_path / f"extracted_prototypes_v6_refactored_{timestamp}.pkl"
            with open(prototypes_file, 'wb') as f:
                pickle.dump(self.prototypes, f)
            
            # 保存摘要
            summary = self._generate_summary()
            summary_file = output_path / f"prototype_summary_v6_refactored_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info("结果保存完成:")
            self.logger.info(f"  原型文件: {prototypes_file}")
            self.logger.info(f"  摘要文件: {summary_file}")
            
            return {
                'prototypes_file': str(prototypes_file),
                'summary_file': str(summary_file),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_summary(self) -> Dict:
        """生成结果摘要"""
        summary = {
            'extraction_method': 'weak_supervised_v6_refactored',
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config.copy(),
            'statistics': self.stats.copy(),
            'quality_evaluation_method': {
                'purity': '原型中霸凌子图的比例',
                'discrimination': '与正常子图的特征差异',
                'coverage': '覆盖的霸凌会话比例',
                'stability': '原型内部特征的一致性'
            },
            'prototypes': []
        }
        
        # 添加原型信息
        for prototype in self.prototypes:
            prototype_info = {
                'id': prototype['id'],
                'name': prototype['name'],
                'subgraph_count': prototype['subgraph_count'],
                'quality_metrics': prototype['quality_metrics'],
                'covered_sessions': len(prototype['session_distribution']),
                'top_sessions': sorted(
                    prototype['session_distribution'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]  # 前5个最多的会话
            }
            summary['prototypes'].append(prototype_info)
        
        return summary
    
    def run_full_extraction(self, 
                          enhanced_dir: str = "data/subgraphs/universal_enhanced",
                          output_dir: str = "data/prototypes") -> Dict:
        """运行完整的原型提取流程"""
        start_time = datetime.now()
        
        try:
            self.logger.info("开始V6重构版原型提取流程...")
            
            # 步骤1：加载会话标签
            if not self.load_session_labels():
                return {'success': False, 'error': '加载会话标签失败'}
            
            # 步骤2：加载异构图
            if not self.load_heterogeneous_graph():
                return {'success': False, 'error': '加载异构图失败'}
            
            # 步骤3：加载增强版子图
            if not self.load_enhanced_subgraphs(enhanced_dir):
                return {'success': False, 'error': '加载增强版子图失败'}
            
            # 步骤4：构建特征矩阵
            if not self.build_feature_matrix():
                return {'success': False, 'error': '构建特征矩阵失败'}
            
            # 步骤5：聚类
            if not self.cluster_subgraphs():
                return {'success': False, 'error': '聚类失败'}
            
            # 步骤6：提取原型
            if not self.extract_prototypes():
                return {'success': False, 'error': '提取原型失败'}
            
            # 步骤7：保存结果
            save_result = self.save_results(output_dir)
            if not save_result['success']:
                return save_result
            
            # 计算总耗时
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info(f"V6重构版原型提取完成，耗时 {duration:.1f} 秒")
            
            return {
                'success': True,
                'duration': duration,
                'statistics': self.stats.copy(),
                'prototypes_count': len(self.prototypes),
                'files': save_result
            }
            
        except Exception as e:
            self.logger.error(f"原型提取流程失败: {e}")
            return {'success': False, 'error': str(e)}


def main():
    """主函数"""
    print("ProtoBully项目 - 原型提取器V6重构版")
    print("基于弱监督学习的原型质量评估")
    print("=" * 50)
    
    # 创建提取器
    config = {
        'purity_threshold': 0.7,
        'min_cluster_size': 15,
        'eps': 0.3,
        'min_samples': 8,
        'max_prototypes': 15,
        'feature_dim': 10,
        'random_state': 42
    }
    
    extractor = PrototypeExtractorV6Refactored(config)
    
    # 运行提取流程
    result = extractor.run_full_extraction()
    
    if result['success']:
        print(f"\n✅ 原型提取成功完成!")
        print(f"📊 统计信息:")
        for key, value in result['statistics'].items():
            print(f"  {key}: {value:,}")
        print(f"⏱️  总耗时: {result['duration']:.1f} 秒")
        print(f"🎯 提取的原型数量: {result['prototypes_count']}")
    else:
        print(f"\n❌ 原型提取失败: {result['error']}")


if __name__ == "__main__":
    main() 