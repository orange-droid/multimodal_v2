#!/usr/bin/env python3
"""
原型提取器V5 - 基于新版本霸凌子图的原型提取

适配新的子图提取模块输出：
1. 使用新版本的霸凌子图数据（data/subgraphs/bullying_new/）
2. 基于改进的情感分数进行原型提取
3. 实现多策略聚类：DBSCAN + 大小聚类 + 强度聚类
4. 提取高质量的霸凌原型
"""

import numpy as np
import torch
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Set, Optional
from collections import defaultdict, Counter
from tqdm import tqdm
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 导入增强情感分析器
try:
    from .enhanced_emotion_analyzer import EnhancedEmotionAnalyzer
    ENHANCED_EMOTION_AVAILABLE = True
except ImportError as e:
    print(f"增强情感分析器导入失败: {e}")
    ENHANCED_EMOTION_AVAILABLE = False

# 导入BERT情感分析器作为备选
BERT_AVAILABLE = False  # 暂时禁用BERT以避免TensorFlow问题

class PrototypeExtractorV5:
    """基于新版本霸凌子图的原型提取器V5"""
    
    def __init__(self, config: Dict = None):
        """
        初始化原型提取器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # 核心参数（增加原型数量）
        self.min_prototype_size = self.config.get('min_prototype_size', 30)  # 降低原型最小子图数量
        self.max_prototypes = self.config.get('max_prototypes', 12)  # 增加最大原型数量
        
        # 聚类参数（调整以获得更多聚类）
        self.dbscan_eps = self.config.get('dbscan_eps', 0.5)  # 降低eps增加聚类数量
        self.dbscan_min_samples = self.config.get('dbscan_min_samples', 8)  # 降低最小样本数
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        
        # 特征权重
        self.feature_weights = self.config.get('feature_weights', {
            'size': 0.2,           # 子图大小
            'emotion': 0.3,        # 情感分数
            'node_composition': 0.2, # 节点组成
            'structure': 0.3       # 结构特征
        })
        
        # 数据存储
        self.bullying_subgraphs = []  # 霸凌子图数据
        self.feature_matrix = None    # 特征矩阵
        self.prototypes = []          # 提取的原型
        
        # 统计信息
        self.stats = {
            'total_subgraphs': 0,
            'clustered_subgraphs': 0,
            'prototypes_count': 0,
            'cluster_distribution': {},
            'processing_time': 0
        }
        
        # 初始化增强情感分析器
        self.emotion_analyzer = None
        self._initialize_emotion_analyzer()
        
    def _setup_logger(self):
        """设置日志"""
        logger_name = 'PrototypeExtractorV5'
        logger = logging.getLogger(logger_name)
        
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _initialize_emotion_analyzer(self):
        """初始化情感分析器"""
        try:
            # 优先使用增强情感分析器
            if ENHANCED_EMOTION_AVAILABLE:
                from .enhanced_emotion_analyzer import EnhancedEmotionAnalyzer
                self.emotion_analyzer = EnhancedEmotionAnalyzer()
                self.logger.info("增强情感分析器初始化成功")
            elif BERT_AVAILABLE:
                from .bert_emotion_analyzer import BERTEmotionAnalyzer
                self.emotion_analyzer = BERTEmotionAnalyzer()
                self.logger.info("BERT情感分析器初始化成功")
            else:
                self.emotion_analyzer = None
                self.logger.warning("未找到可用的情感分析器，将使用基础方法")
        except Exception as e:
            self.logger.warning(f"情感分析器初始化失败: {e}")
            self.emotion_analyzer = None
    
    def _extract_comment_texts(self, subgraph: Dict) -> List[str]:
        """从子图中提取评论文本"""
        comment_texts = []
        
        try:
            # 从子图的DGL图对象中提取评论节点的文本
            graph = subgraph.get('graph')
            if graph is None:
                return comment_texts
            
            # 获取评论节点
            if 'comment' in graph.ntypes:
                comment_nodes = graph.nodes('comment')
                
                # 尝试从节点特征中提取文本信息
                # 这里需要根据实际的数据结构调整
                if 'text' in graph.nodes['comment'].data:
                    texts = graph.nodes['comment'].data['text']
                    for text in texts:
                        if isinstance(text, str) and text.strip():
                            comment_texts.append(text.strip())
                elif 'features' in graph.nodes['comment'].data:
                    # 如果文本被编码为特征，这里暂时跳过
                    # 实际应用中可能需要解码或使用其他方法
                    pass
            
        except Exception as e:
            self.logger.debug(f"提取评论文本失败: {e}")
        
        return comment_texts
    
    def _calculate_enhanced_emotion_score(self, comment_texts: List[str]) -> float:
        """使用增强情感分析器计算情感分数"""
        if not comment_texts or not self.emotion_analyzer:
            return 0.0
        
        try:
            # 批量分析所有评论的情感
            emotion_results = self.emotion_analyzer.analyze_batch_emotions(comment_texts)
            
            # 计算平均攻击性分数
            aggression_scores = [result.get('aggression_score', 0) for result in emotion_results]
            
            if aggression_scores:
                # 使用加权平均，给予更长的评论更高的权重
                weights = [len(text.split()) for text in comment_texts]
                if sum(weights) > 0:
                    weighted_score = sum(score * weight for score, weight in zip(aggression_scores, weights)) / sum(weights)
                else:
                    weighted_score = np.mean(aggression_scores)
                
                return min(weighted_score, 1.0)
            
        except Exception as e:
            self.logger.debug(f"增强情感分析失败: {e}")
        
        return 0.0
    
    def load_bullying_subgraphs(self, bullying_dir: str = "data/subgraphs/bullying_new"):
        """加载新版本的霸凌子图数据"""
        self.logger.info(f"加载霸凌子图数据: {bullying_dir}")
        
        try:
            # 读取霸凌子图索引
            index_path = Path(bullying_dir) / "bullying_subgraph_index.json"
            if not index_path.exists():
                self.logger.error(f"霸凌子图索引文件不存在: {index_path}")
                return False
            
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            self.logger.info(f"霸凌子图索引加载成功:")
            self.logger.info(f"  包含霸凌子图的会话数: {index_data.get('total_sessions', 0)}")
            self.logger.info(f"  总霸凌子图数: {index_data.get('total_bullying_subgraphs', 0)}")
            
            # 加载所有霸凌子图文件
            self.bullying_subgraphs = []
            session_files = index_data.get('session_files', {})
            
            for session_id, filename in tqdm(session_files.items(), desc="加载霸凌子图"):
                file_path = Path(bullying_dir) / filename
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        session_data = pickle.load(f)
                    
                    # 从session_data中提取bullying_subgraphs
                    if isinstance(session_data, dict) and 'bullying_subgraphs' in session_data:
                        bullying_subgraphs = session_data['bullying_subgraphs']
                        
                        # 添加会话标识到每个子图
                        for subgraph in bullying_subgraphs:
                            subgraph['session_id'] = session_id
                            self.bullying_subgraphs.append(subgraph)
                    else:
                        # 兼容旧格式（如果是直接的子图列表）
                        if isinstance(session_data, list):
                            for subgraph in session_data:
                                subgraph['session_id'] = session_id
                                self.bullying_subgraphs.append(subgraph)
            
            self.stats['total_subgraphs'] = len(self.bullying_subgraphs)
            
            self.logger.info(f"霸凌子图数据加载完成")
            self.logger.info(f"   总子图数: {len(self.bullying_subgraphs)}")
            
            # 统计基本信息
            self._analyze_subgraph_distribution()
            
            return True
            
        except Exception as e:
            self.logger.error(f"霸凌子图数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _analyze_subgraph_distribution(self):
        """分析子图分布情况"""
        self.logger.info("分析子图分布...")
        
        # 大小分布
        size_dist = defaultdict(int)
        emotion_scores = []
        session_dist = defaultdict(int)
        
        for subgraph in self.bullying_subgraphs:
            size = subgraph.get('size', 0)
            size_dist[size] += 1
            
            emotion_score = subgraph.get('emotion_score', 0)
            emotion_scores.append(emotion_score)
            
            session_id = subgraph.get('session_id', 'unknown')
            session_dist[session_id] += 1
        
        # 输出统计信息
        self.logger.info(f"大小分布: {dict(sorted(size_dist.items())[:10])}...")
        self.logger.info(f"情感分数范围: [{min(emotion_scores):.3f}, {max(emotion_scores):.3f}]")
        self.logger.info(f"平均情感分数: {np.mean(emotion_scores):.3f}")
        self.logger.info(f"涉及会话数: {len(session_dist)}")
    
    def extract_features(self):
        """从霸凌子图中提取特征向量"""
        self.logger.info("提取子图特征向量...")
        
        features = []
        
        for subgraph in tqdm(self.bullying_subgraphs, desc="提取特征"):
            feature_vector = self._extract_subgraph_features(subgraph)
            features.append(feature_vector)
        
        self.feature_matrix = np.array(features)
        
        self.logger.info(f"特征提取完成")
        self.logger.info(f"   特征矩阵形状: {self.feature_matrix.shape}")
        
        # 特征标准化
        scaler = StandardScaler()
        self.feature_matrix = scaler.fit_transform(self.feature_matrix)
        
        self.logger.info(f"   特征标准化完成")
    
    def _extract_subgraph_features(self, subgraph: Dict) -> np.ndarray:
        """提取单个子图的特征向量"""
        features = []
        
        # 1. 基本特征
        size = subgraph.get('size', 0)
        emotion_score = subgraph.get('emotion_score', 0)
        
        # 如果有增强情感分析器，重新计算更准确的情感分数
        if hasattr(self, 'emotion_analyzer') and self.emotion_analyzer:
            # 尝试从子图中获取评论文本
            comment_texts = self._extract_comment_texts(subgraph)
            if comment_texts:
                enhanced_emotion = self._calculate_enhanced_emotion_score(comment_texts)
                emotion_score = enhanced_emotion  # 使用增强的情感分数
        features.extend([size, emotion_score])
        
        # 2. 节点组成特征
        nodes = subgraph.get('nodes', {})
        node_counts = subgraph.get('node_counts', {})
        
        comment_count = node_counts.get('comment', 0)
        user_count = node_counts.get('user', 0)
        word_count = node_counts.get('word', 0)
        video_count = node_counts.get('video', 0)
        
        # 节点比例特征
        if size > 0:
            comment_ratio = comment_count / size
            user_ratio = user_count / size
            word_ratio = word_count / size
            video_ratio = video_count / size
        else:
            comment_ratio = user_ratio = word_ratio = video_ratio = 0
        
        features.extend([comment_count, user_count, word_count, video_count])
        features.extend([comment_ratio, user_ratio, word_ratio, video_ratio])
        
        # 3. 结构特征
        edges = subgraph.get('edges', {})
        edge_count = sum(len(edge_list) for edge_list in edges.values())
        
        # 边密度
        max_edges = size * (size - 1) // 2 if size > 1 else 1
        edge_density = edge_count / max_edges if max_edges > 0 else 0
        
        features.extend([edge_count, edge_density])
        
        # 4. 会话特征
        session_id = subgraph.get('session_id', '')
        session_idx = int(session_id.split('_')[-1]) if session_id.startswith('media_session_') else 0
        features.append(session_idx % 100)  # 会话特征（取模避免过大）
        
        return np.array(features, dtype=np.float32)
    
    def cluster_subgraphs(self):
        """使用多策略聚类方法对子图进行聚类"""
        self.logger.info("开始多策略聚类...")
        
        start_time = datetime.now()
        
        # 策略1: DBSCAN聚类
        dbscan_clusters = self._dbscan_clustering()
        
        # 策略2: 基于大小的聚类
        size_clusters = self._size_based_clustering()
        
        # 策略3: 基于情感强度的聚类
        emotion_clusters = self._emotion_based_clustering()
        
        # 合并聚类结果
        merged_clusters = self._merge_clustering_results(
            dbscan_clusters, size_clusters, emotion_clusters
        )
        
        # 筛选高质量聚类
        final_clusters = self._filter_quality_clusters(merged_clusters)
        
        self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        self.stats['clustered_subgraphs'] = sum(len(cluster) for cluster in final_clusters.values())
        self.stats['cluster_distribution'] = {k: len(v) for k, v in final_clusters.items()}
        
        self.logger.info(f"聚类完成")
        self.logger.info(f"   聚类数量: {len(final_clusters)}")
        self.logger.info(f"   聚类分布: {self.stats['cluster_distribution']}")
        
        return final_clusters
    
    def _dbscan_clustering(self) -> Dict[int, List[int]]:
        """DBSCAN聚类"""
        self.logger.info("执行DBSCAN聚类...")
        
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        cluster_labels = dbscan.fit_predict(self.feature_matrix)
        
        # 组织聚类结果
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label != -1:  # 忽略噪声点
                clusters[f"dbscan_{label}"].append(idx)
        
        self.logger.info(f"DBSCAN聚类完成: {len(clusters)}个聚类, {sum(cluster_labels == -1)}个噪声点")
        
        return dict(clusters)
    
    def _size_based_clustering(self) -> Dict[str, List[int]]:
        """基于子图大小的聚类"""
        self.logger.info("执行基于大小的聚类...")
        
        size_clusters = defaultdict(list)
        
        for idx, subgraph in enumerate(self.bullying_subgraphs):
            size = subgraph.get('size', 0)
            # 将相近大小的子图归为一类
            size_group = f"size_{size//2*2}_{size//2*2+1}"  # 每2个大小为一组
            size_clusters[size_group].append(idx)
        
        # 过滤小聚类
        filtered_clusters = {k: v for k, v in size_clusters.items() if len(v) >= 20}
        
        self.logger.info(f"大小聚类完成: {len(filtered_clusters)}个聚类")
        
        return filtered_clusters
    
    def _emotion_based_clustering(self) -> Dict[str, List[int]]:
        """基于情感强度的聚类"""
        self.logger.info("执行基于情感强度的聚类...")
        
        emotion_scores = [subgraph.get('emotion_score', 0) for subgraph in self.bullying_subgraphs]
        
        # 基于情感分数分位数进行分组
        percentiles = np.percentile(emotion_scores, [25, 50, 75])
        
        emotion_clusters = defaultdict(list)
        
        for idx, score in enumerate(emotion_scores):
            if score <= percentiles[0]:
                group = "emotion_very_negative"
            elif score <= percentiles[1]:
                group = "emotion_negative"
            elif score <= percentiles[2]:
                group = "emotion_moderate"
            else:
                group = "emotion_mild"
            
            emotion_clusters[group].append(idx)
        
        self.logger.info(f"情感聚类完成: {len(emotion_clusters)}个聚类")
        
        return dict(emotion_clusters)
    
    def _merge_clustering_results(self, *clustering_results) -> Dict[str, List[int]]:
        """合并多个聚类结果"""
        self.logger.info("合并聚类结果...")
        
        merged_clusters = {}
        cluster_id = 0
        
        for i, clusters in enumerate(clustering_results):
            for cluster_name, indices in clusters.items():
                if len(indices) >= self.min_prototype_size:
                    merged_clusters[f"cluster_{cluster_id}_{cluster_name}"] = indices
                    cluster_id += 1
        
        self.logger.info(f"合并后聚类数量: {len(merged_clusters)}")
        
        return merged_clusters
    
    def _filter_quality_clusters(self, clusters: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """筛选高质量聚类"""
        self.logger.info("筛选高质量聚类...")
        
        quality_clusters = {}
        
        for cluster_name, indices in clusters.items():
            if len(indices) >= self.min_prototype_size:
                # 计算聚类质量分数
                quality_score = self._calculate_cluster_quality(indices)
                
                if quality_score > 0.5:  # 质量阈值
                    quality_clusters[cluster_name] = indices
        
        # 如果聚类过多，选择质量最高的几个
        if len(quality_clusters) > self.max_prototypes:
            cluster_qualities = []
            for cluster_name, indices in quality_clusters.items():
                quality = self._calculate_cluster_quality(indices)
                cluster_qualities.append((cluster_name, indices, quality))
            
            # 按质量排序，选择前几个
            cluster_qualities.sort(key=lambda x: x[2], reverse=True)
            quality_clusters = {
                name: indices for name, indices, _ in cluster_qualities[:self.max_prototypes]
            }
        
        self.logger.info(f"高质量聚类数量: {len(quality_clusters)}")
        
        return quality_clusters
    
    def _calculate_cluster_quality(self, indices: List[int]) -> float:
        """计算聚类质量分数"""
        if len(indices) < 2:
            return 0.0
        
        # 提取聚类内的特征
        cluster_features = self.feature_matrix[indices]
        
        # 计算内聚性（聚类内相似度）
        similarities = cosine_similarity(cluster_features)
        # 排除对角线（自相似度）
        mask = np.ones(similarities.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        
        cohesion = np.mean(similarities[mask])
        
        # 考虑聚类大小（适中的大小更好）
        size_score = min(len(indices) / 100, 1.0)  # 归一化到[0,1]
        
        # 综合质量分数
        quality_score = 0.7 * cohesion + 0.3 * size_score
        
        return quality_score
    
    def extract_prototypes(self, clusters: Dict[str, List[int]]):
        """从聚类中提取原型"""
        self.logger.info("从聚类中提取原型...")
        
        self.prototypes = []
        
        for cluster_name, indices in clusters.items():
            prototype = self._generate_prototype(cluster_name, indices)
            if prototype:
                self.prototypes.append(prototype)
        
        self.stats['prototypes_count'] = len(self.prototypes)
        
        self.logger.info(f"原型提取完成")
        self.logger.info(f"   提取原型数: {len(self.prototypes)}")
        
        # 输出原型摘要
        for i, prototype in enumerate(self.prototypes):
            self.logger.info(f"   原型{i}: {prototype['name']}, "
                           f"代表{len(prototype['subgraph_indices'])}个子图, "
                           f"质量分数{prototype['quality_score']:.3f}")
    
    def _generate_prototype(self, cluster_name: str, indices: List[int]) -> Optional[Dict]:
        """生成单个原型"""
        if not indices:
            return None
        
        # 获取聚类中的子图
        cluster_subgraphs = [self.bullying_subgraphs[i] for i in indices]
        
        # 计算代表性特征
        representative_features = self._calculate_representative_features(cluster_subgraphs)
        
        # 选择最具代表性的子图作为原型示例
        exemplar_idx = self._select_exemplar(indices)
        exemplar_subgraph = self.bullying_subgraphs[exemplar_idx]
        
        # 计算原型质量
        quality_score = self._calculate_cluster_quality(indices)
        
        prototype = {
            'name': cluster_name,
            'cluster_id': cluster_name,
            'subgraph_indices': indices,
            'subgraph_count': len(indices),
            'exemplar_subgraph': exemplar_subgraph,
            'representative_features': representative_features,
            'quality_score': quality_score,
            'creation_time': datetime.now().isoformat()
        }
        
        return prototype
    
    def _calculate_representative_features(self, subgraphs: List[Dict]) -> Dict:
        """计算聚类的代表性特征"""
        features = {
            'avg_size': np.mean([sg.get('size', 0) for sg in subgraphs]),
            'avg_emotion_score': np.mean([sg.get('emotion_score', 0) for sg in subgraphs]),
            'size_range': [
                min(sg.get('size', 0) for sg in subgraphs),
                max(sg.get('size', 0) for sg in subgraphs)
            ],
            'emotion_range': [
                min(sg.get('emotion_score', 0) for sg in subgraphs),
                max(sg.get('emotion_score', 0) for sg in subgraphs)
            ]
        }
        
        # 节点组成统计
        node_types = ['comment', 'user', 'word', 'video']
        for node_type in node_types:
            counts = [sg.get('node_counts', {}).get(node_type, 0) for sg in subgraphs]
            features[f'avg_{node_type}_count'] = np.mean(counts)
            features[f'{node_type}_range'] = [min(counts), max(counts)]
        
        return features
    
    def _select_exemplar(self, indices: List[int]) -> int:
        """选择最具代表性的子图作为原型示例"""
        if len(indices) == 1:
            return indices[0]
        
        # 计算聚类中心
        cluster_features = self.feature_matrix[indices]
        cluster_center = np.mean(cluster_features, axis=0)
        
        # 找到最接近中心的子图
        distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
        closest_idx = np.argmin(distances)
        
        return indices[closest_idx]
    
    def save_results(self, output_dir: str = "data/prototypes"):
        """保存原型提取结果"""
        self.logger.info(f"保存原型提取结果到: {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原型
        prototypes_file = output_path / f"extracted_prototypes_v5_{timestamp}.pkl"
        with open(prototypes_file, 'wb') as f:
            pickle.dump(self.prototypes, f)
        
        # 保存统计信息
        stats_file = output_path / f"extraction_stats_v5_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        # 保存原型摘要
        summary_file = output_path / f"prototype_summary_v5_{timestamp}.json"
        summary = self._generate_summary()
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果保存完成")
        self.logger.info(f"   原型文件: {prototypes_file}")
        self.logger.info(f"   统计文件: {stats_file}")
        self.logger.info(f"   摘要文件: {summary_file}")
        
        return {
            'prototypes_file': str(prototypes_file),
            'stats_file': str(stats_file),
            'summary_file': str(summary_file)
        }
    
    def _generate_summary(self) -> Dict:
        """生成原型提取摘要"""
        summary = {
            'extraction_info': {
                'timestamp': datetime.now().isoformat(),
                'total_subgraphs': self.stats['total_subgraphs'],
                'clustered_subgraphs': self.stats['clustered_subgraphs'],
                'prototypes_count': self.stats['prototypes_count'],
                'processing_time': self.stats['processing_time']
            },
            'prototypes': []
        }
        
        for prototype in self.prototypes:
            proto_summary = {
                'name': prototype['name'],
                'subgraph_count': prototype['subgraph_count'],
                'quality_score': prototype['quality_score'],
                'representative_features': prototype['representative_features']
            }
            summary['prototypes'].append(proto_summary)
        
        return summary
    
    def run_full_extraction(self, bullying_dir: str = "data/subgraphs/bullying_new",
                          output_dir: str = "data/prototypes") -> Dict:
        """运行完整的原型提取流程"""
        self.logger.info("开始完整的原型提取流程...")
        
        start_time = datetime.now()
        
        # 1. 加载霸凌子图数据
        if not self.load_bullying_subgraphs(bullying_dir):
            return {'success': False, 'error': '霸凌子图数据加载失败'}
        
        # 2. 提取特征
        self.extract_features()
        
        # 3. 聚类
        clusters = self.cluster_subgraphs()
        
        # 4. 提取原型
        self.extract_prototypes(clusters)
        
        # 5. 保存结果
        save_info = self.save_results(output_dir)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"原型提取流程完成，总耗时: {total_time:.1f}秒")
        
        return {
            'success': True,
            'prototypes_count': len(self.prototypes),
            'total_time': total_time,
            'files': save_info,
            'stats': self.stats
        }

def main():
    """主函数"""
    print("=== ProtoBully 原型提取器 V5 ===")
    
    # 配置
    config = {
        'min_prototype_size': 50,
        'max_prototypes': 5,
        'dbscan_eps': 0.3,
        'dbscan_min_samples': 10
    }
    
    # 创建提取器
    extractor = PrototypeExtractorV5(config)
    
    # 运行提取
    result = extractor.run_full_extraction()
    
    if result['success']:
        print(f"\n✅ 原型提取成功!")
        print(f"   提取原型数: {result['prototypes_count']}")
        print(f"   处理时间: {result['total_time']:.1f}秒")
        print(f"   输出文件: {result['files']}")
    else:
        print(f"\n❌ 原型提取失败: {result.get('error', '未知错误')}")

if __name__ == "__main__":
    main() 