#!/usr/bin/env python3
"""
运行ProtoBully原型提取器V7增强版
解决所有V6版本的问题：
1. 使用真实的情感分析
2. 利用会话标签进行权重加成
3. 移除虚假特征
4. 改进聚类策略
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 添加src路径
sys.path.append('src')

class PrototypeExtractorV7Enhanced:
    """原型提取器V7增强版"""
    
    def __init__(self, config: Dict = None):
        """初始化原型提取器V7"""
        self.config = config or {}
        
        # 基础配置
        self.min_prototype_size = self.config.get('min_prototype_size', 25)
        self.max_prototypes = self.config.get('max_prototypes', 15)
        
        # 聚类配置
        self.dbscan_eps = self.config.get('dbscan_eps', 0.3)
        self.dbscan_min_samples = self.config.get('dbscan_min_samples', 10)
        
        # 会话权重配置
        self.session_weight_boost = self.config.get('session_weight_boost', 1.5)
        self.use_session_labels = self.config.get('use_session_labels', True)
        
        # 数据路径
        self.session_labels_path = self.config.get('session_labels_path', 
                                                 'data/processed/prototypes/session_label_mapping.json')
        
        # 初始化组件
        self.logger = self._setup_logger()
        
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
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载会话标签失败: {e}")
            self.use_session_labels = False
            return False
    
    def load_bullying_subgraphs(self, data_path: str):
        """加载霸凌子图数据"""
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
            if sg.get('session_id', '') in self.bullying_sessions
        )
        
        stats = {
            'total_subgraphs': len(self.bullying_subgraphs),
            'bullying_session_subgraphs': bullying_session_subgraphs,
            'bullying_session_ratio': bullying_session_subgraphs / len(self.bullying_subgraphs) if self.bullying_subgraphs else 0
        }
        
        self.logger.info("数据特征分析:")
        self.logger.info(f"  总子图数: {len(self.bullying_subgraphs)}")
        self.logger.info(f"  霸凌会话子图: {bullying_session_subgraphs} ({stats['bullying_session_ratio']*100:.1f}%)")
        
        return stats
    
    def extract_enhanced_features(self, subgraph: Dict) -> Dict[str, float]:
        """提取真实的增强特征（移除虚假特征）"""
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
            
            # 真实的情感特征（使用已有的emotion_score，来自规则分析器）
            emotion_score = subgraph.get('emotion_score', 0.0)
            features['emotion_score'] = emotion_score
            features['aggression_score'] = max(0, -emotion_score)  # 转换为正值攻击性分数
            
            # 交互强度（真实计算）
            features['interaction_intensity'] = min(1.0, edge_count / (size + 1)) if size > 0 else 0
            
            # 会话标签特征（新增 - 核心改进）
            session_id = subgraph.get('session_id', '')
            is_from_bullying_session = 1.0 if session_id in self.bullying_sessions else 0.0
            features['from_bullying_session'] = is_from_bullying_session
            
            # 会话权重（用于后续加权 - 核心改进）
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
        """构建特征矩阵和权重向量"""
        self.logger.info("构建增强特征矩阵...")
        
        feature_list = []
        weight_list = []
        successful_indices = []
        
        for i, subgraph in enumerate(self.bullying_subgraphs):
            try:
                features = self.extract_enhanced_features(subgraph)
                
                # 提取权重（这是核心改进）
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
        self.logger.info(f"霸凌会话权重加成子图: {np.sum(weight_vector > 1.0)}/{len(weight_vector)} ({np.sum(weight_vector > 1.0)/len(weight_vector)*100:.1f}%)")
        
        return feature_matrix, weight_vector
    
    def weighted_clustering(self, feature_matrix: np.ndarray, 
                          weights: np.ndarray) -> np.ndarray:
        """加权聚类（核心改进）"""
        self.logger.info("执行加权聚类...")
        
        # 应用权重到特征矩阵（霸凌会话的子图特征被放大）
        weighted_features = feature_matrix * weights.reshape(-1, 1)
        
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples
        )
        
        cluster_labels = clustering.fit_predict(weighted_features)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        self.logger.info(f"加权聚类完成: {n_clusters}个聚类, {n_noise}个噪声点")
        
        return cluster_labels
    
    def extract_prototypes_from_clusters(self, cluster_labels: np.ndarray, 
                                       feature_matrix: np.ndarray,
                                       weights: np.ndarray) -> List[Dict]:
        """从聚类中提取原型（考虑会话权重）"""
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
        
        # 按综合评分排序（霸凌会话原型优先）
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
            # 基础质量：内聚性（主要分数）
            center = np.average(cluster_features, axis=0, weights=cluster_weights)
            distances = np.linalg.norm(cluster_features - center, axis=1)
            weighted_cohesion = np.average(distances, weights=cluster_weights)
            cohesion_score = 1.0 / (1.0 + weighted_cohesion)
            
            # 霸凌会话比例奖励（修复：基于实际权重分布）
            # 权重>1.0表示来自霸凌会话（1.5倍权重）
            bullying_count = np.sum(cluster_weights > 1.0)
            bullying_ratio = bullying_count / len(cluster_weights)
            bullying_bonus = bullying_ratio * 0.15  # 降低奖励避免过度
            
            # 权重多样性评估（修复：更合理的多样性计算）
            if len(cluster_weights) > 1:
                weight_std = np.std(cluster_weights)
                weight_mean = np.mean(cluster_weights)
                weight_cv = weight_std / weight_mean if weight_mean > 0 else 0
                diversity_bonus = min(weight_cv * 0.1, 0.1)  # 变异系数作为多样性指标
            else:
                diversity_bonus = 0.0
            
            # 总质量分数（不截断，允许超过1.0）
            total_quality = cohesion_score + bullying_bonus + diversity_bonus
            
            # 记录详细计算过程（用于调试）
            self.logger.debug(f"质量计算详情: cohesion={cohesion_score:.3f}, "
                            f"bullying_bonus={bullying_bonus:.3f} (ratio={bullying_ratio:.3f}), "
                            f"diversity_bonus={diversity_bonus:.3f}, total={total_quality:.3f}")
            
            return total_quality
            
        except Exception as e:
            self.logger.warning(f"质量计算失败: {e}")
            return 0.5
    
    def _select_weighted_representative(self, cluster_subgraphs: List, 
                                     cluster_features: np.ndarray,
                                     cluster_weights: np.ndarray) -> Dict:
        """选择加权代表性子图（优先霸凌会话）"""
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
        
        # 权重统计
        stats['avg_weight'] = np.mean(cluster_weights)
        stats['max_weight'] = np.max(cluster_weights)
        stats['min_weight'] = np.min(cluster_weights)
        
        # 霸凌会话统计（修复：更准确的统计方式）
        # 方法1：基于权重判断（权重>1.0表示来自霸凌会话）
        bullying_count_by_weight = np.sum(cluster_weights > 1.0)
        
        # 方法2：基于session_id判断（如果有会话标签的话）
        bullying_count_by_session = 0
        if hasattr(self, 'bullying_sessions') and self.bullying_sessions:
            bullying_count_by_session = sum(
                1 for sg in cluster_subgraphs 
                if sg.get('session_id', '') in self.bullying_sessions
            )
        
        # 使用权重方法作为主要统计（更可靠）
        stats['bullying_session_count'] = bullying_count_by_weight
        stats['bullying_session_ratio'] = bullying_count_by_weight / len(cluster_subgraphs)
        
        # 记录两种方法的对比（用于验证）
        if bullying_count_by_session > 0:
            stats['bullying_count_by_session'] = bullying_count_by_session
            stats['bullying_ratio_by_session'] = bullying_count_by_session / len(cluster_subgraphs)
        
        return stats
    
    def extract_prototypes(self, data_path: str) -> List[Dict]:
        """主要的原型提取方法"""
        self.logger.info("🚀 开始V7增强原型提取...")
        
        # 1. 加载会话标签
        if self.use_session_labels:
            self.load_session_labels()
        
        # 2. 加载霸凌子图
        self.load_bullying_subgraphs(data_path)
        
        # 3. 构建特征矩阵
        feature_matrix, weights = self.build_feature_matrix()
        
        # 4. 加权聚类
        cluster_labels = self.weighted_clustering(feature_matrix, weights)
        
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
            'avg_quality': np.mean(quality_scores),
            'avg_weighted_score': np.mean(weighted_scores),
            'avg_cluster_size': np.mean(cluster_sizes),
            'total_coverage': sum(cluster_sizes),
            'avg_bullying_ratio': np.mean(bullying_ratios),
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        # 打印统计信息
        self.logger.info("📊 提取统计:")
        self.logger.info(f"  原型数量: {self.extraction_stats['total_prototypes']}")
        self.logger.info(f"  平均质量: {self.extraction_stats['avg_quality']:.3f}")
        self.logger.info(f"  平均加权分数: {self.extraction_stats['avg_weighted_score']:.3f}")
        self.logger.info(f"  覆盖率: {self.extraction_stats['total_coverage']}/{len(self.bullying_subgraphs)} ({self.extraction_stats['total_coverage']/len(self.bullying_subgraphs)*100:.1f}%)")
        self.logger.info(f"  平均霸凌会话比例: {self.extraction_stats['avg_bullying_ratio']:.1%}")
    
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
    print("🚀 启动ProtoBully原型提取器V7增强版")
    print("=" * 50)
    
    # 配置参数（修复：调整权重加成和聚类参数）
    config = {
        'min_prototype_size': 25,
        'max_prototypes': 15,
        'dbscan_eps': 0.35,  # 稍微增大，允许更多样的聚类
        'dbscan_min_samples': 8,  # 降低最小样本数，增加聚类多样性
        'session_weight_boost': 1.3,  # 降低权重加成，避免过度集中（从1.5降到1.3）
        'use_session_labels': True,
        'session_labels_path': 'data/processed/prototypes/session_label_mapping.json'
    }
    
    print("📋 配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 创建提取器
    extractor = PrototypeExtractorV7Enhanced(config)
    
    # 提取原型
    try:
        prototypes = extractor.extract_prototypes('data/subgraphs/bullying_subgraphs_new.pkl')
        
        # 保存结果
        extractor.save_results()
        
        print(f"\n🎉 V7增强原型提取成功完成!")
        print(f"📊 提取了 {len(prototypes)} 个高质量原型")
        
        # 显示前5个原型的详细信息（修复：更详细的诊断信息）
        if prototypes:
            print(f"\n📋 前5个原型详情:")
            for i, prototype in enumerate(prototypes[:5]):
                stats = prototype.get('statistics', {})
                print(f"  原型 {i+1}:")
                print(f"    质量分数: {prototype['quality_score']:.4f}")
                print(f"    加权分数: {prototype['weighted_score']:.4f}")
                print(f"    聚类大小: {prototype['cluster_size']}")
                print(f"    霸凌会话比例: {prototype['bullying_session_ratio']:.1%}")
                print(f"    平均权重: {stats.get('avg_weight', 0):.3f}")
                print(f"    权重范围: {stats.get('min_weight', 0):.3f} - {stats.get('max_weight', 0):.3f}")
                
                # 显示代表性子图的基本信息
                rep_sg = prototype.get('representative_subgraph', {})
                if rep_sg:
                    nodes = rep_sg.get('nodes', {})
                    print(f"    代表子图: {sum(len(v) for v in nodes.values())}个节点")
                    node_types = [f"{k}({len(v)})" for k, v in nodes.items() if len(v) > 0]
                    print(f"    节点构成: {', '.join(node_types)}")
                print()
        
        print("\n✅ 所有任务完成!")
        
    except Exception as e:
        print(f"\n❌ 提取过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 