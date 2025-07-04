import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
import os
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 神经网络相关导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

class BullyingDetectionNN(nn.Module):
    """
    霸凌检测神经网络模型
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], dropout_rate: float = 0.3):
        super(BullyingDetectionNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


class NeuralNetworkWrapper:
    """
    包装PyTorch模型以兼容sklearn接口
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).int().numpy()
            # 确保返回的是1维数组
            if predictions.ndim == 0:
                predictions = np.array([predictions])
            return predictions
    
    def predict_proba(self, X):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            proba_positive = outputs.numpy()
            # 确保返回的是1维数组
            if proba_positive.ndim == 0:
                proba_positive = np.array([proba_positive])
            proba_negative = 1 - proba_positive
            return np.column_stack([proba_negative, proba_positive])


class CyberbullyingDetectorV6:
    """
    霸凌检测模块V6
    基于原型匹配结果进行会话级霸凌检测
    """
    
    def __init__(self, config_path: str = None):
        """初始化霸凌检测器"""
        self.logger = self._setup_logger()
        self.prototypes = None
        self.session_labels = None
        self.universal_subgraphs = {}
        self.models = {}
        self.feature_scaler = StandardScaler()
        self.feature_names = []
        
        # 多权重组合预设
        self.weight_combinations = [
            (0.5, 0.3, 0.2),  # 结构主导
            (0.4, 0.4, 0.2),  # 结构+聚合情感平衡
            (0.4, 0.3, 0.3),  # 三者相对平衡
            (0.6, 0.2, 0.2),  # 更重视结构
            (0.3, 0.5, 0.2),  # 聚合情感主导
            (0.3, 0.4, 0.3),  # 情感主导(聚合+分层)
        ]
        
        self.best_weights = (0.5, 0.3, 0.2)  # 默认权重
        self.best_performance = 0.0
        
        self.logger.info("CyberbullyingDetectorV6 initialized")
    
    def _setup_logger(self):
        """设置日志系统"""
        logger = logging.getLogger('CyberbullyingDetectorV6')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_prototypes(self, prototype_path: str):
        """加载霸凌原型数据"""
        self.logger.info(f"Loading prototypes from {prototype_path}")
        
        try:
            with open(prototype_path, 'rb') as f:
                prototype_data = pickle.load(f)
            
            if isinstance(prototype_data, dict) and 'prototypes' in prototype_data:
                self.prototypes = prototype_data['prototypes']
            else:
                self.prototypes = prototype_data
            
            self.logger.info(f"Loaded {len(self.prototypes)} prototypes")
            for i, prototype in enumerate(self.prototypes):
                # 原型数据结构: 每个原型有一个exemplar_subgraph和subgraph_indices
                subgraph_count = prototype.get('subgraph_count', 0)
                quality = prototype.get('quality_score', 0.0)
                self.logger.info(f"Prototype {i+1}: {subgraph_count} subgraphs, quality={quality:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading prototypes: {e}")
            return False
    
    def load_session_labels(self, labels_path: str):
        """加载会话标签数据"""
        self.logger.info(f"Loading session labels from {labels_path}")
        
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.session_labels = json.load(f)
            
            # 统计标签分布
            bullying_count = sum(1 for session in self.session_labels if session.get('is_bullying', 0) > 0)
            total_count = len(self.session_labels)
            
            self.logger.info(f"Loaded {total_count} session labels")
            self.logger.info(f"Bullying sessions: {bullying_count} ({bullying_count/total_count*100:.1f}%)")
            self.logger.info(f"Normal sessions: {total_count-bullying_count} ({(total_count-bullying_count)/total_count*100:.1f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading session labels: {e}")
            return False
    
    def load_universal_subgraphs(self, subgraph_dir: str):
        """加载所有会话的子图数据"""
        self.logger.info(f"Loading universal subgraphs from {subgraph_dir}")
        
        subgraph_path = Path(subgraph_dir)
        if not subgraph_path.exists():
            self.logger.error(f"Subgraph directory not found: {subgraph_dir}")
            return False
        
        try:
            total_subgraphs = 0
            session_count = 0
            
            # 遍历所有子图文件
            for subgraph_file in subgraph_path.glob("*.pkl"):
                session_id = subgraph_file.stem.replace('_subgraphs', '')
                
                with open(subgraph_file, 'rb') as f:
                    session_data = pickle.load(f)
                
                # 处理数据格式
                if isinstance(session_data, dict) and 'subgraphs' in session_data:
                    # 新格式：字典包含subgraphs列表
                    session_subgraphs = session_data['subgraphs']
                    self.universal_subgraphs[session_id] = session_subgraphs
                    total_subgraphs += len(session_subgraphs)
                elif isinstance(session_data, list):
                    # 旧格式：直接是列表
                    self.universal_subgraphs[session_id] = session_data
                    total_subgraphs += len(session_data)
                else:
                    self.logger.warning(f"Unexpected data format in {subgraph_file}: {type(session_data)}")
                    continue
                session_count += 1
                
                if session_count % 100 == 0:
                    self.logger.info(f"Loaded {session_count} sessions, {total_subgraphs} subgraphs")
            
            self.logger.info(f"Successfully loaded {total_subgraphs} subgraphs from {session_count} sessions")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading universal subgraphs: {e}")
            return False
    
    def calculate_structural_similarity(self, subgraph: Dict, prototype_subgraph: Dict) -> float:
        """计算结构相似度"""
        try:
            # 节点类型分布
            sg_nodes = subgraph.get('nodes', {})
            pt_nodes = prototype_subgraph.get('nodes', {})
            
            # 获取所有可能的节点类型
            all_node_types = set(sg_nodes.keys()) | set(pt_nodes.keys())
            
            sg_node_dist = []
            pt_node_dist = []
            
            for node_type in sorted(all_node_types):
                sg_count = len(sg_nodes.get(node_type, []))
                pt_count = len(pt_nodes.get(node_type, []))
                sg_node_dist.append(sg_count)
                pt_node_dist.append(pt_count)
            
            # 归一化
            sg_total = sum(sg_node_dist) or 1
            pt_total = sum(pt_node_dist) or 1
            sg_node_dist = [x/sg_total for x in sg_node_dist]
            pt_node_dist = [x/pt_total for x in pt_node_dist]
            
            # 计算余弦相似度
            if sum(sg_node_dist) > 0 and sum(pt_node_dist) > 0:
                node_sim = 1 - cosine(sg_node_dist, pt_node_dist)
            else:
                node_sim = 0.0
            
            # 边类型分布
            sg_edges = subgraph.get('edges', {})
            pt_edges = prototype_subgraph.get('edges', {})
            
            all_edge_types = set(sg_edges.keys()) | set(pt_edges.keys())
            
            sg_edge_dist = []
            pt_edge_dist = []
            
            for edge_type in sorted(all_edge_types):
                sg_count = len(sg_edges.get(edge_type, []))
                pt_count = len(pt_edges.get(edge_type, []))
                sg_edge_dist.append(sg_count)
                pt_edge_dist.append(pt_count)
            
            # 归一化
            sg_total = sum(sg_edge_dist) or 1
            pt_total = sum(pt_edge_dist) or 1
            sg_edge_dist = [x/sg_total for x in sg_edge_dist]
            pt_edge_dist = [x/pt_total for x in pt_edge_dist]
            
            # 计算余弦相似度
            if sum(sg_edge_dist) > 0 and sum(pt_edge_dist) > 0:
                edge_sim = 1 - cosine(sg_edge_dist, pt_edge_dist)
            else:
                edge_sim = 0.0
            
            # 加权合并
            structural_sim = 0.6 * node_sim + 0.4 * edge_sim
            return max(0.0, min(1.0, structural_sim))  # 确保在[0,1]范围内
            
        except Exception as e:
            self.logger.warning(f"Error calculating structural similarity: {e}")
            return 0.0
    
    def extract_emotion_profile(self, subgraph: Dict) -> Dict[str, float]:
        """提取子图的聚合情感特征"""
        try:
            # 使用子图级别的情感分数（如果存在）
            if 'emotion_score' in subgraph:
                emotion_score = float(subgraph['emotion_score'])
                return {
                    'avg_emotion': emotion_score,
                    'avg_comment_emotion': emotion_score,
                    'avg_user_emotion': emotion_score,
                    'emotion_variance': 0.0,
                    'negative_ratio': 1.0 if emotion_score < 0 else 0.0,
                    'extreme_negative_ratio': 1.0 if emotion_score < -0.5 else 0.0
                }
            
            # 如果没有子图级别的情感分数，使用节点数量作为代理特征
            nodes = subgraph.get('nodes', {})
            comment_count = len(nodes.get('comment', []))
            user_count = len(nodes.get('user', []))
            total_nodes = sum(len(node_list) for node_list in nodes.values())
            
            # 基于节点数量的简单情感估计
            # 更多评论节点可能意味着更多互动，可能包含更多情感
            estimated_emotion = -0.3 if comment_count > 3 else -0.1
            
            return {
                'avg_emotion': estimated_emotion,
                'avg_comment_emotion': estimated_emotion,
                'avg_user_emotion': estimated_emotion,
                'emotion_variance': 0.1,
                'negative_ratio': 0.7 if comment_count > 3 else 0.3,
                'extreme_negative_ratio': 0.2 if comment_count > 5 else 0.0
            }
            
        except Exception as e:
            self.logger.warning(f"Error extracting emotion profile: {e}")
            return {
                'avg_emotion': 0.0,
                'avg_comment_emotion': 0.0,
                'avg_user_emotion': 0.0,
                'emotion_variance': 0.0,
                'negative_ratio': 0.0,
                'extreme_negative_ratio': 0.0
            }
    
    def calculate_emotion_aggregate_similarity(self, subgraph: Dict, prototype_subgraph: Dict) -> float:
        """计算聚合情感相似度"""
        try:
            sg_profile = self.extract_emotion_profile(subgraph)
            pt_profile = self.extract_emotion_profile(prototype_subgraph)
            
            # 转换为向量
            sg_vector = list(sg_profile.values())
            pt_vector = list(pt_profile.values())
            
            # 计算余弦相似度
            if sum(abs(x) for x in sg_vector) > 0 and sum(abs(x) for x in pt_vector) > 0:
                similarity = 1 - cosine(sg_vector, pt_vector)
                return max(0.0, min(1.0, similarity))
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Error calculating emotion aggregate similarity: {e}")
            return 0.0
    
    def calculate_layered_emotion_similarity(self, subgraph: Dict, prototype_subgraph: Dict) -> float:
        """计算分层情感相似度"""
        try:
            similarities = []
            
            # 使用子图级别的情感分数
            sg_emotion = subgraph.get('emotion_score', 0.0)
            pt_emotion = prototype_subgraph.get('emotion_score', 0.0)
            
            if sg_emotion != 0.0 or pt_emotion != 0.0:
                emotion_sim = 1 - min(abs(sg_emotion - pt_emotion) / 2, 1.0)
                similarities.append(('emotion', emotion_sim, 0.8))
            
            # 节点数量相似度作为补充
            sg_nodes = subgraph.get('nodes', {})
            pt_nodes = prototype_subgraph.get('nodes', {})
            
            sg_comment_count = len(sg_nodes.get('comment', []))
            pt_comment_count = len(pt_nodes.get('comment', []))
            
            if sg_comment_count > 0 or pt_comment_count > 0:
                max_count = max(sg_comment_count, pt_comment_count, 1)
                count_sim = 1 - abs(sg_comment_count - pt_comment_count) / max_count
                similarities.append(('count', count_sim, 0.2))
            
            # 加权平均
            if similarities:
                weighted_sum = sum(sim * weight for _, sim, weight in similarities)
                total_weight = sum(weight for _, _, weight in similarities)
                return weighted_sum / total_weight
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Error calculating layered emotion similarity: {e}")
            return 0.0
    
    def comprehensive_similarity(self, subgraph: Dict, prototype_subgraph: Dict, weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> Dict[str, float]:
        """计算综合相似度"""
        struct_sim = self.calculate_structural_similarity(subgraph, prototype_subgraph)
        emotion_agg_sim = self.calculate_emotion_aggregate_similarity(subgraph, prototype_subgraph)
        emotion_layer_sim = self.calculate_layered_emotion_similarity(subgraph, prototype_subgraph)
        
        w1, w2, w3 = weights
        total_sim = w1 * struct_sim + w2 * emotion_agg_sim + w3 * emotion_layer_sim
        
        return {
            'total_similarity': total_sim,
            'structural_similarity': struct_sim,
            'emotion_aggregate_similarity': emotion_agg_sim,
            'emotion_layered_similarity': emotion_layer_sim
        }
    
    def extract_session_features(self, session_id: str, weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> Optional[List[float]]:
        """提取会话级特征向量"""
        if session_id not in self.universal_subgraphs:
            self.logger.warning(f"Session {session_id} not found in universal subgraphs")
            return None
        
        session_subgraphs = self.universal_subgraphs[session_id]
        if not session_subgraphs:
            self.logger.warning(f"No subgraphs found for session {session_id}")
            return None
        
        # 计算与所有原型的相似度
        all_similarities = []
        prototype_similarities = [[] for _ in range(len(self.prototypes))]
        
        for subgraph in session_subgraphs:
            for prototype_idx, prototype in enumerate(self.prototypes):
                # 与原型的exemplar_subgraph计算相似度
                prototype_subgraph = prototype['exemplar_subgraph']
                sim_result = self.comprehensive_similarity(subgraph, prototype_subgraph, weights)
                similarity = sim_result['total_similarity']
                
                prototype_similarities[prototype_idx].append(similarity)
                all_similarities.append(similarity)
        
        # 提取特征
        features = []
        
        # 1. 原型匹配特征 (15维)
        if all_similarities:
            features.extend([
                max(all_similarities),  # 最高相似度
                np.mean(all_similarities),  # 平均相似度
                np.std(all_similarities),  # 相似度标准差
                sum(1 for sim in all_similarities if sim > 0.5) / len(all_similarities),  # 高相似度比例(>0.5)
                sum(1 for sim in all_similarities if sim > 0.7) / len(all_similarities),  # 极高相似度比例(>0.7)
            ])
        else:
            features.extend([0.0] * 5)
        
        # 每个原型的最佳匹配相似度
        for proto_sims in prototype_similarities:
            if proto_sims:
                features.extend([
                    max(proto_sims),  # 该原型的最高相似度
                    np.mean(proto_sims),  # 该原型的平均相似度
                ])
            else:
                features.extend([0.0, 0.0])
        
        # 原型偏好度（哪个原型匹配最好）
        prototype_max_sims = [max(proto_sims) if proto_sims else 0.0 for proto_sims in prototype_similarities]
        best_prototype_idx = np.argmax(prototype_max_sims) if any(prototype_max_sims) else 0
        features.append(float(best_prototype_idx))
        
        # 2. 会话结构特征 (8维)
        total_nodes = sum(len(subgraph.get('nodes', {})) for subgraph in session_subgraphs)
        total_edges = sum(sum(len(edges) for edges in subgraph.get('edges', {}).values()) for subgraph in session_subgraphs)
        
        features.extend([
            len(session_subgraphs),  # 子图数量
            total_nodes / len(session_subgraphs) if session_subgraphs else 0.0,  # 平均节点数
            total_edges / len(session_subgraphs) if session_subgraphs else 0.0,  # 平均边数
            total_edges / total_nodes if total_nodes > 0 else 0.0,  # 平均密度
        ])
        
        # 子图大小分布
        subgraph_sizes = [sum(len(nodes) for nodes in subgraph.get('nodes', {}).values()) for subgraph in session_subgraphs]
        if subgraph_sizes:
            features.extend([
                min(subgraph_sizes),  # 最小子图大小
                max(subgraph_sizes),  # 最大子图大小
                np.std(subgraph_sizes),  # 子图大小标准差
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 节点类型多样性
        all_node_types = set()
        for subgraph in session_subgraphs:
            all_node_types.update(subgraph.get('nodes', {}).keys())
        features.append(len(all_node_types))
        
        # 3. 多模态融合特征 (5维)
        # 使用子图级别的情感分数
        subgraph_emotions = []
        for subgraph in session_subgraphs:
            emotion_score = subgraph.get('emotion_score', 0.0)
            if emotion_score != 0.0:
                subgraph_emotions.append(emotion_score)
        
        if subgraph_emotions:
            features.extend([
                np.mean(subgraph_emotions),  # 平均情感分数
                np.std(subgraph_emotions),  # 情感分数标准差
                sum(1 for e in subgraph_emotions if e < 0) / len(subgraph_emotions),  # 负面情感比例
                sum(1 for e in subgraph_emotions if e < -0.5) / len(subgraph_emotions),  # 极端负面比例
            ])
            avg_emotion = np.mean(subgraph_emotions)
        else:
            # 使用基于节点数量的估计
            comment_counts = [len(sg.get('nodes', {}).get('comment', [])) for sg in session_subgraphs]
            avg_comment_count = np.mean(comment_counts) if comment_counts else 0
            estimated_emotion = -0.3 if avg_comment_count > 3 else -0.1
            
            features.extend([
                estimated_emotion,  # 估计的平均情感分数
                0.1,  # 估计的情感分数标准差
                0.7 if avg_comment_count > 3 else 0.3,  # 估计的负面情感比例
                0.2 if avg_comment_count > 5 else 0.0,  # 估计的极端负面比例
            ])
            avg_emotion = estimated_emotion
        
        # 综合霸凌评分
        bullying_score = max(all_similarities) * 0.4 + np.mean(all_similarities) * 0.3 + avg_emotion * 0.3
        features.append(bullying_score)
        
        return features
    
    def extract_all_session_features(self, weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """提取所有会话的特征"""
        self.logger.info("Extracting features for all sessions...")
        
        session_ids = []
        features_list = []
        labels = []
        
        # 创建会话ID到标签的映射
        session_label_map = {}
        for session_data in self.session_labels:
            post_id = session_data.get('post_id', '')
            if post_id.startswith('session_'):
                # 转换 session_X 为 media_session_X
                session_num = post_id.replace('session_', '')
                session_key = f"media_session_{session_num}"
                is_bullying = session_data.get('is_bullying', 0)
                session_label_map[session_key] = 1 if is_bullying > 0 else 0
        
        # 提取特征
        processed_count = 0
        for session_id in self.universal_subgraphs.keys():
            if session_id in session_label_map:
                features = self.extract_session_features(session_id, weights)
                if features is not None:
                    session_ids.append(session_id)
                    features_list.append(features)
                    labels.append(session_label_map[session_id])
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        self.logger.info(f"Processed {processed_count} sessions")
        
        self.logger.info(f"Successfully extracted features for {len(features_list)} sessions")
        
        # 生成特征名称
        if not self.feature_names:
            self.feature_names = self._generate_feature_names()
        
        return np.array(features_list), np.array(labels), session_ids
    
    def _generate_feature_names(self) -> List[str]:
        """生成特征名称列表"""
        names = []
        
        # 原型匹配特征
        names.extend([
            'max_similarity', 'avg_similarity', 'similarity_std',
            'high_similarity_ratio_05', 'high_similarity_ratio_07'
        ])
        
        # 每个原型的特征
        for i in range(len(self.prototypes)):
            names.extend([f'prototype_{i}_max_sim', f'prototype_{i}_avg_sim'])
        
        names.append('best_prototype_idx')
        
        # 会话结构特征
        names.extend([
            'subgraph_count', 'avg_node_count', 'avg_edge_count', 'avg_density',
            'min_subgraph_size', 'max_subgraph_size', 'subgraph_size_std',
            'node_type_diversity'
        ])
        
        # 多模态融合特征
        names.extend([
            'avg_emotion_score', 'emotion_score_std', 'negative_emotion_ratio',
            'extreme_negative_ratio', 'composite_bullying_score'
        ])
        
        return names
    
    def train_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.3):
        """训练机器学习模型"""
        self.logger.info("Starting model training...")
        
        # 数据集划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 特征标准化
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        self.logger.info(f"Training set: {len(X_train)} samples")
        self.logger.info(f"Test set: {len(X_test)} samples")
        self.logger.info(f"Training set bullying ratio: {np.mean(y_train):.3f}")
        self.logger.info(f"Test set bullying ratio: {np.mean(y_test):.3f}")
        
        # RandomForest
        self.logger.info("Training RandomForest...")
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        rf_grid.fit(X_train_scaled, y_train)
        self.models['RandomForest'] = rf_grid.best_estimator_
        
        self.logger.info(f"RandomForest best params: {rf_grid.best_params_}")
        self.logger.info(f"RandomForest best CV score: {rf_grid.best_score_:.3f}")
        
        # Logistic Regression
        self.logger.info("Training Logistic Regression...")
        lr_param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'class_weight': ['balanced']
        }
        
        lr_grid = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            lr_param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        lr_grid.fit(X_train_scaled, y_train)
        self.models['LogisticRegression'] = lr_grid.best_estimator_
        
        self.logger.info(f"LogisticRegression best params: {lr_grid.best_params_}")
        self.logger.info(f"LogisticRegression best CV score: {lr_grid.best_score_:.3f}")
        
        # Neural Network
        self.logger.info("Training Neural Network...")
        nn_model = self._train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
        if nn_model is not None:
            self.models['NeuralNetwork'] = nn_model
        
        # 评估模型
        results = {}
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            results[model_name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            self.logger.info(f"\n{model_name} Performance:")
            self.logger.info(f"Accuracy: {results[model_name]['accuracy']:.3f}")
            self.logger.info(f"Precision: {results[model_name]['precision']:.3f}")
            self.logger.info(f"Recall: {results[model_name]['recall']:.3f}")
            self.logger.info(f"F1-score: {results[model_name]['f1']:.3f}")
        
        return results, (X_train_scaled, X_test_scaled, y_train, y_test)
    
    def optimize_weights_and_train(self) -> Dict[str, Any]:
        """优化权重组合并训练最佳模型"""
        self.logger.info("Starting weight optimization...")
        
        best_results = None
        best_weights = None
        best_f1 = 0.0
        
        for i, weights in enumerate(self.weight_combinations):
            self.logger.info(f"\nTesting weight combination {i+1}/{len(self.weight_combinations)}: {weights}")
            
            try:
                # 提取特征
                X, y, session_ids = self.extract_all_session_features(weights)
                
                if len(X) == 0:
                    self.logger.warning("No features extracted, skipping this weight combination")
                    continue
                
                # 训练模型
                results, data_splits = self.train_models(X, y)
                
                # 评估RandomForest性能
                rf_f1 = results['RandomForest']['f1']
                
                if rf_f1 > best_f1:
                    best_f1 = rf_f1
                    best_weights = weights
                    best_results = {
                        'weights': weights,
                        'results': results,
                        'data_splits': data_splits,
                        'session_ids': session_ids,
                        'features': X,
                        'labels': y
                    }
                    
                    self.logger.info(f"New best F1 score: {best_f1:.3f} with weights {weights}")
                
            except Exception as e:
                self.logger.error(f"Error with weight combination {weights}: {e}")
                continue
        
        if best_results:
            self.best_weights = best_weights
            self.best_performance = best_f1
            self.logger.info(f"\nOptimization completed!")
            self.logger.info(f"Best weights: {best_weights}")
            self.logger.info(f"Best F1 score: {best_f1:.3f}")
            return best_results
        else:
            self.logger.error("No successful weight combinations found!")
            return None
    
    def predict_session(self, session_id: str, model_name: str = 'RandomForest') -> Dict[str, Any]:
        """预测单个会话"""
        if model_name not in self.models:
            return {'status': 'error', 'message': f'Model {model_name} not trained'}
        
        features = self.extract_session_features(session_id, self.best_weights)
        if features is None:
            return {'status': 'error', 'message': f'Cannot extract features for session {session_id}'}
        
        # 标准化特征
        features_scaled = self.feature_scaler.transform([features])
        
        # 预测
        prediction = self.models[model_name].predict(features_scaled)[0]
        confidence = self.models[model_name].predict_proba(features_scaled)[0].max()
        
        return {
            'session_id': session_id,
            'status': 'success',
            'prediction': 'bullying' if prediction == 1 else 'normal',
            'confidence': float(confidence),
            'model_used': model_name
        }
    
    def save_models(self, output_dir: str):
        """保存训练好的模型"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        for model_name, model in self.models.items():
            model_file = output_path / f"{model_name.lower()}_model_{timestamp}.pkl"
            
            try:
                # 对于神经网络模型，保存PyTorch模型的state_dict
                if isinstance(model, NeuralNetworkWrapper):
                    torch_model_file = output_path / f"{model_name.lower()}_pytorch_model_{timestamp}.pth"
                    torch.save(model.model.state_dict(), torch_model_file)
                    self.logger.info(f"Saved {model_name} PyTorch model to {torch_model_file}")
                    
                    # 保存模型架构信息
                    model_info = {
                        'model_type': 'NeuralNetworkWrapper',
                        'input_dim': model.model.network[0].in_features,
                        'hidden_dims': [128, 64, 32],  # 默认架构
                        'dropout_rate': 0.3,
                        'state_dict_file': torch_model_file.name
                    }
                    
                    info_file = output_path / f"{model_name.lower()}_model_info_{timestamp}.json"
                    with open(info_file, 'w', encoding='utf-8') as f:
                        json.dump(model_info, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"Saved {model_name} model info to {info_file}")
                else:
                    # 对于sklearn模型，正常保存
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)
                    self.logger.info(f"Saved {model_name} model to {model_file}")
                    
            except Exception as e:
                self.logger.error(f"Error saving {model_name} model: {e}")
                continue
        
        # 保存特征标准化器
        scaler_file = output_path / f"feature_scaler_{timestamp}.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        # 保存特征名称
        feature_names_file = output_path / f"feature_names_{timestamp}.json"
        with open(feature_names_file, 'w', encoding='utf-8') as f:
            json.dump(self.feature_names, f, indent=2, ensure_ascii=False)
        
        # 保存配置
        config = {
            'best_weights': self.best_weights,
            'best_performance': self.best_performance,
            'feature_count': len(self.feature_names),
            'prototype_count': len(self.prototypes) if self.prototypes else 0,
            'timestamp': timestamp
        }
        
        config_file = output_path / f"detector_config_{timestamp}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"All models and configurations saved to {output_path}")
        return timestamp
    
    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_test: np.ndarray, y_test: np.ndarray) -> Optional[Dict]:
        """训练神经网络模型"""
        try:
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 初始化模型
            input_dim = X_train.shape[1]
            model = BullyingDetectionNN(input_dim)
            
            # 损失函数和优化器
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # 训练参数
            epochs = 100
            best_val_f1 = 0.0
            patience = 15
            patience_counter = 0
            
            # 训练循环
            model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                # 验证
                if epoch % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_test_tensor)
                        val_pred = (val_outputs > 0.5).float()
                        
                        # 计算F1分数
                        val_f1 = f1_score(y_test, val_pred.numpy())
                        val_loss = criterion(val_outputs, y_test_tensor).item()
                        
                        scheduler.step(val_loss)
                        
                        if val_f1 > best_val_f1:
                            best_val_f1 = val_f1
                            best_model_state = model.state_dict().copy()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if epoch % 20 == 0:
                            self.logger.info(f"Epoch {epoch}: Train Loss={total_loss/len(train_loader):.4f}, "
                                           f"Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}")
                    
                    model.train()
                    
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # 加载最佳模型
            model.load_state_dict(best_model_state)
            model.eval()
            
            self.logger.info(f"Neural Network training completed. Best validation F1: {best_val_f1:.4f}")
            
            # 包装模型以兼容sklearn接口
            return NeuralNetworkWrapper(model)
            
        except Exception as e:
            self.logger.error(f"Error training neural network: {e}")
            import traceback
            traceback.print_exc()
            return None 