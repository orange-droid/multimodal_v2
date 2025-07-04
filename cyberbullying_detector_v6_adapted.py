#!/usr/bin/env python3
"""
霸凌检测模块V6 - 适配版本
适配V6原型提取器的数据结构，进行会话级霸凌检测
"""

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
    """霸凌检测神经网络模型"""
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
    """包装PyTorch模型以兼容sklearn接口"""
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).int().numpy()
            if predictions.ndim == 0:
                predictions = np.array([predictions])
            return predictions
    
    def predict_proba(self, X):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            proba_positive = outputs.numpy()
            if proba_positive.ndim == 0:
                proba_positive = np.array([proba_positive])
            proba_negative = 1 - proba_positive
            return np.column_stack([proba_negative, proba_positive])

class CyberbullyingDetectorV6Adapted:
    """
    霸凌检测模块V6 - 适配版本
    基于V6原型匹配结果进行会话级霸凌检测
    """
    
    def __init__(self):
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
        
        self.logger.info("CyberbullyingDetectorV6Adapted initialized")
        
        # 数据存储
        self.prototypes = []
        self.session_labels = {}
        self.universal_subgraphs = {}
        self.heterogeneous_graph = None  # 存储异构图以访问真实特征
        
        # 模型和特征
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_scaler = StandardScaler()  # 特征标准化器
        self.feature_names = []
        
        # 优化结果
        self.best_weights = None
        self.best_performance = 0.0
        
        # 权重组合候选
        self.weight_combinations = [
            (0.5, 0.3, 0.2),  # 平衡型
            (0.4, 0.4, 0.2),  # 特征重视型
            (0.4, 0.3, 0.3),  # 分层特征重视型
            (0.6, 0.2, 0.2),  # 结构重视型
            (0.8, 0.1, 0.1),  # 极端结构重视型（新增）
            (0.3, 0.5, 0.2),  # 真实特征重视型
            (0.3, 0.4, 0.3),  # 特征平衡型
        ]
    
    def _setup_logger(self):
        """设置日志系统"""
        logger = logging.getLogger('CyberbullyingDetectorV6Adapted')
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
        """加载V6霸凌原型数据"""
        self.logger.info(f"Loading V6 prototypes from {prototype_path}")
        
        try:
            with open(prototype_path, 'rb') as f:
                prototype_data = pickle.load(f)
            
            # V6原型数据是直接的列表
            self.prototypes = prototype_data
            
            self.logger.info(f"Loaded {len(self.prototypes)} V6 prototypes")
            for i, prototype in enumerate(self.prototypes):
                # V6原型数据结构适配
                subgraph_count = prototype.get('subgraph_count', 0)
                quality_metrics = prototype.get('quality_metrics', {})
                quality = quality_metrics.get('discrimination', 0.0)  # 使用区分度作为质量指标
                purity = quality_metrics.get('purity', 0.0)
                
                self.logger.info(f"Prototype {i+1}: {subgraph_count} subgraphs, "
                               f"discrimination={quality:.3f}, purity={purity:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading V6 prototypes: {e}")
            return False
    
    def load_session_labels(self, labels_path: str):
        """加载V6会话标签数据"""
        self.logger.info(f"Loading V6 session labels from {labels_path}")
        
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                session_labels_raw = json.load(f)
            
            # 转换V6标签格式为检测器期望的格式
            self.session_labels = {}
            for session_id, label in session_labels_raw.items():
                self.session_labels[session_id] = {
                    'is_bullying': int(label),  # 0 -> 0, 1 -> 1
                    'session_id': session_id
                }
            
            # 统计标签分布
            bullying_count = sum(1 for session in self.session_labels.values() 
                               if session.get('is_bullying', 0) > 0)
            total_count = len(self.session_labels)
            
            self.logger.info(f"Loaded {total_count} session labels")
            self.logger.info(f"Bullying sessions: {bullying_count} ({bullying_count/total_count*100:.1f}%)")
            self.logger.info(f"Normal sessions: {total_count-bullying_count} "
                           f"({(total_count-bullying_count)/total_count*100:.1f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading V6 session labels: {e}")
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
                
                # 处理V6子图数据格式
                if isinstance(session_data, dict) and 'subgraphs' in session_data:
                    # V6新格式：字典包含subgraphs列表
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
            return max(0.0, min(1.0, structural_sim))
            
        except Exception as e:
            self.logger.warning(f"Error calculating structural similarity: {e}")
            return 0.0
    
    # 移除伪造的情感特征提取方法 - 现在使用真实节点特征
    
    def calculate_emotion_aggregate_similarity(self, subgraph: Dict, prototype_subgraph: Dict) -> float:
        """计算基于真实特征的相似度"""
        try:
            # 基于真实节点特征计算相似度
            sg_features = self.extract_real_node_features(subgraph)
            pt_features = self.extract_real_node_features(prototype_subgraph)
            
            similarities = []
            
            # Comment特征相似度
            if sg_features['comment_features'] and pt_features['comment_features']:
                sg_comment_mean = np.array(sg_features['comment_features']).mean(axis=0)
                pt_comment_mean = np.array(pt_features['comment_features']).mean(axis=0)
                comment_sim = 1.0 - np.linalg.norm(sg_comment_mean - pt_comment_mean) / 8.0
                similarities.append(max(0.0, comment_sim))
            
            # User特征相似度  
            if sg_features['user_features'] and pt_features['user_features']:
                sg_user_mean = np.array(sg_features['user_features']).mean(axis=0)[:10]  # 取前10维
                pt_user_mean = np.array(pt_features['user_features']).mean(axis=0)[:10]
                user_sim = 1.0 - np.linalg.norm(sg_user_mean - pt_user_mean) / 10.0
                similarities.append(max(0.0, user_sim))
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating feature-based similarity: {e}")
            return 0.0
    
    def calculate_layered_emotion_similarity(self, subgraph: Dict, prototype_subgraph: Dict) -> float:
        """计算分层情感相似度"""
        try:
            # 简化版本：基于节点类型的分层情感
            sg_nodes = subgraph.get('nodes', {})
            pt_nodes = prototype_subgraph.get('nodes', {})
            
            # 计算各层的相似度
            comment_sim = 0.0
            user_sim = 0.0
            word_sim = 0.0
            
            # 评论层相似度
            sg_comments = len(sg_nodes.get('comment', []))
            pt_comments = len(pt_nodes.get('comment', []))
            if sg_comments > 0 or pt_comments > 0:
                max_comments = max(sg_comments, pt_comments, 1)
                comment_sim = 1.0 - abs(sg_comments - pt_comments) / max_comments
            
            # 用户层相似度
            sg_users = len(sg_nodes.get('user', []))
            pt_users = len(pt_nodes.get('user', []))
            if sg_users > 0 or pt_users > 0:
                max_users = max(sg_users, pt_users, 1)
                user_sim = 1.0 - abs(sg_users - pt_users) / max_users
            
            # 词汇层相似度
            sg_words = len(sg_nodes.get('word', []))
            pt_words = len(pt_nodes.get('word', []))
            if sg_words > 0 or pt_words > 0:
                max_words = max(sg_words, pt_words, 1)
                word_sim = 1.0 - abs(sg_words - pt_words) / max_words
            
            # 加权平均
            layered_sim = 0.5 * comment_sim + 0.3 * user_sim + 0.2 * word_sim
            return max(0.0, min(1.0, layered_sim))
            
        except Exception as e:
            self.logger.warning(f"Error calculating layered emotion similarity: {e}")
            return 0.0
    
    def comprehensive_similarity(self, subgraph: Dict, prototype_subgraph: Dict, 
                               weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> Dict[str, float]:
        """计算综合相似度"""
        structural_sim = self.calculate_structural_similarity(subgraph, prototype_subgraph)
        emotion_agg_sim = self.calculate_emotion_aggregate_similarity(subgraph, prototype_subgraph)
        emotion_layer_sim = self.calculate_layered_emotion_similarity(subgraph, prototype_subgraph)
        
        # 加权综合相似度
        w_struct, w_emotion_agg, w_emotion_layer = weights
        comprehensive_sim = (w_struct * structural_sim + 
                           w_emotion_agg * emotion_agg_sim + 
                           w_emotion_layer * emotion_layer_sim)
        
        return {
            'structural': structural_sim,
            'emotion_aggregate': emotion_agg_sim,
            'emotion_layered': emotion_layer_sim,
            'comprehensive': comprehensive_sim
        }
    
    def load_heterogeneous_graph(self, graph_path: str) -> bool:
        """加载异构图以访问真实节点特征"""
        try:
            self.logger.info(f"Loading heterogeneous graph from {graph_path}")
            
            with open(graph_path, 'rb') as f:
                self.heterogeneous_graph = pickle.load(f)
            
            self.logger.info("Heterogeneous graph loaded successfully")
            self.logger.info(f"Node types: {self.heterogeneous_graph.node_types}")
            
            # 打印各类型节点的特征维度
            for node_type in self.heterogeneous_graph.node_types:
                if node_type in self.heterogeneous_graph.x_dict:
                    features = self.heterogeneous_graph.x_dict[node_type]
                    self.logger.info(f"{node_type}: {features.shape[0]} nodes, {features.shape[1]} dims")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load heterogeneous graph: {e}")
            return False

    def extract_real_node_features(self, subgraph: Dict) -> Dict[str, np.ndarray]:
        """提取子图中节点的真实特征向量"""
        if self.heterogeneous_graph is None:
            self.logger.warning("Heterogeneous graph not loaded")
            return {}
        
        try:
            features = {
                'comment_features': [],
                'user_features': [],
                'word_features': [],
                'media_session_features': []
            }
            
            # 提取各类型节点的真实特征
            nodes = subgraph.get('nodes', {})
            
            # 1. Comment节点特征 (8维：攻击词数、攻击词比例等关键霸凌指标)
            if 'comment' in nodes and 'comment' in self.heterogeneous_graph.x_dict:
                comment_features_tensor = self.heterogeneous_graph.x_dict['comment']
                for comment_id in nodes['comment']:
                    if comment_id < len(comment_features_tensor):
                        comment_feat = comment_features_tensor[comment_id].numpy()
                        features['comment_features'].append(comment_feat)
            
            # 2. User节点特征 (53维：攻击性评论数、攻击性比例等行为指标)
            if 'user' in nodes and 'user' in self.heterogeneous_graph.x_dict:
                user_features_tensor = self.heterogeneous_graph.x_dict['user']
                for user_id in nodes['user']:
                    if user_id < len(user_features_tensor):
                        user_feat = user_features_tensor[user_id].numpy()
                        features['user_features'].append(user_feat)
            
            # 3. Word节点特征 (4维：语义特征)
            if 'word' in nodes and 'word' in self.heterogeneous_graph.x_dict:
                word_features_tensor = self.heterogeneous_graph.x_dict['word']
                for word_id in nodes['word']:
                    if word_id < len(word_features_tensor):
                        word_feat = word_features_tensor[word_id].numpy()
                        features['word_features'].append(word_feat)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Error extracting real node features: {e}")
            return {}

    def aggregate_node_features(self, node_features: Dict[str, List]) -> List[float]:
        """聚合节点特征为子图级特征"""
        aggregated = []
        
        # Comment特征聚合 (8维特征)
        if node_features['comment_features']:
            comment_matrix = np.array(node_features['comment_features'])
            
            # 基础统计特征 (8维平均值 + 8维最大值 = 16维)
            aggregated.extend(comment_matrix.mean(axis=0).tolist())  # 8维
            aggregated.extend(comment_matrix.max(axis=0).tolist())   # 8维
            
            # 重点关注攻击性特征（基于特征分析）
            # 假设维度4-5是攻击词相关特征
            if comment_matrix.shape[1] >= 6:
                attack_related_mean = comment_matrix[:, 4:6].mean()  # 攻击词相关平均值
                attack_related_max = comment_matrix[:, 4:6].max()    # 攻击词相关最大值
                aggregated.extend([attack_related_mean, attack_related_max])  # 2维
            else:
                aggregated.extend([0.0, 0.0])
                
            # 评论数量统计
            aggregated.append(len(node_features['comment_features']))  # 1维
        else:
            aggregated.extend([0.0] * 19)  # 16 + 2 + 1
        
        # User特征聚合 (53维特征)
        if node_features['user_features']:
            user_matrix = np.array(node_features['user_features'])
            
            # 取前10维的统计特征（避免特征维度过高）
            key_user_features = user_matrix[:, :10] if user_matrix.shape[1] >= 10 else user_matrix
            aggregated.extend(key_user_features.mean(axis=0).tolist())  # 前10维平均值
            
            # 重点关注攻击行为特征（假设维度0-1是攻击性指标）
            if user_matrix.shape[1] >= 2:
                attack_behavior_mean = user_matrix[:, 0:2].mean()  # 攻击行为平均值
                attack_behavior_max = user_matrix[:, 0:2].max()    # 攻击行为最大值
                aggregated.extend([attack_behavior_mean, attack_behavior_max])  # 2维
            else:
                aggregated.extend([0.0, 0.0])
                
            # 用户数量统计
            aggregated.append(len(node_features['user_features']))  # 1维
        else:
            # 补充空特征 (10 + 2 + 1 = 13维)
            aggregated.extend([0.0] * 13)
        
        # Word特征聚合 (4维特征)
        if node_features['word_features']:
            word_matrix = np.array(node_features['word_features'])
            
            # 词汇语义特征统计
            aggregated.extend(word_matrix.mean(axis=0).tolist())  # 4维平均值
            aggregated.extend(word_matrix.std(axis=0).tolist())   # 4维标准差
            
            # 词汇数量统计
            aggregated.append(len(node_features['word_features']))  # 1维
        else:
            aggregated.extend([0.0] * 9)  # 4 + 4 + 1
        
        return aggregated

    def extract_session_features(self, session_id: str, 
                               weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> Optional[List[float]]:
        """提取会话的综合特征（原型匹配 + 真实节点特征）"""
        if session_id not in self.universal_subgraphs:
            self.logger.warning(f"Session {session_id} not found in subgraphs")
            return None
        
        if not self.prototypes:
            self.logger.error("No prototypes loaded")
            return None
        
        if self.heterogeneous_graph is None:
            self.logger.error("Heterogeneous graph not loaded")
            return None
        
        session_subgraphs = self.universal_subgraphs[session_id]
        if not session_subgraphs:
            self.logger.warning(f"No subgraphs found for session {session_id}")
            return None
        
        features = []
        
        # 第一部分：原型匹配特征
        for prototype in self.prototypes:
            prototype_subgraph = prototype.get('representative_subgraph', {})
            if not prototype_subgraph:
                self.logger.warning(f"No representative_subgraph in prototype {prototype.get('id', 'unknown')}")
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            
            # 计算会话中所有子图与该原型的相似度
            similarities = []
            for subgraph in session_subgraphs:
                sim_scores = self.comprehensive_similarity(subgraph, prototype_subgraph, weights)
                similarities.append(sim_scores['comprehensive'])
            
            if similarities:
                max_sim = max(similarities)
                avg_sim = np.mean(similarities)
                std_sim = np.std(similarities) if len(similarities) > 1 else 0.0
                match_count = sum(1 for sim in similarities if sim > 0.5)
                match_ratio = match_count / len(similarities)
                
                features.extend([max_sim, avg_sim, std_sim, match_count, match_ratio])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 第二部分：真实节点特征聚合
        all_comment_features = []
        all_user_features = []
        all_word_features = []
        
        # 从所有子图中收集真实节点特征
        for subgraph in session_subgraphs:
            node_features = self.extract_real_node_features(subgraph)
            
            if node_features['comment_features']:
                all_comment_features.extend(node_features['comment_features'])
            if node_features['user_features']:
                all_user_features.extend(node_features['user_features'])
            if node_features['word_features']:
                all_word_features.extend(node_features['word_features'])
        
        # 聚合会话级别的真实特征
        session_node_features = {
            'comment_features': all_comment_features,
            'user_features': all_user_features,
            'word_features': all_word_features
        }
        
        real_features = self.aggregate_node_features(session_node_features)
        features.extend(real_features)
        
        # 第三部分：会话级别的图结构统计特征（保持兼容性）
        total_subgraphs = len(session_subgraphs)
        total_comments = sum(len(sg.get('nodes', {}).get('comment', [])) for sg in session_subgraphs)
        total_users = sum(len(sg.get('nodes', {}).get('user', [])) for sg in session_subgraphs)
        total_words = sum(len(sg.get('nodes', {}).get('word', [])) for sg in session_subgraphs)
        total_edges = sum(sum(len(edge_list) for edge_list in sg.get('edges', {}).values()) 
                         for sg in session_subgraphs)
        
        # 添加会话统计特征
        features.extend([
            total_subgraphs,
            total_comments,
            total_users, 
            total_words,
            total_edges,
            total_comments / total_subgraphs if total_subgraphs > 0 else 0,
            total_users / total_subgraphs if total_subgraphs > 0 else 0,
            total_words / total_subgraphs if total_subgraphs > 0 else 0,
            total_edges / total_subgraphs if total_subgraphs > 0 else 0
        ])
        
        return features
    
    def extract_all_session_features(self, weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """提取所有会话的特征"""
        self.logger.info("Extracting features for all sessions...")
        
        X = []
        y = []
        session_ids = []
        
        for session_id in self.session_labels.keys():
            if session_id in self.universal_subgraphs:
                features = self.extract_session_features(session_id, weights)
                if features is not None:
                    X.append(features)
                    y.append(self.session_labels[session_id]['is_bullying'])
                    session_ids.append(session_id)
        
        X = np.array(X)
        y = np.array(y)
        
        self.logger.info(f"Extracted features for {len(X)} sessions")
        self.logger.info(f"Feature dimension: {X.shape[1] if len(X) > 0 else 0}")
        
        # 生成特征名称
        feature_names = self._generate_feature_names()
        
        return X, y, session_ids
    
    def _generate_feature_names(self) -> List[str]:
        """生成特征名称"""
        feature_names = []
        
        # 每个原型的5个特征
        for i, prototype in enumerate(self.prototypes):
            prototype_name = prototype.get('name', f'Prototype_{i+1}')
            feature_names.extend([
                f'{prototype_name}_max_sim',
                f'{prototype_name}_avg_sim', 
                f'{prototype_name}_std_sim',
                f'{prototype_name}_match_count',
                f'{prototype_name}_match_ratio'
            ])
        
        # 真实节点特征名称
        # Comment特征 (19维)
        for i in range(8):
            feature_names.append(f'comment_mean_{i}')
        for i in range(8):
            feature_names.append(f'comment_max_{i}')
        feature_names.extend(['comment_attack_mean', 'comment_attack_max', 'comment_count'])
        
        # User特征 (13维)
        for i in range(10):
            feature_names.append(f'user_key_mean_{i}')
        feature_names.extend(['user_attack_mean', 'user_attack_max', 'user_count'])
        
        # Word特征 (9维)
        for i in range(4):
            feature_names.append(f'word_mean_{i}')
        for i in range(4):
            feature_names.append(f'word_std_{i}')
        feature_names.append('word_count')
        
        # 会话统计特征
        feature_names.extend([
            'total_subgraphs',
            'total_comments',
            'total_users',
            'total_words', 
            'total_edges',
            'avg_comments_per_subgraph',
            'avg_users_per_subgraph',
            'avg_words_per_subgraph',
            'avg_edges_per_subgraph'
        ])
        
        self.feature_names = feature_names
        return feature_names
    
    def train_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.3):
        """训练多个检测模型"""
        self.logger.info(f"Training models with {len(X)} samples, test_size={test_size}")
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 特征标准化
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        models_config = {
            'RandomForest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            }
        }
        
        results = {}
        
        for model_name, config in models_config.items():
            self.logger.info(f"Training {model_name}...")
            
            # 网格搜索
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='f1',
                n_jobs=-1
            )
            
            # 训练
            if model_name == 'LogisticRegression':
                grid_search.fit(X_train_scaled, y_train)
                y_pred = grid_search.predict(X_test_scaled)
                y_pred_proba = grid_search.predict_proba(X_test_scaled)[:, 1]
            else:
                grid_search.fit(X_train, y_train)
                y_pred = grid_search.predict(X_test)
                y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
            
            # 评估
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results[model_name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'test_labels': y_test
            }
            
            self.logger.info(f"{model_name} - Accuracy: {accuracy:.3f}, "
                           f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # 训练神经网络
        nn_result = self._train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
        if nn_result:
            results['NeuralNetwork'] = nn_result
        
        self.models = results
        return results
    
    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_test: np.ndarray, y_test: np.ndarray) -> Optional[Dict]:
        """训练神经网络模型"""
        try:
            self.logger.info("Training Neural Network...")
            
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 创建模型
            model = BullyingDetectionNN(input_dim=X_train.shape[1])
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 训练
            model.train()
            for epoch in range(100):
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 20 == 0:
                    avg_loss = total_loss / len(train_loader)
                    self.logger.info(f"Epoch {epoch+1}/100, Loss: {avg_loss:.4f}")
            
            # 评估
            model.eval()
            with torch.no_grad():
                y_pred_proba = model(X_test_tensor).numpy()
                y_pred = (y_pred_proba > 0.5).astype(int)
            
            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            self.logger.info(f"NeuralNetwork - Accuracy: {accuracy:.3f}, "
                           f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            return {
                'model': NeuralNetworkWrapper(model),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'test_labels': y_test
            }
            
        except Exception as e:
            self.logger.error(f"Error training neural network: {e}")
            return None
    
    def optimize_weights_and_train(self) -> Dict[str, Any]:
        """优化权重组合并训练最佳模型"""
        if self.heterogeneous_graph is None:
            self.logger.error("Heterogeneous graph not loaded. Please load it first.")
            return {}
            
        self.logger.info("Optimizing weight combinations...")
        
        best_f1 = 0.0
        best_weights = (0.5, 0.3, 0.2)  # 默认权重
        best_results = None
        
        for weights in self.weight_combinations:
            self.logger.info(f"Testing weights: {weights}")
            
            # 提取特征
            X, y, session_ids = self.extract_all_session_features(weights)
            
            if len(X) == 0:
                self.logger.warning(f"No features extracted for weights {weights}")
                continue
            
            # 训练模型
            results = self.train_models(X, y)
            
            # 选择最佳F1分数
            avg_f1 = np.mean([result['f1'] for result in results.values()])
            
            self.logger.info(f"Weights {weights} - Average F1: {avg_f1:.3f}")
            
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_weights = weights
                best_results = results
        
        self.best_weights = best_weights
        self.best_performance = best_f1
        self.models = best_results
        
        self.logger.info(f"Best weights: {best_weights}, Best F1: {best_f1:.3f}")
        
        return {
            'best_weights': best_weights,
            'best_f1': best_f1,
            'best_results': best_results
        }
    
    def predict_session(self, session_id: str, model_name: str = 'RandomForest') -> Dict[str, Any]:
        """预测单个会话的霸凌概率"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        features = self.extract_session_features(session_id, self.best_weights)
        if features is None:
            return {'error': f'Cannot extract features for session {session_id}'}
        
        features = np.array(features).reshape(1, -1)
        
        # 特征标准化（如果需要）
        if model_name == 'LogisticRegression':
            features = self.feature_scaler.transform(features)
        
        model = self.models[model_name]['model']
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        return {
            'session_id': session_id,
            'prediction': int(prediction),
            'probability': float(probability),
            'model': model_name
        }
    
    def save_models(self, output_dir: str):
        """保存训练好的模型"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        for model_name, model_data in self.models.items():
            if model_name != 'NeuralNetwork':  # 神经网络需要特殊处理
                model_file = output_path / f"{model_name.lower()}_model_{timestamp}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data['model'], f)
        
        # 保存特征缩放器
        scaler_file = output_path / f"feature_scaler_{timestamp}.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        # 保存特征名称
        feature_names_file = output_path / f"feature_names_{timestamp}.json"
        with open(feature_names_file, 'w', encoding='utf-8') as f:
            json.dump(self.feature_names, f, ensure_ascii=False, indent=2)
        
        # 保存配置
        config = {
            'best_weights': self.best_weights,
            'best_performance': self.best_performance,
            'timestamp': timestamp,
            'num_prototypes': len(self.prototypes) if self.prototypes else 0,
            'feature_dim': len(self.feature_names)
        }
        
        config_file = output_path / f"detector_config_{timestamp}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Models saved to {output_dir}")
        
        return {
            'output_dir': str(output_dir),
            'timestamp': timestamp,
            'files_saved': {
                'models': len(self.models),
                'scaler': str(scaler_file),
                'feature_names': str(feature_names_file),
                'config': str(config_file)
            }
        } 