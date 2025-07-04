#!/usr/bin/env python3
"""
霸凌子图筛选器（Bullying Subgraph Filter）- 全面优化版

优化内容：
1. 集成BERT情感分析器和用户攻击性分析器
2. BERT和规则分析的加权组合
3. 支持评论文本的深度语义分析
4. 预计算用户攻击性分析以提高效率
5. 更准确的攻击性评论统计
"""

import os
import pickle
import logging
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
import time
import json
from datetime import datetime
from pathlib import Path

# 导入新的分析器
try:
    from ..utils.bert_emotion_analyzer import BERTEmotionAnalyzer
    from ..utils.user_aggression_analyzer import UserAggressionAnalyzer
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from bert_emotion_analyzer import BERTEmotionAnalyzer
    from user_aggression_analyzer import UserAggressionAnalyzer

class BullyingSubgraphFilter:
    """霸凌子图筛选器 - 全面优化版，集成BERT和用户分析"""
    
    def __init__(self, config: Dict = None):
        """
        初始化霸凌子图筛选器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 情感分数阈值配置 (使用正数阈值，因为新的分析器输出正数)
        self.emotion_threshold = self.config.get('emotion_threshold', 0.4)  # 改为正数
        self.min_attack_comments = self.config.get('min_attack_comments', 1)
        
        # 权重配置
        self.comment_weight = self.config.get('comment_weight', 0.6)
        self.user_weight = self.config.get('user_weight', 0.3)
        self.word_weight = self.config.get('word_weight', 0.1)
        
        # 攻击性判断阈值
        self.attack_word_threshold = self.config.get('attack_word_threshold', 0.1)
        self.uppercase_threshold = self.config.get('uppercase_threshold', 0.3)
        
        # BERT分析配置
        self.use_bert_analysis = self.config.get('use_bert_analysis', True)
        self.bert_weight = self.config.get('bert_weight', 0.7)  # BERT分析的权重
        self.rule_weight = self.config.get('rule_weight', 0.3)  # 规则分析的权重
        
        # 用户分析配置
        self.use_user_analysis = self.config.get('use_user_analysis', True)
        self.user_aggression_threshold = self.config.get('user_aggression_threshold', 0.3)
        
        # 批处理配置
        self.batch_size = self.config.get('batch_size', 100)
        
        # 日志
        self.logger = self._setup_logger()
        
        # 数据存储
        self.filtered_subgraphs = {}  # session_id -> bullying_subgraphs
        
        # 初始化分析器
        self.bert_analyzer = None
        self.user_analyzer = None
        
        if self.use_bert_analysis:
            try:
                self.bert_analyzer = BERTEmotionAnalyzer()
                self.logger.info("BERT情感分析器已初始化")
            except Exception as e:
                self.logger.warning(f"BERT分析器初始化失败: {e}")
                self.use_bert_analysis = False
        
        if self.use_user_analysis:
            try:
                self.user_analyzer = UserAggressionAnalyzer(self.bert_analyzer)
                self.logger.info("用户攻击性分析器已初始化")
            except Exception as e:
                self.logger.warning(f"用户分析器初始化失败: {e}")
                self.use_user_analysis = False
        
        # 评论文本缓存
        self.comment_texts = {}
        self.comment_texts_loaded = False
        
        # 用户攻击性缓存
        self.user_aggression_cache = {}
        
    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger('BullyingSubgraphFilter')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def filter_all_subgraphs(self, graph: HeteroData, 
                           input_dir: str = "data/subgraphs/universal_optimized",
                           output_dir: str = "data/subgraphs/bullying_optimized",
                           comment_text_file: str = "data/processed/text/processed_text_data.json") -> Dict[str, Any]:
        """
        筛选所有子图中的霸凌子图 - 全面优化版
        
        Args:
            graph: 异构图对象
            input_dir: 输入子图目录
            output_dir: 输出霸凌子图目录
            comment_text_file: 评论文本数据文件
            
        Returns:
            筛选统计信息
        """
        self.logger.info("=== 开始全面优化版霸凌子图筛选 ===")
        self.logger.info(f"集成BERT分析: {self.use_bert_analysis}")
        self.logger.info(f"集成用户分析: {self.use_user_analysis}")
        self.logger.info(f"情感阈值: {self.emotion_threshold}")
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载评论文本数据
        if self.use_bert_analysis:
            self._load_comment_texts(comment_text_file)
        
        # 预计算用户攻击性分析
        if self.use_user_analysis:
            self._precompute_user_analysis(graph)
        
        # 获取所有会话子图文件
        input_path = Path(input_dir)
        session_files = list(input_path.glob("media_session_*_subgraphs.pkl"))
        
        self.logger.info(f"发现 {len(session_files)} 个会话子图文件")
        
        # 筛选每个会话的霸凌子图
        total_input_subgraphs = 0
        total_bullying_subgraphs = 0
        session_stats = []
        
        for i, session_file in enumerate(session_files):
            if i % 20 == 0:
                self.logger.info(f"筛选进度: {i}/{len(session_files)} - 已筛选 {total_bullying_subgraphs} 个霸凌子图")
            
            # 加载会话子图
            session_subgraphs = self._load_session_subgraphs(str(session_file))
            if not session_subgraphs:
                continue
            
            session_id = session_file.stem.replace('_subgraphs', '')
            total_input_subgraphs += len(session_subgraphs)
            
            # 筛选霸凌子图
            bullying_subgraphs = self._filter_session_bullying_subgraphs_comprehensive(
                session_id, session_subgraphs, graph
            )
            
            if bullying_subgraphs:
                self.filtered_subgraphs[session_id] = bullying_subgraphs
                total_bullying_subgraphs += len(bullying_subgraphs)
                
                # 保存该会话的霸凌子图
                self._save_bullying_subgraphs(session_id, bullying_subgraphs, output_dir)
                
                # 记录会话统计
                session_stats.append({
                    'session_id': session_id,
                    'input_subgraphs': len(session_subgraphs),
                    'bullying_subgraphs': len(bullying_subgraphs),
                    'filtering_ratio': len(bullying_subgraphs) / len(session_subgraphs)
                })
        
        # 保存筛选索引和详细统计
        self._save_comprehensive_filtering_stats(output_dir, session_stats)
        
        # 统计信息
        filtering_ratio = total_bullying_subgraphs / total_input_subgraphs if total_input_subgraphs > 0 else 0
        
        stats = {
            'total_input_subgraphs': total_input_subgraphs,
            'total_bullying_subgraphs': total_bullying_subgraphs,
            'filtering_ratio': filtering_ratio,
            'sessions_with_bullying': len(self.filtered_subgraphs),
            'emotion_threshold': self.emotion_threshold,
            'analysis_methods': {
                'bert_analysis': self.use_bert_analysis,
                'user_analysis': self.use_user_analysis,
                'bert_weight': self.bert_weight if self.use_bert_analysis else 0,
                'rule_weight': self.rule_weight
            },
            'filtering_timestamp': datetime.now().isoformat(),
            'session_statistics': {
                'avg_filtering_ratio': np.mean([s['filtering_ratio'] for s in session_stats]) if session_stats else 0,
                'max_filtering_ratio': np.max([s['filtering_ratio'] for s in session_stats]) if session_stats else 0,
                'min_filtering_ratio': np.min([s['filtering_ratio'] for s in session_stats]) if session_stats else 0
            }
        }
        
        self.logger.info(f"全面优化版筛选完成:")
        self.logger.info(f"  - 输入子图: {total_input_subgraphs}")
        self.logger.info(f"  - 霸凌子图: {total_bullying_subgraphs}")
        self.logger.info(f"  - 筛选率: {filtering_ratio:.1%}")
        self.logger.info(f"  - 有霸凌的会话: {len(self.filtered_subgraphs)}")
        
        return stats
    
    def _load_comment_texts(self, comment_text_file: str):
        """加载评论文本数据"""
        try:
            self.logger.info("加载评论文本数据...")
            with open(comment_text_file, 'r', encoding='utf-8') as f:
                text_data = json.load(f)
            
            # 构建评论ID到文本的映射
            for item in text_data.get('comments', []):
                comment_id = str(item.get('comment_id', ''))
                text = item.get('processed_text', '') or item.get('text', '')
                if comment_id and text:
                    self.comment_texts[comment_id] = text
            
            self.comment_texts_loaded = True
            self.logger.info(f"加载了 {len(self.comment_texts)} 条评论文本")
            
        except Exception as e:
            self.logger.error(f"加载评论文本失败: {e}")
            self.comment_texts_loaded = False
    
    def _precompute_user_analysis(self, graph: HeteroData):
        """预计算用户攻击性分析"""
        if not self.use_user_analysis or not self.user_analyzer:
            return
        
        try:
            self.logger.info("开始预计算用户攻击性分析...")
            
            # 预计算所有用户的攻击性
            self.user_aggression_cache = self.user_analyzer.precompute_all_users(graph)
            
            self.logger.info(f"预计算完成，缓存了 {len(self.user_aggression_cache)} 个用户的攻击性分数")
            
        except Exception as e:
            self.logger.error(f"预计算用户攻击性失败: {e}")
            self.use_user_analysis = False
    
    def _load_session_subgraphs(self, session_file: str) -> List[Dict[str, Any]]:
        """加载会话子图"""
        try:
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)
            return session_data.get('subgraphs', [])
        except Exception as e:
            self.logger.error(f"加载会话子图失败 {session_file}: {e}")
            return []
    
    def _filter_session_bullying_subgraphs_comprehensive(self, session_id: str, 
                                                       session_subgraphs: List[Dict[str, Any]], 
                                                       graph: HeteroData) -> List[Dict[str, Any]]:
        """
        筛选会话中的霸凌子图 - 全面优化版
        
        Args:
            session_id: 会话ID
            session_subgraphs: 会话子图列表
            graph: 异构图对象
            
        Returns:
            霸凌子图列表
        """
        bullying_subgraphs = []
        
        for subgraph in session_subgraphs:
            # 计算全面优化版情感分数
            emotion_score = self._calculate_subgraph_emotion_score_comprehensive(subgraph, graph)
            subgraph['emotion_score'] = emotion_score
            
            # 判断是否为霸凌子图
            if self._is_bullying_subgraph_comprehensive(subgraph, graph):
                bullying_subgraphs.append(subgraph)
        
        return bullying_subgraphs
    
    def _calculate_subgraph_emotion_score_comprehensive(self, subgraph: Dict[str, Any], graph: HeteroData) -> float:
        """
        计算子图的情感分数 - 全面优化版，集成BERT和用户分析
        
        Args:
            subgraph: 子图数据
            graph: 异构图对象
            
        Returns:
            情感分数（越负越攻击性强）
        """
        nodes = subgraph.get('nodes', {})
        
        # 初始化各类型节点分数
        comment_score = 0.0
        user_score = 0.0
        word_score = 0.0
        
        # 1. Comment节点分数（集成BERT分析）
        comment_nodes = nodes.get('comment', [])
        if comment_nodes:
            comment_score = self._calculate_comment_emotion_score_comprehensive(comment_nodes, graph)
        
        # 2. User节点分数（集成用户攻击性分析）
        user_nodes = nodes.get('user', [])
        if user_nodes:
            user_score = self._calculate_user_emotion_score_comprehensive(user_nodes, graph)
        
        # 3. Word节点分数（保持原有逻辑）
        word_nodes = nodes.get('word', [])
        if word_nodes:
            word_score = self._calculate_word_emotion_score(word_nodes, graph)
        
        # 4. 加权平均（归一化）
        total_nodes = sum(len(node_list) for node_type, node_list in nodes.items() 
                         if node_type != 'video')  # 排除视频节点
        
        if total_nodes == 0:
            return 0.0
        
        comment_weight_norm = len(comment_nodes) / total_nodes * self.comment_weight
        user_weight_norm = len(user_nodes) / total_nodes * self.user_weight
        word_weight_norm = len(word_nodes) / total_nodes * self.word_weight
        
        # 归一化权重
        total_weight = comment_weight_norm + user_weight_norm + word_weight_norm
        if total_weight == 0:
            return 0.0
        
        emotion_score = (
            comment_score * comment_weight_norm +
            user_score * user_weight_norm +
            word_score * word_weight_norm
        ) / total_weight
        
        # 根据子图大小进行微调（大子图可能包含更多攻击性内容）
        size_factor = min(total_nodes / 15.0, 1.0)  # 15个节点为基准
        emotion_score = emotion_score * (1 + size_factor * 0.1)
        
        return emotion_score
    
    def _calculate_comment_emotion_score_comprehensive(self, comment_indices: List[int], graph: HeteroData) -> float:
        """基于BERT和规则的评论情感分数计算 - 全面优化版"""
        if not comment_indices or 'comment' not in graph.node_types:
            return -0.3
        
        comment_features = graph['comment'].x
        
        # 提取评论文本
        comment_texts = []
        valid_comment_indices = []
        
        for comment_idx in comment_indices:
            if comment_idx < comment_features.shape[0]:
                # 获取评论ID（假设在特征的最后一列）
                comment_id = str(int(comment_features[comment_idx, -1].item()))
                comment_text = self.comment_texts.get(comment_id, '')
                
                if comment_text:
                    comment_texts.append(comment_text)
                    valid_comment_indices.append(comment_idx)
        
        if not comment_texts:
            # 如果没有文本，回退到规则方法
            return self._calculate_comment_emotion_score_rules(comment_indices, graph)
        
        # BERT分析
        bert_score = 0.0
        if self.use_bert_analysis and self.bert_analyzer:
            try:
                bert_results = self.bert_analyzer.analyze_batch_texts(comment_texts)
                # 转换为负数分数（越负越攻击性强）
                bert_aggression_scores = [result['aggression_score'] for result in bert_results]
                avg_bert_aggression = np.mean(bert_aggression_scores)
                bert_score = -0.2 - avg_bert_aggression * 0.8  # 转换为负数范围
            except Exception as e:
                self.logger.warning(f"BERT分析失败: {e}")
                bert_score = -0.3
        
        # 规则分析
        rule_score = self._calculate_comment_emotion_score_rules(valid_comment_indices, graph)
        
        # 组合BERT和规则分数
        if self.use_bert_analysis and bert_score != 0.0:
            combined_score = bert_score * self.bert_weight + rule_score * self.rule_weight
        else:
            combined_score = rule_score
        
        return combined_score
    
    def _calculate_comment_emotion_score_rules(self, comment_indices: List[int], graph: HeteroData) -> float:
        """基于规则的评论情感分数计算（原有逻辑）"""
        if not comment_indices or 'comment' not in graph.node_types:
            return -0.3
        
        comment_features = graph['comment'].x
        total_score = 0.0
        valid_comments = 0
        
        for comment_idx in comment_indices:
            if comment_idx < comment_features.shape[0]:
                features = comment_features[comment_idx]
                
                # 特征索引：[长度, 词数, 感叹号, 问号, 大写比例, 攻击词数, 攻击词比例, ID]
                if features.shape[0] >= 7:
                    exclamation_count = features[2].item()
                    question_count = features[3].item()
                    uppercase_ratio = features[4].item()
                    attack_word_count = features[5].item()
                    attack_word_ratio = features[6].item()
                    
                    # 计算攻击性分数
                    attack_score = (
                        exclamation_count * 0.1 +
                        question_count * 0.05 +
                        uppercase_ratio * 0.5 +
                        attack_word_count * 0.2 +
                        attack_word_ratio * 1.0
                    )
                    
                    # 转换为负面情感分数（越负越攻击性强）
                    comment_score = -0.2 - min(attack_score, 1.0) * 0.5
                    total_score += comment_score
                    valid_comments += 1
        
        return total_score / valid_comments if valid_comments > 0 else -0.3
    
    def _calculate_user_emotion_score_comprehensive(self, user_indices: List[int], graph: HeteroData) -> float:
        """基于用户攻击性分析的情感分数计算 - 全面优化版"""
        if not user_indices:
            return -0.3
        
        total_score = 0.0
        valid_users = 0
        
        for user_idx in user_indices:
            if self.use_user_analysis and self.user_analyzer:
                # 使用预计算的用户攻击性分数
                user_aggression_score = self.user_aggression_cache.get(user_idx, 0.0)
                # 转换为情感分数（攻击性越高，分数越负）
                user_score = -0.2 - user_aggression_score * 0.8
                total_score += user_score
                valid_users += 1
            else:
                # 回退到规则方法
                rule_score = self._calculate_user_emotion_score_rules(user_idx, graph)
                total_score += rule_score
                valid_users += 1
        
        return total_score / valid_users if valid_users > 0 else -0.3
    
    def _calculate_user_emotion_score_rules(self, user_idx: int, graph: HeteroData) -> float:
        """基于规则的用户情感分数计算（原有逻辑）"""
        if 'user' not in graph.node_types:
            return -0.3
        
        user_features = graph['user'].x
        
        if user_idx >= user_features.shape[0]:
            return -0.3
        
        features = user_features[user_idx]
        
        # 特征索引：[评论数, 平均情感, 攻击性评论数, 攻击性比例, ID]
        if features.shape[0] >= 4:
            comment_count = features[0].item()
            avg_emotion = features[1].item()
            attack_comment_count = features[2].item()
            attack_ratio = features[3].item()
            
            # 基于用户的攻击性行为计算分数
            user_score = avg_emotion  # 使用平均情感作为基础
            
            # 根据攻击性比例调整
            if attack_ratio > 0.3:  # 如果超过30%的评论是攻击性的
                user_score -= attack_ratio * 0.5
            
            # 根据攻击性评论数量调整
            if attack_comment_count > 2:
                user_score -= min(attack_comment_count * 0.1, 0.3)
            
            return max(user_score, -1.0)  # 限制最低分数
        
        return -0.3
    
    def _calculate_word_emotion_score(self, word_indices: List[int], graph: HeteroData) -> float:
        """计算词汇节点的情感分数"""
        if not word_indices or 'word' not in graph.node_types:
            return -0.3
        
        # 简单的词汇攻击性评估（基于词汇特征）
        word_features = graph['word'].x
        total_score = 0.0
        valid_words = 0
        
        for word_idx in word_indices:
            if word_idx < word_features.shape[0]:
                # 假设词汇特征包含攻击性评分
                word_score = -0.3  # 默认中性偏负
                total_score += word_score
                valid_words += 1
        
        return total_score / valid_words if valid_words > 0 else -0.3
    
    def _is_bullying_subgraph_comprehensive(self, subgraph: Dict[str, Any], graph: HeteroData) -> bool:
        """判断子图是否为霸凌子图 - 全面优化版"""
        emotion_score = subgraph.get('emotion_score', 0.0)
        nodes = subgraph.get('nodes', {})
        
        # 基本阈值检查
        if emotion_score > self.emotion_threshold:
            return False
        
        # 检查是否有评论节点
        comment_nodes = nodes.get('comment', [])
        if not comment_nodes:
            return False
        
        # 检查是否有真正的攻击性评论
        attack_comment_count = self._count_attack_comments_comprehensive(comment_nodes, graph)
        if attack_comment_count < self.min_attack_comments:
            return False
        
        return True
    
    def _count_attack_comments_comprehensive(self, comment_indices: List[int], graph: HeteroData) -> int:
        """统计攻击性评论数量 - 全面优化版，集成BERT分析"""
        if not comment_indices or 'comment' not in graph.node_types:
            return 0
        
        attack_count = 0
        comment_features = graph['comment'].x
        
        # 提取评论文本用于BERT分析
        comment_texts = []
        valid_indices = []
        
        for comment_idx in comment_indices:
            if comment_idx < comment_features.shape[0]:
                # 获取评论ID
                comment_id = str(int(comment_features[comment_idx, -1].item()))
                comment_text = self.comment_texts.get(comment_id, '')
                
                if comment_text:
                    comment_texts.append(comment_text)
                    valid_indices.append(comment_idx)
        
        # BERT分析攻击性评论
        if self.use_bert_analysis and self.bert_analyzer and comment_texts:
            try:
                bert_results = self.bert_analyzer.analyze_batch_texts(comment_texts)
                for result in bert_results:
                    if result['aggression_score'] > 0.5:  # 攻击性阈值
                        attack_count += 1
            except Exception as e:
                self.logger.warning(f"BERT攻击性分析失败: {e}")
        
        # 规则分析作为补充
        rule_attack_count = self._count_attack_comments_rules(comment_indices, graph)
        
        # 取两种方法的最大值
        return max(attack_count, rule_attack_count)
    
    def _count_attack_comments_rules(self, comment_indices: List[int], graph: HeteroData) -> int:
        """基于规则统计攻击性评论数量（原有逻辑）"""
        if not comment_indices or 'comment' not in graph.node_types:
            return 0
        
        comment_features = graph['comment'].x
        attack_count = 0
        
        for comment_idx in comment_indices:
            if comment_idx < comment_features.shape[0]:
                features = comment_features[comment_idx]
                
                if features.shape[0] >= 7:
                    uppercase_ratio = features[4].item()
                    attack_word_ratio = features[6].item()
                    
                    # 判断是否为攻击性评论
                    if (attack_word_ratio > self.attack_word_threshold or 
                        uppercase_ratio > self.uppercase_threshold):
                        attack_count += 1
        
        return attack_count
    
    def _save_bullying_subgraphs(self, session_id: str, bullying_subgraphs: List[Dict[str, Any]], output_dir: str):
        """保存霸凌子图到文件"""
        # 构建文件数据
        session_data = {
            'session_id': session_id,
            'total_bullying_subgraphs': len(bullying_subgraphs),
            'emotion_stats': self._calculate_emotion_stats(bullying_subgraphs),
            'filtering_timestamp': datetime.now().isoformat(),
            'bullying_subgraphs': bullying_subgraphs
        }
        
        # 保存到文件
        output_file = f"{output_dir}/{session_id}_bullying_subgraphs.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(session_data, f)
        
        self.logger.debug(f"保存会话 {session_id} 的 {len(bullying_subgraphs)} 个霸凌子图到 {output_file}")
    
    def _calculate_emotion_stats(self, subgraphs: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算情感统计信息"""
        if not subgraphs:
            return {}
        
        emotion_scores = [sg.get('emotion_score', 0.0) for sg in subgraphs]
        
        return {
            'avg_emotion_score': float(np.mean(emotion_scores)),
            'min_emotion_score': float(np.min(emotion_scores)),
            'max_emotion_score': float(np.max(emotion_scores)),
            'std_emotion_score': float(np.std(emotion_scores))
        }
    
    def _save_comprehensive_filtering_stats(self, output_dir: str, session_stats: List[Dict[str, Any]]):
        """保存筛选索引和详细统计"""
        index_data = {
            'filtering_timestamp': datetime.now().isoformat(),
            'total_sessions_with_bullying': len(self.filtered_subgraphs),
            'total_bullying_subgraphs': sum(len(subgraphs) for subgraphs in self.filtered_subgraphs.values()),
            'session_files': {
                session_id: f"{session_id}_bullying_subgraphs.pkl"
                for session_id in self.filtered_subgraphs.keys()
            },
            'filtering_config': {
                'emotion_threshold': self.emotion_threshold,
                'min_attack_comments': self.min_attack_comments,
                'comment_weight': self.comment_weight,
                'user_weight': self.user_weight,
                'word_weight': self.word_weight,
                'attack_word_threshold': self.attack_word_threshold,
                'uppercase_threshold': self.uppercase_threshold,
                'use_bert_analysis': self.use_bert_analysis,
                'use_user_analysis': self.use_user_analysis,
                'bert_weight': self.bert_weight,
                'rule_weight': self.rule_weight
            },
            'size_statistics': self._calculate_bullying_size_statistics(),
            'session_statistics': session_stats,
            'analysis_methods': {
                'bert_available': self.bert_analyzer is not None,
                'user_analysis_available': self.user_analyzer is not None,
                'comment_texts_loaded': self.comment_texts_loaded,
                'user_cache_size': len(self.user_aggression_cache)
            }
        }
        
        index_file = f"{output_dir}/bullying_subgraph_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"保存霸凌子图索引到 {index_file}")
    
    def _calculate_bullying_size_statistics(self) -> Dict[str, Any]:
        """计算霸凌子图大小统计"""
        all_sizes = []
        for subgraphs in self.filtered_subgraphs.values():
            all_sizes.extend(sg.get('total_nodes', sg.get('size', 0)) for sg in subgraphs)
        
        if not all_sizes:
            return {}
        
        return {
            'size_distribution': dict(Counter(all_sizes)),
            'avg_size': float(np.mean(all_sizes)),
            'min_size': int(min(all_sizes)),
            'max_size': int(max(all_sizes)),
            'total_bullying_subgraphs': len(all_sizes)
                 }
        if not user_indices:
            return -0.3
        
        total_score = 0.0
        valid_users = 0
        
        for user_idx in user_indices:
            if self.use_user_analysis and self.user_analyzer:
                # 使用预计算的用户攻击性分数
                user_aggression_score = self.user_aggression_cache.get(user_idx, -0.3)
                # 转换为情感分数（攻击性越高，分数越负）
                user_score = -0.2 - user_aggression_score * 0.8
                total_score += user_score
                valid_users += 1
            else:
                # 回退到原有的规则方法
                user_score = self._calculate_user_emotion_score_rules(user_idx, graph)
                total_score += user_score
                valid_users += 1
        
        return total_score / valid_users if valid_users > 0 else -0.3
    
    def _calculate_user_emotion_score_rules(self, user_idx: int, graph: HeteroData) -> float:
        """基于规则的用户情感分数计算（原有逻辑）"""
        if ('user', 'posts', 'comment') not in graph.edge_types:
            return -0.3
        
        user_comment_edges = graph[('user', 'posts', 'comment')].edge_index
        comment_features = graph['comment'].x
        
        # 找到该用户发布的所有评论
        user_comments = user_comment_edges[1][user_comment_edges[0] == user_idx]
        
        if len(user_comments) == 0:
            return -0.3
        
        # 计算该用户攻击性评论的比例
        attack_comments = 0
        for comment_idx in user_comments:
            if comment_idx < comment_features.shape[0]:
                features = comment_features[comment_idx]
                if features.shape[0] >= 7:
                    attack_word_ratio = features[6].item()
                    uppercase_ratio = features[4].item()
                    
                    # 判断是否为攻击性评论
                    if attack_word_ratio > self.attack_word_threshold or uppercase_ratio > self.uppercase_threshold:
                        attack_comments += 1
        
        # 计算攻击性比例
        attack_ratio = attack_comments / len(user_comments)
        
        # 转换为情感分数（攻击性越高，分数越负）
        user_score = -0.2 - attack_ratio * 0.8
        return user_score
    
    def _calculate_word_emotion_score(self, word_indices: List[int], graph: HeteroData) -> float:
        """基于词汇的攻击性倾向计算情感分数（保持原有逻辑）"""
        # 简化实现：基于词汇数量和分布
        if not word_indices:
            return -0.3
        
        # 假设更多的词汇表示更复杂的情感表达
        word_density = len(word_indices) / 10.0  # 10个词汇为基准
        word_score = -0.3 - min(word_density, 1.0) * 0.2
        
        return word_score
    
    def _is_bullying_subgraph_comprehensive(self, subgraph: Dict[str, Any], graph: HeteroData) -> bool:
        """判断子图是否为霸凌子图 - 全面优化版"""
        emotion_score = subgraph.get('emotion_score', 0.0)
        nodes = subgraph.get('nodes', {})
        
        # 基本条件：情感分数低于阈值
        if emotion_score > self.emotion_threshold:
            return False
        
        # 额外条件：至少包含一定数量的攻击性评论
        comment_nodes = nodes.get('comment', [])
        if len(comment_nodes) < self.min_attack_comments:
            return False
        
        # 检查是否有真正的攻击性评论
        attack_comment_count = self._count_attack_comments_comprehensive(comment_nodes, graph)
        if attack_comment_count < self.min_attack_comments:
            return False
        
        return True
    
    def _count_attack_comments_comprehensive(self, comment_indices: List[int], graph: HeteroData) -> int:
        """统计攻击性评论数量 - 全面优化版，集成BERT分析"""
        if not comment_indices or 'comment' not in graph.node_types:
            return 0
        
        comment_features = graph['comment'].x
        
        # 如果使用BERT分析
        if self.use_bert_analysis and self.bert_analyzer and self.comment_texts_loaded:
            # 提取评论文本
            comment_texts = []
            for comment_idx in comment_indices:
                if comment_idx < comment_features.shape[0]:
                    comment_id = str(int(comment_features[comment_idx, -1].item()))
                    comment_text = self.comment_texts.get(comment_id, '')
                    if comment_text:
                        comment_texts.append(comment_text)
            
            if comment_texts:
                try:
                    # 使用BERT分析
                    bert_results = self.bert_analyzer.analyze_batch_texts(comment_texts)
                    attack_count = sum(1 for result in bert_results 
                                     if result['aggression_score'] > 0.3)  # 攻击性阈值
                    return attack_count
                except Exception as e:
                    self.logger.warning(f"BERT攻击性评论统计失败: {e}")
        
        # 回退到规则方法
        return self._count_attack_comments_rules(comment_indices, graph)
    
    def _count_attack_comments_rules(self, comment_indices: List[int], graph: HeteroData) -> int:
        """统计攻击性评论数量 - 规则方法（原有逻辑）"""
        if not comment_indices or 'comment' not in graph.node_types:
            return 0
        
        comment_features = graph['comment'].x
        attack_count = 0
        
        for comment_idx in comment_indices:
            if comment_idx < comment_features.shape[0]:
                features = comment_features[comment_idx]
                if features.shape[0] >= 7:
                    attack_word_ratio = features[6].item()
                    uppercase_ratio = features[4].item()
                    attack_word_count = features[5].item()
                    
                    # 攻击性评论判断条件
                    is_attack = (
                        attack_word_ratio > self.attack_word_threshold or
                        uppercase_ratio > self.uppercase_threshold or
                        attack_word_count >= 2
                    )
                    
                    if is_attack:
                        attack_count += 1
        
        return attack_count
    
    def _save_bullying_subgraphs(self, session_id: str, bullying_subgraphs: List[Dict[str, Any]], output_dir: str):
        """保存霸凌子图到文件"""
        # 构建文件数据
        session_data = {
            'session_id': session_id,
            'total_bullying_subgraphs': len(bullying_subgraphs),
            'size_distribution': Counter(sg.get('size', 0) for sg in bullying_subgraphs),
            'emotion_score_stats': self._calculate_emotion_stats(bullying_subgraphs),
            'filtering_timestamp': datetime.now().isoformat(),
            'bullying_subgraphs': bullying_subgraphs
        }
        
        # 保存到文件
        output_file = f"{output_dir}/{session_id}_bullying_subgraphs.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(session_data, f)
        
        self.logger.debug(f"保存会话 {session_id} 的 {len(bullying_subgraphs)} 个霸凌子图到 {output_file}")
    
    def _calculate_emotion_stats(self, subgraphs: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算情感分数统计"""
        if not subgraphs:
            return {}
        
        emotion_scores = [sg.get('emotion_score', 0.0) for sg in subgraphs]
        
        return {
            'mean': np.mean(emotion_scores),
            'std': np.std(emotion_scores),
            'min': np.min(emotion_scores),
            'max': np.max(emotion_scores),
            'median': np.median(emotion_scores)
        }
    
    def _save_comprehensive_filtering_stats(self, output_dir: str, session_stats: List[Dict[str, Any]]):
        """保存筛选索引和详细统计"""
        index_data = {
            'filtering_timestamp': datetime.now().isoformat(),
            'total_sessions': len(self.filtered_subgraphs),
            'total_bullying_subgraphs': sum(len(subgraphs) for subgraphs in self.filtered_subgraphs.values()),
            'session_files': {
                session_id: f"{session_id}_bullying_subgraphs.pkl"
                for session_id in self.filtered_subgraphs.keys()
            },
            'filtering_config': {
                'emotion_threshold': self.emotion_threshold,
                'min_attack_comments': self.min_attack_comments,
                'attack_word_threshold': self.attack_word_threshold,
                'uppercase_threshold': self.uppercase_threshold
            },
            'size_statistics': self._calculate_bullying_size_statistics(),
            'session_statistics': session_stats
        }
        
        index_file = f"{output_dir}/bullying_subgraph_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"保存霸凌子图索引到 {index_file}")
    
    def _calculate_bullying_size_statistics(self) -> Dict[str, Any]:
        """计算霸凌子图大小统计"""
        all_sizes = []
        all_emotion_scores = []
        
        for subgraphs in self.filtered_subgraphs.values():
            all_sizes.extend(sg.get('size', 0) for sg in subgraphs)
            all_emotion_scores.extend(sg.get('emotion_score', 0.0) for sg in subgraphs)
        
        if not all_sizes:
            return {}
        
        return {
            'size_distribution': dict(Counter(all_sizes)),
            'avg_size': np.mean(all_sizes),
            'min_size': min(all_sizes),
            'max_size': max(all_sizes),
            'total_bullying_subgraphs': len(all_sizes),
            'emotion_score_stats': {
                'mean': np.mean(all_emotion_scores),
                'std': np.std(all_emotion_scores),
                'min': np.min(all_emotion_scores),
                'max': np.max(all_emotion_scores)
            }
        }

def main():
    """主函数"""
    # 配置
    config = {
        'emotion_threshold': 0.4,
        'min_attack_comments': 1,
        'attack_word_threshold': 0.1,
        'uppercase_threshold': 0.3,
        'comment_weight': 0.5,
        'user_weight': 0.3,
        'video_weight': 0.0,  # 视频节点不参与计算
        'word_weight': 0.2
    }
    
    # 创建筛选器
    filter = BullyingSubgraphFilter(config)
    
    # 加载图数据
    print("加载异构图...")
    with open('data/graphs/heterogeneous_graph_final.pkl', 'rb') as f:
        graph = pickle.load(f)
    
    # 筛选霸凌子图
    print("开始筛选霸凌子图...")
    stats = filter.filter_all_subgraphs(
        graph, 
        input_dir="data/subgraphs/universal_optimized",
        output_dir="data/subgraphs/bullying_optimized",
        comment_text_file="data/processed/text/processed_text_data.json"
    )
    
    print("筛选完成！")
    print(f"统计信息: {stats}")

if __name__ == "__main__":
    main() 