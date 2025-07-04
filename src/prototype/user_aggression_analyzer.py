#!/usr/bin/env python3
"""
用户攻击性分析器
用于分析用户在多个会话中的攻击性行为模式
"""

import pickle
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict, Counter
from pathlib import Path
import torch
from torch_geometric.data import HeteroData

from .bert_emotion_analyzer import BERTEmotionAnalyzer

class UserAggressionAnalyzer:
    """用户攻击性分析器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化用户攻击性分析器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # 初始化BERT情感分析器
        self.bert_analyzer = BERTEmotionAnalyzer()
        
        # 攻击性阈值配置
        self.aggression_threshold = self.config.get('aggression_threshold', 0.3)
        self.min_comments_per_user = self.config.get('min_comments_per_user', 2)
        
        # 缓存用户分析结果
        self.user_analysis_cache = {}
        self.user_comments_cache = {}
        
    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger('UserAggressionAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_all_users(self, graph: HeteroData, 
                         comment_text_file: str = "data/processed/text/processed_text_data.json") -> Dict[str, Any]:
        """
        分析所有用户的攻击性行为
        
        Args:
            graph: 异构图对象
            comment_text_file: 评论文本数据文件
            
        Returns:
            用户攻击性分析结果
        """
        self.logger.info("开始分析所有用户的攻击性行为")
        
        # 加载评论文本数据
        comment_texts = self._load_comment_texts(comment_text_file)
        
        # 提取用户-评论映射
        user_comments_mapping = self._extract_user_comments_mapping(graph, comment_texts)
        
        # 分析每个用户
        user_analysis_results = {}
        total_users = len(user_comments_mapping)
        
        for i, (user_idx, user_data) in enumerate(user_comments_mapping.items()):
            if i % 50 == 0:
                self.logger.info(f"分析进度: {i}/{total_users}")
            
            # 分析单个用户
            user_analysis = self._analyze_single_user(user_idx, user_data)
            user_analysis_results[user_idx] = user_analysis
        
        # 计算全局统计
        global_stats = self._calculate_global_stats(user_analysis_results)
        
        # 保存结果
        self._save_user_analysis_results(user_analysis_results, global_stats)
        
        self.logger.info(f"用户攻击性分析完成: {len(user_analysis_results)} 个用户")
        
        return {
            'user_analysis': user_analysis_results,
            'global_stats': global_stats,
            'total_users': len(user_analysis_results)
        }
    
    def _load_comment_texts(self, comment_text_file: str) -> Dict[str, str]:
        """加载评论文本数据"""
        try:
            with open(comment_text_file, 'r', encoding='utf-8') as f:
                text_data = json.load(f)
            
            # 构建评论ID到文本的映射
            comment_texts = {}
            for item in text_data.get('comments', []):
                comment_id = str(item.get('comment_id', ''))
                text = item.get('processed_text', '') or item.get('text', '')
                if comment_id and text:
                    comment_texts[comment_id] = text
            
            self.logger.info(f"加载评论文本: {len(comment_texts)} 条")
            return comment_texts
            
        except Exception as e:
            self.logger.error(f"加载评论文本失败: {e}")
            return {}
    
    def _extract_user_comments_mapping(self, graph: HeteroData, comment_texts: Dict[str, str]) -> Dict[int, Dict[str, Any]]:
        """提取用户-评论映射关系"""
        user_comments_mapping = defaultdict(lambda: {
            'comment_indices': [],
            'comment_texts': [],
            'sessions': set()
        })
        
        # 检查图中是否有用户-评论边
        if ('user', 'posts', 'comment') not in graph.edge_types:
            self.logger.warning("图中未找到用户-评论边关系")
            return {}
        
        # 获取用户-评论边
        user_comment_edges = graph[('user', 'posts', 'comment')].edge_index
        
        # 获取评论特征（包含评论ID）
        comment_features = graph['comment'].x
        
        # 构建用户-评论映射
        for edge_idx in range(user_comment_edges.shape[1]):
            user_idx = user_comment_edges[0, edge_idx].item()
            comment_idx = user_comment_edges[1, edge_idx].item()
            
            # 获取评论ID（假设在特征的最后一列）
            if comment_idx < comment_features.shape[0]:
                comment_id = str(int(comment_features[comment_idx, -1].item()))
                
                # 获取评论文本
                comment_text = comment_texts.get(comment_id, '')
                
                if comment_text:
                    user_comments_mapping[user_idx]['comment_indices'].append(comment_idx)
                    user_comments_mapping[user_idx]['comment_texts'].append(comment_text)
                    
                    # 尝试确定会话（基于评论所属的会话）
                    session_id = self._get_comment_session(graph, comment_idx)
                    if session_id is not None:
                        user_comments_mapping[user_idx]['sessions'].add(session_id)
        
        # 过滤掉评论数量太少的用户
        filtered_mapping = {
            user_idx: user_data 
            for user_idx, user_data in user_comments_mapping.items()
            if len(user_data['comment_texts']) >= self.min_comments_per_user
        }
        
        self.logger.info(f"提取用户-评论映射: {len(filtered_mapping)} 个有效用户")
        return filtered_mapping
    
    def _get_comment_session(self, graph: HeteroData, comment_idx: int) -> int:
        """获取评论所属的会话"""
        if ('media_session', 'contains', 'comment') not in graph.edge_types:
            return None
        
        session_comment_edges = graph[('media_session', 'contains', 'comment')].edge_index
        
        # 找到包含该评论的会话
        comment_mask = session_comment_edges[1] == comment_idx
        if comment_mask.any():
            session_indices = session_comment_edges[0][comment_mask]
            return session_indices[0].item()
        
        return None
    
    def _analyze_single_user(self, user_idx: int, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析单个用户的攻击性行为"""
        comment_texts = user_data['comment_texts']
        comment_indices = user_data['comment_indices']
        sessions = user_data['sessions']
        
        # 使用BERT分析器分析所有评论
        bert_results = self.bert_analyzer.calculate_user_aggression_score(comment_texts)
        
        # 计算额外的统计信息
        analysis_result = {
            'user_idx': user_idx,
            'total_comments': len(comment_texts),
            'total_sessions': len(sessions),
            'sessions': list(sessions),
            
            # BERT分析结果
            'aggression_ratio': bert_results['aggression_ratio'],
            'avg_aggression': bert_results['avg_aggression'],
            'avg_toxicity': bert_results['avg_toxicity'],
            'aggressive_comments': bert_results['aggressive_comments'],
            
            # 额外的行为特征
            'comments_per_session': len(comment_texts) / max(len(sessions), 1),
            'aggression_level': self._classify_aggression_level(bert_results['aggression_ratio']),
            
            # 文本特征统计
            'text_stats': self._calculate_text_stats(comment_texts),
            
            # 攻击性评论示例（用于调试和验证）
            'aggressive_examples': self._extract_aggressive_examples(comment_texts, bert_results)
        }
        
        return analysis_result
    
    def _classify_aggression_level(self, aggression_ratio: float) -> str:
        """分类用户的攻击性级别"""
        if aggression_ratio >= 0.7:
            return 'high'
        elif aggression_ratio >= 0.3:
            return 'medium'
        elif aggression_ratio >= 0.1:
            return 'low'
        else:
            return 'minimal'
    
    def _calculate_text_stats(self, comment_texts: List[str]) -> Dict[str, float]:
        """计算文本统计特征"""
        if not comment_texts:
            return {}
        
        # 基本统计
        text_lengths = [len(text) for text in comment_texts]
        word_counts = [len(text.split()) for text in comment_texts]
        
        # 大写字母比例
        uppercase_ratios = [
            sum(1 for c in text if c.isupper()) / max(len(text), 1) 
            for text in comment_texts
        ]
        
        # 标点符号统计
        exclamation_counts = [text.count('!') for text in comment_texts]
        question_counts = [text.count('?') for text in comment_texts]
        
        return {
            'avg_text_length': float(np.mean(text_lengths)),
            'avg_word_count': float(np.mean(word_counts)),
            'avg_uppercase_ratio': float(np.mean(uppercase_ratios)),
            'avg_exclamation_count': float(np.mean(exclamation_counts)),
            'avg_question_count': float(np.mean(question_counts)),
            'max_text_length': float(np.max(text_lengths)),
            'max_word_count': float(np.max(word_counts))
        }
    
    def _extract_aggressive_examples(self, comment_texts: List[str], bert_results: Dict[str, Any]) -> List[str]:
        """提取攻击性评论示例"""
        # 重新分析每条评论以获取个别分数
        individual_results = self.bert_analyzer.analyze_batch_texts(comment_texts)
        
        # 提取攻击性评论
        aggressive_examples = []
        for i, (text, result) in enumerate(zip(comment_texts, individual_results)):
            if result['aggression_score'] > self.aggression_threshold:
                aggressive_examples.append({
                    'text': text,
                    'aggression_score': result['aggression_score'],
                    'toxicity_score': result['toxicity_score']
                })
        
        # 按攻击性分数排序，取前5个
        aggressive_examples.sort(key=lambda x: x['aggression_score'], reverse=True)
        return aggressive_examples[:5]
    
    def _calculate_global_stats(self, user_analysis_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """计算全局统计信息"""
        if not user_analysis_results:
            return {}
        
        # 提取所有用户的指标
        aggression_ratios = [result['aggression_ratio'] for result in user_analysis_results.values()]
        avg_aggressions = [result['avg_aggression'] for result in user_analysis_results.values()]
        total_comments = [result['total_comments'] for result in user_analysis_results.values()]
        
        # 攻击性级别分布
        aggression_levels = [result['aggression_level'] for result in user_analysis_results.values()]
        level_distribution = Counter(aggression_levels)
        
        # 全局统计
        global_stats = {
            'total_users': len(user_analysis_results),
            'aggression_ratio_stats': {
                'mean': float(np.mean(aggression_ratios)),
                'median': float(np.median(aggression_ratios)),
                'std': float(np.std(aggression_ratios)),
                'min': float(np.min(aggression_ratios)),
                'max': float(np.max(aggression_ratios))
            },
            'avg_aggression_stats': {
                'mean': float(np.mean(avg_aggressions)),
                'median': float(np.median(avg_aggressions)),
                'std': float(np.std(avg_aggressions)),
                'min': float(np.min(avg_aggressions)),
                'max': float(np.max(avg_aggressions))
            },
            'comments_stats': {
                'total_comments': sum(total_comments),
                'avg_comments_per_user': float(np.mean(total_comments)),
                'median_comments_per_user': float(np.median(total_comments))
            },
            'aggression_level_distribution': dict(level_distribution),
            'high_aggression_users': sum(1 for level in aggression_levels if level == 'high'),
            'medium_aggression_users': sum(1 for level in aggression_levels if level == 'medium')
        }
        
        return global_stats
    
    def _save_user_analysis_results(self, user_analysis_results: Dict[int, Dict[str, Any]], 
                                   global_stats: Dict[str, Any]):
        """保存用户分析结果"""
        # 创建输出目录
        output_dir = Path("data/analysis/user_aggression")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        output_file = output_dir / "user_aggression_analysis.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump({
                'user_analysis': user_analysis_results,
                'global_stats': global_stats
            }, f)
        
        # 保存JSON格式的摘要
        summary_file = output_dir / "user_aggression_summary.json"
        summary_data = {
            'global_stats': global_stats,
            'top_aggressive_users': self._get_top_aggressive_users(user_analysis_results, 10),
            'analysis_timestamp': str(Path(output_file).stat().st_mtime)
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"用户分析结果已保存到 {output_dir}")
    
    def _get_top_aggressive_users(self, user_analysis_results: Dict[int, Dict[str, Any]], 
                                 top_n: int = 10) -> List[Dict[str, Any]]:
        """获取最具攻击性的用户"""
        # 按攻击性比例排序
        sorted_users = sorted(
            user_analysis_results.items(),
            key=lambda x: x[1]['aggression_ratio'],
            reverse=True
        )
        
        top_users = []
        for user_idx, user_data in sorted_users[:top_n]:
            top_users.append({
                'user_idx': user_idx,
                'aggression_ratio': user_data['aggression_ratio'],
                'avg_aggression': user_data['avg_aggression'],
                'total_comments': user_data['total_comments'],
                'aggressive_comments': user_data['aggressive_comments'],
                'aggression_level': user_data['aggression_level']
            })
        
        return top_users
    
    def get_user_aggression_score(self, user_idx: int) -> float:
        """获取特定用户的攻击性分数"""
        if user_idx in self.user_analysis_cache:
            return self.user_analysis_cache[user_idx]['aggression_ratio']
        
        # 如果缓存中没有，返回默认值
        return 0.0
    
    def load_user_analysis_cache(self, cache_file: str = "data/analysis/user_aggression/user_aggression_analysis.pkl"):
        """加载用户分析缓存"""
        try:
            if Path(cache_file).exists():
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.user_analysis_cache = cache_data.get('user_analysis', {})
                self.logger.info(f"加载用户分析缓存: {len(self.user_analysis_cache)} 个用户")
                return True
        except Exception as e:
            self.logger.warning(f"加载用户分析缓存失败: {e}")
        
        return False 