import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import os
from collections import defaultdict, Counter
import dgl
import torch

from .bert_emotion_analyzer import BERTEmotionAnalyzer

logger = logging.getLogger(__name__)

class UserAggressionAnalyzer:
    """
    用户攻击性分析器
    分析用户在多个会话中的攻击性行为模式
    """
    
    def __init__(self, emotion_analyzer: Optional[BERTEmotionAnalyzer] = None, 
                 cache_file='data/user_aggression_cache.pkl'):
        self.emotion_analyzer = emotion_analyzer or BERTEmotionAnalyzer()
        self.cache_file = cache_file
        self.user_cache = self._load_cache()
        
        # 用户攻击性统计
        self.user_stats = defaultdict(lambda: {
            'total_comments': 0,
            'aggressive_comments': 0,
            'aggression_scores': [],
            'sessions': set(),
            'avg_aggression': 0.0,
            'max_aggression': 0.0,
            'aggression_ratio': 0.0
        })
    
    def _load_cache(self):
        """加载用户攻击性缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"加载用户攻击性缓存失败: {e}")
        return {}
    
    def _save_cache(self):
        """保存用户攻击性缓存"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.user_cache, f)
        except Exception as e:
            logger.warning(f"保存用户攻击性缓存失败: {e}")
    
    def analyze_user_from_graph(self, graph: dgl.DGLGraph, user_id: int, 
                               session_id: Optional[str] = None) -> float:
        """从图中分析单个用户的攻击性"""
        try:
            # 获取用户节点索引
            user_nodes = torch.where(graph.ndata['node_type'] == 1)[0]  # 用户节点类型为1
            user_node_idx = None
            
            # 查找对应的用户节点
            for node_idx in user_nodes:
                if graph.ndata.get('user_id') is not None:
                    if graph.ndata['user_id'][node_idx].item() == user_id:
                        user_node_idx = node_idx
                        break
            
            if user_node_idx is None:
                return 0.0
            
            # 获取用户发表的评论
            comment_nodes = torch.where(graph.ndata['node_type'] == 0)[0]  # 评论节点类型为0
            user_comments = []
            
            # 查找用户发表的评论 (通过边连接关系)
            if graph.num_edges() > 0:
                # 获取从用户到评论的边
                edges = graph.edges()
                src_nodes, dst_nodes = edges[0], edges[1]
                
                for i in range(len(src_nodes)):
                    if src_nodes[i] == user_node_idx and dst_nodes[i] in comment_nodes:
                        comment_idx = dst_nodes[i]
                        if 'comment_text' in graph.ndata:
                            comment_text = graph.ndata['comment_text'][comment_idx]
                            if isinstance(comment_text, torch.Tensor):
                                # 如果是tensor，需要解码
                                comment_text = str(comment_text.item())
                            user_comments.append(str(comment_text))
            
            # 如果没有找到评论，尝试其他方法
            if not user_comments and 'comment_text' in graph.ndata:
                # 检查是否有用户ID匹配的评论
                for comment_idx in comment_nodes:
                    if 'user_id' in graph.ndata:
                        comment_user_id = graph.ndata['user_id'][comment_idx]
                        if isinstance(comment_user_id, torch.Tensor):
                            comment_user_id = comment_user_id.item()
                        if comment_user_id == user_id:
                            comment_text = graph.ndata['comment_text'][comment_idx]
                            if isinstance(comment_text, torch.Tensor):
                                comment_text = str(comment_text.item())
                            user_comments.append(str(comment_text))
            
            if not user_comments:
                return 0.0
            
            # 分析评论的攻击性
            aggression_scores = self.emotion_analyzer.analyze_batch(user_comments)
            
            # 计算用户攻击性分数
            avg_aggression = np.mean(aggression_scores) if aggression_scores else 0.0
            max_aggression = np.max(aggression_scores) if aggression_scores else 0.0
            aggressive_count = sum(1 for score in aggression_scores if score > 0.3)
            aggression_ratio = aggressive_count / len(aggression_scores) if aggression_scores else 0.0
            
            # 综合攻击性分数 (平均分数 * 0.6 + 最大分数 * 0.2 + 攻击性比例 * 0.2)
            user_aggression = avg_aggression * 0.6 + max_aggression * 0.2 + aggression_ratio * 0.2
            
            # 更新用户统计
            self._update_user_stats(user_id, user_comments, aggression_scores, session_id)
            
            return user_aggression
            
        except Exception as e:
            logger.error(f"分析用户 {user_id} 攻击性失败: {e}")
            return 0.0
    
    def _update_user_stats(self, user_id: int, comments: List[str], 
                          aggression_scores: List[float], session_id: Optional[str] = None):
        """更新用户统计信息"""
        stats = self.user_stats[user_id]
        
        stats['total_comments'] += len(comments)
        stats['aggressive_comments'] += sum(1 for score in aggression_scores if score > 0.3)
        stats['aggression_scores'].extend(aggression_scores)
        
        if session_id:
            stats['sessions'].add(session_id)
        
        # 重新计算统计指标
        if stats['aggression_scores']:
            stats['avg_aggression'] = np.mean(stats['aggression_scores'])
            stats['max_aggression'] = np.max(stats['aggression_scores'])
            stats['aggression_ratio'] = stats['aggressive_comments'] / stats['total_comments']
    
    def analyze_users_batch(self, graph: dgl.DGLGraph, user_ids: List[int], 
                           session_id: Optional[str] = None) -> Dict[int, float]:
        """批量分析多个用户的攻击性"""
        results = {}
        
        for user_id in user_ids:
            # 检查缓存
            cache_key = f"{user_id}_{session_id}" if session_id else str(user_id)
            if cache_key in self.user_cache:
                results[user_id] = self.user_cache[cache_key]
            else:
                aggression_score = self.analyze_user_from_graph(graph, user_id, session_id)
                results[user_id] = aggression_score
                self.user_cache[cache_key] = aggression_score
        
        # 保存缓存
        if len(self.user_cache) % 50 == 0:
            self._save_cache()
        
        return results
    
    def get_user_aggression_level(self, user_id: int) -> str:
        """获取用户攻击性等级"""
        if user_id not in self.user_stats:
            return "unknown"
        
        avg_aggression = self.user_stats[user_id]['avg_aggression']
        
        if avg_aggression >= 0.7:
            return "high"
        elif avg_aggression >= 0.4:
            return "medium"
        elif avg_aggression >= 0.1:
            return "low"
        else:
            return "minimal"
    
    def get_global_stats(self) -> Dict:
        """获取全局用户攻击性统计"""
        if not self.user_stats:
            return {
                'total_users': 0,
                'avg_user_aggression': 0.0,
                'high_aggression_users': 0,
                'medium_aggression_users': 0,
                'low_aggression_users': 0,
                'minimal_aggression_users': 0
            }
        
        all_scores = []
        level_counts = {'high': 0, 'medium': 0, 'low': 0, 'minimal': 0}
        
        for user_id, stats in self.user_stats.items():
            all_scores.append(stats['avg_aggression'])
            level = self.get_user_aggression_level(user_id)
            level_counts[level] += 1
        
        return {
            'total_users': len(self.user_stats),
            'avg_user_aggression': np.mean(all_scores) if all_scores else 0.0,
            'high_aggression_users': level_counts['high'],
            'medium_aggression_users': level_counts['medium'],
            'low_aggression_users': level_counts['low'],
            'minimal_aggression_users': level_counts['minimal'],
            'aggression_distribution': {
                'mean': np.mean(all_scores) if all_scores else 0.0,
                'std': np.std(all_scores) if all_scores else 0.0,
                'min': np.min(all_scores) if all_scores else 0.0,
                'max': np.max(all_scores) if all_scores else 0.0,
                'median': np.median(all_scores) if all_scores else 0.0
            }
        }
    
    def get_user_detailed_stats(self, user_id: int) -> Dict:
        """获取用户详细统计信息"""
        if user_id not in self.user_stats:
            return {}
        
        stats = self.user_stats[user_id].copy()
        stats['sessions'] = list(stats['sessions'])  # 转换set为list以便序列化
        stats['aggression_level'] = self.get_user_aggression_level(user_id)
        
        return stats
    
    def precompute_all_users(self, graph: dgl.DGLGraph, session_id: Optional[str] = None):
        """预计算所有用户的攻击性分数"""
        try:
            # 获取所有用户ID
            user_nodes = torch.where(graph.ndata['node_type'] == 1)[0]
            user_ids = []
            
            if 'user_id' in graph.ndata:
                for node_idx in user_nodes:
                    user_id = graph.ndata['user_id'][node_idx]
                    if isinstance(user_id, torch.Tensor):
                        user_id = user_id.item()
                    user_ids.append(int(user_id))
            
            if user_ids:
                logger.info(f"开始预计算 {len(user_ids)} 个用户的攻击性分数")
                results = self.analyze_users_batch(graph, user_ids, session_id)
                logger.info(f"预计算完成，平均攻击性分数: {np.mean(list(results.values())):.3f}")
                
                # 保存缓存
                self._save_cache()
                
                return results
            else:
                logger.warning("未找到用户节点")
                return {}
                
        except Exception as e:
            logger.error(f"预计算用户攻击性失败: {e}")
            return {}
    
    def __del__(self):
        """析构函数，保存缓存"""
        try:
            self._save_cache()
        except:
            pass 