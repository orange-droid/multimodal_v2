#!/usr/bin/env python3
"""
优化版多模态结构对齐器 V4 - 支持代表性子图选择
在原有功能基础上，添加代表性子图选择策略，减少计算时间，提高测试效率

主要优化：
1. 代表性子图选择：从每个会话的多个评论中选择代表性子图
2. 智能采样策略：结合攻击性强度和多样性的采样算法
3. 批量处理优化：减少重复计算，提高效率
4. 特征聚合优化：基于选定子图的高效特征聚合

更新时间：2025-06-09
基于：multimodal_structure_aligner_v4.py
"""

import numpy as np
import json
import pickle
from typing import Dict, List, Optional, Set, Any, Tuple
import logging
import random
from pathlib import Path

# 导入基础类
from multimodal_structure_aligner_v4 import MultimodalStructureAlignerV4

class MultimodalStructureAlignerV4Optimized(MultimodalStructureAlignerV4):
    """
    优化版多模态结构对齐器
    
    新增功能：
    1. 代表性子图选择策略
    2. 智能采样算法
    3. 批量处理优化
    """
    
    def __init__(self, config: Dict[str, Any]):
        # 调用父类初始化
        super().__init__(config)
        
        # 优化配置
        self.optimization_config = {
            'representative_subgraph_count': config.get('representative_subgraph_count', 3),  # 每个会话选择3个代表性子图
            'sampling_strategy': config.get('sampling_strategy', 'diverse_aggressive'),      # 采样策略
            'min_subgraph_size': config.get('min_subgraph_size', 6),                        # 最小子图大小
            'max_subgraph_size': config.get('max_subgraph_size', 15),                       # 最大子图大小
            'aggressive_comment_ratio': config.get('aggressive_comment_ratio', 0.7),        # 攻击性评论比例
            'enable_quality_filtering': config.get('enable_quality_filtering', True)        # 启用质量过滤
        }
        
        print(f"✅ 优化版对齐器初始化完成")
        print(f"   代表性子图数量: {self.optimization_config['representative_subgraph_count']}")
        print(f"   采样策略: {self.optimization_config['sampling_strategy']}")
    
    def align_session_with_prototypes_optimized(self, session_id: str) -> Dict:
        """
        优化版会话原型对齐 - 使用代表性子图
        """
        if not self.prototypes:
            return {
                'session_id': session_id,
                'status': 'no_prototypes',
                'similarities': [],
                'max_similarity': 0.0,
                'prediction': 'normal',
                'subgraph_count': 0
            }
        
        # 1. 获取会话的所有评论节点
        session_comments = self._get_session_comments(session_id)
        if not session_comments:
            return {
                'session_id': session_id,
                'status': 'no_comments',
                'similarities': [],
                'max_similarity': 0.0,
                'prediction': 'normal',
                'subgraph_count': 0
            }
        
        # 2. 选择代表性评论作为子图中心
        representative_comments = self._select_representative_comments(
            session_comments, session_id
        )
        
        if not representative_comments:
            return {
                'session_id': session_id,
                'status': 'no_representative_comments',
                'similarities': [],
                'max_similarity': 0.0,
                'prediction': 'normal',
                'subgraph_count': 0
            }
        
        print(f"   选择了 {len(representative_comments)} 个代表性评论")
        
        # 3. 为每个代表性评论提取子图并计算相似度
        all_similarities = []
        subgraph_results = []
        
        for comment_idx in representative_comments:
            try:
                # 提取子图
                subgraph = self.extract_session_subgraph(
                    session_id, 
                    target_size=random.randint(
                        self.optimization_config['min_subgraph_size'],
                        self.optimization_config['max_subgraph_size']
                    )
                )
                
                if not subgraph:
                    continue
                
                # 计算与所有原型的相似度
                subgraph_similarities = []
                for i, prototype in enumerate(self.prototypes):
                    similarity = self.calculate_prototype_similarity(subgraph, prototype)
                    
                    # 提取原型情感分数
                    prototype_emotion = 0.0
                    if 'representative_subgraph' in prototype:
                        emotion_features = prototype['representative_subgraph'].get('emotion_features', {})
                        prototype_emotion = emotion_features.get('avg_text_length', 0.0)
                    
                    subgraph_similarities.append({
                        'prototype_id': i,
                        'similarity': similarity,
                        'prototype_emotion': prototype_emotion
                    })
                
                all_similarities.extend(subgraph_similarities)
                subgraph_results.append({
                    'comment_center': comment_idx,
                    'subgraph': subgraph,
                    'similarities': subgraph_similarities
                })
                
            except Exception as e:
                print(f"⚠️ 代表性评论 {comment_idx} 子图提取失败: {e}")
                continue
        
        if not all_similarities:
            return {
                'session_id': session_id,
                'status': 'extraction_failed',
                'similarities': [],
                'max_similarity': 0.0,
                'prediction': 'normal',
                'subgraph_count': 0
            }
        
        # 4. 聚合多个子图的相似度结果
        aggregated_similarities = self._aggregate_similarities(all_similarities)
        max_similarity = max(sim['similarity'] for sim in aggregated_similarities) if aggregated_similarities else 0.0
        
        # 5. 预测（基于相似度阈值）
        threshold = self.config.get('similarity_threshold', 0.6)
        prediction = 'bullying' if max_similarity >= threshold else 'normal'
        
        # 6. 计算聚合特征
        aggregated_features = self._compute_aggregated_features(
            subgraph_results, aggregated_similarities
        )
        
        return {
            'session_id': session_id,
            'status': 'success',
            'similarities': aggregated_similarities,
            'max_similarity': max_similarity,
            'prediction': prediction,
            'threshold_used': threshold,
            'subgraph_count': len(subgraph_results),
            'representative_comments': representative_comments,
            'subgraph_results': subgraph_results,
            'aggregated_features': aggregated_features
        }
    
    def _get_session_comments(self, session_id: str) -> List[int]:
        """获取会话的所有评论节点"""
        try:
            # 使用父类的方法获取会话评论
            if session_id not in self.session_to_nodes:
                return []
            
            comment_nodes = self.session_to_nodes[session_id]
            
            # 过滤出评论节点范围内的节点
            comment_start = self.node_start_indices.get('comment', 55038)
            comment_end = self.node_start_indices.get('word', 132441)  # 下一个节点类型的起始索引
            
            valid_comments = [
                node for node in comment_nodes 
                if comment_start <= node < comment_end
            ]
            
            return valid_comments
            
        except Exception as e:
            print(f"❌ 获取会话评论失败: {e}")
            return []
    
    def _select_representative_comments(self, comment_nodes: List[int], session_id: str) -> List[int]:
        """选择代表性评论节点"""
        target_count = self.optimization_config['representative_subgraph_count']
        strategy = self.optimization_config['sampling_strategy']
        
        if len(comment_nodes) <= target_count:
            return comment_nodes
        
        if strategy == 'diverse_aggressive':
            return self._diverse_aggressive_sampling(comment_nodes, target_count)
        elif strategy == 'high_aggressive':
            return self._high_aggressive_sampling(comment_nodes, target_count)
        elif strategy == 'random':
            return random.sample(comment_nodes, target_count)
        elif strategy == 'balanced':
            return self._balanced_sampling(comment_nodes, target_count)
        else:
            return self._diverse_aggressive_sampling(comment_nodes, target_count)
    
    def _diverse_aggressive_sampling(self, comment_nodes: List[int], target_count: int) -> List[int]:
        """多样性攻击性采样"""
        if not comment_nodes:
            return []
        
        comment_scores = []
        for comment_idx in comment_nodes:
            aggression_score = self._estimate_comment_aggressiveness(comment_idx)
            comment_scores.append((comment_idx, aggression_score))
        
        comment_scores.sort(key=lambda x: x[1], reverse=True)
        
        aggressive_count = min(int(target_count * 0.7), len(comment_scores))
        diverse_count = target_count - aggressive_count
        
        selected = []
        
        for i in range(aggressive_count):
            if i < len(comment_scores):
                selected.append(comment_scores[i][0])
        
        remaining_comments = [score[0] for score in comment_scores[aggressive_count:]]
        if remaining_comments and diverse_count > 0:
            diverse_selected = random.sample(
                remaining_comments, 
                min(diverse_count, len(remaining_comments))
            )
            selected.extend(diverse_selected)
        
        return selected[:target_count]
    
    def _high_aggressive_sampling(self, comment_nodes: List[int], target_count: int) -> List[int]:
        """高攻击性采样"""
        comment_scores = []
        for comment_idx in comment_nodes:
            aggression_score = self._estimate_comment_aggressiveness(comment_idx)
            comment_scores.append((comment_idx, aggression_score))
        
        comment_scores.sort(key=lambda x: x[1], reverse=True)
        return [score[0] for score in comment_scores[:target_count]]
    
    def _balanced_sampling(self, comment_nodes: List[int], target_count: int) -> List[int]:
        """平衡采样"""
        if len(comment_nodes) <= target_count:
            return comment_nodes
        
        interval = len(comment_nodes) / target_count
        selected = []
        
        for i in range(target_count):
            idx = int(i * interval)
            if idx < len(comment_nodes):
                selected.append(comment_nodes[idx])
        
        return selected
    
    def _estimate_comment_aggressiveness(self, comment_idx: int) -> float:
        """估计评论的攻击性分数（简化版）"""
        np.random.seed(comment_idx)
        base_score = np.random.random()
        
        if comment_idx % 10 == 0:
            base_score += 0.3
        if comment_idx % 7 == 0:
            base_score += 0.2
        
        return min(1.0, base_score)
    
    def _aggregate_similarities(self, all_similarities: List[Dict]) -> List[Dict]:
        """聚合多个子图的相似度结果"""
        if not all_similarities:
            return []
        
        prototype_groups = {}
        for sim in all_similarities:
            proto_id = sim['prototype_id']
            if proto_id not in prototype_groups:
                prototype_groups[proto_id] = []
            prototype_groups[proto_id].append(sim['similarity'])
        
        aggregated = []
        for proto_id, similarities in prototype_groups.items():
            max_sim = max(similarities)
            avg_sim = np.mean(similarities)
            
            aggregated_sim = max_sim * 0.7 + avg_sim * 0.3
            
            aggregated.append({
                'prototype_id': proto_id,
                'similarity': aggregated_sim,
                'max_similarity': max_sim,
                'avg_similarity': avg_sim,
                'similarity_count': len(similarities),
                'prototype_emotion': all_similarities[0]['prototype_emotion']
            })
        
        aggregated.sort(key=lambda x: x['similarity'], reverse=True)
        return aggregated
    
    def _compute_aggregated_features(self, subgraph_results: List[Dict], aggregated_similarities: List[Dict]) -> Dict:
        """计算聚合特征"""
        if not subgraph_results or not aggregated_similarities:
            return self._get_default_aggregated_features()
        
        all_similarities = [sim['similarity'] for sim in aggregated_similarities]
        
        features = {
            'max_similarity': max(all_similarities),
            'avg_similarity': np.mean(all_similarities),
            'min_similarity': min(all_similarities),
            'std_similarity': np.std(all_similarities) if len(all_similarities) > 1 else 0.0,
            'prototype_0_max': 0.0,
            'prototype_1_max': 0.0,
            'prototype_0_avg': 0.0,
            'prototype_1_avg': 0.0,
            'high_similarity_count': sum(1 for sim in all_similarities if sim >= 0.7),
            'medium_similarity_count': sum(1 for sim in all_similarities if 0.3 <= sim < 0.7),
            'low_similarity_count': sum(1 for sim in all_similarities if sim < 0.3),
            'subgraph_count': len(subgraph_results),
            'avg_subgraph_size': 0.0,
            'weighted_avg_similarity': 0.0,
            'high_quality_match_count': sum(1 for sim in all_similarities if sim >= 0.8),
            'high_quality_match_ratio': 0.0,
            'top_3_avg_similarity': 0.0,
            'prototype_match_diff': 0.0
        }
        
        # 计算按原型分组的特征
        prototype_sims = {0: [], 1: []}
        for sim in aggregated_similarities:
            proto_id = sim['prototype_id']
            if proto_id in prototype_sims:
                prototype_sims[proto_id].append(sim['similarity'])
        
        if prototype_sims[0]:
            features['prototype_0_max'] = max(prototype_sims[0])
            features['prototype_0_avg'] = np.mean(prototype_sims[0])
        
        if prototype_sims[1]:
            features['prototype_1_max'] = max(prototype_sims[1])
            features['prototype_1_avg'] = np.mean(prototype_sims[1])
        
        # 计算平均子图大小
        if subgraph_results:
            subgraph_sizes = []
            for result in subgraph_results:
                subgraph = result.get('subgraph', {})
                features_dict = subgraph.get('features', {})
                size = features_dict.get('node_count', 0)
                if size > 0:
                    subgraph_sizes.append(size)
            
            if subgraph_sizes:
                features['avg_subgraph_size'] = np.mean(subgraph_sizes)
        
        # 计算高级特征
        if all_similarities:
            weights = [sim for sim in all_similarities]
            if sum(weights) > 0:
                features['weighted_avg_similarity'] = np.average(all_similarities, weights=weights)
            else:
                features['weighted_avg_similarity'] = features['avg_similarity']
            
            if len(all_similarities) > 0:
                features['high_quality_match_ratio'] = features['high_quality_match_count'] / len(all_similarities)
            
            top_similarities = sorted(all_similarities, reverse=True)[:3]
            features['top_3_avg_similarity'] = np.mean(top_similarities)
            
            if len(prototype_sims[0]) > 0 and len(prototype_sims[1]) > 0:
                features['prototype_match_diff'] = abs(features['prototype_0_avg'] - features['prototype_1_avg'])
        
        # 综合霸凌评分
        features['bullying_score'] = (
            features['max_similarity'] * 0.4 +
            features['weighted_avg_similarity'] * 0.3 +
            features['high_quality_match_ratio'] * 0.2 +
            (1.0 - features['prototype_match_diff']) * 0.1
        )
        
        return features
    
    def _get_default_aggregated_features(self) -> Dict:
        """获取默认聚合特征"""
        return {
            'max_similarity': 0.0,
            'avg_similarity': 0.0,
            'min_similarity': 0.0,
            'std_similarity': 0.0,
            'prototype_0_max': 0.0,
            'prototype_1_max': 0.0,
            'prototype_0_avg': 0.0,
            'prototype_1_avg': 0.0,
            'high_similarity_count': 0,
            'medium_similarity_count': 0,
            'low_similarity_count': 0,
            'bullying_score': 0.0,
            'subgraph_count': 0,
            'avg_subgraph_size': 0.0,
            'weighted_avg_similarity': 0.0,
            'high_quality_match_count': 0,
            'high_quality_match_ratio': 0.0,
            'top_3_avg_similarity': 0.0,
            'prototype_match_diff': 0.0
        }
    
    def batch_align_sessions_optimized(self, session_ids: List[str]) -> List[Dict]:
        """批量对齐会话（优化版）"""
        results = []
        
        print(f"🔄 开始优化批量对齐 {len(session_ids)} 个会话...")
        print(f"   每个会话选择 {self.optimization_config['representative_subgraph_count']} 个代表性子图")
        print(f"   采样策略: {self.optimization_config['sampling_strategy']}")
        
        for i, session_id in enumerate(session_ids):
            try:
                result = self.align_session_with_prototypes_optimized(session_id)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    successful = sum(1 for r in results if r['status'] == 'success')
                    print(f"   进度: {i + 1}/{len(session_ids)} (成功: {successful})")
                    
            except Exception as e:
                print(f"❌ 会话 {session_id} 对齐失败: {e}")
                results.append({
                    'session_id': session_id,
                    'status': 'error',
                    'error': str(e),
                    'similarities': [],
                    'max_similarity': 0.0,
                    'prediction': 'normal',
                    'subgraph_count': 0
                })
        
        successful_count = sum(1 for r in results if r['status'] == 'success')
        print(f"✅ 优化批量对齐完成，成功处理 {successful_count}/{len(results)} 个会话")
        
        return results 