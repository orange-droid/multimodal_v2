"""
多模态结构对齐模块 V4
基于SubgraphPrototypeExtractorV3的原型匹配系统

主要功能：
1. 加载从SubgraphPrototypeExtractorV3提取的原型
2. 为待检测的会话提取子图
3. 计算子图与原型的结构相似度
4. 兼容单个和多个原型的情况
"""

import pickle
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import torch_geometric
from torch_geometric.data import HeteroData
from prototype_extractor_v4_fixed import PrototypeExtractorV4Fixed

class MultimodalStructureAlignerV4:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.extractor = None
        self.prototypes = []
        self.graph = None
        
        # 加载图数据
        self._load_graph()
        
        # 初始化原型提取器
        self._initialize_extractor()
        
        # 加载现有原型
        self._load_prototypes()
    
    def _load_graph(self):
        """加载图数据"""
        graph_path = self.config.get('graph_path', 'data/graphs/media_session_graph_20250601_183403.pkl')
        
        try:
            with open(graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            print(f"✅ 图数据加载成功: {self.graph.num_nodes:,}个节点, {self.graph.num_edges:,}条边")
        except Exception as e:
            raise RuntimeError(f"❌ 图数据加载失败: {e}")
    
    def _initialize_extractor(self):
        """初始化原型提取器"""
        try:
            self.extractor = PrototypeExtractorV4Fixed(self.config)
            
            # 加载必要的数据
            graph_path = self.config.get('graph_path', 'data/graphs/text_graph_20250602_175019.pkl')
            labels_path = self.config.get('labels_path', 'data/session_labels_20250602_175019.json')
            
            # 加载数据到提取器
            self.extractor.load_data(graph_path, labels_path)
            
            # 建立邻接表
            self.adj_list = self.extractor._build_adjacency_list()
            print(f"✅ 邻接表建立成功: {len(self.adj_list):,}个节点")
            
            # 计算节点类型起始索引
            self._calculate_node_start_indices()
            
            print("✅ 原型提取器初始化成功")
        except Exception as e:
            raise RuntimeError(f"❌ 原型提取器初始化失败: {e}")
    
    def _calculate_node_start_indices(self):
        """计算各种节点类型的起始索引"""
        if not self.graph:
            raise RuntimeError("图数据未加载")
        
        self.comment_start_idx = 0
        self.user_start_idx = 0
        
        # 计算评论节点起始索引
        for node_type in self.graph.node_types:
            if node_type == 'comment':
                break
            self.comment_start_idx += self.graph[node_type].x.size(0)
        
        # 计算用户节点起始索引
        for node_type in self.graph.node_types:
            if node_type == 'user':
                break
            self.user_start_idx += self.graph[node_type].x.size(0)
        
        print(f"📍 节点索引计算完成:")
        print(f"   评论节点起始索引: {self.comment_start_idx}")
        print(f"   用户节点起始索引: {self.user_start_idx}")
    
    def _load_prototypes(self):
        """加载现有原型"""
        prototype_dir = Path(self.config.get('prototype_output_dir', 'ProtoBully/data/prototype_v4_fixed_20250602_193953'))
        
        # 查找v4原型文件
        prototype_file = prototype_dir / 'prototypes_v4_fixed_20250602_193953.pkl'
        
        if not prototype_file.exists():
            print(f"⚠️ 未找到v4原型文件: {prototype_file}")
            return
        
        try:
            with open(prototype_file, 'rb') as f:
                prototypes_data = pickle.load(f)
                
            # 处理不同的数据格式
            if isinstance(prototypes_data, list):
                self.prototypes = prototypes_data
            elif isinstance(prototypes_data, dict) and 'prototypes' in prototypes_data:
                self.prototypes = prototypes_data['prototypes']
            else:
                print(f"⚠️ 原型文件格式不识别: {type(prototypes_data)}")
                return
                
            print(f"✅ 原型加载成功: {prototype_file.name}")
            print(f"📊 总共加载了 {len(self.prototypes)} 个原型")
            
            # 打印原型基本信息
            for i, prototype in enumerate(self.prototypes):
                if isinstance(prototype, dict):
                    prototype_id = prototype.get('prototype_id', f'prototype_{i}')
                    cluster_size = prototype.get('cluster_size', 'N/A')
                    quality_score = prototype.get('quality_score', 'N/A')
                    print(f"   原型 {i} ({prototype_id}): 聚类大小={cluster_size}, 质量={quality_score}")
                    
        except Exception as e:
            print(f"❌ 原型加载失败 {prototype_file.name}: {e}")
        
        if not self.prototypes:
            print("⚠️ 未成功加载任何原型")
    
    def extract_session_subgraph(self, session_id: str, target_size: int = 10) -> Optional[Dict]:
        """为指定会话提取子图 - 使用verified方法"""
        if not self.extractor:
            print("❌ 原型提取器未初始化")
            return None
        
        try:
            # 从会话映射文件中查找评论
            session_comments = []
            
            # 加载会话映射文件
            session_mapping_path = self.config.get('session_mapping_path', 'data/graphs/complete_session_to_nodes_mapping.json')
            
            try:
                with open(session_mapping_path, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                
                session_mapping = mapping_data.get('session_to_nodes_mapping', {})
                
                if session_id in session_mapping:
                    comment_indices = session_mapping[session_id].get('comment_indices', [])
                    # 转换为全局索引
                    session_comments = [c + self.comment_start_idx for c in comment_indices]
                    print(f"   找到 {len(session_comments)} 个评论节点")
                else:
                    print(f"⚠️ 会话 {session_id} 不在映射文件中")
                    
            except Exception as e:
                print(f"❌ 加载会话映射失败: {e}")
            
            if not session_comments:
                print(f"⚠️ 会话 {session_id} 中没有找到评论")
                return None
            
            # 选择第一个评论作为中心
            center_comment = session_comments[0]
            print(f"   使用中心评论: {center_comment}")
            
            # 使用verified的子图节点收集方法
            if not hasattr(self, 'adj_list') or not self.adj_list:
                print("   ❌ 邻接表未建立，使用原有方法")
                # 回退到原有方法 - 使用找到的评论节点
                node_indices = session_comments[:target_size]  # 限制大小
                
                subgraph_features = self._compute_subgraph_features(node_indices)
                return {
                    'session_id': session_id,
                    'node_indices': node_indices,
                    'features': subgraph_features,
                    'size': len(node_indices),
                    'extraction_method': 'fallback_method'
                }
            
            # 使用verified方法收集多样化子图节点
            subgraph_nodes = self._collect_diverse_subgraph_nodes(
                center_comment, self.adj_list, target_size
            )
            
            print(f"   多样化节点收集: {len(subgraph_nodes)}个节点")
            
            # 使用verified方法提取子图结构
            structure = self._extract_subgraph_structure_v2(subgraph_nodes)
            
            print(f"   子图边数: {structure['structural_features']['total_edges']}")
            print(f"   子图密度: {structure['structural_features']['density']:.3f}")
            
            # 转换为列表以保持兼容性
            node_list = list(subgraph_nodes)
            
            # 构建features用于相似度计算
            features = {
                'node_count': len(node_list),
                'comment_nodes': len(structure['nodes_by_type'].get('comment', [])),
                'user_nodes': len(structure['nodes_by_type'].get('user', [])),
                'avg_emotion_score': 0.0,  # 简化版
                'avg_attack_ratio': 0.0,   # 简化版
                'structural_features': structure['structural_features']
            }
            
            print(f"✅ 会话 {session_id} 子图提取成功: {len(node_list)}个节点")
            print(f"   节点范围: {min(node_list)}-{max(node_list)}")
            print(f"   特征: 密度={features['structural_features']['density']:.3f}")
            
            return {
                'session_id': session_id,
                'node_indices': node_list,
                'features': features,
                'size': len(node_list),
                'structure': structure,
                'extraction_method': 'verified_diverse_collection'
            }
            
        except Exception as e:
            print(f"❌ 会话 {session_id} 子图提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _compute_subgraph_features(self, node_indices: List[int]) -> Dict:
        """计算子图特征"""
        features = {
            'node_count': len(node_indices),
            'comment_nodes': 0,
            'user_nodes': 0,
            'emotion_scores': [],
            'attack_ratios': [],
            'structural_features': {}
        }
        
        try:
            # 统计节点类型和特征
            for node_idx in node_indices:
                if self.comment_start_idx <= node_idx < self.user_start_idx:
                    # 评论节点
                    features['comment_nodes'] += 1
                    comment_idx = node_idx - self.comment_start_idx
                    
                    # 提取评论特征
                    comment_features = self.extractor.graph['comment'].x[comment_idx]
                    emotion_score = comment_features[1].item()
                    attack_ratio = comment_features[2].item()
                    
                    features['emotion_scores'].append(emotion_score)
                    features['attack_ratios'].append(attack_ratio)
                    
                elif node_idx >= self.user_start_idx:
                    # 用户节点
                    features['user_nodes'] += 1
            
            # 计算统计特征
            if features['emotion_scores']:
                features['avg_emotion_score'] = np.mean(features['emotion_scores'])
                features['std_emotion_score'] = np.std(features['emotion_scores'])
                features['min_emotion_score'] = np.min(features['emotion_scores'])
            else:
                features['avg_emotion_score'] = 0.0
                features['std_emotion_score'] = 0.0
                features['min_emotion_score'] = 0.0
            
            if features['attack_ratios']:
                features['avg_attack_ratio'] = np.mean(features['attack_ratios'])
                features['max_attack_ratio'] = np.max(features['attack_ratios'])
            else:
                features['avg_attack_ratio'] = 0.0
                features['max_attack_ratio'] = 0.0
            
            # 计算结构特征
            features['structural_features'] = self._compute_structural_features(node_indices)
            
        except Exception as e:
            print(f"❌ 子图特征计算失败: {e}")
        
        return features
    
    def _compute_structural_features(self, node_indices: List[int]) -> Dict:
        """计算结构特征"""
        structural = {
            'edge_count': 0,
            'density': 0.0,
            'comment_to_user_edges': 0,
            'user_to_comment_edges': 0,
            'comment_to_comment_edges': 0
        }
        
        try:
            node_set = set(node_indices)
            
            # 统计不同类型的边
            edge_types = [
                ('comment', 'posted_by', 'user'),
                ('user', 'posted', 'comment'),
                ('comment', 'replies_to', 'comment')
            ]
            
            for src_type, edge_type, dst_type in edge_types:
                if (src_type, edge_type, dst_type) in self.graph.edge_types:
                    edge_index = self.graph[src_type, edge_type, dst_type].edge_index
                    
                    # 调整节点索引到对应类型的本地索引
                    for i in range(edge_index.size(1)):
                        src_global = edge_index[0, i].item()
                        dst_global = edge_index[1, i].item()
                        
                        # 转换为全局索引
                        if src_type == 'comment':
                            src_global += self.comment_start_idx
                        elif src_type == 'user':
                            src_global += self.user_start_idx
                        
                        if dst_type == 'comment':
                            dst_global += self.comment_start_idx
                        elif dst_type == 'user':
                            dst_global += self.user_start_idx
                        
                        # 检查边是否在子图内
                        if src_global in node_set and dst_global in node_set:
                            structural['edge_count'] += 1
                            
                            if edge_type == 'posted_by':
                                structural['comment_to_user_edges'] += 1
                            elif edge_type == 'posted':
                                structural['user_to_comment_edges'] += 1
                            elif edge_type == 'replies_to':
                                structural['comment_to_comment_edges'] += 1
            
            # 计算密度
            n = len(node_indices)
            if n > 1:
                max_edges = n * (n - 1)  # 有向图
                structural['density'] = structural['edge_count'] / max_edges
            
        except Exception as e:
            print(f"❌ 结构特征计算失败: {e}")
        
        return structural
    
    def calculate_prototype_similarity(self, session_subgraph: Dict, prototype: Dict) -> float:
        """计算会话子图与原型的相似度"""
        try:
            session_features = session_subgraph['features']
            
            # 处理v4原型格式
            if 'representative_subgraph' in prototype:
                proto_subgraph = prototype['representative_subgraph']
            else:
                print(f"⚠️ 原型格式不符合预期，键: {list(prototype.keys()) if isinstance(prototype, dict) else 'Not dict'}")
                return 0.0
            
            # 从原型中提取特征
            proto_features = self._extract_prototype_features(proto_subgraph)
            session_features_normalized = self._normalize_session_features(session_features)
            
            # 特征相似度权重
            weights = {
                'structural': 0.4,
                'node_composition': 0.3,
                'size': 0.3
            }
            
            similarity_scores = []
            
            # 1. 结构相似度（密度）
            session_density = session_features_normalized.get('density', 0.0)
            proto_density = proto_features.get('density', 0.0)
            
            if session_density > 0 and proto_density > 0:
                density_sim = 1.0 - abs(session_density - proto_density) / max(session_density, proto_density)
            elif session_density == 0 and proto_density == 0:
                density_sim = 1.0  # 都是0，完全相似
            else:
                density_sim = 0.0  # 一个为0一个不为0
            
            density_sim = max(0.0, min(1.0, density_sim))
            similarity_scores.append(('structural', density_sim, weights['structural']))
            
            # 2. 节点组成相似度
            session_comment_ratio = session_features_normalized.get('comment_ratio', 0.0)
            proto_comment_ratio = proto_features.get('comment_ratio', 0.0)
            
            composition_sim = 1.0 - abs(session_comment_ratio - proto_comment_ratio)
            composition_sim = max(0.0, min(1.0, composition_sim))
            
            similarity_scores.append(('composition', composition_sim, weights['node_composition']))
            
            # 3. 大小相似度
            session_size = session_features_normalized.get('node_count', 0)
            proto_size = proto_features.get('node_count', 0)
            
            if session_size > 0 and proto_size > 0:
                size_sim = 1.0 - abs(session_size - proto_size) / max(session_size, proto_size)
            elif session_size == proto_size == 0:
                size_sim = 1.0
            else:
                size_sim = 0.0
            
            size_sim = max(0.0, min(1.0, size_sim))
            similarity_scores.append(('size', size_sim, weights['size']))
            
            # 计算加权总相似度
            total_similarity = sum(score * weight for _, score, weight in similarity_scores)
            
            # 调试输出
            print(f"   相似度详情: 结构={density_sim:.3f}, 组成={composition_sim:.3f}, 大小={size_sim:.3f}")
            print(f"   总相似度: {total_similarity:.3f}")
            
            return min(1.0, max(0.0, total_similarity))
            
        except Exception as e:
            print(f"❌ 相似度计算失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _extract_prototype_features(self, proto_subgraph: Dict) -> Dict:
        """从原型子图中提取标准化特征"""
        features = {}
        
        try:
            # 提取结构特征
            structural_features = proto_subgraph.get('structural_features', {})
            features['density'] = structural_features.get('density', 0.0)
            features['node_count'] = structural_features.get('total_nodes', 0)
            features['edge_count'] = structural_features.get('total_edges', 0)
            
            # 提取节点组成
            nodes_by_type = proto_subgraph.get('nodes_by_type', {})
            comment_count = len(nodes_by_type.get('comment', []))
            total_nodes = features['node_count']
            
            if total_nodes > 0:
                features['comment_ratio'] = comment_count / total_nodes
            else:
                features['comment_ratio'] = 0.0
            
            # 提取情感特征（如果有）
            emotion_features = proto_subgraph.get('emotion_features', {})
            features['avg_emotion_score'] = emotion_features.get('avg_text_length', 0.0)
            
            return features
            
        except Exception as e:
            print(f"⚠️ 原型特征提取失败: {e}")
            return {'density': 0.0, 'node_count': 0, 'comment_ratio': 0.0}
    
    def _normalize_session_features(self, session_features: Dict) -> Dict:
        """标准化会话特征"""
        features = {}
        
        try:
            # 基本特征
            features['node_count'] = session_features.get('node_count', 0)
            features['comment_nodes'] = session_features.get('comment_nodes', 0)
            features['user_nodes'] = session_features.get('user_nodes', 0)
            
            # 计算比例
            if features['node_count'] > 0:
                features['comment_ratio'] = features['comment_nodes'] / features['node_count']
            else:
                features['comment_ratio'] = 0.0
            
            # 结构特征
            structural_features = session_features.get('structural_features', {})
            features['density'] = structural_features.get('density', 0.0)
            features['edge_count'] = structural_features.get('total_edges', 0)
            
            return features
            
        except Exception as e:
            print(f"⚠️ 会话特征标准化失败: {e}")
            return {'node_count': 0, 'comment_ratio': 0.0, 'density': 0.0}
    
    def align_session_with_prototypes(self, session_id: str) -> Dict:
        """将单个会话与所有原型对齐"""
        if not self.prototypes:
            return {
                'session_id': session_id,
                'status': 'no_prototypes',
                'similarities': [],
                'max_similarity': 0.0,
                'prediction': 'normal'
            }
        
        # 提取会话子图
        session_subgraph = self.extract_session_subgraph(session_id)
        
        if not session_subgraph:
            return {
                'session_id': session_id,
                'status': 'extraction_failed',
                'similarities': [],
                'max_similarity': 0.0,
                'prediction': 'normal'
            }
        
        # 与所有原型计算相似度
        similarities = []
        
        for i, prototype in enumerate(self.prototypes):
            similarity = self.calculate_prototype_similarity(session_subgraph, prototype)
            
            # 正确提取原型情感分数
            prototype_emotion = 0.0
            if 'representative_subgraph' in prototype:
                emotion_features = prototype['representative_subgraph'].get('emotion_features', {})
                prototype_emotion = emotion_features.get('avg_text_length', 0.0)
            
            similarities.append({
                'prototype_id': i,
                'similarity': similarity,
                'prototype_emotion': prototype_emotion
            })
        
        # 排序，获取最高相似度
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        max_similarity = similarities[0]['similarity'] if similarities else 0.0
        
        # 预测（基于相似度阈值）
        threshold = self.config.get('similarity_threshold', 0.6)
        prediction = 'bullying' if max_similarity >= threshold else 'normal'
        
        return {
            'session_id': session_id,
            'status': 'success',
            'session_subgraph': session_subgraph,
            'similarities': similarities,
            'max_similarity': max_similarity,
            'prediction': prediction,
            'threshold_used': threshold
        }
    
    def batch_align_sessions(self, session_ids: List[str]) -> List[Dict]:
        """批量对齐会话"""
        results = []
        
        print(f"🔄 开始批量对齐 {len(session_ids)} 个会话...")
        
        for i, session_id in enumerate(session_ids):
            try:
                result = self.align_session_with_prototypes(session_id)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"   进度: {i + 1}/{len(session_ids)}")
                    
            except Exception as e:
                print(f"❌ 会话 {session_id} 对齐失败: {e}")
                results.append({
                    'session_id': session_id,
                    'status': 'error',
                    'error': str(e),
                    'similarities': [],
                    'max_similarity': 0.0,
                    'prediction': 'normal'
                })
        
        print(f"✅ 批量对齐完成，成功处理 {len(results)} 个会话")
        return results
    
    def save_alignment_results(self, results: List[Dict], output_path: str):
        """保存对齐结果"""
        try:
            # 准备保存的数据
            save_data = {
                'total_sessions': len(results),
                'prototype_count': len(self.prototypes),
                'config': self.config,
                'results': results,
                'statistics': self._compute_alignment_statistics(results)
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 对齐结果已保存至: {output_path}")
            
        except Exception as e:
            print(f"❌ 对齐结果保存失败: {e}")
    
    def _compute_alignment_statistics(self, results: List[Dict]) -> Dict:
        """计算对齐统计信息"""
        stats = {
            'successful_alignments': 0,
            'failed_alignments': 0,
            'bullying_predictions': 0,
            'normal_predictions': 0,
            'avg_max_similarity': 0.0,
            'similarity_distribution': {}
        }
        
        similarities = []
        
        for result in results:
            if result['status'] == 'success':
                stats['successful_alignments'] += 1
                similarities.append(result['max_similarity'])
                
                if result['prediction'] == 'bullying':
                    stats['bullying_predictions'] += 1
                else:
                    stats['normal_predictions'] += 1
            else:
                stats['failed_alignments'] += 1
        
        if similarities:
            stats['avg_max_similarity'] = np.mean(similarities)
            stats['similarity_distribution'] = {
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities)),
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'median': float(np.median(similarities))
            }
        
        return stats
    
    # =============================================================================
    # 从prototype_extractor_v4_fixed.py复制的verified方法
    # =============================================================================
    
    def _collect_diverse_subgraph_nodes(self, center_comment: int, adj_list: Dict, target_size: int) -> Set[int]:
        """收集多样化的子图节点，支持不同的扩展策略"""
        
        subgraph_nodes = {center_comment}
        candidates = set()
        
        # 添加中心评论的直接邻居
        if center_comment in adj_list:
            for neighbor in adj_list[center_comment]:
                candidates.add(neighbor)
        
        # 根据目标大小选择不同的扩展策略
        if target_size <= 12:
            # 小型子图：紧密连接，选择高优先级节点
            strategy = "compact"
        elif target_size <= 21:
            # 中型子图：平衡扩展
            strategy = "balanced"
        else:
            # 大型子图：广泛扩展，包含更多节点类型
            strategy = "extensive"
        
        # 按策略扩展子图
        while len(subgraph_nodes) < target_size and candidates:
            if strategy == "compact":
                next_node = self._select_priority_node(candidates, subgraph_nodes)
            elif strategy == "balanced":
                next_node = self._select_balanced_node(candidates, subgraph_nodes)
            else:  # extensive
                next_node = self._select_diverse_node(candidates, subgraph_nodes)
                
            if next_node is None:
                break
                
            subgraph_nodes.add(next_node)
            candidates.remove(next_node)
            
            # 添加新节点的邻居作为候选
            if next_node in adj_list:
                for neighbor in adj_list[next_node]:
                    if neighbor not in subgraph_nodes:
                        candidates.add(neighbor)
        
        return subgraph_nodes
    
    def _select_balanced_node(self, candidates: Set[int], existing_nodes: Set[int]) -> int:
        """平衡选择策略：在优先级和多样性之间平衡"""
        # 50%概率按优先级选择，50%概率随机选择
        if np.random.random() < 0.5:
            return self._select_priority_node(candidates, existing_nodes)
        else:
            return next(iter(candidates)) if candidates else None
    
    def _select_diverse_node(self, candidates: Set[int], existing_nodes: Set[int]) -> int:
        """多样性选择策略：优先选择图中缺少的节点类型"""
        existing_types = {self._get_node_type(node) for node in existing_nodes}
        
        # 优先选择当前子图中缺少的节点类型
        for node_type in ['word', 'time', 'location', 'user', 'comment']:
            if node_type not in existing_types:
                type_candidates = [
                    node for node in candidates 
                    if self._get_node_type(node) == node_type
                ]
                if type_candidates:
                    return type_candidates[0]
        
        # 如果所有类型都有，随机选择
        return next(iter(candidates)) if candidates else None
    
    def _select_priority_node(self, candidates: Set[int], existing_nodes: Set[int]) -> int:
        """按优先级选择下一个节点：评论 > 用户 > 词汇 > 其他"""
        priority_order = ['comment', 'user', 'word', 'time', 'location']
        
        for node_type in priority_order:
            type_candidates = [
                node for node in candidates 
                if self._get_node_type(node) == node_type
            ]
            if type_candidates:
                return type_candidates[0]
        
        # 如果没有优先类型，返回任意候选
        return next(iter(candidates)) if candidates else None
    
    def _get_node_type(self, node_idx: int) -> str:
        """根据节点索引确定节点类型"""
        if node_idx < 55038:
            return 'user'
        elif node_idx < 132441:
            return 'comment'
        elif node_idx < 133441:
            return 'word'
        elif node_idx < 134400:
            return 'video'
        else:
            return 'other'
    
    def _extract_subgraph_edges_v2(self, subgraph_nodes: Set[int]) -> Dict:
        """从原型提取器复制的verified边提取方法"""
        subgraph_edges = {}
        
        # 使用已经建立的邻接表
        if hasattr(self, 'adj_list') and self.adj_list:
            edge_count = 0
            for node in subgraph_nodes:
                if node in self.adj_list:
                    for neighbor in self.adj_list[node]:
                        if neighbor in subgraph_nodes:
                            # 确定边类型
                            edge_type = self._determine_edge_type(node, neighbor)
                            if edge_type not in subgraph_edges:
                                subgraph_edges[edge_type] = []
                            subgraph_edges[edge_type].append((node, neighbor))
                            edge_count += 1
            
            print(f"   发现子图内部边: {edge_count}条")
        else:
            print("   ❌ 邻接表未建立，无法提取边")
        
        return subgraph_edges
    
    def _determine_edge_type(self, node1: int, node2: int) -> str:
        """确定两个节点间的边类型"""
        type1 = self._get_node_type(node1)
        type2 = self._get_node_type(node2)
        
        # 返回标准化的边类型名称
        if (type1 == 'user' and type2 == 'comment') or (type1 == 'comment' and type2 == 'user'):
            return 'user_posts_comment'
        elif (type1 == 'comment' and type2 == 'word') or (type1 == 'word' and type2 == 'comment'):
            return 'comment_contains_word'
        elif type1 == 'user' and type2 == 'user':
            return 'user_interacts_user'
        elif (type1 == 'video' and type2 == 'comment') or (type1 == 'comment' and type2 == 'video'):
            return 'video_contains_comment'
        else:
            return f'{type1}_{type2}'
    
    def _extract_subgraph_structure_v2(self, subgraph_nodes: Set[int]) -> Dict:
        """从原型提取器复制的verified结构提取方法"""
        # 按类型分组节点
        nodes_by_type = {}
        for node in subgraph_nodes:
            node_type = self._get_node_type(node)
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)
        
        # 提取边
        edges = self._extract_subgraph_edges_v2(subgraph_nodes)
        
        # 计算特征
        structural_features = self._calculate_subgraph_features_v2(nodes_by_type, edges)
        
        return {
            'nodes_by_type': nodes_by_type,
            'edges': edges,
            'structural_features': structural_features
        }
    
    def _calculate_subgraph_features_v2(self, nodes_by_type: Dict, edges: Dict) -> Dict:
        """从原型提取器复制的verified特征计算方法"""
        features = {}
        
        # 节点统计
        features['node_counts'] = {k: len(v) for k, v in nodes_by_type.items()}
        features['total_nodes'] = sum(features['node_counts'].values())
        
        # 边统计
        features['edge_counts'] = {k: len(v) for k, v in edges.items()}
        features['total_edges'] = sum(features['edge_counts'].values())
        
        # 密度
        total_nodes = features['total_nodes']
        if total_nodes > 1:
            max_edges = total_nodes * (total_nodes - 1) / 2
            features['density'] = features['total_edges'] / max_edges
        else:
            features['density'] = 0.0
        
        # 多样性
        features['node_type_diversity'] = len(features['node_counts'])
        features['edge_type_diversity'] = len(features['edge_counts'])
        
        return features 