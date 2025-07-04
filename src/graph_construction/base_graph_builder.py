"""
基础图构建器

提供所有图构建器共享的基础功能，包括图的创建、保存、加载等。
"""

import os
import json
import pickle
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import networkx as nx
import torch
from torch_geometric.data import Data, HeteroData
import numpy as np
from datetime import datetime

class BaseGraphBuilder(ABC):
    """基础图构建器抽象类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化基础图构建器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # 图存储
        self.graphs = {}
        self.node_mappings = {}
        self.edge_mappings = {}
        
        # 统计信息
        self.stats = {
            'num_graphs': 0,
            'total_nodes': 0,
            'total_edges': 0,
            'node_types': {},
            'edge_types': {}
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @abstractmethod
    def build_graph(self, data: Dict[str, Any], **kwargs) -> Union[Data, HeteroData, nx.Graph]:
        """
        构建图的抽象方法
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            构建的图对象
        """
        pass
    
    def create_node_mapping(self, nodes: List[str]) -> Dict[str, int]:
        """
        创建节点映射
        
        Args:
            nodes: 节点列表
            
        Returns:
            节点到索引的映射
        """
        return {node: idx for idx, node in enumerate(nodes)}
    
    def create_edge_list(self, edges: List[Tuple[str, str]], 
                        node_mapping: Dict[str, int]) -> List[Tuple[int, int]]:
        """
        创建边列表
        
        Args:
            edges: 边列表（节点名称）
            node_mapping: 节点映射
            
        Returns:
            边列表（节点索引）
        """
        edge_list = []
        for src, dst in edges:
            if src in node_mapping and dst in node_mapping:
                edge_list.append((node_mapping[src], node_mapping[dst]))
        return edge_list
    
    def normalize_features(self, features: np.ndarray, 
                          method: str = 'minmax') -> np.ndarray:
        """
        标准化特征
        
        Args:
            features: 特征矩阵
            method: 标准化方法 ('minmax', 'zscore')
            
        Returns:
            标准化后的特征
        """
        if method == 'minmax':
            min_vals = np.min(features, axis=0)
            max_vals = np.max(features, axis=0)
            # 避免除零
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1
            return (features - min_vals) / range_vals
        elif method == 'zscore':
            mean_vals = np.mean(features, axis=0)
            std_vals = np.std(features, axis=0)
            # 避免除零
            std_vals[std_vals == 0] = 1
            return (features - mean_vals) / std_vals
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def add_graph(self, graph_id: str, graph: Union[Data, HeteroData, nx.Graph]):
        """
        添加图到存储
        
        Args:
            graph_id: 图ID
            graph: 图对象
        """
        self.graphs[graph_id] = graph
        self._update_stats(graph)
        self.logger.info(f"Added graph {graph_id}")
    
    def _update_stats(self, graph: Union[Data, HeteroData, nx.Graph]):
        """更新统计信息"""
        self.stats['num_graphs'] += 1
        
        if isinstance(graph, nx.Graph):
            self.stats['total_nodes'] += graph.number_of_nodes()
            self.stats['total_edges'] += graph.number_of_edges()
        elif isinstance(graph, Data):
            self.stats['total_nodes'] += graph.num_nodes
            self.stats['total_edges'] += graph.num_edges
        elif isinstance(graph, HeteroData):
            for node_type in graph.node_types:
                num_nodes = graph[node_type].num_nodes
                self.stats['total_nodes'] += num_nodes
                self.stats['node_types'][node_type] = \
                    self.stats['node_types'].get(node_type, 0) + num_nodes
            
            for edge_type in graph.edge_types:
                num_edges = graph[edge_type].num_edges
                self.stats['total_edges'] += num_edges
                edge_type_str = f"{edge_type[0]}-{edge_type[1]}-{edge_type[2]}"
                self.stats['edge_types'][edge_type_str] = \
                    self.stats['edge_types'].get(edge_type_str, 0) + num_edges
    
    def get_graph(self, graph_id: str) -> Optional[Union[Data, HeteroData, nx.Graph]]:
        """
        获取图
        
        Args:
            graph_id: 图ID
            
        Returns:
            图对象或None
        """
        return self.graphs.get(graph_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def save_graphs(self, output_dir: str, format: str = 'pickle'):
        """
        保存图到文件
        
        Args:
            output_dir: 输出目录
            format: 保存格式 ('pickle', 'json')
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if format == 'pickle':
            # 保存图对象
            graphs_file = os.path.join(output_dir, 'graphs.pkl')
            with open(graphs_file, 'wb') as f:
                pickle.dump(self.graphs, f)
            
            # 保存映射
            mappings_file = os.path.join(output_dir, 'mappings.pkl')
            with open(mappings_file, 'wb') as f:
                pickle.dump({
                    'node_mappings': self.node_mappings,
                    'edge_mappings': self.edge_mappings
                }, f)
        
        elif format == 'json':
            # 保存统计信息和配置
            metadata = {
                'stats': self.stats,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            metadata_file = os.path.join(output_dir, 'metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(self.graphs)} graphs to {output_dir}")
    
    def load_graphs(self, input_dir: str, format: str = 'pickle'):
        """
        从文件加载图
        
        Args:
            input_dir: 输入目录
            format: 加载格式 ('pickle', 'json')
        """
        if format == 'pickle':
            # 加载图对象
            graphs_file = os.path.join(input_dir, 'graphs.pkl')
            if os.path.exists(graphs_file):
                with open(graphs_file, 'rb') as f:
                    self.graphs = pickle.load(f)
            
            # 加载映射
            mappings_file = os.path.join(input_dir, 'mappings.pkl')
            if os.path.exists(mappings_file):
                with open(mappings_file, 'rb') as f:
                    mappings = pickle.load(f)
                    self.node_mappings = mappings.get('node_mappings', {})
                    self.edge_mappings = mappings.get('edge_mappings', {})
        
        elif format == 'json':
            # 加载元数据
            metadata_file = os.path.join(input_dir, 'metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.stats = metadata.get('stats', {})
                    self.config.update(metadata.get('config', {}))
        
        self.logger.info(f"Loaded {len(self.graphs)} graphs from {input_dir}")
    
    def validate_graph(self, graph: Union[Data, HeteroData, nx.Graph]) -> bool:
        """
        验证图的有效性
        
        Args:
            graph: 图对象
            
        Returns:
            是否有效
        """
        try:
            if isinstance(graph, nx.Graph):
                return graph.number_of_nodes() > 0
            elif isinstance(graph, Data):
                return graph.num_nodes > 0 and hasattr(graph, 'edge_index')
            elif isinstance(graph, HeteroData):
                return len(graph.node_types) > 0
            return False
        except Exception as e:
            self.logger.error(f"Graph validation error: {e}")
            return False
    
    def clear_graphs(self):
        """清空所有图"""
        self.graphs.clear()
        self.node_mappings.clear()
        self.edge_mappings.clear()
        self.stats = {
            'num_graphs': 0,
            'total_nodes': 0,
            'total_edges': 0,
            'node_types': {},
            'edge_types': {}
        }
        self.logger.info("Cleared all graphs") 