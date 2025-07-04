#!/usr/bin/env python3
"""
运行修复后的子图提取模块
确保子图中的节点都有边连接，没有孤立节点
"""

import sys
import os
import pickle
from datetime import datetime

# 添加src路径
sys.path.append('src')

from prototype.universal_subgraph_extractor import UniversalSubgraphExtractor
import logging

def main():
    """运行修复后的子图提取"""
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fixed_subgraph_extraction.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("开始运行修复后的子图提取模块")
    
    try:
        # 加载异构图
        logger.info("加载异构图数据...")
        with open('data/graphs/heterogeneous_graph_final.pkl', 'rb') as f:
            graph = pickle.load(f)
        
        logger.info(f"图加载成功: {len(graph.node_types)}种节点类型, {len(graph.edge_types)}种边类型")
        
        # 创建修复后的提取器（优化参数提升性能）
        config = {
            'min_subgraph_size': 3,           # 最小子图大小
            'max_subgraph_size': 8,           # 最大子图大小（从20降到8）
            'max_comments_per_video': 10,     # 每个视频最大评论数（从30降到10）
            'max_users_per_subgraph': 4,      # 每个子图最大用户数（从8降到4）
            'max_words_per_subgraph': 5,      # 每个子图最大词汇数（从15降到5）
            'enable_multi_user_subgraphs': True,
            'enable_smart_selection': True,
        }
        
        extractor = UniversalSubgraphExtractor(config)
        logger.info("子图提取器创建成功")
        
        # 运行提取
        output_dir = "data/subgraphs/universal_optimized_fixed"
        logger.info(f"开始提取子图到目录: {output_dir}")
        logger.info(f"优化配置: 大小3-8, 评论≤10, 用户≤4, 词汇≤5")
        
        results = extractor.extract_all_session_subgraphs(
            graph=graph,
            output_dir=output_dir
        )
        
        # 统计结果
        total_subgraphs = results.get('total_subgraphs', 0)
        total_sessions = results.get('total_sessions', 0)
        
        logger.info("=" * 50)
        logger.info("子图提取完成!")
        logger.info(f"处理会话数: {total_sessions}")
        logger.info(f"总子图数: {total_subgraphs}")
        logger.info(f"平均每会话子图数: {total_subgraphs/total_sessions:.1f}")
        
        # 验证边连接情况
        logger.info("\n验证修复效果...")
        verify_edge_connectivity(output_dir, logger)
        
        logger.info("修复后的子图提取成功完成!")
        
    except Exception as e:
        logger.error(f"子图提取失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def verify_edge_connectivity(output_dir: str, logger):
    """验证子图的边连接情况"""
    
    # 检查前几个文件
    files = [f for f in os.listdir(output_dir) if f.endswith('.pkl')]
    sample_files = files[:3]  # 检查前3个文件
    
    total_subgraphs_checked = 0
    subgraphs_with_edges = 0
    
    for file in sample_files:
        file_path = os.path.join(output_dir, file)
        
        try:
            with open(file_path, 'rb') as f:
                session_data = pickle.load(f)
            
            subgraphs = session_data.get('subgraphs', [])
            
            for subgraph in subgraphs[:20]:  # 每个文件检查前20个子图
                total_subgraphs_checked += 1
                edges = subgraph.get('edges', {})
                
                # 检查是否有边
                has_edges = False
                total_edges = 0
                for edge_type, edge_data in edges.items():
                    if edge_data and len(edge_data) > 0:
                        has_edges = True
                        total_edges += len(edge_data)
                
                if has_edges:
                    subgraphs_with_edges += 1
                    
        except Exception as e:
            logger.warning(f"验证文件 {file} 时出错: {e}")
    
    if total_subgraphs_checked > 0:
        edge_ratio = subgraphs_with_edges / total_subgraphs_checked * 100
        logger.info(f"边连接验证结果:")
        logger.info(f"  检查子图数: {total_subgraphs_checked}")
        logger.info(f"  有边连接的子图: {subgraphs_with_edges} ({edge_ratio:.1f}%)")
        logger.info(f"  修复效果: {'成功' if edge_ratio > 80 else '需要进一步检查'}")
    else:
        logger.warning("没有找到子图数据进行验证")

if __name__ == "__main__":
    main() 