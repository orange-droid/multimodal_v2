#!/usr/bin/env python3
"""
完整的多尺寸子图提取和按会话保存
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pickle
import time
from collections import defaultdict
from src.prototype.subgraph_extractor import SubgraphExtractor
from src.utils.logger import setup_logger

def run_complete_extraction():
    """运行完整的子图提取"""
    logger = setup_logger('complete_extraction')
    
    logger.info("=== 开始完整的多尺寸子图提取 ===")
    
    # 加载异构图
    logger.info("Loading heterogeneous graph...")
    start_time = time.time()
    with open('data/graphs/heterogeneous_graph_final.pkl', 'rb') as f:
        graph = pickle.load(f)
    load_time = time.time() - start_time
    logger.info(f"Loaded graph with {graph.num_nodes} nodes and {graph.num_edges} edges in {load_time:.2f}s")
    
    # 创建子图提取器
    extractor = SubgraphExtractor()
    
    # 第一步：识别负面评论
    logger.info("=== Step 1: Identifying negative comments ===")
    start_time = time.time()
    negative_comments = extractor._identify_negative_comments(graph)
    identification_time = time.time() - start_time
    logger.info(f"Found {len(negative_comments)} negative comments in {identification_time:.2f} seconds")
    
    # 按会话统计
    session_stats = defaultdict(int)
    for neg_comment in negative_comments:
        session_stats[neg_comment['session_id']] += 1
    
    # 过滤掉unknown会话
    valid_sessions = {k: v for k, v in session_stats.items() if k != 'media_session_unknown'}
    logger.info(f"Negative comments found in {len(valid_sessions)} valid sessions")
    
    # 显示统计信息
    top_sessions = sorted(valid_sessions.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top 10 sessions with most negative comments:")
    for session_id, count in top_sessions:
        logger.info(f"  {session_id}: {count} negative comments")
    
    # 第二步：多尺寸子图提取
    logger.info("=== Step 2: Multi-size subgraph extraction ===")
    
    # 过滤有效的负面评论
    valid_negative_comments = [nc for nc in negative_comments if nc['session_id'] != 'media_session_unknown']
    logger.info(f"Processing {len(valid_negative_comments)} valid negative comments from {len(valid_sessions)} sessions")
    
    start_time = time.time()
    all_subgraphs = extractor._extract_session_based_subgraphs(graph, valid_negative_comments)
    extraction_time = time.time() - start_time
    
    logger.info(f"Successfully extracted {len(all_subgraphs)} subgraphs in {extraction_time:.2f} seconds")
    logger.info(f"Average extraction speed: {len(all_subgraphs) / extraction_time:.1f} subgraphs/second")
    
    if not all_subgraphs:
        logger.error("No subgraphs extracted! Stopping.")
        return
    
    # 第三步：分析和统计
    logger.info("=== Step 3: Analysis and statistics ===")
    
    # 基本统计
    size_distribution = defaultdict(int)
    session_distribution = defaultdict(int)
    total_nodes = 0
    total_edges = 0
    
    for subgraph in all_subgraphs:
        size = subgraph.get('size', 0)
        session_id = subgraph.get('session_id', 'unknown')
        
        size_distribution[size] += 1
        session_distribution[session_id] += 1
        
        # 统计节点和边
        nodes = subgraph.get('nodes', {})
        edges = subgraph.get('edges', {})
        
        subgraph_nodes = sum(len(node_list) for node_list in nodes.values())
        subgraph_edges = sum(len(edge_list) for edge_list in edges.values())
        
        total_nodes += subgraph_nodes
        total_edges += subgraph_edges
    
    logger.info("=== 提取统计结果 ===")
    logger.info(f"总子图数量: {len(all_subgraphs)}")
    logger.info(f"涉及会话数: {len(session_distribution)}")
    logger.info(f"平均每个子图节点数: {total_nodes / len(all_subgraphs):.1f}")
    logger.info(f"平均每个子图边数: {total_edges / len(all_subgraphs):.1f}")
    logger.info(f"平均每个会话子图数: {len(all_subgraphs) / len(session_distribution):.1f}")
    
    logger.info("子图大小分布:")
    for size in sorted(size_distribution.keys()):
        logger.info(f"  大小 {size}: {size_distribution[size]} 个子图 ({size_distribution[size]/len(all_subgraphs)*100:.1f}%)")
    
    logger.info("子图数量最多的前10个会话:")
    top_subgraph_sessions = sorted(session_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
    for session_id, count in top_subgraph_sessions:
        logger.info(f"  {session_id}: {count} 个子图")
    
    # 第四步：按会话保存子图
    logger.info("=== Step 4: Saving subgraphs by session ===")
    
    output_dir = 'data/subgraphs/multi_size_by_session'
    start_time = time.time()
    index_path = extractor.save_subgraphs_by_session(all_subgraphs, output_dir)
    save_time = time.time() - start_time
    
    if index_path:
        logger.info(f"Successfully saved all subgraphs in {save_time:.2f} seconds")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Index file: {index_path}")
        
        # 第五步：验证保存结果
        logger.info("=== Step 5: Verification ===")
        
        # 加载索引文件验证
        index_data = extractor.get_subgraph_index(output_dir)
        if index_data:
            logger.info("Index file verification:")
            logger.info(f"  Total sessions: {index_data['total_sessions']}")
            logger.info(f"  Total subgraphs: {index_data['total_subgraphs']}")
            logger.info(f"  Saved files: {len(index_data['saved_files'])}")
            
            # 测试加载一个会话的子图
            sample_session = list(index_data['session_summary'].keys())[0]
            logger.info(f"Testing load for session: {sample_session}")
            
            session_data = extractor.load_session_subgraphs(sample_session, output_dir)
            if session_data:
                logger.info(f"  Successfully loaded {session_data['total_subgraphs']} subgraphs")
                logger.info(f"  Size distribution: {session_data['size_distribution']}")
            
            # 显示文件大小信息
            logger.info("File size information:")
            total_size = 0
            for filepath in index_data['saved_files']:
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    total_size += size
            
            logger.info(f"  Total storage size: {total_size / 1024 / 1024:.2f} MB")
            logger.info(f"  Average file size: {total_size / len(index_data['saved_files']) / 1024:.2f} KB")
        
        logger.info("=== 完整提取流程成功完成 ===")
        logger.info(f"所有子图已按会话保存到: {output_dir}")
        logger.info("后续模块可以通过以下方式使用:")
        logger.info("1. 加载索引文件获取所有会话信息")
        logger.info("2. 根据需要加载特定会话的子图")
        logger.info("3. 每个会话的子图已按大小排序，支持包含关系分析")
        
    else:
        logger.error("Failed to save subgraphs!")

def test_load_functionality():
    """测试加载功能"""
    logger = setup_logger('test_load')
    extractor = SubgraphExtractor()
    
    output_dir = 'data/subgraphs/multi_size_by_session'
    
    # 测试索引加载
    logger.info("Testing index loading...")
    index_data = extractor.get_subgraph_index(output_dir)
    
    if index_data:
        logger.info(f"Index loaded successfully: {index_data['total_sessions']} sessions")
        
        # 测试加载前3个会话的子图
        sample_sessions = list(index_data['session_summary'].keys())[:3]
        
        for session_id in sample_sessions:
            session_data = extractor.load_session_subgraphs(session_id, output_dir)
            if session_data:
                logger.info(f"{session_id}: {session_data['total_subgraphs']} subgraphs loaded")
            else:
                logger.warning(f"Failed to load {session_id}")
    else:
        logger.error("Failed to load index")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_load_functionality()
    else:
        run_complete_extraction() 