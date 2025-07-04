#!/usr/bin/env python3
"""
分析ProtoBully异构图结构，理解霸凌判断的关键要素
"""

import pickle
import torch
import numpy as np
import os

def analyze_graph_structure():
    """分析异构图的完整结构"""
    print("=== ProtoBully异构图结构分析 ===\n")
    
    # 加载异构图
    with open('data/graphs/heterogeneous_graph_final.pkl', 'rb') as f:
        graph = pickle.load(f)
    
    print("1. 图基本信息:")
    print(f"   图类型: {type(graph)}")
    print(f"   节点类型数: {len(graph.node_types)}")
    print(f"   边类型数: {len(graph.edge_types)}")
    print()
    
    print("2. 节点类型详细分析:")
    for node_type in graph.node_types:
        node_data = graph[node_type]
        features = node_data.x if hasattr(node_data, 'x') and node_data.x is not None else None
        
        print(f"   {node_type}:")
        if features is not None:
            print(f"     数量: {features.shape[0]:,}")
            print(f"     特征维度: {features.shape[1]}")
            print(f"     特征范围: [{features.min().item():.4f}, {features.max().item():.4f}]")
            
            # 特殊分析
            if node_type == 'media_session':
                analyze_media_session_labels(features)
            elif node_type == 'comment':
                analyze_comment_features(features)
            elif node_type == 'user':
                analyze_user_features(features)
        else:
            print(f"     数量: 未知 (无特征数据)")
        print()
    
    print("3. 边类型详细分析:")
    for edge_type in graph.edge_types:
        edge_data = graph[edge_type]
        edge_index = edge_data.edge_index if hasattr(edge_data, 'edge_index') else None
        
        if edge_index is not None:
            print(f"   {edge_type}: {edge_index.shape[1]:,}条边")
        else:
            print(f"   {edge_type}: 无边数据")
    print()
    
    print("4. 子图数据结构分析:")
    analyze_subgraph_structure()

def analyze_media_session_labels(features):
    """分析media_session节点的标签分布"""
    labels = features[:, 0]  # 标签在第0列
    unique_labels = torch.unique(labels)
    
    print(f"     标签分布:")
    for label in unique_labels:
        count = (labels == label).sum().item()
        percentage = count / features.shape[0] * 100
        label_type = "霸凌" if label > 0 else "正常"
        print(f"       {label:.4f} ({label_type}): {count}个 ({percentage:.1f}%)")

def analyze_comment_features(features):
    """分析comment节点的特征结构"""
    print(f"     特征结构分析:")
    feature_names = [
        '文本长度', '词数', '感叹号数', '问号数', 
        '大写比例', '攻击词数', '攻击词比例', '内部ID'
    ]
    
    for i in range(min(len(feature_names), features.shape[1])):
        feature_col = features[:, i]
        non_zero_count = (feature_col != 0).sum().item()
        print(f"       维度{i} ({feature_names[i]}): "
              f"范围[{feature_col.min():.3f}, {feature_col.max():.3f}], "
              f"均值{feature_col.mean():.3f}, "
              f"非零率{non_zero_count/features.shape[0]*100:.1f}%")

def analyze_user_features(features):
    """分析user节点的特征结构"""
    print(f"     用户特征分析:")
    if features.shape[1] >= 4:
        print(f"       评论数范围: [{features[:, 0].min():.0f}, {features[:, 0].max():.0f}]")
        print(f"       平均情感范围: [{features[:, 1].min():.3f}, {features[:, 1].max():.3f}]")
        print(f"       攻击性评论数范围: [{features[:, 2].min():.0f}, {features[:, 2].max():.0f}]")
        print(f"       攻击性比例范围: [{features[:, 3].min():.3f}, {features[:, 3].max():.3f}]")

def analyze_subgraph_structure():
    """分析子图数据结构"""
    # 分析4.1增强版子图
    enhanced_dir = 'data/subgraphs/universal_enhanced'
    if os.path.exists(enhanced_dir):
        sample_file = os.path.join(enhanced_dir, 'media_session_0_subgraphs.pkl')
        if os.path.exists(sample_file):
            with open(sample_file, 'rb') as f:
                subgraphs = pickle.load(f)
            
            print(f"   4.1增强版子图 (样本: 会话0):")
            print(f"     子图数量: {len(subgraphs)}")
            
            if subgraphs:
                sample_sg = subgraphs[0]
                print(f"     子图结构示例:")
                print(f"       session_id: {sample_sg.get('session_id', 'N/A')}")
                print(f"       subgraph_id: {sample_sg.get('subgraph_id', 'N/A')}")
                print(f"       total_nodes: {sample_sg.get('total_nodes', 'N/A')}")
                print(f"       subgraph_type: {sample_sg.get('subgraph_type', 'N/A')}")
                
                nodes = sample_sg.get('nodes', {})
                print(f"       节点组成:")
                for node_type, node_list in nodes.items():
                    print(f"         {node_type}: {len(node_list)}个")
                
                edges = sample_sg.get('edges', {})
                print(f"       边类型: {len(edges)}种")
                
                # 分析子图类型分布
                type_counts = {}
                for sg in subgraphs[:100]:  # 分析前100个
                    sg_type = sg.get('subgraph_type', 'unknown')
                    type_counts[sg_type] = type_counts.get(sg_type, 0) + 1
                
                print(f"     子图类型分布 (前100个):")
                for sg_type, count in type_counts.items():
                    print(f"       {sg_type}: {count}个")

def analyze_bullying_detection_logic():
    """分析霸凌检测的关键逻辑"""
    print("\n5. 霸凌检测关键逻辑:")
    print("   基于代码分析，霸凌子图的判断依据:")
    print("   a) 情感分数计算:")
    print("      - Comment节点: 基于攻击词比例、大写比例、感叹号等")
    print("      - User节点: 基于用户的攻击性评论比例")
    print("      - Word节点: 基于词汇的攻击性倾向")
    print("      - 加权组合: comment(0.4-0.6) + user(0.3) + word(0.1-0.2)")
    print()
    print("   b) 筛选阈值:")
    print("      - 情感分数阈值: -0.3 到 -0.4 (越负越攻击性)")
    print("      - 最小攻击评论数: >= 1")
    print("      - 攻击词阈值: > 0.1")
    print("      - 大写比例阈值: > 0.3")
    print()
    print("   c) 关键特征:")
    print("      - Comment特征[5]: 攻击词数")
    print("      - Comment特征[6]: 攻击词比例")
    print("      - Comment特征[4]: 大写比例")
    print("      - Comment特征[2]: 感叹号数")

if __name__ == "__main__":
    analyze_graph_structure()
    analyze_bullying_detection_logic() 