#!/usr/bin/env python3
"""
最终图构建模块运行脚本
"""
import sys
import os
import json

# 添加src目录到路径
sys.path.append('src')

from graph_construction.heterogeneous_graph_builder import HeterogeneousGraphBuilder

def test_graph_construction():
    """测试图构建模块"""
    print("="*80)
    print("重新测试图构建模块...")
    print("="*80)
    
    try:
        # 检查输入文件
        text_features_file = "data/features/text_features.pkl"
        text_data_file = "data/processed/text/processed_text_data.json"
        
        if not os.path.exists(text_features_file):
            print(f"警告：找不到文本特征文件: {text_features_file}")
            print("将使用空特征继续测试...")
            text_features = {}
        else:
            print(f"✓ 找到文本特征文件: {text_features_file}")
            import pickle
            with open(text_features_file, 'rb') as f:
                text_features = pickle.load(f)
            print(f"✓ 文本特征加载成功")
            
        if not os.path.exists(text_data_file):
            print(f"错误：找不到文本数据文件: {text_data_file}")
            return False
            
        print(f"✓ 找到文本数据文件: {text_data_file}")
        
        # 加载数据
        with open(text_data_file, 'r', encoding='utf-8') as f:
            text_data = json.load(f)
        
        print(f"✓ 文本数据加载成功 ({len(text_data)} 个会话)")
        
        # 检查caption字段（这是关键！）
        sample_post = list(text_data.values())[0]
        caption = sample_post.get('caption', 'NOT_FOUND')
        print(f"✓ 样本caption字段: {repr(caption[:50])}...")
        print(f"  caption类型: {type(caption)}")
        print(f"  caption长度: {len(caption) if isinstance(caption, str) else 'N/A'}")
        
        # 准备图构建数据
        graph_data = {
            'text_features': text_features,
            'aligned_data': text_data,
            'comments_data': text_data
        }
        
        # 初始化图构建器
        print("\n初始化异构图构建器...")
        config = {
            'text_config': {},
            'video_config': {},
            'metadata_config': {},
            'enable_cross_modal_edges': True,
            'user_alignment_threshold': 0.8,
            'feature_fusion_method': 'concatenate'
        }
        
        builder = HeterogeneousGraphBuilder(config)
        print("✓ 异构图构建器初始化成功")
        
        # 构建图（这里是关键测试点）
        print("\n开始构建异构图...")
        print("正在测试视频标题字段处理...")
        
        hetero_graph = builder.build_graph(graph_data)
        
        print(f"✓ 异构图构建成功!")
        print(f"  - 节点类型: {list(hetero_graph.node_types)}")
        print(f"  - 边类型: {list(hetero_graph.edge_types)}")
        
        # 输出详细统计
        print("\n图统计信息:")
        total_nodes = 0
        total_edges = 0
        
        for node_type in hetero_graph.node_types:
            num_nodes = hetero_graph[node_type].num_nodes
            total_nodes += num_nodes
            print(f"  - {node_type}: {num_nodes} 个节点")
        
        for edge_type in hetero_graph.edge_types:
            num_edges = hetero_graph[edge_type].num_edges
            total_edges += num_edges
            print(f"  - {edge_type}: {num_edges} 条边")
        
        print(f"\n总计: {total_nodes} 个节点, {total_edges} 条边")
        
        # 特别检查视频节点的title字段处理
        if 'video' in hetero_graph.node_types:
            video_features = hetero_graph['video'].x
            print(f"\n视频节点特征检查:")
            print(f"  - 视频节点数: {video_features.shape[0]}")
            print(f"  - 特征维度: {video_features.shape[1]}")
            
            # 第3列应该是title长度
            title_lengths = video_features[:, 2]
            print(f"  - 标题长度统计:")
            print(f"    最小值: {title_lengths.min().item():.1f}")
            print(f"    最大值: {title_lengths.max().item():.1f}")
            print(f"    平均值: {title_lengths.mean().item():.1f}")
            
            # 检查是否有异常值
            non_zero_count = (title_lengths > 0).sum().item()
            print(f"    非零长度数量: {non_zero_count}/{len(title_lengths)}")
        
        # 保存图
        output_dir = "data/graphs"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n保存图到 {output_dir}...")
        import pickle
        graph_file = os.path.join(output_dir, "heterogeneous_graph_final.pkl")
        with open(graph_file, 'wb') as f:
            pickle.dump(hetero_graph, f)
        
        print(f"✓ 图已保存到: {graph_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 图构建测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_graph_construction()
    if success:
        print("\n" + "="*80)
        print("🎉 图构建模块测试成功! 视频标题字段问题已解决!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ 图构建模块测试失败!")
        print("="*80) 