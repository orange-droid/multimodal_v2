#!/usr/bin/env python3
"""
继续运行增强版通用子图提取器

只处理缺失的会话（945-958）
"""

import os
import sys
import pickle
import logging
import time
import glob
from datetime import datetime

# 添加src目录到路径
sys.path.append('src')

from prototype.universal_subgraph_extractor import UniversalSubgraphExtractor

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'continue_enhanced_subgraph_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def get_missing_sessions(output_dir: str) -> list:
    """获取缺失的会话列表"""
    # 获取已存在的会话文件
    existing_files = glob.glob(f'{output_dir}/media_session_*_subgraphs.pkl')
    existing_sessions = set()
    
    for file in existing_files:
        filename = os.path.basename(file)
        session_num = int(filename.replace('media_session_', '').replace('_subgraphs.pkl', ''))
        existing_sessions.add(session_num)
    
    # 找出缺失的会话
    all_sessions = set(range(959))
    missing_sessions = sorted(list(all_sessions - existing_sessions))
    
    return missing_sessions

class ContinueExtractor(UniversalSubgraphExtractor):
    """继续提取的子图提取器"""
    
    def __init__(self, config=None, target_sessions=None):
        super().__init__(config)
        self.target_sessions = target_sessions or []
    
    def _get_all_sessions(self, graph):
        """只返回目标会话"""
        return [f"media_session_{i}" for i in self.target_sessions]

def main():
    """主函数"""
    print("=== 继续增强版通用子图提取器 ===")
    print("只处理缺失的会话")
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 检查输出目录
    output_dir = "data/subgraphs/universal_enhanced"
    if not os.path.exists(output_dir):
        logger.error(f"输出目录不存在: {output_dir}")
        return
    
    # 获取缺失的会话
    missing_sessions = get_missing_sessions(output_dir)
    
    if not missing_sessions:
        logger.info("所有会话都已提取完成，无需继续提取")
        return
    
    logger.info(f"发现 {len(missing_sessions)} 个缺失的会话: {missing_sessions}")
    
    # 增强配置
    config = {
        # 基本配置
        'min_subgraph_size': 6,
        'max_subgraph_size': 15,
        'size_step': 1,
        
        # 节点数量限制
        'max_comments_per_video': 50,
        'max_users_per_subgraph': 8,
        'max_words_per_subgraph': 25,
        'max_seeds_per_session': 30,
        
        # 多用户交互配置
        'enable_multi_user_subgraphs': True,
        'min_interacting_users': 2,
        'max_multi_user_combinations': 100,
        
        # 完全枚举配置
        'enable_smart_selection': True,
        'max_enumeration_combinations': 500,
        
        # 其他策略配置
        'enable_single_type_subgraphs': True,
        'single_type_min_size': 6,
        'single_type_max_size': 15,
        'random_sampling_ratio': 0.3,
        'diversity_boost': True,
        'enable_large_subgraphs': True,
        
        # 策略权重
        'strategy_weights': {
            'multi_user_interaction': 0.25,  # 多用户交互
            'smart_selection': 0.25,         # 智能选取
            'traditional': 0.2,              # 传统多类型
            'single_type': 0.15,             # 单类型
            'random_combo': 0.15             # 随机组合
        }
    }
    
    logger.info("配置参数:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 创建继续提取器
    logger.info("创建继续增强版子图提取器...")
    extractor = ContinueExtractor(config, missing_sessions)
    
    # 加载图数据
    graph_path = "data/graphs/heterogeneous_graph_final.pkl"
    logger.info(f"加载异构图数据: {graph_path}")
    
    if not os.path.exists(graph_path):
        logger.error(f"图文件不存在: {graph_path}")
        return
    
    try:
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        
        logger.info(f"图加载成功:")
        logger.info(f"  节点类型: {graph.node_types}")
        logger.info(f"  边类型: {graph.edge_types}")
        logger.info(f"  节点数量: {[f'{nt}({graph[nt].x.shape[0]})' for nt in graph.node_types]}")
        logger.info(f"  边数量: {[f'{et}({graph[et].edge_index.size(1)})' for et in graph.edge_types]}")
        
    except Exception as e:
        logger.error(f"加载图数据失败: {e}")
        return
    
    # 开始提取
    logger.info("开始继续增强版子图提取...")
    start_time = time.time()
    
    try:
        stats = extractor.extract_all_session_subgraphs(graph, output_dir)
        
        end_time = time.time()
        extraction_time = end_time - start_time
        
        # 输出统计结果
        logger.info("=" * 60)
        logger.info("继续增强版子图提取完成！")
        logger.info("=" * 60)
        logger.info(f"总耗时: {extraction_time:.2f} 秒")
        logger.info(f"处理会话数: {stats['total_sessions']}")
        logger.info(f"成功会话数: {stats['successful_sessions']}")
        logger.info(f"总子图数: {stats['total_subgraphs']}")
        logger.info(f"平均每会话子图数: {stats['avg_subgraphs_per_session']:.1f}")
        
        logger.info("\n子图类型分布:")
        for subgraph_type, count in stats['subgraph_type_distribution'].items():
            percentage = (count / stats['total_subgraphs']) * 100 if stats['total_subgraphs'] > 0 else 0
            logger.info(f"  {subgraph_type}: {count} ({percentage:.1f}%)")
        
        logger.info("\n子图大小分布:")
        size_items = sorted(stats['size_distribution'].items())
        for size, count in size_items[:10]:  # 只显示前10个
            percentage = (count / stats['total_subgraphs']) * 100 if stats['total_subgraphs'] > 0 else 0
            logger.info(f"  {size}节点: {count} ({percentage:.1f}%)")
        
        if len(size_items) > 10:
            logger.info(f"  ... 还有 {len(size_items) - 10} 种大小")
        
        logger.info("\n增强特性统计:")
        enhancement_features = stats['enhancement_features']
        logger.info(f"  多用户交互启用: {enhancement_features['multi_user_enabled']}")
        logger.info(f"  智能选取启用: {enhancement_features['smart_selection_enabled']}")
        logger.info(f"  用户交互关系数: {enhancement_features['user_interactions_loaded']}")
        logger.info(f"  节点数量范围: {enhancement_features['size_range']}")
        
        # 计算性能指标
        if extraction_time > 0:
            sessions_per_second = stats['successful_sessions'] / extraction_time
            subgraphs_per_second = stats['total_subgraphs'] / extraction_time
            logger.info(f"\n性能指标:")
            logger.info(f"  处理速度: {sessions_per_second:.2f} 会话/秒")
            logger.info(f"  生成速度: {subgraphs_per_second:.2f} 子图/秒")
        
        logger.info(f"\n结果保存在: {output_dir}")
        logger.info("=" * 60)
        
        # 验证是否完成
        final_missing = get_missing_sessions(output_dir)
        if not final_missing:
            logger.info("🎉 所有会话子图提取完成！")
        else:
            logger.warning(f"仍有 {len(final_missing)} 个会话未完成: {final_missing}")
        
        return stats
        
    except Exception as e:
        logger.error(f"子图提取过程中出错: {e}")
        import traceback
        logger.error(f"错误详情:\n{traceback.format_exc()}")
        return None

if __name__ == "__main__":
    main() 