#!/usr/bin/env python3
"""
原型提取模块运行脚本 V6 - 基于会话标签的弱监督学习
使用175万个增强版子图和会话标签进行智能原型提取
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import time
from pathlib import Path
from src.prototype.prototype_extractor_v6_session_enhanced import PrototypeExtractorV6SessionEnhanced
from src.utils.logger import setup_logger

# 全局logger
logger = setup_logger('prototype_extraction_v6')

def main():
    """主函数"""
    logger.info("=== ProtoBully 原型提取模块 V6 (会话标签增强版) ===")
    logger.info("基于会话标签的弱监督学习原型提取")
    
    start_time = time.time()
    
    # 检查输入数据
    enhanced_dir = "data/subgraphs/universal_enhanced"
    session_labels_file = "data/processed/prototypes/session_label_mapping.json"
    graph_file = "data/graphs/heterogeneous_graph_final.pkl"
    
    # 数据完整性检查
    missing_files = []
    if not Path(enhanced_dir).exists():
        missing_files.append(f"增强版子图目录: {enhanced_dir}")
    if not Path(session_labels_file).exists():
        missing_files.append(f"会话标签文件: {session_labels_file}")
    if not Path(graph_file).exists():
        missing_files.append(f"异构图文件: {graph_file}")
    
    if missing_files:
        logger.error("缺少必要的输入文件:")
        for file in missing_files:
            logger.error(f"  - {file}")
        logger.info("请确保已运行:")
        logger.info("  1. 子图提取模块: python fix_overwritten_sessions.py")
        logger.info("  2. 图构建模块: python src/graph_construction/graph_construction_runner.py")
        return
    
    logger.info("输入数据检查通过")
    logger.info(f"   增强版子图目录: {enhanced_dir}")
    logger.info(f"   会话标签文件: {session_labels_file}")
    logger.info(f"   异构图文件: {graph_file}")
    
    # 配置原型提取器V6
    config = {
        # 核心参数
        'min_prototype_size': 50,          # 原型最小子图数量
        'max_prototypes': 15,              # 最大原型数量
        'session_label_weight': 0.3,       # 会话标签权重
        
        # 分层筛选参数
        'bullying_session_priority': 2.0,  # 霸凌会话优先级
        'normal_sample_ratio': 0.3,        # 正常子图采样比例
        
        # 聚类参数
        'dbscan_eps': 0.4,                 # DBSCAN距离阈值
        'dbscan_min_samples': 10,          # DBSCAN最小样本数
        'similarity_threshold': 0.75,      # 相似度阈值
        
        # 特征权重
        'feature_weights': {
            'session_signal': 0.25,        # 会话标签信号权重
            'attack_features': 0.3,        # 攻击性特征权重
            'user_behavior': 0.2,          # 用户行为特征权重
            'structure': 0.25              # 图结构特征权重
        }
    }
    
    logger.info("原型提取器V6配置:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"   {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"     {sub_key}: {sub_value}")
        else:
            logger.info(f"   {key}: {value}")
    
    # 创建原型提取器V6
    logger.info("初始化原型提取器V6...")
    extractor = PrototypeExtractorV6SessionEnhanced(config)
    
    # 运行原型提取
    logger.info("开始V6原型提取流程...")
    logger.info("流程概述:")
    logger.info("  第一层筛选: 基于会话标签分离霸凌/正常子图")
    logger.info("  第二层筛选: 基于特征规则筛选高质量霸凌子图")
    logger.info("  第三层筛选: 对比学习分析判别性特征")
    logger.info("  聚类分析: 提取代表性原型")
    
    result = extractor.run_full_extraction(
        enhanced_dir=enhanced_dir,
        output_dir="data/prototypes"
    )
    
    # 输出结果
    total_time = time.time() - start_time
    
    if result['success']:
        logger.info("原型提取V6成功完成!")
        logger.info("=" * 60)
        logger.info("V6提取结果摘要:")
        
        stats = result['stats']
        logger.info(f"   总会话数: {stats['total_sessions']}")
        logger.info(f"   霸凌会话数: {stats['bullying_sessions']} ({stats['bullying_sessions']/stats['total_sessions']*100:.1f}%)")
        logger.info(f"   正常会话数: {stats['normal_sessions']} ({stats['normal_sessions']/stats['total_sessions']*100:.1f}%)")
        
        logger.info(f"   总子图数: {stats['total_subgraphs']:,}")
        logger.info(f"   霸凌候选子图: {stats['bullying_candidate_subgraphs']:,}")
        logger.info(f"   正常参考子图: {stats['normal_reference_subgraphs']:,}")
        logger.info(f"   高质量霸凌子图: {stats['high_quality_bullying_subgraphs']:,}")
        
        if stats['bullying_candidate_subgraphs'] > 0:
            filter_rate = stats['high_quality_bullying_subgraphs'] / stats['bullying_candidate_subgraphs'] * 100
            logger.info(f"   筛选通过率: {filter_rate:.1f}%")
        
        logger.info(f"   处理时间: {stats['processing_time']:.1f}秒")
        logger.info(f"   总运行时间: {total_time:.1f}秒")
        
        # 输出文件信息
        if result['files']:
            logger.info("输出文件:")
            for file_type, file_path in result['files'].items():
                if file_path:
                    logger.info(f"   {file_type}: {file_path}")
        
        logger.info("=" * 60)
        logger.info("V6技术创新点:")
        logger.info("   ✓ 会话标签弱监督学习")
        logger.info("   ✓ 分层筛选策略")
        logger.info("   ✓ 对比学习机制")
        logger.info("   ✓ 175万子图规模效应")
        
        logger.info("下一步建议:")
        logger.info("   1. 对比V5和V6版本的原型质量")
        logger.info("   2. 使用V6原型进行霸凌检测测试")
        logger.info("   3. 分析判别性特征的有效性")
        
    else:
        logger.error("原型提取V6失败!")
        logger.error(f"   错误信息: {result.get('error', '未知错误')}")
        logger.info("可能的解决方案:")
        logger.info("   1. 检查会话标签文件格式")
        logger.info("   2. 确认增强版子图数据完整性")
        logger.info("   3. 调整筛选阈值参数")
        logger.info("   4. 检查系统内存是否充足")
        
        # 提供回退方案
        logger.info("可以回退到V5版本:")
        logger.info("   python run_prototype_extraction_v5.py")

if __name__ == "__main__":
    main() 