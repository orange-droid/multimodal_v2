#!/usr/bin/env python3
"""
原型提取模块运行脚本 V5
使用新版本霸凌子图提取器的输出作为输入，进行改进的原型提取
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import time
from pathlib import Path
from src.prototype.prototype_extractor_v5_new import PrototypeExtractorV5
from src.utils.logger import setup_logger

# 全局logger
logger = setup_logger('prototype_extraction_v5')

def main():
    """主函数"""
    logger.info("=== ProtoBully 原型提取模块 V5 ===")
    logger.info("基于新版本霸凌子图数据进行原型提取")
    
    start_time = time.time()
    
    # 检查输入数据
    bullying_dir = "data/subgraphs/bullying_new"
    if not Path(bullying_dir).exists():
        logger.error(f"霸凌子图目录不存在: {bullying_dir}")
        logger.info("请先运行子图提取模块: python run_new_subgraph_extraction.py")
        return
    
    # 检查索引文件
    index_file = Path(bullying_dir) / "bullying_subgraph_index.json"
    if not index_file.exists():
        logger.error(f"霸凌子图索引文件不存在: {index_file}")
        return
    
    logger.info("输入数据检查通过")
    logger.info(f"   霸凌子图目录: {bullying_dir}")
    
    # 配置原型提取器（增加原型数量）
    config = {
        # 核心参数
        'min_prototype_size': 30,       # 降低原型最小子图数量以获得更多原型
        'max_prototypes': 12,           # 增加最大原型数量
        
        # 聚类参数
        'dbscan_eps': 0.5,             # 降低DBSCAN距离阈值以获得更多聚类
        'dbscan_min_samples': 8,        # 降低DBSCAN最小样本数
        'similarity_threshold': 0.7,    # 降低相似度阈值
        
        # 特征权重
        'feature_weights': {
            'size': 0.15,              # 子图大小权重
            'emotion': 0.4,            # 情感分数权重（增加）
            'node_composition': 0.25,  # 节点组成权重
            'structure': 0.2           # 结构特征权重
        }
    }
    
    logger.info("原型提取器配置:")
    for key, value in config.items():
        logger.info(f"   {key}: {value}")
    
    # 创建原型提取器
    extractor = PrototypeExtractorV5(config)
    
    # 运行原型提取
    logger.info("开始原型提取流程...")
    
    result = extractor.run_full_extraction(
        bullying_dir=bullying_dir,
        output_dir="data/prototypes"
    )
    
    # 输出结果
    total_time = time.time() - start_time
    
    if result['success']:
        logger.info("原型提取成功完成!")
        logger.info("=" * 50)
        logger.info("提取结果摘要:")
        logger.info(f"   输入霸凌子图数: {result['stats']['total_subgraphs']}")
        logger.info(f"   参与聚类的子图数: {result['stats']['clustered_subgraphs']}")
        logger.info(f"   提取的原型数: {result['prototypes_count']}")
        logger.info(f"   聚类分布: {result['stats']['cluster_distribution']}")
        logger.info(f"   特征提取时间: {result['stats']['processing_time']:.1f}秒")
        logger.info(f"   总处理时间: {total_time:.1f}秒")
        
        logger.info("输出文件:")
        for file_type, file_path in result['files'].items():
            logger.info(f"   {file_type}: {file_path}")
        
        # 输出原型详情
        logger.info("原型详情:")
        for i, prototype in enumerate(extractor.prototypes):
            logger.info(f"   原型 {i+1}: {prototype['name']}")
            logger.info(f"     代表子图数: {prototype['subgraph_count']}")
            logger.info(f"     质量分数: {prototype['quality_score']:.3f}")
            
            features = prototype['representative_features']
            logger.info(f"     平均大小: {features['avg_size']:.1f}")
            logger.info(f"     平均情感分数: {features['avg_emotion_score']:.3f}")
            logger.info(f"     评论节点数: {features['avg_comment_count']:.1f}")
            logger.info(f"     用户节点数: {features['avg_user_count']:.1f}")
        
        logger.info("=" * 50)
        logger.info("原型提取模块运行完成")
        logger.info("接下来可以:")
        logger.info("   1. 查看原型摘要文件了解详细信息")
        logger.info("   2. 使用原型进行霸凌检测模型训练")
        logger.info("   3. 进行多模态对齐优化")
        
    else:
        logger.error("原型提取失败!")
        logger.error(f"   错误信息: {result.get('error', '未知错误')}")
        logger.info("可能的解决方案:")
        logger.info("   1. 检查霸凌子图数据是否完整")
        logger.info("   2. 调整聚类参数")
        logger.info("   3. 检查系统内存是否充足")

if __name__ == "__main__":
    main() 