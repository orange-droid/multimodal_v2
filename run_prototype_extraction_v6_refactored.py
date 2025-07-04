#!/usr/bin/env python3
"""
ProtoBully项目 - 原型提取器V6重构版运行脚本
基于弱监督学习的原型质量评估

运行方式：
python run_prototype_extraction_v6_refactored.py

作者：AI Assistant
日期：2025-06-28
版本：V6 Refactored
"""

import os
import sys
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.prototype.prototype_extractor_v6_session_enhanced import PrototypeExtractorV6Refactored


def main():
    """主函数"""
    print("=" * 80)
    print("ProtoBully项目 - 原型提取器V6重构版")
    print("基于弱监督学习的原型质量评估方法")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 配置参数
    config = {
        'purity_threshold': 0.7,      # 纯度阈值：原型中霸凌子图比例需要 > 70%
        'min_cluster_size': 15,       # 最小聚类大小：至少15个子图才能形成原型
        'eps': 0.3,                   # DBSCAN eps参数：聚类半径
        'min_samples': 8,             # DBSCAN min_samples参数：核心点最少邻居数
        'max_prototypes': 15,         # 最大原型数量：最多提取15个原型
        'feature_dim': 10,            # 特征维度
        'random_state': 42            # 随机种子
    }
    
    print("📋 配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 创建原型提取器
    print("🚀 初始化原型提取器...")
    extractor = PrototypeExtractorV6Refactored(config)
    
    # 运行完整提取流程
    print("🔄 开始原型提取流程...")
    start_time = time.time()
    
    result = extractor.run_full_extraction(
        enhanced_dir="data/subgraphs/universal_optimized_fixed",  # 使用最新修复的子图数据
        output_dir="data/prototypes"
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print()
    print("=" * 80)
    
    if result['success']:
        print("✅ 原型提取成功完成!")
        print()
        
        print("📊 统计信息:")
        stats = result['statistics']
        print(f"  总会话数: {stats['total_sessions']:,}")
        print(f"  霸凌会话: {stats['bullying_sessions']:,} ({stats['bullying_sessions']/stats['total_sessions']*100:.1f}%)")
        print(f"  正常会话: {stats['normal_sessions']:,} ({stats['normal_sessions']/stats['total_sessions']*100:.1f}%)")
        print(f"  总子图数: {stats['total_subgraphs']:,}")
        print(f"  霸凌子图: {stats['bullying_subgraphs']:,} ({stats['bullying_subgraphs']/stats['total_subgraphs']*100:.1f}%)")
        print(f"  正常子图: {stats['normal_subgraphs']:,} ({stats['normal_subgraphs']/stats['total_subgraphs']*100:.1f}%)")
        print(f"  提取原型: {stats['extracted_prototypes']}")
        print()
        
        print("⏱️  性能指标:")
        print(f"  总耗时: {total_time:.1f} 秒")
        print(f"  平均处理速度: {stats['total_subgraphs']/total_time:.0f} 子图/秒")
        print()
        
        print("📁 输出文件:")
        files = result['files']
        print(f"  原型文件: {files['prototypes_file']}")
        print(f"  摘要文件: {files['summary_file']}")
        print()
        
        print("🎯 核心创新:")
        print("  ✓ 弱监督学习：利用会话标签作为子图弱监督信号")
        print("  ✓ 4个科学指标：纯度、区分度、覆盖度、稳定性")
        print("  ✓ 数据驱动选择：基于统计特性而非主观权重")
        print("  ✓ 避免质量分数：不再依赖不可靠的内部计算评分")
        print()
        
        print("🔍 质量评估方法:")
        print("  • 原型纯度: 霸凌子图占比 > 70%")
        print("  • 原型区分度: 与正常子图的特征差异")
        print("  • 原型覆盖度: 覆盖的霸凌会话比例")
        print("  • 原型稳定性: 内部特征的一致性")
        
    else:
        print("❌ 原型提取失败!")
        print(f"错误信息: {result['error']}")
        return 1
    
    print()
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("V6重构版原型提取完成! 🎉")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 