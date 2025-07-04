#!/usr/bin/env python3
"""
运行霸凌检测模块V6适配版本
基于V6原型进行霸凌检测测试
"""

import sys
import os
from datetime import datetime
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cyberbullying_detector_v6_adapted import CyberbullyingDetectorV6Adapted

def setup_logging():
    """设置日志系统"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"cyberbullying_detection_v6_adapted_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_filename

def main():
    """主函数"""
    print("🚀 启动霸凌检测模块V6适配版本测试...")
    
    # 设置日志
    log_filename = setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 初始化检测器
        detector = CyberbullyingDetectorV6Adapted()
        
        # 数据路径配置 - 使用最新基于修复子图的原型
        prototype_path = "data/prototypes/extracted_prototypes_v6_refactored_20250628_100456.pkl"  # 最新原型
        session_labels_path = "data/processed/prototypes/session_label_mapping.json"
        subgraph_dir = "data/subgraphs/universal_optimized_fixed"  # 最新修复的子图数据
        graph_path = "data/graphs/heterogeneous_graph_final.pkl"  # 异构图（包含真实特征）
        
        print(f"📁 数据路径配置:")
        print(f"   原型文件: {prototype_path}")
        print(f"   会话标签: {session_labels_path}")
        print(f"   子图目录: {subgraph_dir}")
        print(f"   异构图: {graph_path}")
        
        # 验证文件存在性
        if not Path(prototype_path).exists():
            raise FileNotFoundError(f"原型文件不存在: {prototype_path}")
        if not Path(session_labels_path).exists():
            raise FileNotFoundError(f"会话标签文件不存在: {session_labels_path}")
        if not Path(subgraph_dir).exists():
            raise FileNotFoundError(f"子图目录不存在: {subgraph_dir}")
        if not Path(graph_path).exists():
            raise FileNotFoundError(f"异构图文件不存在: {graph_path}")
        
        # 加载数据
        print("\n📊 加载数据...")
        
        print("   加载异构图（包含真实特征）...")
        if not detector.load_heterogeneous_graph(graph_path):
            raise Exception("Failed to load heterogeneous graph")
        
        print("   加载V6原型数据...")
        if not detector.load_prototypes(prototype_path):
            raise Exception("Failed to load prototypes")
        
        print("   加载会话标签...")
        if not detector.load_session_labels(session_labels_path):
            raise Exception("Failed to load session labels")
        
        print("   加载子图数据...")
        if not detector.load_universal_subgraphs(subgraph_dir):
            raise Exception("Failed to load subgraphs")
        
        print("✅ 数据加载完成!")
        
        # 优化权重并训练模型
        print("\n🔧 开始权重优化和模型训练...")
        optimization_results = detector.optimize_weights_and_train()
        
        print(f"\n🎯 优化结果:")
        print(f"   最佳权重组合: {optimization_results['best_weights']}")
        print(f"   最佳F1分数: {optimization_results['best_f1']:.3f}")
        
        # 详细结果展示
        best_results = optimization_results['best_results']
        print(f"\n📈 各模型详细性能:")
        for model_name, result in best_results.items():
            print(f"   {model_name}:")
            print(f"     准确率: {result['accuracy']:.3f}")
            print(f"     精确率: {result['precision']:.3f}")
            print(f"     召回率: {result['recall']:.3f}")
            print(f"     F1分数: {result['f1']:.3f}")
        
        # 保存模型
        output_dir = "data/models/cyberbullying_v6_adapted"
        print(f"\n💾 保存模型到: {output_dir}")
        save_results = detector.save_models(output_dir)
        
        print(f"✅ 模型保存完成!")
        print(f"   时间戳: {save_results['timestamp']}")
        print(f"   保存的模型数量: {save_results['files_saved']['models']}")
        
        # 测试几个会话的预测
        print(f"\n🧪 测试会话预测...")
        test_sessions = ['media_session_0', 'media_session_1', 'media_session_10']
        
        for session_id in test_sessions:
            if session_id in detector.session_labels:
                true_label = detector.session_labels[session_id]['is_bullying']
                
                for model_name in ['RandomForest', 'LogisticRegression']:
                    if model_name in detector.models:
                        prediction = detector.predict_session(session_id, model_name)
                        
                        print(f"   {session_id} ({model_name}):")
                        print(f"     真实标签: {true_label}")
                        print(f"     预测结果: {prediction['prediction']}")
                        print(f"     预测概率: {prediction['probability']:.3f}")
                        print(f"     预测正确: {'✅' if prediction['prediction'] == true_label else '❌'}")
        
        # 生成总结报告
        print(f"\n📋 V6适配版检测器测试总结:")
        print(f"   原型数量: {len(detector.prototypes)}")
        print(f"   会话数量: {len(detector.session_labels)}")
        print(f"   子图数量: {sum(len(subgraphs) for subgraphs in detector.universal_subgraphs.values())}")
        print(f"   特征维度: {len(detector.feature_names)}")
        print(f"   最佳权重: {detector.best_weights}")
        print(f"   最佳性能: {detector.best_performance:.3f}")
        print(f"   日志文件: {log_filename}")
        
        print(f"\n🎉 V6适配版霸凌检测器测试完成!")
        
        return {
            'success': True,
            'best_weights': detector.best_weights,
            'best_performance': detector.best_performance,
            'models_trained': len(detector.models),
            'log_file': log_filename
        }
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        print(f"❌ 测试失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'log_file': log_filename
        }

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result['success'] else 1) 