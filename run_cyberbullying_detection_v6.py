#!/usr/bin/env python3
"""
ProtoBully霸凌检测模块V6运行脚本
基于原型匹配结果进行会话级霸凌检测
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from prototype.cyberbullying_detector_v6 import CyberbullyingDetectorV6
import logging
from datetime import datetime
import json

def setup_logging():
    """设置日志系统"""
    log_dir = "cyberbullying_detection_v6"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"cyberbullying_detection_v6_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def main():
    """主函数"""
    print("="*60)
    print("ProtoBully霸凌检测模块V6")
    print("基于原型匹配结果进行会话级霸凌检测")
    print("="*60)
    
    # 设置日志
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ProtoBully Cyberbullying Detection V6")
    
    try:
        # 初始化检测器
        detector = CyberbullyingDetectorV6()
        
        # 数据路径配置
        prototype_path = "data/prototypes/extracted_prototypes_v5_20250624_025403.pkl"
        session_labels_path = "data/processed/prototypes/session_labels_from_graph.json"
        universal_subgraphs_dir = "data/subgraphs/universal_new"
        
        # 检查文件是否存在
        if not os.path.exists(prototype_path):
            logger.error(f"Prototype file not found: {prototype_path}")
            return False
        
        if not os.path.exists(session_labels_path):
            logger.error(f"Session labels file not found: {session_labels_path}")
            return False
        
        if not os.path.exists(universal_subgraphs_dir):
            logger.error(f"Universal subgraphs directory not found: {universal_subgraphs_dir}")
            return False
        
        # 加载数据
        logger.info("Step 1: Loading prototypes...")
        if not detector.load_prototypes(prototype_path):
            logger.error("Failed to load prototypes")
            return False
        
        logger.info("Step 2: Loading session labels...")
        if not detector.load_session_labels(session_labels_path):
            logger.error("Failed to load session labels")
            return False
        
        logger.info("Step 3: Loading universal subgraphs...")
        if not detector.load_universal_subgraphs(universal_subgraphs_dir):
            logger.error("Failed to load universal subgraphs")
            return False
        
        # 优化权重并训练模型
        logger.info("Step 4: Optimizing weights and training models...")
        best_results = detector.optimize_weights_and_train()
        
        if best_results is None:
            logger.error("Training failed!")
            return False
        
        # 保存模型
        logger.info("Step 5: Saving trained models...")
        models_dir = "data/models/cyberbullying_v6"
        timestamp = detector.save_models(models_dir)
        
        # 生成详细报告
        logger.info("Step 6: Generating evaluation report...")
        generate_detailed_report(best_results, log_file, timestamp)
        
        # 测试预测功能
        logger.info("Step 7: Testing prediction functionality...")
        test_predictions(detector, best_results['session_ids'][:5])
        
        logger.info("="*60)
        logger.info("ProtoBully霸凌检测模块V6训练完成!")
        logger.info(f"最佳权重组合: {best_results['weights']}")
        logger.info(f"最佳F1分数: {detector.best_performance:.3f}")
        logger.info(f"模型保存位置: {models_dir}")
        logger.info(f"详细日志: {log_file}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def generate_detailed_report(best_results, log_file, timestamp):
    """生成详细评估报告"""
    logger = logging.getLogger(__name__)
    
    results = best_results['results']
    weights = best_results['weights']
    
    # 创建报告目录
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # 生成文本报告
    report_file = os.path.join(report_dir, f"cyberbullying_detection_v6_report_{timestamp}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("ProtoBully霸凌检测模块V6评估报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"最佳权重组合: {weights}\n")
        f.write(f"特征数量: {len(best_results['features'][0])}\n")
        f.write(f"训练样本数: {len(best_results['labels'])}\n")
        f.write(f"霸凌样本比例: {best_results['labels'].mean():.3f}\n\n")
        
        # 模型性能对比
        f.write("模型性能对比\n")
        f.write("-"*40 + "\n")
        for model_name, result in results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  准确率: {result['accuracy']:.3f}\n")
            f.write(f"  精确率: {result['precision']:.3f}\n")
            f.write(f"  召回率: {result['recall']:.3f}\n")
            f.write(f"  F1分数: {result['f1']:.3f}\n")
            
            # 混淆矩阵
            cm = result['confusion_matrix']
            f.write(f"  混淆矩阵:\n")
            f.write(f"    实际\\预测  正常  霸凌\n")
            f.write(f"    正常     {cm[0,0]:4d}  {cm[0,1]:4d}\n")
            f.write(f"    霸凌     {cm[1,0]:4d}  {cm[1,1]:4d}\n")
        
        # 权重组合分析
        f.write("\n\n权重组合分析\n")
        f.write("-"*40 + "\n")
        f.write(f"结构相似度权重: {weights[0]:.1f}\n")
        f.write(f"聚合情感相似度权重: {weights[1]:.1f}\n")
        f.write(f"分层情感相似度权重: {weights[2]:.1f}\n")
    
    logger.info(f"详细报告已保存: {report_file}")

def test_predictions(detector, test_session_ids):
    """测试预测功能"""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing prediction functionality...")
    
    for session_id in test_session_ids:
        # RandomForest预测
        rf_result = detector.predict_session(session_id, 'RandomForest')
        
        # LogisticRegression预测
        lr_result = detector.predict_session(session_id, 'LogisticRegression')
        
        # NeuralNetwork预测
        nn_result = detector.predict_session(session_id, 'NeuralNetwork')
        
        logger.info(f"Session {session_id}:")
        logger.info(f"  RandomForest: {rf_result['prediction']} (confidence: {rf_result['confidence']:.3f})")
        logger.info(f"  LogisticRegression: {lr_result['prediction']} (confidence: {lr_result['confidence']:.3f})")
        if nn_result['status'] == 'success':
            logger.info(f"  NeuralNetwork: {nn_result['prediction']} (confidence: {nn_result['confidence']:.3f})")
        else:
            logger.info(f"  NeuralNetwork: {nn_result['message']}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 