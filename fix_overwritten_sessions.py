#!/usr/bin/env python3
"""
完全重新进行增强版子图提取
覆盖所有会话（0-958），确保使用最新的增强版逻辑
"""

import os
import sys
import pickle
import logging
import time
from datetime import datetime
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入增强版子图提取器
from src.prototype.universal_subgraph_extractor import UniversalSubgraphExtractor

def setup_logging():
    """设置日志"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"complete_enhanced_subgraph_extraction_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__), log_file

def load_heterogeneous_graph(graph_path: str):
    """加载异构图"""
    print(f"📊 加载异构图: {graph_path}")
    
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    # 获取会话信息
    media_session_nodes = graph['media_session'].x
    total_sessions = len(media_session_nodes)
    
    print(f"✅ 异构图加载完成")
    print(f"   总会话数: {total_sessions}")
    print(f"   节点类型: {list(graph.node_types)}")
    print(f"   边类型: {list(graph.edge_types)}")
    
    return graph, total_sessions

def main():
    """主函数"""
    print("🚀 开始完全重新进行增强版子图提取...")
    print("📋 本次将覆盖所有会话（0-958），使用最新的增强版逻辑")
    
    # 设置日志
    logger, log_file = setup_logging()
    logger.info("开始完全重新进行增强版子图提取")
    
    start_time = time.time()
    
    try:
        # 配置参数
        config = {
            'min_subgraph_size': 6,
            'max_subgraph_size': 15,
            'max_enumeration_combinations': 500,  # 使用完整配置，不限制
            'enable_multi_user_interactions': True,
            'enable_multi_size_extraction': True,
            'enable_complete_enumeration': True,
            'interaction_threshold': 0.1,
            'batch_size': 50,
            'save_frequency': 10
        }
        
        # 数据路径
        graph_path = "data/graphs/heterogeneous_graph_final.pkl"
        output_dir = "data/subgraphs/universal_enhanced"
        
        # 检查输入文件
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"异构图文件不存在: {graph_path}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 输出目录: {output_dir}")
        
        # 加载异构图
        graph, total_sessions = load_heterogeneous_graph(graph_path)
        
        # 创建子图提取器
        print("🔧 初始化增强版子图提取器...")
        extractor = UniversalSubgraphExtractor(config)
        
        # 提取所有会话的子图（带进度条）
        print(f"\n🎯 开始提取所有 {total_sessions} 个会话的子图...")
        
        total_subgraphs = 0
        failed_sessions = []
        
        # 使用tqdm显示进度条
        with tqdm(total=total_sessions, desc="提取进度", unit="会话") as pbar:
            for session_id in range(total_sessions):
                try:
                    # 更新进度条描述
                    pbar.set_description(f"提取会话 {session_id}")
                    
                    # 提取单个会话的子图
                    session_subgraphs = extractor._extract_session_subgraphs_enhanced(
                        graph, f"media_session_{session_id}"
                    )
                    
                    if session_subgraphs:
                        # 保存子图
                        output_file = os.path.join(output_dir, f"media_session_{session_id}_subgraphs.pkl")
                        with open(output_file, 'wb') as f:
                            pickle.dump(session_subgraphs, f)
                        
                        subgraph_count = len(session_subgraphs)
                        total_subgraphs += subgraph_count
                        
                        # 更新进度条后缀信息
                        pbar.set_postfix({
                            '子图数': subgraph_count,
                            '总计': total_subgraphs,
                            '失败': len(failed_sessions)
                        })
                        
                        logger.info(f"会话 {session_id}: 生成 {subgraph_count} 个子图")
                    else:
                        failed_sessions.append(session_id)
                        logger.warning(f"会话 {session_id}: 未生成任何子图")
                        
                        pbar.set_postfix({
                            '子图数': 0,
                            '总计': total_subgraphs,
                            '失败': len(failed_sessions)
                        })
                
                except Exception as e:
                    failed_sessions.append(session_id)
                    logger.error(f"会话 {session_id} 处理失败: {e}")
                    
                    pbar.set_postfix({
                        '子图数': 0,
                        '总计': total_subgraphs,
                        '失败': len(failed_sessions)
                    })
                
                # 更新进度条
                pbar.update(1)
        
        # 计算耗时
        end_time = time.time()
        duration = end_time - start_time
        
        # 生成统计报告
        successful_sessions = total_sessions - len(failed_sessions)
        success_rate = (successful_sessions / total_sessions) * 100
        avg_subgraphs_per_session = total_subgraphs / successful_sessions if successful_sessions > 0 else 0
        
        print(f"\n🎉 完全重新提取完成！")
        print(f"📊 统计结果:")
        print(f"   总会话数: {total_sessions}")
        print(f"   成功处理: {successful_sessions} ({success_rate:.1f}%)")
        print(f"   失败会话: {len(failed_sessions)}")
        print(f"   总子图数: {total_subgraphs:,}")
        print(f"   平均每会话: {avg_subgraphs_per_session:.1f} 个子图")
        print(f"   总耗时: {duration:.2f} 秒")
        print(f"   平均速度: {total_sessions/duration:.1f} 会话/秒")
        
        # 记录到日志
        logger.info("完全重新提取完成")
        logger.info(f"成功处理: {successful_sessions}/{total_sessions} ({success_rate:.1f}%)")
        logger.info(f"总子图数: {total_subgraphs:,}")
        logger.info(f"平均每会话: {avg_subgraphs_per_session:.1f} 个子图")
        logger.info(f"总耗时: {duration:.2f} 秒")
        
        if failed_sessions:
            print(f"\n⚠️  失败的会话: {failed_sessions[:10]}{'...' if len(failed_sessions) > 10 else ''}")
            logger.warning(f"失败的会话: {failed_sessions}")
        
        print(f"\n📝 详细日志已保存至: {log_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"完全重新提取失败: {e}")
        print(f"❌ 完全重新提取失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 