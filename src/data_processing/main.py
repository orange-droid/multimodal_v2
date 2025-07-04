"""
数据处理模块主入口

运行所有数据处理步骤，包括文本、视频、元数据处理和多模态对齐
"""
import os
import sys
import argparse
import time
from tqdm import tqdm

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# 添加ProtoBully项目目录到路径
protobully_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(protobully_root)

from src.utils.config import DATA_CONFIG, PREPROCESSING_CONFIG
from src.utils.logger import setup_logger
from src.utils.utils import ensure_dir

# 导入各个处理器
from src.data_processing.text_processor import TextProcessor
from src.data_processing.video_processor import VideoProcessor
from src.data_processing.metadata_processor import MetadataProcessor
from src.data_processing.multimodal_processor import MultimodalProcessor

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ProtoBully数据处理模块")
    parser.add_argument("--force", action="store_true", help="强制重新处理所有数据")
    parser.add_argument("--text-only", action="store_true", help="只处理文本数据")
    parser.add_argument("--video-only", action="store_true", help="只处理视频数据")
    parser.add_argument("--metadata-only", action="store_true", help="只处理元数据")
    parser.add_argument("--multimodal-only", action="store_true", help="只处理多模态对齐")
    parser.add_argument("--summary", action="store_true", help="只显示处理后的数据统计摘要")
    
    return parser.parse_args()

def process_text_data(force_reprocess=False):
    """处理文本数据"""
    print("\n" + "="*80)
    print("开始处理文本数据...")
    print("="*80)
    
    start_time = time.time()
    processor = TextProcessor()
    data = processor.get_processed_data(force_reprocess)
    
    elapsed_time = time.time() - start_time
    print(f"文本数据处理完成，处理了 {len(data)} 个帖子，耗时 {elapsed_time:.2f} 秒")
    
    return data

def process_video_data(force_reprocess=False):
    """处理视频数据"""
    print("\n" + "="*80)
    print("开始处理视频数据...")
    print("="*80)
    
    start_time = time.time()
    processor = VideoProcessor()
    data = processor.get_processed_data(force_reprocess)
    
    elapsed_time = time.time() - start_time
    print(f"视频数据处理完成，处理了 {len(data)} 个帖子，耗时 {elapsed_time:.2f} 秒")
    
    return data

def process_metadata(force_reprocess=False):
    """处理元数据"""
    print("\n" + "="*80)
    print("开始处理元数据...")
    print("="*80)
    
    start_time = time.time()
    processor = MetadataProcessor()
    data = processor.get_processed_data(force_reprocess)
    
    elapsed_time = time.time() - start_time
    print(f"元数据处理完成，处理了 {len(data)} 个帖子，耗时 {elapsed_time:.2f} 秒")
    
    return data

def process_multimodal_data(force_reprocess=False):
    """处理多模态数据"""
    print("\n" + "="*80)
    print("开始处理多模态数据...")
    print("="*80)
    
    start_time = time.time()
    processor = MultimodalProcessor()
    data = processor.get_processed_data(force_reprocess)
    
    elapsed_time = time.time() - start_time
    print(f"多模态数据处理完成，对齐了 {len(data['aligned_data'])} 个帖子，耗时 {elapsed_time:.2f} 秒")
    
    # 打印数据集划分统计
    train_ids = data['dataset_splits']['train']
    val_ids = data['dataset_splits']['val']
    test_ids = data['dataset_splits']['test']
    
    train_labels = [data['aligned_data'][post_id]['label'] for post_id in train_ids]
    val_labels = [data['aligned_data'][post_id]['label'] for post_id in val_ids]
    test_labels = [data['aligned_data'][post_id]['label'] for post_id in test_ids]
    
    train_bullying = sum(train_labels)
    train_non_bullying = len(train_labels) - train_bullying
    
    val_bullying = sum(val_labels)
    val_non_bullying = len(val_labels) - val_bullying
    
    test_bullying = sum(test_labels)
    test_non_bullying = len(test_labels) - test_bullying
    
    print("\n数据集划分统计:")
    print(f"训练集: {len(train_ids)} 条 (霸凌: {train_bullying}, 非霸凌: {train_non_bullying})")
    print(f"验证集: {len(val_ids)} 条 (霸凌: {val_bullying}, 非霸凌: {val_non_bullying})")
    print(f"测试集: {len(test_ids)} 条 (霸凌: {test_bullying}, 非霸凌: {test_non_bullying})")
    
    return data

def display_data_summary():
    """显示处理后的数据统计摘要"""
    print("\n" + "="*80)
    print("数据处理摘要")
    print("="*80)
    
    # 检查各个处理后的数据文件是否存在
    text_file = os.path.join(DATA_CONFIG["text_data_path"], 'processed_text_data.json')
    video_file = os.path.join(DATA_CONFIG["video_data_path"], 'processed_video_data.json')
    metadata_file = os.path.join(DATA_CONFIG["metadata_data_path"], 'processed_metadata.json')
    multimodal_file = os.path.join(DATA_CONFIG["aligned_data_path"], 'multimodal_data.json')
    
    text_exists = os.path.exists(text_file)
    video_exists = os.path.exists(video_file)
    metadata_exists = os.path.exists(metadata_file)
    multimodal_exists = os.path.exists(multimodal_file)
    
    print(f"文本数据处理完成: {'是' if text_exists else '否'}")
    print(f"视频数据处理完成: {'是' if video_exists else '否'}")
    print(f"元数据处理完成: {'是' if metadata_exists else '否'}")
    print(f"多模态数据处理完成: {'是' if multimodal_exists else '否'}")
    
    # 如果多模态数据已处理，显示其统计信息
    if multimodal_exists:
        processor = MultimodalProcessor()
        data = processor.load_processed_data(multimodal_file)
        
        aligned_count = len(data['aligned_data'])
        bullying_count = sum(1 for post_id in data['aligned_data'] if data['aligned_data'][post_id]['label'] == 1)
        non_bullying_count = aligned_count - bullying_count
        
        print(f"\n多模态数据统计:")
        print(f"总样本数: {aligned_count}")
        print(f"霸凌样本数: {bullying_count} ({bullying_count/aligned_count*100:.2f}%)")
        print(f"非霸凌样本数: {non_bullying_count} ({non_bullying_count/aligned_count*100:.2f}%)")
        
        # 打印数据集划分统计
        train_ids = data['dataset_splits']['train']
        val_ids = data['dataset_splits']['val']
        test_ids = data['dataset_splits']['test']
        
        print("\n数据集划分统计:")
        print(f"训练集: {len(train_ids)} 条 ({len(train_ids)/aligned_count*100:.2f}%)")
        print(f"验证集: {len(val_ids)} 条 ({len(val_ids)/aligned_count*100:.2f}%)")
        print(f"测试集: {len(test_ids)} 条 ({len(test_ids)/aligned_count*100:.2f}%)")

def main():
    """主函数"""
    args = parse_args()
    
    # 确保输出目录存在
    for path_key in ["processed_data_path", "text_data_path", "video_data_path", 
                     "metadata_data_path", "aligned_data_path"]:
        ensure_dir(DATA_CONFIG[path_key])
    
    # 如果只显示摘要
    if args.summary:
        display_data_summary()
        return
    
    # 决定处理哪些模态
    should_process_text = not (args.video_only or args.metadata_only or args.multimodal_only)
    should_process_video = not (args.text_only or args.metadata_only or args.multimodal_only)
    should_process_metadata = not (args.text_only or args.video_only or args.multimodal_only)
    should_process_multimodal = not (args.text_only or args.video_only or args.metadata_only)
    
    # 如果没有指定任何特定模态，则处理所有模态
    if not (should_process_text or should_process_video or should_process_metadata or should_process_multimodal):
        should_process_text = should_process_video = should_process_metadata = should_process_multimodal = True
    
    # 处理各个模态的数据
    if should_process_text:
        process_text_data(args.force)
    
    if should_process_video:
        process_video_data(args.force)
    
    if should_process_metadata:
        process_metadata(args.force)
    
    if should_process_multimodal:
        process_multimodal_data(args.force)
    
    print("\n" + "="*80)
    print("所有数据处理完成！")
    print("="*80)

if __name__ == "__main__":
    main() 