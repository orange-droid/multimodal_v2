"""
工具函数模块
"""
import os
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm

def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def load_json_data(file_path, limit=None):
    """
    加载JSON数据文件
    
    参数:
        file_path: JSON文件路径
        limit: 可选，限制加载的数据量
        
    返回:
        加载的JSON数据
    """
    print(f"正在加载 {file_path}...")
    
    try:
        # 尝试作为完整JSON对象加载
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 如果数据是字典类型，直接返回
            if isinstance(data, dict):
                print(f"成功加载JSON对象，包含 {len(data)} 个键值对")
                return data
            
            # 如果数据是列表类型，并且设置了限制，裁剪列表
            if isinstance(data, list) and limit is not None:
                data = data[:limit]
                print(f"成功加载 {len(data)} 条记录")
                return data
            
            print(f"成功加载 {len(data)} 条记录")
            return data
            
    except json.JSONDecodeError:
        # 如果不是完整的JSON对象，尝试按行加载
        print("无法作为完整JSON对象加载，尝试按行加载...")
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            # 显示进度条
            file_size = os.path.getsize(file_path)
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="加载JSON数据") as pbar:
                line_count = 0
                for line in f:
                    if line.strip():  # 跳过空行
                        try:
                            json_obj = json.loads(line.strip())
                            data.append(json_obj)
                            line_count += 1
                            
                            # 如果设置了限制，达到限制后停止
                            if limit is not None and line_count >= limit:
                                break
                        except json.JSONDecodeError:
                            print(f"警告: 无法解析JSON行: {line[:50]}...")
                    
                    # 更新进度条
                    pbar.update(len(line.encode('utf-8')))
        
        print(f"成功按行加载 {len(data)} 条记录")
        return data

def load_csv_data(file_path, limit=None):
    """
    加载CSV数据文件
    
    参数:
        file_path: CSV文件路径
        limit: 可选，限制加载的数据量
        
    返回:
        加载的DataFrame
    """
    print(f"正在加载 {file_path}...")
    
    if limit is not None:
        # 如果指定了限制，使用nrows参数
        df = pd.read_csv(file_path, nrows=limit)
    else:
        # 显示进度条的CSV加载方式
        # 首先获取总行数
        total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1  # 减去标题行
        
        # 使用chunksize分批读取
        chunks = []
        with tqdm(total=total_rows, desc="加载CSV数据") as pbar:
            for chunk in pd.read_csv(file_path, chunksize=10000):
                chunks.append(chunk)
                pbar.update(len(chunk))
        
        # 合并所有数据块
        df = pd.concat(chunks)
    
    print(f"成功加载 {len(df)} 行数据")
    return df

def save_dataframe(df, file_path, index=False):
    """
    保存DataFrame到CSV文件
    
    参数:
        df: 要保存的DataFrame
        file_path: 保存路径
        index: 是否保存索引
    """
    # 确保目录存在
    ensure_dir(os.path.dirname(file_path))
    
    print(f"正在保存数据到 {file_path}...")
    df.to_csv(file_path, index=index)
    print(f"数据成功保存到 {file_path}")

def save_json(data, file_path):
    """
    保存数据到JSON文件
    
    参数:
        data: 要保存的数据
        file_path: 保存路径
    """
    # 确保目录存在
    ensure_dir(os.path.dirname(file_path))
    
    print(f"正在保存数据到 {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据成功保存到 {file_path}")

def split_dataset(df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
    """
    将数据集分割为训练集、验证集和测试集
    
    参数:
        df: 要分割的DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
        
    返回:
        train_df, val_df, test_df: 分割后的数据集
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "比例之和必须为1"
    
    # 打乱数据
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # 计算分割点
    train_end = int(len(df) * train_ratio)
    val_end = train_end + int(len(df) * val_ratio)
    
    # 分割数据
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    print(f"数据集分割完成，训练集: {len(train_df)}，验证集: {len(val_df)}，测试集: {len(test_df)}")
    
    return train_df, val_df, test_df

def load_json(file_path):
    """
    加载JSON文件
    
    参数:
        file_path: JSON文件路径
        
    返回:
        加载的JSON数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_pickle(data, file_path):
    """
    保存数据到pickle文件
    
    参数:
        data: 要保存的数据
        file_path: 保存路径
    """
    # 确保目录存在
    ensure_dir(os.path.dirname(file_path))
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    """
    从pickle文件加载数据
    
    参数:
        file_path: pickle文件路径
        
    返回:
        加载的数据
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def get_timestamp():
    """
    获取当前时间戳字符串
    
    返回:
        时间戳字符串，格式为 YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S") 