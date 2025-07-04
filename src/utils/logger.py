"""
日志工具模块
"""
import logging
import os
import sys
from datetime import datetime

def setup_logger(log_dir="logs", log_level=logging.INFO):
    """
    设置日志记录器
    
    参数:
        log_dir: 日志文件保存目录
        log_level: 日志级别
    
    返回:
        logger: 配置好的日志记录器
    """
    # 创建日志目录（如果不存在）
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建日志记录器
    logger = logging.getLogger("RSBully")
    logger.setLevel(log_level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # 创建文件处理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rsbully_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"日志系统初始化完成，日志文件：{log_file}")
    
    return logger 