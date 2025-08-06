# helper function like logger, profiling

import logging
import cProfile
from io import StringIO
import pstats
import time
from logging.handlers import RotatingFileHandler

# logger
def setup_logger(name, log_file, level=logging.INFO, max_size=10*1024*1024, backup_count=100):
    """
    设置一个自定义logger
    
    参数:
    name: 日志器名称
    log_file: 日志文件路径
    level: 日志级别，默认为INFO
    max_size: 日志文件最大大小(MB)，默认为10MB
    backup_count: 保留的备份文件数量，默认为5个
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建文件处理器
        file_handler = RotatingFileHandler(log_file, maxBytes=max_size, backupCount=backup_count)
        file_handler.setLevel(level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # 创建格式化器并添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger



# profile
def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        print(f"Profile for {func.__name__}:\n{s.getvalue()}")
        return result
    return wrapper

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行原函数
        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算执行时间
        print(f"函数 {func.__name__} 运行时间: {execution_time:.4f} 秒")
        return result  # 返回原函数的返回值
    return wrapper

