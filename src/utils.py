import transformers
import torch
import logging


def setup_logger(log_dir: str = "log") -> logging.Logger:
    """
    配置并返回一个日志记录器实例。
    该日志记录器会将INFO及以上级别的信息同时输出到控制台和
    一个位于指定目录下的、带时间戳的日志文件中。
    Args:
        log_dir (str): 用于存放日志文件的目录路径。
    Returns:
        logging.Logger: 一个配置好的根日志记录器实例。
    """
    pass
