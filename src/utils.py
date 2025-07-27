import transformers
import torch
import logging
import json
from datetime import datetime
import os

from typing import List, Optional

import argparse

from transformers.modeling_utils import PreTrainedModel


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
    # 1. 获取根日志记录器
    # 使用 getLogger() 获取根 logger，这样配置将是全局的。
    logger = logging.getLogger()

    # 2. 设置日志级别
    # 只处理 INFO 及以上级别的日志
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler
    # 如果 logger 已经有 handlers，先清空，避免重复打印日志
    if logger.hasHandlers():
        logger.handlers.clear()

    # 3. 创建格式化器 (Formatter)
    # 定义日志输出的格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 4. 配置控制台处理器 (StreamHandler)
    # 将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 5. 配置文件处理器 (FileHandler)
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 创建带时间戳的日志文件名
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"app_{current_time}.log")

    # 创建文件处理器，将日志写入文件
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_prompts_from_jsonl(file_path: str, max_prompts: Optional[int] = None) -> List[str]:
    """
    从 JSONL 文件中加载 prompts。

    该函数逐行读取文件，解析每一行的JSON对象。它假定每个JSON对象
    都有一个名为 "turns" 的键，其值为一个字符串列表。函数将提取
    此列表的第一个元素作为 prompt。
    Args:
        file_path (str): .jsonl 文件的路径。
        max_prompts (Optional[int]): 要加载的最大 prompt 数量。
                                    如果为 None，则加载所有 prompts。
    Returns:
        List[str]: 从文件中提取的 prompt 字符串列表。

    Raises:
        FileNotFoundError: 如果指定的文件路径不存在。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")
    prompts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # 如果达到了最大数量限制，则停止读取
            if max_prompts is not None and len(prompts) >= max_prompts:
                break

            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                data = json.loads(line)

                # 验证 "turns" 字段是否存在且为非空列表
                if "turns" in data and isinstance(data["turns"], list) and data["turns"]:
                    # 提取第一个 "turn" 作为 prompt
                    prompts.append(data["turns"][0])
                else:
                    print(f"警告: 第 {i+1} 行缺少有效的 'turns' 字段，已跳过。")

            except json.JSONDecodeError:
                print(f"警告: 无法解析第 {i+1} 行的JSON，已跳过。")
    return prompts


def split_list_into_chunks(lst: List[str], m: int) -> List[List[str]]:
    """
    将列表分成m份，如果不够，最后一份可以少一些。

    Args:
        lst: 要分割的列表
        m: 要分成的份数

    Returns:
        包含m个子列表的列表，每个子列表包含原列表的一部分元素
    """
    n = len(lst)
    chunk_size = n // m
    remainder = n % m
    chunks = []
    start = 0
    for i in range(m):
        end = start + chunk_size + (1 if i < remainder else 0)
        if start < n:
            chunks.append(lst[start:end])
        start = end
    return chunks


def parse_args():
    parser = argparse.ArgumentParser(description="Run autoregressive decoding speed benchmark.")
    parser.add_argument(
        "--model",
        type=str,
        default="tiny-vicuna-1b",
        help="The model to benchmark. Options: tiny-vicuna-1b, vicuna-13b-v1.5",
    )
    parser.add_argument(
        "--max-token",
        type=int,
        default=128,
        help="Maximum number of tokens to generate in each decoding step.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="data/mt_bench.jsonl",
        help="The benchmark to run.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of warmup steps before measuring speed.",
    )
    parser.add_argument(
        "--draft_model",
        type=str,
        default="model/vicuna/vicuna-68m",
        help="Path to the draft model for speculative decoding.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of prompts to process in each batch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on. Default is 'cuda:0'.",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="speculative_decoding",
        choices=["autoregressive_decoding", "speculative_decoding", "tridecoding"],
        help="Evaluation mode to use. Default is 'speculative_decoding'.",
    )
    parser.add_argument(
        "--compile_optimization",
        type=bool,
        default=True,
        help="Whether to use compile optimization for the model. Default is True.",
    )
    parser.add_argument(
        "--little_model_path",
        type=str,
        required=False,
        help="Path to the little model for speculative decoding.",
    )
    return parser.parse_args()


model_dict = {
    "tiny-vicuna-1b": "model/vicuna/tiny-vicuna-1b",
    "vicuna-13b-v1.5": "model/vicuna/vicuna-13b-v1.5",
    "vicuna-68m": "model/vicuna/vicuna-68m",
}
