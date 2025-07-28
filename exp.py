import subprocess
import logging
import sys

# --- 1. 配置日志记录 ---
# 设置日志，将信息输出到控制台和文件，方便后续分析
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiment_log.txt"), logging.StreamHandler(sys.stdout)],
)

# --- 2. 模型和实验参数定义 ---

# 模型路径字典
model_dict = {
    "vicuna-13b-v1.5": "model/vicuna/vicuna-13b-v1.5",
    "vicuna-7b-v1.5": "model/vicuna/vicuna-7b-v1.5",
    "tiny-vicuna-1b": "model/vicuna/tiny-vicuna-1b",
    "vicuna-160m": "model/vicuna/vicuna-160m",
    "vicuna-68m": "model/vicuna/vicuna-68m",
}

# 定义所有实验的组合
# 'little', 'draft', 'target' 分别对应 little_model_path, draft_model, model 参数
# gamma1_list 对应 γ_l (Little -> Draft), gamma2_list 对应 γ_d (Draft -> Target)
experiment_configs = [
    {
        "little": "vicuna-68m",
        "draft": "tiny-vicuna-1b",
        "target": "vicuna-7b-v1.5",
        "gamma1_list": [8, 10, 12],
        "gamma2_list": [4, 5, 6, 7],
    },
    {
        "little": "vicuna-68m",
        "draft": "tiny-vicuna-1b",
        "target": "vicuna-13b-v1.5",
        "gamma1_list": [8, 10],
        "gamma2_list": [3, 4, 5],
    },
    {
        "little": "vicuna-160m",
        "draft": "tiny-vicuna-1b",
        "target": "vicuna-7b-v1.5",
        "gamma1_list": [10, 12, 14],
        "gamma2_list": [5, 6, 8],
    },
    {
        "little": "vicuna-160m",
        "draft": "tiny-vicuna-1b",
        "target": "vicuna-13b-v1.5",
        "gamma1_list": [10, 12],
        "gamma2_list": [4, 5, 6],
    },
]

# --- 3. 实验启动函数 (已修改默认GPU) ---


def launch_benchmark(
    model_path,
    draft_model_path,
    little_model_path,
    gamma1,
    gamma2,
    max_token=128,
    benchmark="data/mt_bench.jsonl",
    warmup_steps=10,
    batch_size=8,
    device="cuda:1",  # <--- 修改后的默认设备
):
    """
    构建并执行单个基准测试命令。
    返回 subprocess.CompletedProcess 对象。
    """
    # 注意：eval_mode 固定为 tridecoding
    # 注意：gamma 参数可能在 tridecoding 模式下未使用，我们传一个默认值0
    cmd = [
        "python",
        "src/benchmark.py",
        "--model",
        str(model_path),
        "--draft_model",
        str(draft_model_path),
        "--little_model",
        str(little_model_path),
        "--gamma1",
        str(gamma1),
        "--gamma2",
        str(gamma2),
        "--gamma",
        "0",  # 默认值，可能未使用
        "--eval_mode",
        "tridecoding",
        "--max-token",
        str(max_token),
        "--benchmark",
        str(benchmark),
        "--warmup-steps",
        str(warmup_steps),
        "--batch_size",
        str(batch_size),
        "--device",
        str(device),
        "--compile_optimization",
        "False",
    ]

    logging.info(f"Executing command: {' '.join(cmd)}")

    # 使用 subprocess.run，并捕获输出，不因错误退出 (check=False)
    # 这允许我们检查 stderr 来判断是否 OOM
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result


# --- 4. 主执行循环 ---


def main():
    # 计算总的实验次数
    total_runs = sum(len(cfg["gamma1_list"]) * len(cfg["gamma2_list"]) for cfg in experiment_configs)
    completed_runs = 0
    oom_skips = 0

    logging.info(f"Starting experiment suite with a total of {total_runs} runs.")

    for config in experiment_configs:
        little_model_name = config["little"]
        draft_model_name = config["draft"]
        target_model_name = config["target"]

        # 从字典获取完整路径
        little_model_path = model_dict[little_model_name]
        draft_model_path = model_dict[draft_model_name]
        target_model_path = model_dict[target_model_name]

        for g1 in config["gamma1_list"]:
            for g2 in config["gamma2_list"]:
                # 检查逻辑约束：gamma2不应大于gamma1，这种组合效率低
                if g2 > g1:
                    logging.warning(f"Skipping combination where gamma2({g2}) > gamma1({g1}).")
                    total_runs -= 1  # 从总数中减去，以免计数错误
                    continue

                completed_runs += 1
                logging.info(f"--- [ RUN {completed_runs}/{total_runs} ] ---")
                logging.info(f"Models: L={little_model_name}, D={draft_model_name}, T={target_model_name}")
                logging.info(f"Gammas: gamma1={g1}, gamma2={g2}")

                # 调用 launch_benchmark 时，它将使用新的默认值 "cuda:1"
                result = launch_benchmark(
                    model_path=target_model_path,
                    draft_model_path=draft_model_path,
                    little_model_path=little_model_path,
                    gamma1=g1,
                    gamma2=g2,
                )

                # 检查执行结果
                if result.returncode == 0:
                    logging.info(f"Run {completed_runs} completed SUCCESSFULLY.")
                    # 打印吞吐量等关键结果 (如果存在于stdout中)
                    logging.info("STDOUT:\n" + result.stdout[-500:])  # 只打印最后一部分输出
                else:
                    # 检查是否是显存溢出错误
                    if "CUDA out of memory" in result.stderr or "OutOfMemoryError" in result.stderr:
                        logging.warning(f"Run {completed_runs} FAILED due to CUDA Out of Memory. Skipping.")
                        oom_skips += 1
                    else:
                        # 其他未知错误
                        logging.error(f"Run {completed_runs} FAILED with an unknown error.")
                        logging.error("STDERR:\n" + result.stderr)

                logging.info("-" * 40 + "\n")

    logging.info("======== EXPERIMENT SUITE FINISHED ========")
    logging.info(f"Total runs planned: {total_runs}")
    logging.info(f"Successful/Failed runs attempted: {completed_runs - oom_skips}")
    logging.info(f"Runs skipped due to OOM: {oom_skips}")
    logging.info("Logs saved to experiment_log.txt")


if __name__ == "__main__":
    main()
