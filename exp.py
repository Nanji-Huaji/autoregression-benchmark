import subprocess
import logging
import sys

# --- 1. 配置日志记录 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiment_log.txt"), logging.StreamHandler(sys.stdout)],
)

# --- 2. 模型和实验参数定义 ---

# 模型路径字典 (保持不变)
model_dict = {
    "vicuna-13b-v1.5": "model/vicuna/vicuna-13b-v1.5",
    "vicuna-7b-v1.5": "model/vicuna/vicuna-7b-v1.5",
    "tiny-vicuna-1b": "model/vicuna/tiny-vicuna-1b",
    "vicuna-160m": "model/vicuna/vicuna-160m",
    "vicuna-68m": "model/vicuna/vicuna-68m",
}

# --- 2a. Tridecoding 实验配置 ---
# (原 experiment_configs，已重命名以示区分)
tridecoding_configs = [
    {
        "little": "vicuna-68m", "draft": "tiny-vicuna-1b", "target": "vicuna-7b-v1.5",
        "gamma1_list": [8, 10, 12], "gamma2_list": [4, 5, 6, 7],
    },
    {
        "little": "vicuna-68m", "draft": "tiny-vicuna-1b", "target": "vicuna-13b-v1.5",
        "gamma1_list": [8, 10], "gamma2_list": [3, 4, 5],
    },
    {
        "little": "vicuna-160m", "draft": "tiny-vicuna-1b", "target": "vicuna-7b-v1.5",
        "gamma1_list": [10, 12, 14], "gamma2_list": [5, 6, 8],
    },
    {
        "little": "vicuna-160m", "draft": "tiny-vicuna-1b", "target": "vicuna-13b-v1.5",
        "gamma1_list": [10, 12], "gamma2_list": [4, 5, 6],
    },
]

# --- 2b. Speculative Decoding 实验配置 (新增) ---
# 只需 'draft', 'target' 和 'gamma_list'
speculative_configs = [
    # {
    #     "draft": "tiny-vicuna-1b", "target": "vicuna-7b-v1.5",
    #     "gamma_list": [4, 6, 8, 10], # 您可以根据需要调整gamma值
    # },
    {
        "draft": "tiny-vicuna-1b", "target": "vicuna-13b-v1.5",
        "gamma_list": [i for i in range(3, 21)],
    },
    # {
    #     "draft": "vicuna-160m", "target": "vicuna-7b-v1.5",
    #     "gamma_list": [5, 7, 9],
    # }
]


# --- 3. 实验启动函数 ---

def _run_subprocess(cmd):
    """一个通用的子进程执行函数，用于捕获输出和日志记录。"""
    logging.info(f"Executing command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result

def launch_tridecoding_benchmark(
    model_path, draft_model_path, little_model_path, gamma1, gamma2,
    max_token=128, benchmark="data/mt_bench.jsonl", warmup_steps=10,
    batch_size=8, device="auto"
):
    """为 Tridecoding 构建并执行命令。"""
    cmd = [
        "python", "src/benchmark.py",
        "--model", str(model_path),
        "--draft_model", str(draft_model_path),
        "--little_model_path", str(little_model_path), # 确保参数名与 benchmark.py 匹配
        "--gamma1", str(gamma1),
        "--gamma2", str(gamma2),
        "--eval_mode", "tridecoding",
        "--max-token", str(max_token),
        "--benchmark", str(benchmark),
        "--warmup-steps", str(warmup_steps),
        "--batch_size", str(batch_size),
        "--device", str(device),
        "--compile_optimization", "False",
    ]
    return _run_subprocess(cmd)

def launch_speculative_benchmark(
    model_path, draft_model_path, gamma,
    max_token=128, benchmark="data/mt_bench.jsonl", warmup_steps=10,
    batch_size=8, device="auto"
):
    """为 Speculative Decoding 构建并执行命令。"""
    cmd = [
        "python", "src/benchmark.py",
        "--model", str(model_path),
        "--draft_model", str(draft_model_path),
        # 注意：没有 little_model_path, gamma1, gamma2
        "--gamma", str(gamma),
        "--eval_mode", "speculative_decoding",
        "--max-token", str(max_token),
        "--benchmark", str(benchmark),
        "--warmup-steps", str(warmup_steps),
        "--batch_size", str(batch_size),
        "--device", str(device),
        "--compile_optimization", "False",
    ]
    return _run_subprocess(cmd)

# --- 4. 主执行循环 ---

def main():
    # 计算总的实验次数
    total_tridecoding_runs = sum(len(cfg["gamma1_list"]) * len(cfg["gamma2_list"]) for cfg in tridecoding_configs)
    total_speculative_runs = sum(len(cfg["gamma_list"]) for cfg in speculative_configs)
    total_runs = total_tridecoding_runs + total_speculative_runs
    
    completed_runs = 0
    oom_skips = 0

    logging.info(f"Starting experiment suite with a total of {total_runs} runs.")
    logging.info(f"({total_tridecoding_runs} Tridecoding runs, {total_speculative_runs} Speculative Decoding runs)")
    logging.info("=" * 60)

    # --- 4a. 执行 Tridecoding 实验 ---
    # logging.info("\n" + "="*20 + " STARTING TRIDECODING EXPERIMENTS " + "="*20)
    # for config in tridecoding_configs:
    #     little_model_path = model_dict[config["little"]]
    #     draft_model_path = model_dict[config["draft"]]
    #     target_model_path = model_dict[config["target"]]

    #     for g1 in config["gamma1_list"]:
    #         for g2 in config["gamma2_list"]:
    #             if g2 > g1:
    #                 logging.warning(f"Skipping combination where gamma2({g2}) > gamma1({g1}).")
    #                 continue

    #             completed_runs += 1
    #             logging.info(f"--- [ RUN {completed_runs}/{total_runs} | Mode: Tridecoding ] ---")
    #             logging.info(f"Models: L={config['little']}, D={config['draft']}, T={config['target']}")
    #             logging.info(f"Gammas: gamma1={g1}, gamma2={g2}")
                
    #             result = launch_tridecoding_benchmark(
    #                 model_path=target_model_path, draft_model_path=draft_model_path,
    #                 little_model_path=little_model_path, gamma1=g1, gamma2=g2
    #             )
                
    #             # 统一的结果处理
    #             if result.returncode == 0:
    #                 logging.info(f"Run {completed_runs} completed SUCCESSFULLY.")
    #                 logging.info("STDOUT (last 500 chars):\n" + result.stdout[-500:])
    #             else:
    #                 if "CUDA out of memory" in result.stderr or "OutOfMemoryError" in result.stderr:
    #                     logging.warning(f"Run {completed_runs} FAILED due to CUDA Out of Memory. Skipping.")
    #                     oom_skips += 1
    #                 else:
    #                     logging.error(f"Run {completed_runs} FAILED with an unknown error.")
    #                     logging.error("STDERR:\n" + result.stderr)
    #             logging.info("-" * 40 + "\n")

    # --- 4b. 执行 Speculative Decoding 实验 ---
    logging.info("\n" + "="*15 + " STARTING SPECULATIVE DECODING EXPERIMENTS " + "="*15)
    for config in speculative_configs:
        draft_model_path = model_dict[config["draft"]]
        target_model_path = model_dict[config["target"]]

        for g in config["gamma_list"]:
            completed_runs += 1
            logging.info(f"--- [ RUN {completed_runs}/{total_runs} | Mode: Speculative ] ---")
            logging.info(f"Models: D={config['draft']}, T={config['target']}")
            logging.info(f"Gamma: {g}")
            
            result = launch_speculative_benchmark(
                model_path=target_model_path, draft_model_path=draft_model_path, gamma=g
            )

            # 统一的结果处理
            if result.returncode == 0:
                logging.info(f"Run {completed_runs} completed SUCCESSFULLY.")
                logging.info("STDOUT (last 500 chars):\n" + result.stdout[-500:])
            else:
                if "CUDA out of memory" in result.stderr or "OutOfMemoryError" in result.stderr:
                    logging.warning(f"Run {completed_runs} FAILED due to CUDA Out of Memory. Skipping.")
                    oom_skips += 1
                else:
                    logging.error(f"Run {completed_runs} FAILED with an unknown error.")
                    logging.error("STDERR:\n" + result.stderr)
            logging.info("-" * 40 + "\n")


    # --- 5. 最终总结 ---
    logging.info("======== EXPERIMENT SUITE FINISHED ========")
    logging.info(f"Total runs planned: {total_runs}")
    # 减去因为逻辑约束（g2 > g1）而实际未运行的次数
    final_attempted_runs = completed_runs
    logging.info(f"Total runs attempted: {final_attempted_runs}")
    logging.info(f"Runs skipped due to OOM: {oom_skips}")
    logging.info("Logs saved to experiment_log.txt")


if __name__ == "__main__":
    main()
