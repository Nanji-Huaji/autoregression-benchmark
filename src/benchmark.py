import argparse
from utils import setup_logger, load_prompts_from_jsonl, split_list_into_chunks
from model import BenchmarkModel

from datetime import datetime

import json
import os

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Run autoregressive decoding speed benchmark.")
    parser.add_argument(
        "--model",
        type=str,
        default="vicuna-13b-v1.5",
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
    return parser.parse_args()


def run_benchmark(args):
    model = args.model
    if model not in ["tiny-vicuna-1b", "vicuna-13b-v1.5"]:
        raise NotImplementedError(
            f"Model {model} is not supported. Please choose from 'tiny-vicuna-1b' or 'vicuna-13b-v1.5'."
        )
    model_path_dict = {
        "tiny-vicuna-1b": "model/vicuna/tiny-vicuna-1b",
        "vicuna-13b-v1.5": "model/vicuna/vicuna-13b-v1.5",
    }
    model = model_path_dict[model]
    max_token = args.max_token
    benchmark_file = args.benchmark
    warmup_steps = args.warmup_steps
    print(f"Running benchmark with model: {model}, max_token: {max_token}, benchmark_file: {benchmark_file}")
    logger = setup_logger()
    model_instance = BenchmarkModel(model, logger)

    prompts = load_prompts_from_jsonl(benchmark_file)
    assert prompts, f"No prompts found in {benchmark_file}. Please check the file."
    # warm up
    assert warmup_steps >= 0, "Warmup steps must be non-negative."
    warmup_prompts = prompts[:warmup_steps]
    model_instance.warmup(warmup_prompts, max_token)

    # run benchmark
    batch_list = split_list_into_chunks(prompts, 4)
    results = []
    for batch in tqdm(batch_list, desc="Running benchmark batches"):
        batch_results = model_instance.autoregressive_decoding(batch, max_token)
        results.append(batch_results)

    # # Aggregate results
    total_prompts = sum(result["num_prompts"] for result in results)
    total_generated_tokens = sum(result["total_generated_tokens"] for result in results)
    total_prefill_time_sec = sum(result["total_prefill_time_sec"] for result in results)
    total_decode_time_sec = sum(result["total_decode_time_sec"] for result in results)
    decode_throughput = total_generated_tokens / total_decode_time_sec if total_decode_time_sec > 0 else 0.0
    prefill_throughput = total_prompts / total_prefill_time_sec if total_prefill_time_sec > 0 else 0.0
    results = {
        "model": model,
        "num_prompts": total_prompts,
        "total_generated_tokens": total_generated_tokens,
        "total_prefill_time_sec": total_prefill_time_sec,
        "total_decode_time_sec": total_decode_time_sec,
        "decode_throughput_tokens_per_sec": decode_throughput,
        "prefill_throughput_prompts_per_sec": prefill_throughput,
    }
    logger.info(f"Benchmark results: {results}")

    # Save results to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/benchmark_results_{model}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
