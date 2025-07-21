import argparse
from utils import setup_logger, load_prompts_from_jsonl, split_list_into_chunks
from model import BenchmarkModel

from datetime import datetime

import json
import os

from tqdm import tqdm


import math


import torch


from model import BenchmarkResults


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
        default="autoregression_decoding",
        help="Evaluation mode to use. Default is 'autoregression_decoding'.",
    )
    parser.add_argument(
        "--compile_optimization",
        type=bool,
        default=True,
        help="Whether to use compile optimization for the model. Default is True.",
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
    batch_size = args.batch_size
    device = args.device
    print(f"Running benchmark with model: {model}, max_token: {max_token}, benchmark_file: {benchmark_file}")
    logger = setup_logger()
    model_instance = BenchmarkModel(model, logger, args, device)

    prompts = load_prompts_from_jsonl(benchmark_file)
    assert prompts, f"No prompts found in {benchmark_file}. Please check the file."
    # warm up
    assert warmup_steps >= 0, "Warmup steps must be non-negative."
    warmup_prompts = prompts[:warmup_steps]
    model_instance.warmup(warmup_prompts, max_token)

    # run benchmark
    if args.eval_mode == "autoregression_decoding":
        if args.compile_optimization:
            logger.info("Using compile optimization for autoregressive decoding.")
            logger.info("compile optimization is enabled, which may improve performance.")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream=torch.cuda.current_stream())
        result = model_instance.autoregressive_decoding(prompts, max_token)
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()  # Wait for the events to complete
        decode_time_sec = start_event.elapsed_time(end_event) / 1000.0
        results = [result]  # Wrap single result in a list for consistency

    elif args.eval_mode == "vanilla_autoregressive_decoding":
        if args.compile_optimization:
            logger.info("Using compile optimization for serial autoregressive decoding.")
            logger.info("compile optimization is enabled, which may improve performance.")
        result = model_instance.serial_autoregression_decoding(prompts, max_token)
        results = [result]  # Wrap single result in a list for consistency

    elif args.eval_mode == "speculative_decoding":
        raise NotImplementedError("Speculative decoding is not yet implemented in the benchmark script.")
        model_instance.add_draft_model(args.draft_model)
        results = model_instance.speculative_decoding(prompts, max_token)

    else:
        raise ValueError(f"Unknown eval_mode: {args.eval_mode}")

    # # Aggregate results
    total_prompts = sum(result["num_prompts"] for result in results)
    total_generated_tokens = sum(result["total_generated_tokens"] for result in results)
    total_decode_time_sec = (
        decode_time_sec if "decode_time_sec" in locals() else sum(result["total_decode_time_sec"] for result in results)
    )
    decode_throughput = total_generated_tokens / total_decode_time_sec if total_decode_time_sec > 0 else 0.0
    results = {
        "model": model,
        "num_prompts": total_prompts,
        "total_generated_tokens": total_generated_tokens,
        "total_decode_time_sec": total_decode_time_sec,
        "decode_throughput_tokens_per_sec": decode_throughput,
    }
    logger.info(f"Benchmark results: {results}")
    argument = {
        "model": model,
        "max_token": max_token,
        "benchmark_file": benchmark_file,
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "device": device,
        "eval_mode": args.eval_mode,
    }
    # Save results to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/benchmark_results_{model}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dump_data = {
        "arguments": argument,
        "results": results,
    }
    with open(output_file, "w") as f:
        json.dump(dump_data, f, indent=4)
    logger.info(f"Results saved to {output_file}")


def find_batch_size_ralationship(args):

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
    device = args.device
    print(f"Running benchmark with model: {model}, max_token: {max_token}, benchmark_file: {benchmark_file}")
    logger = setup_logger()
    model_instance = BenchmarkModel(model, logger, device)

    # Load prompts
    prompts = load_prompts_from_jsonl(benchmark_file)
    assert prompts, f"No prompts found in {benchmark_file}. Please check the file."

    # warm up
    assert warmup_steps >= 0, "Warmup steps must be non-negative."
    warmup_prompts = prompts[:warmup_steps]
    model_instance.warmup(warmup_prompts, max_token)

    batch_size = (2**i for i in range(math.floor(math.log2(len(prompts)) + 1)))

    # Run benchmark for each batch size
    assert (
        args.eval_mode == "autoregression_decoding"
    ), "Only autoregression decoding is supported for batch size relationship."
    results = []
    for batch in tqdm(batch_size, desc="Running batch size relationship"):
        try:
            batch_list = split_list_into_chunks(prompts, batch)
            batch_results = []
            for batch_chunk in tqdm(batch_list, desc=f"Processing batch size {batch}"):
                batch_result = model_instance.autoregressive_decoding(batch_chunk, max_token)
                batch_results.append(batch_result)
            # Aggregate results for this batch size
            total_prompts = sum(result["num_prompts"] for result in batch_results)
            total_generated_tokens = sum(result["total_generated_tokens"] for result in batch_results)
            total_prefill_time_sec = sum(result["total_prefill_time_sec"] for result in batch_results)
            total_decode_time_sec = sum(result["total_decode_time_sec"] for result in batch_results)
            decode_throughput = total_generated_tokens / total_decode_time_sec if total_decode_time_sec > 0 else 0.0
            prefill_throughput = total_prompts / total_prefill_time_sec if total_prefill_time_sec > 0 else 0.0
            results.append(
                {
                    "batch_size": batch,
                    "num_prompts": total_prompts,
                    "total_generated_tokens": total_generated_tokens,
                    "total_prefill_time_sec": total_prefill_time_sec,
                    "total_decode_time_sec": total_decode_time_sec,
                    "decode_throughput_tokens_per_sec": decode_throughput,
                    "prefill_throughput_prompts_per_sec": prefill_throughput,
                }
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning(f"CUDA out of memory for batch size {batch}, skipping this batch size. Error: {e}")
                continue
            else:
                raise e
    logger.info(f"Batch size relationship results: {results}")
    argument = {
        "model": model,
        "max_token": max_token,
        "benchmark_file": benchmark_file,
        "warmup_steps": warmup_steps,
        "device": device,
        "eval_mode": args.eval_mode,
    }
    # Save results to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/batch_size_relationship_{model}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dump_data = {
        "arguments": argument,
        "results": results,
    }
    with open(output_file, "w") as f:
        json.dump(dump_data, f, indent=4)
    logger.info(f"Batch size relationship results saved to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
    # find_batch_size_ralationship(args)
