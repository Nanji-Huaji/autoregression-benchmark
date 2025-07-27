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

from utils import parse_args
from utils import model_dict


def run_benchmark(args):
    model = args.model
    model = model_dict[model]
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
    # model_instance.warmup(warmup_prompts, max_token)

    # run benchmark
    decoding_dict = {
        "autoregressive_decoding": model_instance.autoregressive_decoding,
        "speculative_decoding": model_instance.speculative_decoding,
        "tridecoding": model_instance.tridecoding,
    }
    assert (
        args.eval_mode in decoding_dict
    ), f"Unsupported eval_mode: {args.eval_mode}. Supported modes: {list(decoding_dict.keys())}."
    decode_function = decoding_dict[args.eval_mode]
    result, model_answer = decode_function(prompts=prompts, max_token=max_token, gamma1=args.gamma1, gamma2=args.gamma2)

    # # Aggregate results
    results = result
    print(f"Benchmark results: {results}")
    logger.info(f"Benchmark results: {results}")
    argument = {
        "model": model,
        "max_token": max_token,
        "benchmark_file": benchmark_file,
        "warmup_steps": warmup_steps,
        "gamma": args.gamma,
        "gamma1": args.gamma1,
        "gamma2": args.gamma2,
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
        "model_answer": model_answer,
    }
    with open(output_file, "w") as f:
        json.dump(dump_data, f, indent=4)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
    # find_batch_size_ralationship(args)
