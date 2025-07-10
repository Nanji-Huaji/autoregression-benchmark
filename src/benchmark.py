import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run autoregressive decoding speed benchmark.")
    parser.add_argument(
        "--model",
        type=str,
        default="tiny-vicuna-1b",
        help="The model to benchmark. Options: tiny-vicuna-1b, vicuna-13b-v1.5",
    )


def run_benchmark(args):
    pass


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
