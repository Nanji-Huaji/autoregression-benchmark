import transformers
from model import BenchmarkModel
from utils import setup_logger, parse_args
import torch

logger = setup_logger()

args = parse_args()

model_instance = BenchmarkModel(
    model_name="/home/tiantianyi/code/autoregression-benchmark/model/vicuna/tiny-vicuna-1b",
    logger_instance=logger,
    args=args,
)

if __name__ == "__main__":
    # --- 测试用例 1: 基本功能 ---
    print("\n" + "=" * 50)
    print("--- Test Case 1: Basic Generation ---")
    print("=" * 50)
    prompt_text = "The capital of France is"
    max_new_tokens = 30
    gamma = 4
    # 1. 准备输入
    prompt_ids = model_instance.tokenizer(prompt_text, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]

    print(f"Prompt: '{prompt_text}'")
    print(f"Prompt tensor shape: {prompt_ids.shape}")
    print(f"Max new tokens: {max_new_tokens}, Gamma: {gamma}")
    # 2. 调用待测函数
    output_ids = model_instance._speculative_decoding_n_tokens(
        prompt_ids=prompt_ids, max_new_tokens=max_new_tokens, gamma=gamma
    )

    # 3. 验证输出
    print("\n--- Verification for Test Case 1 ---")
    print(f"Output tensor shape: {output_ids.shape}")
    # 检查输出形状
    assert output_ids.dim() == 2, "Output tensor should have 2 dimensions"
    assert output_ids.shape[0] == 1, "Output tensor should have a batch size of 1"
    assert output_ids.shape[1] > prompt_len, "Output length should be greater than prompt length"
    assert (
        output_ids.shape[1] <= prompt_len + max_new_tokens
    ), "Output length should not exceed prompt_len + max_new_tokens"
    print("Assertion PASSED: Output shape is correct.")
    # 检查 prompt 部分是否被保留
    prompt_ids = prompt_ids.to("cuda")
    assert torch.equal(output_ids[:, :prompt_len], prompt_ids), "The start of the output should match the input prompt"
    prompt_ids = prompt_ids.cpu()  # 将 prompt_ids 移回 CPU 以避免 CUDA 内存泄漏
    print("Assertion PASSED: Prompt is correctly preserved in the output.")
    # 解码并打印结果以供人工检查
    generated_text = model_instance.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nFull Generated Text:\n'{generated_text}'")

    print("\nTest Case 1 Completed Successfully!")
    # --- 测试用例 2: 边界条件 (max_new_tokens = 0) ---
    print("\n" + "=" * 50)
    print("--- Test Case 2: Edge Case (max_new_tokens = 0) ---")
    print("=" * 50)
    # 1. 调用待测函数
    output_ids_zero = model_instance._speculative_decoding_n_tokens(
        prompt_ids=prompt_ids, max_new_tokens=0, gamma=gamma
    )
    # 2. 验证输出
    print(f"Output shape for max_new_tokens=0: {output_ids_zero.shape}")
    assert torch.equal(output_ids_zero, prompt_ids), "Output should be identical to input when max_new_tokens is 0"
    print("Assertion PASSED: Correctly returned the original prompt.")

    print("\nTest Case 2 Completed Successfully!")
    print("\n" + "=" * 50)
    print("All tests passed!")
