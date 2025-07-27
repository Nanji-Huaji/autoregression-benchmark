import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import TypedDict, Union, List, Dict, Optional
import logging
from tqdm import tqdm
import numpy as np

import torch._dynamo

import argparse

torch._dynamo.config.suppress_errors = True


from typing import Tuple


import torch.nn.functional as F


class BenchmarkResults(TypedDict):
    num_prompts: int
    total_generated_tokens: int
    wall_time: float
    decode_throughput_tokens_per_sec: float
    acceptance_rate: float


class BenchmarkModel:

    model_id: str
    device: str
    torch_dtype: torch.dtype
    attn_implementation: Optional[str]

    target_model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        model_name: str,
        logger_instance: logging.Logger,
        args: argparse.Namespace,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.logger = logger_instance
        self.model_id = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.target_model = self._load_model()
        if args.compile_optimization:
            self.logger.info("Using compile optimization for target model.")
            self.logger.info("compile optimization is enabled, which may improve performance.")
            self.target_model = torch.compile(self.target_model, fullgraph=True, mode="reduce-overhead")  # type: ignore
        self.draft_model = None
        self.tokenizer = self._load_tokenizer()
        self.args = args

        if self.args.eval_mode == "speculative_decoding":
            self.add_draft_model(self.args.draft_model_path)

        self.logger.info(f"--- Initializing BenchmarkModel for: {self.model_id} ---")
        self.target_model.eval()

    def add_draft_model(self, draft_model_path: str) -> None:
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path, torch_dtype=self.torch_dtype, device_map=self.device
        ).eval()
        if self.args.compile_optimization:
            self.logger.info("Using compile optimization for draft model.")
            self.logger.info("compile optimization is enabled, which may improve performance.")
            self.draft_model = torch.compile(self.draft_model, fullgraph=True)
        self.logger.info(f"Draft model loaded from {draft_model_path}.")
        self.draft_model.config.use_static_cache = True
        self.draft_model.config.use_cache = True
        self.draft_model.config.is_assistant = True  # Set this to True for draft models

    def _load_model(self) -> PreTrainedModel:
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, device_map=self.device
        )
        self.logger.info(f"Model {self.model_id} loaded successfully.")
        return model

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.padding_side is None:
            tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
        self.logger.info(f"Tokenizer for {self.model_id} loaded successfully.")
        return tokenizer

    @torch.inference_mode()
    def warmup(self, warmup_prompts: List[str], max_token: int = 128) -> None:
        self.logger.info(f"Starting GPU warmup with {len(warmup_prompts)} prompts...")

        if not warmup_prompts:
            self.logger.warning("No warmup prompts provided. Skipping warmup.")
            return

        for prompt in tqdm(
            warmup_prompts,
            desc="Warming up",
        ):
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            self.target_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_token,
                do_sample=False,
                use_cache=True,
                num_beams=1,
            )  # type: ignore
        torch.cuda.synchronize()
        self.logger.info("Warmup completed successfully.")

    @torch.inference_mode()
    def autoregressive_decoding(
        self, prompts: List[str], max_token: int = 128, **kwargs
    ) -> Tuple[BenchmarkResults, List[Dict]]:
        self.logger.info(
            f"Starting benchmark for {len(prompts)} prompts, " f"generating up to {max_token} tokens each."
        )

        if not prompts:
            raise ValueError("No prompts provided for benchmarking.")

        model_answer = []
        generated_tokens = 0
        wall_time = 0.0
        acceptance_rate = 1.0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream=torch.cuda.current_stream())
        for prompt in tqdm(prompts, desc="Processing prompts"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_len = inputs.input_ids.shape[1]
            output = self.target_model.generate(**inputs, max_new_tokens=max_token, do_sample=False, use_cache=True)  # type: ignore
            new_generated_text = self.tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
            new_tokens_count = output.shape[1] - inputs.input_ids.shape[1]  # 计算生成的 token 数量
            model_answer.append(
                {"prompt": prompt, "new_generated_text": new_generated_text, "new_tokens_count": new_tokens_count}
            )
            generated_tokens += new_tokens_count
            wall_time += output.generation_info.get("generation_time", 0.0)
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()  # 确保所有操作完成
        wall_time += start_event.elapsed_time(end_event) / 1000.0  # 将毫秒转换为秒
        decode_throughput = generated_tokens / wall_time if wall_time > 0 else 0.0
        res = BenchmarkResults(
            num_prompts=len(prompts),
            total_generated_tokens=generated_tokens,
            wall_time=wall_time,
            decode_throughput_tokens_per_sec=decode_throughput,
            acceptance_rate=acceptance_rate,
        )
        return res, model_answer

    @torch.inference_mode()
    def _decode_n_token(self, model: PreTrainedModel, prefix: torch.Tensor, n: int) -> torch.Tensor:
        """
        使用给定的模型，从一个前缀（prefix）开始，自回归地解码 n 个 token。
        Args:
            model (PreTrainedModel): 用于生成的 Hugging Face Transformer 模型。
            prefix (torch.Tensor): 输入的 token ID 张量，形状为 (batch_size, seq_len)。
            n (int): 要生成的新 token 的数量。
        Returns:
            torch.Tensor: 生成的 n 个新 token 的张量，形状为 (batch_size, n)。
        """
        # 将输入移动到与模型相同的设备上
        prompt_tokens = prefix.to(model.device)

        # 如果输入是1D的 (seq_len,)，将其扩展为 (1, seq_len) 以处理 batch_size=1 的情况
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)
        # 用于存储生成的 token
        generated_tokens = []

        # 初始化 KV 缓存。在第一次模型调用后，它将被填充。
        past_key_values = None

        # 当前的输入是整个前缀
        current_input = prompt_tokens
        # 循环 n 次，每次生成一个 token
        for _ in range(n):
            # 1. 模型前向传播
            #    - `use_cache=True` 告诉模型返回 KV 缓存。
            #    - `past_key_values` 传入上一步的缓存，以避免重复计算。
            outputs = model(input_ids=current_input, past_key_values=past_key_values, use_cache=True)

            # 2. 获取 logits 和更新后的 KV 缓存
            logits = outputs.logits
            past_key_values = outputs.past_key_values  # 缓存将被用于下一次迭代
            # 3. 获取下一个 token 的预测
            #    我们只关心序列中最后一个 token 的 logits
            #    logits 的形状是 (batch_size, sequence_length, vocab_size)
            next_token_logits = logits[:, -1, :]

            # 4. 从 logits 中选择 token (此处使用贪心策略, 即选择概率最高的 token)
            #    也可以使用其他采样策略，如 top-k, top-p (nucleus) sampling
            #    例如: next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            next_token = torch.argmax(next_token_logits, dim=-1)  # 形状: (batch_size,)
            # 5. 存储生成的 token
            generated_tokens.append(next_token)

            # 6. 准备下一次迭代的输入
            #    下一次的输入只需要新的 token 即可，因为历史信息已经存储在 past_key_values 中
            #    需要将其形状从 (batch_size,) 调整为 (batch_size, 1)
            current_input = next_token.unsqueeze(-1)
        # 将生成的 token 列表连接成一个张量
        # 结果张量的形状为 (batch_size, n)
        result = torch.cat([t.unsqueeze(1) for t in generated_tokens], dim=1)

        return result

    @torch.inference_mode()
    def speculative_decoding(
        self, prompts: List[str], max_token: int = 128, gamma: int = 4, **kwargs
    ) -> Tuple[BenchmarkResults, List[Dict]]:
        """
        实现思辨解码：使用草稿模型生成候选token，目标模型验证。

        Args:
            prompts (List[str]): 输入的提示语列表。
            max_token (int): 每个提示语最多生成的新 token 数量。
            gamma (int): 每次由草稿模型预测的候选 token 数量（lookahead）。
        """
        self.logger.info(
            f"Starting speculative decoding for {len(prompts)} prompts, " f"max_token={max_token}, gamma={gamma}."
        )
        if not prompts:
            raise ValueError("No prompts provided for speculative decoding.")
        if not hasattr(self, "draft_model"):
            raise AttributeError("Draft model `self.draft_model` is required for speculative decoding.")
        if self.draft_model is None:
            raise ValueError("Draft model is not initialized. Please add a draft model before decoding.")
        model_answers = []
        total_generated_tokens = 0
        total_accepted_tokens = 0
        # --- 精确的 GPU 计时 ---
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream=torch.cuda.current_stream())
        for prompt in tqdm(prompts, desc="Speculative Decoding"):
            # 1. 初始化
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            generated_sequence = input_ids.clone()
            prompt_len = input_ids.shape[1]
            for _ in range(max_token):
                # --- 2. 起草阶段 (Drafting) ---
                # 使用草稿模型快速生成 gamma 个候选 token
                # 注意: draft_outputs 是完整的序列 (prompt + new_tokens)
                draft_outputs = self.draft_model.generate(
                    generated_sequence,
                    max_new_tokens=gamma,
                    do_sample=True,  # 采样以获得多样性
                    temperature=0.9,  # 建议使用略高的温度
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                # 提取草稿模型生成的 gamma 个新 token
                draft_tokens = draft_outputs[0, generated_sequence.shape[1] :]
                if not draft_tokens.numel():  # 如果草稿模型没生成任何东西，则终止
                    break
                # --- 3. 验证阶段 (Verification) ---
                # 将原始序列和草稿 token 拼接起来
                verified_sequence = torch.cat([generated_sequence, draft_tokens.unsqueeze(0)], dim=1)

                # 使用目标模型和草稿模型进行一次前向传播，获取 logits
                # 只需要对新生成的部分进行 logits 计算
                target_logits = self.target_model(verified_sequence).logits[:, -len(draft_tokens) - 1 :, :]
                draft_logits = self.draft_model(verified_sequence).logits[:, -len(draft_tokens) - 1 :, :]
                accepted_count = 0
                for i in range(len(draft_tokens)):
                    p_target = F.softmax(target_logits[:, i, :], dim=-1).squeeze()
                    p_draft = F.softmax(draft_logits[:, i, :], dim=-1).squeeze()
                    draft_token_id = draft_tokens[i]

                    # --- 4. 接受/拒绝逻辑 ---
                    if torch.rand(1).item() < (p_target[draft_token_id] / p_draft[draft_token_id]):
                        # 接受这个 token
                        accepted_count += 1
                    else:
                        # 拒绝，并重新从修正后的分布中采样一个 token
                        # 修正分布 p' = normalize(max(0, p_target - p_draft))
                        diff_dist = F.relu(p_target - p_draft)
                        sum_diff = diff_dist.sum()
                        if sum_diff == 0:  # 如果分布完全相同，则按目标分布采样
                            resampled_token = torch.multinomial(p_target, 1)
                        else:
                            resampled_token = torch.multinomial(diff_dist / sum_diff, 1)

                        # 将接受的 token 和重新采样的 token 加入序列
                        accepted_tokens_seq = draft_tokens[:accepted_count]
                        final_new_tokens = torch.cat([accepted_tokens_seq, resampled_token], dim=0)
                        break
                else:
                    # 如果所有草稿 token 都被接受了 (循环正常结束)
                    # 我们需要从目标模型的最后一个位置采样一个 "奖励" token
                    last_pos_dist = F.softmax(target_logits[:, -1, :], dim=-1).squeeze()
                    bonus_token = torch.multinomial(last_pos_dist, 1)
                    final_new_tokens = torch.cat([draft_tokens, bonus_token], dim=0)
                # 更新主序列
                generated_sequence = torch.cat([generated_sequence, final_new_tokens.unsqueeze(0)], dim=1)

                # 更新统计数据
                total_accepted_tokens += accepted_count
                total_generated_tokens += len(final_new_tokens)
                # 检查是否生成了结束符或达到最大长度
                if (
                    self.tokenizer.eos_token_id in final_new_tokens
                    or generated_sequence.shape[1] >= prompt_len + max_token
                ):
                    break

            # 将最终结果保存
            generated_text = self.tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
            new_tokens_count = generated_sequence.shape[1] - prompt_len
            model_answers.append(
                {"prompt": prompt, "generated_text": generated_text, "new_tokens_count": new_tokens_count}
            )
        # --- 结束计时并计算指标 ---
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()  # 等待所有 CUDA核心完成工作

        wall_time_ms = start_event.elapsed_time(end_event)
        wall_time = wall_time_ms / 1000.0
        # 计算接受率和吞吐量
        acceptance_rate = (
            total_accepted_tokens / (total_generated_tokens - len(prompts))
            if (total_generated_tokens - len(prompts)) > 0
            else 0.0
        )
        # 注意: 在计算接受率时，分母是草稿模型尝试生成的总token数，约等于总生成数减去采样修正的token数
        decode_throughput = total_generated_tokens / wall_time if wall_time > 0 else 0.0
        res = BenchmarkResults(
            num_prompts=len(prompts),
            total_generated_tokens=total_generated_tokens,
            wall_time=wall_time,
            decode_throughput_tokens_per_sec=decode_throughput,
            acceptance_rate=acceptance_rate,
        )
        return res, model_answers

    @torch.inference_mode()
    def tridecoding(
        self, prompts: List[str], max_token: int = 128, gamma1: int = 4, gamma2: int = 4
    ) -> Tuple[BenchmarkResults, List[Dict]]:
        raise NotImplementedError("Tridecoding is not implemented yet.")
