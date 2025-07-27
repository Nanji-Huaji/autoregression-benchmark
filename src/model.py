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

import time


class BenchmarkResults(TypedDict):
    num_prompts: int
    total_generated_tokens: int
    draft_model_generated_tokens: int
    little_model_generated_tokens: int
    wall_time: float
    decode_throughput_tokens_per_sec: float
    acceptance_rate: float
    little_model_acceptance_rate: float
    total_target_model_forward_time: int
    total_draft_model_forward_time: int
    total_little_model_forward_time: int


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
            self.add_draft_model(self.args.draft_model)
        if self.args.eval_mode == "tridecoding":
            self.add_draft_model(self.args.draft_model)
            self.little_model = AutoModelForCausalLM.from_pretrained(
                self.args.little_model_path, torch_dtype=self.torch_dtype, device_map=self.device
            ).eval()

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
        self.logger.info(f"Starting benchmark for {len(prompts)} prompts, generating up to {max_token} tokens each.")

        if not prompts:
            raise ValueError("No prompts provided for benchmarking.")

        model_answer = []
        generated_tokens = 0
        wall_time = 0.0
        acceptance_rate = 1.0
        total_target_model_forward_time = 0

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
            # generate 的 forward 次数 = 生成 token 数 + 1（初始 prompt）
            total_target_model_forward_time += new_tokens_count + 1
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()  # 确保所有操作完成
        wall_time += start_event.elapsed_time(end_event) / 1000.0  # 将毫秒转换为秒
        decode_throughput = generated_tokens / wall_time if wall_time > 0 else 0.0
        res = BenchmarkResults(
            num_prompts=len(prompts),
            total_generated_tokens=generated_tokens,
            draft_model_generated_tokens=0,
            little_model_generated_tokens=0,
            wall_time=wall_time,
            decode_throughput_tokens_per_sec=decode_throughput,
            acceptance_rate=acceptance_rate,
            little_model_acceptance_rate=1.0,
            total_target_model_forward_time=total_target_model_forward_time,
            total_draft_model_forward_time=0,
            total_little_model_forward_time=0,
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
        思辨解码实现，并根据指定的 BenchmarkResults 结构统计信息。
        """
        if not hasattr(self, "draft_model") or self.draft_model is None:
            raise AttributeError("Draft model `self.draft_model` is required for speculative decoding.")

        model_answers = []
        total_generated_tokens = 0

        # [MODIFIED] 初始化统计信息字典，用于计数
        # 'forward_time' 在此被解释为 'forward_count' (调用次数)
        total_stats = {
            "target_forward_count": 0,
            "draft_forward_count": 0,
            "total_drafted_tokens": 0,
            "total_accepted_tokens": 0,
        }
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream=torch.cuda.current_stream())
        for prompt in tqdm(prompts, desc="Speculative Decoding"):
            # --- 1. 初始化和 Prompt 处理 ---
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            if inputs.input_ids.shape[1] == 0:
                self.logger.warning(f"Skipping an empty or invalid prompt: '{prompt}'")
                model_answers.append({"prompt": prompt, "generated_text": "", "new_tokens_count": 0})
                continue

            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            prompt_len = input_ids.shape[1]
            # --- 预计算 Prompt 的 KV 缓存 ---
            target_outputs = self.target_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            total_stats["target_forward_count"] += 1
            past_key_values_target = target_outputs.past_key_values
            draft_outputs = self.draft_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            total_stats["draft_forward_count"] += 1
            past_key_values_draft = draft_outputs.past_key_values
            generated_sequence = input_ids
            n_generated_for_prompt = 0

            while n_generated_for_prompt < max_token:
                # --- 2. 起草阶段 (Drafting) ---
                draft_tokens = []
                draft_logits_list = []

                current_draft_kv = past_key_values_draft
                next_token = generated_sequence[:, -1:]

                for _ in range(gamma):
                    draft_step_outputs = self.draft_model(
                        input_ids=next_token, use_cache=True, past_key_values=current_draft_kv
                    )
                    total_stats["draft_forward_count"] += 1
                    logits = draft_step_outputs.logits[:, -1, :]
                    logits = logits / 0.9  # temperature
                    probs = F.softmax(logits, dim=-1)
                    sampled_token = torch.multinomial(probs, num_samples=1)
                    draft_tokens.append(sampled_token.squeeze(0))
                    draft_logits_list.append(draft_step_outputs.logits)

                    next_token = sampled_token
                    current_draft_kv = draft_step_outputs.past_key_values

                if not draft_tokens:
                    break
                draft_tokens = torch.cat(draft_tokens)
                draft_logits = torch.cat(draft_logits_list, dim=1)

                n_draft = len(draft_tokens)
                total_stats["total_drafted_tokens"] += n_draft
                # --- 3. 验证阶段 (Verification) ---
                target_outputs = self.target_model(
                    input_ids=draft_tokens.unsqueeze(0),
                    use_cache=True,
                    past_key_values=past_key_values_target,
                )
                total_stats["target_forward_count"] += 1
                target_logits = target_outputs.logits

                # --- 4. 接受/拒绝逻辑 ---
                accepted_count = 0
                for i in range(n_draft):
                    p_target = F.softmax(target_logits[:, i, :], dim=-1).squeeze()
                    p_draft = F.softmax(draft_logits[:, i, :], dim=-1).squeeze()
                    draft_token_id = draft_tokens[i]
                    if p_draft[draft_token_id] > 1e-9 and torch.rand(1).item() < (
                        p_target[draft_token_id] / p_draft[draft_token_id]
                    ):
                        accepted_count += 1
                    else:
                        diff_dist = F.relu(p_target - p_draft)
                        sum_diff = diff_dist.sum()
                        resampled_token = torch.multinomial(p_target if sum_diff < 1e-9 else diff_dist / sum_diff, 1)
                        final_new_tokens = torch.cat([draft_tokens[:accepted_count], resampled_token], dim=0)
                        break
                else:
                    last_pos_dist = F.softmax(target_logits[:, -1, :], dim=-1).squeeze()
                    bonus_token = torch.multinomial(last_pos_dist, 1)
                    final_new_tokens = torch.cat([draft_tokens, bonus_token], dim=0)
                total_stats["total_accepted_tokens"] += accepted_count
                # --- 5. 更新状态 ---
                final_tokens_for_update = final_new_tokens.unsqueeze(0)

                target_update_outputs = self.target_model(
                    input_ids=final_tokens_for_update, use_cache=True, past_key_values=past_key_values_target
                )
                total_stats["target_forward_count"] += 1
                past_key_values_target = target_update_outputs.past_key_values
                draft_update_outputs = self.draft_model(
                    input_ids=final_tokens_for_update, use_cache=True, past_key_values=past_key_values_draft
                )
                total_stats["draft_forward_count"] += 1
                past_key_values_draft = draft_update_outputs.past_key_values

                n_new = len(final_new_tokens)
                generated_sequence = torch.cat([generated_sequence, final_new_tokens.unsqueeze(0)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones(1, n_new, device=self.device)], dim=1)
                n_generated_for_prompt += n_new
                if self.tokenizer.eos_token_id in final_new_tokens:
                    break

            total_generated_tokens += n_generated_for_prompt
            new_tokens_count = generated_sequence.shape[1] - prompt_len
            generated_text = self.tokenizer.decode(generated_sequence[0, prompt_len:], skip_special_tokens=True)
            model_answers.append(
                {"prompt": prompt, "generated_text": generated_text, "new_tokens_count": new_tokens_count}
            )
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        wall_time = start_event.elapsed_time(end_event) / 1000.0
        # [MODIFIED] 构建完全符合您要求的 BenchmarkResults 字典
        acceptance_rate = (
            total_stats["total_accepted_tokens"] / total_stats["total_drafted_tokens"]
            if total_stats["total_drafted_tokens"] > 0
            else 0.0
        )
        decode_throughput = total_generated_tokens / wall_time if wall_time > 0 else 0.0

        res: BenchmarkResults = {
            "num_prompts": len(prompts),
            "total_generated_tokens": total_generated_tokens,
            "draft_model_generated_tokens": total_stats["total_drafted_tokens"],
            "little_model_generated_tokens": 0,  # 此实现中无 little model
            "wall_time": wall_time,
            "decode_throughput_tokens_per_sec": decode_throughput,
            "acceptance_rate": acceptance_rate,
            "little_model_acceptance_rate": 0.0,  # 无 little model，接受率为 0
            "total_target_model_forward_time": total_stats["target_forward_count"],
            "total_draft_model_forward_time": total_stats["draft_forward_count"],
            "total_little_model_forward_time": 0,  # 无 little model，调用次数为 0
        }

        return res, model_answers

    @torch.inference_mode()
    def _speculative_decoding_n_tokens(self, prompt_ids: torch.Tensor, max_new_tokens: int, gamma: int) -> torch.Tensor:
        """
        使用思辨解码为一个 prompt 生成最多 n 个 token。
        这是一个核心辅助函数，直接操作 token 张量。
        Args:
            prompt_ids (torch.Tensor): 输入的 token ID 张量，形状为 `[1, prompt_len]`。
            max_new_tokens (int): 要生成的最大新 token 数量。
            gamma (int): 每轮验证中，草稿模型生成的候选 token 数量。
        Returns:
            torch.Tensor: 包含原始 prompt 和生成 token 的完整序列，
                        形状为 `[1, prompt_len + generated_len]`。
        """
        # --- 0. 输入校验和初始化 ---
        if prompt_ids.dim() != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("`prompt_ids` a single prompt, shape [1, seq_len].")
        if max_new_tokens <= 0:
            return prompt_ids
        if self.draft_model is None:
            raise ValueError("Draft model is not set. Please add a draft model before using speculative decoding.")
        input_ids = prompt_ids.to(self.device)
        prompt_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        # --- 1. 预计算 Prompt 的 KV 缓存 ---
        # 为目标模型和草稿模型生成初始的 KV 缓存
        target_outputs = self.target_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past_key_values_target = target_outputs.past_key_values

        draft_outputs = self.draft_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past_key_values_draft = draft_outputs.past_key_values

        generated_sequence = input_ids
        num_generated = 0
        # --- 2. 主解码循环 ---
        while num_generated < max_new_tokens:

            # --- 2.1. 起草阶段 (Drafting) ---
            # 使用手动循环精确控制 KV 缓存，生成 gamma 个草稿 token
            draft_tokens = []
            draft_logits_list = []

            current_draft_kv = past_key_values_draft
            next_token_for_draft = generated_sequence[:, -1:]
            for _ in range(gamma):
                draft_step_outputs = self.draft_model(
                    input_ids=next_token_for_draft, use_cache=True, past_key_values=current_draft_kv
                )

                # 从 logits 中采样一个 token
                # 注意：这里可以根据需要调整采样策略（如 temperature, top_p 等）
                logits = draft_step_outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                sampled_token = torch.multinomial(probs, num_samples=1)

                draft_tokens.append(sampled_token.squeeze(0))
                draft_logits_list.append(draft_step_outputs.logits)

                # 为下一次起草迭代更新输入和 KV 缓存
                next_token_for_draft = sampled_token
                current_draft_kv = draft_step_outputs.past_key_values
            if not draft_tokens:  # 如果没有生成任何草稿，则终止
                break

            draft_tokens = torch.cat(draft_tokens)
            draft_logits = torch.cat(draft_logits_list, dim=1)

            # --- 2.2. 验证阶段 (Verification) ---
            # 将草稿 token 作为一个序列，通过目标模型进行一次前向传播来验证
            target_outputs = self.target_model(
                input_ids=draft_tokens.unsqueeze(0),
                use_cache=True,
                past_key_values=past_key_values_target,
            )
            target_logits = target_outputs.logits
            # --- 2.3. 接受/拒绝/重采样 ---
            n_draft = len(draft_tokens)
            final_new_tokens = None  # 用于存储本轮最终确认的 token 序列
            for i in range(n_draft):
                p_target = F.softmax(target_logits[:, i, :], dim=-1).squeeze()
                p_draft = F.softmax(draft_logits[:, i, :], dim=-1).squeeze()
                draft_token_id = draft_tokens[i]
                # 接受条件：随机数 < p_target / p_draft
                if p_draft[draft_token_id] > 1e-6 and torch.rand(1).item() < (
                    p_target[draft_token_id] / p_draft[draft_token_id]
                ):
                    # 接受此 token，继续检查下一个
                    continue
                else:
                    # 拒绝此 token，从差值分布中重采样一个，然后终止本轮验证
                    diff_dist = F.relu(p_target - p_draft)
                    sum_diff = diff_dist.sum()

                    # 如果差值分布几乎为0，则退回到从目标分布中采样
                    resample_dist = p_target if sum_diff < 1e-9 else diff_dist / sum_diff
                    resampled_token = torch.multinomial(resample_dist, 1)

                    # 最终序列 = 已接受的 + 重采样的
                    final_new_tokens = torch.cat([draft_tokens[:i], resampled_token], dim=0)
                    break

            # 如果所有草稿都被接受，则额外生成一个“奖励”token
            if final_new_tokens is None:
                last_pos_dist = F.softmax(target_logits[:, -1, :], dim=-1).squeeze()
                bonus_token = torch.multinomial(last_pos_dist, 1)
                final_new_tokens = torch.cat([draft_tokens, bonus_token], dim=0)
            # --- 2.4. 更新状态 ---
            # 使用最终确认的 token 序列统一更新两个模型的 KV 缓存
            final_tokens_for_update = final_new_tokens.unsqueeze(0)
            # 更新目标模型
            target_update_outputs = self.target_model(
                input_ids=final_tokens_for_update, use_cache=True, past_key_values=past_key_values_target
            )
            past_key_values_target = target_update_outputs.past_key_values

            # 更新草稿模型
            draft_update_outputs = self.draft_model(
                input_ids=final_tokens_for_update, use_cache=True, past_key_values=past_key_values_draft
            )
            past_key_values_draft = draft_update_outputs.past_key_values
            # 更新主序列和已生成 token 计数
            n_new = final_new_tokens.shape[0]
            generated_sequence = torch.cat([generated_sequence, final_tokens_for_update], dim=1)
            num_generated += n_new
            # 检查是否生成了 EOS token
            if self.tokenizer.eos_token_id in final_new_tokens:
                break

        return generated_sequence

    @torch.inference_mode()
    def _tridecoding_single_prompt(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        gamma1: int,
        gamma2: int,
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        为单个 prompt 执行三阶段推测解码的核心逻辑。
        返回生成的 token IDs 和用于基准测试的统计数据。
        """
        # --- 0. 初始化 ---
        input_ids = prompt_ids.to(self.device)

        # 为所有三个模型预先计算 prompt 的 KV 缓存
        past_kv_target = self.target_model(input_ids, use_cache=True).past_key_values
        if self.draft_model is None:
            raise ValueError("Draft model is not set. Please add a draft model before using tridecoding.")
        past_kv_draft = self.draft_model(input_ids, use_cache=True).past_key_values
        past_kv_little = self.little_model(input_ids, use_cache=True).past_key_values
        generated_sequence = input_ids
        num_generated = 0
        # 用于基准测试的计数器
        stats = {
            "accepted_by_target": 0,
            "drafted_for_target": 0,
            "accepted_by_draft": 0,
            "drafted_for_draft": 0,
            "target_model_forward_time": 0,
            "draft_model_forward_time": 0,
            "little_model_forward_time": 0,
        }
        while num_generated < max_new_tokens:
            # --- 1. 小小模型起草 (Little Model Drafting) ---
            little_draft_tokens = []
            little_logits_list = []
            current_kv_little = past_kv_little
            next_token = generated_sequence[:, -1:]
            for _ in range(gamma1):
                outputs = self.little_model(input_ids=next_token, use_cache=True, past_key_values=current_kv_little)
                stats["little_model_forward_time"] += 1  # 每次调用 little_model 都计数
                logits = outputs.logits[:, -1, :]
                sampled_token = torch.multinomial(F.softmax(logits, dim=-1), 1)

                little_draft_tokens.append(sampled_token.squeeze(0))
                little_logits_list.append(outputs.logits)
                next_token = sampled_token
                current_kv_little = outputs.past_key_values

            if not little_draft_tokens:
                break  # 如果 gamma1=0 或其他原因导致没有草稿，则退出

            little_draft_tokens = torch.cat(little_draft_tokens)
            little_logits = torch.cat(little_logits_list, dim=1)
            # --- 2. 草稿模型验证与再起草 (Draft Model Verification & Redrafting) ---
            draft_outputs_for_little = self.draft_model(
                little_draft_tokens.unsqueeze(0), use_cache=True, past_key_values=past_kv_draft
            )
            draft_logits_for_little = draft_outputs_for_little.logits
            stats["draft_model_forward_time"] += 1  # 每次调用 draft_model 都计数
            # 验证 little_model 的草稿
            verified_little_tokens = None
            stats["drafted_for_draft"] += len(little_draft_tokens)
            for i in range(len(little_draft_tokens)):
                p_draft = F.softmax(draft_logits_for_little[:, i, :], -1).squeeze()
                p_little = F.softmax(little_logits[:, i, :], -1).squeeze()
                token_id = little_draft_tokens[i]
                stats["draft_model_forward_time"] += 1  # 每次调用 draft_model 都计数
                if torch.rand(1).item() < (p_draft[token_id] / (p_little[token_id] + 1e-9)):
                    stats["accepted_by_draft"] += 1
                    continue
                else:  # 拒绝 & 重采样
                    diff_dist = F.relu(p_draft - p_little)
                    resampled_token = torch.multinomial(diff_dist / diff_dist.sum(), 1)
                    verified_little_tokens = torch.cat([little_draft_tokens[:i], resampled_token], 0)
                    break

            if verified_little_tokens is None:  # 全部接受
                verified_little_tokens = little_draft_tokens
            # 草稿模型在其验证后的序列基础上继续起草
            current_kv_draft_redraft = self.draft_model(
                verified_little_tokens.unsqueeze(0), use_cache=True, past_key_values=past_kv_draft
            ).past_key_values

            draft_model_tokens = []
            draft_logits_list_redraft = []
            next_token = verified_little_tokens[-1:].unsqueeze(0)

            for _ in range(gamma2):
                outputs = self.draft_model(
                    input_ids=next_token, use_cache=True, past_key_values=current_kv_draft_redraft
                )
                logits = outputs.logits[:, -1, :]
                sampled_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
                stats["draft_model_forward_time"] += 1  # 每次调用 draft_model 都计数

                draft_model_tokens.append(sampled_token.squeeze(0))
                draft_logits_list_redraft.append(outputs.logits)
                next_token = sampled_token
                current_kv_draft_redraft = outputs.past_key_values
            # 组合成完整的草稿序列
            full_draft_tokens = verified_little_tokens
            full_draft_logits = draft_logits_for_little[:, : len(verified_little_tokens), :]
            if draft_model_tokens:
                full_draft_tokens = torch.cat([full_draft_tokens, torch.cat(draft_model_tokens)])
                full_draft_logits = torch.cat([full_draft_logits, torch.cat(draft_logits_list_redraft, dim=1)], dim=1)
            # --- 3. 目标模型最终验证 (Target Model Final Verification) ---
            target_outputs = self.target_model(
                full_draft_tokens.unsqueeze(0), use_cache=True, past_key_values=past_kv_target
            )
            target_logits = target_outputs.logits

            final_new_tokens = None
            n_draft = len(full_draft_tokens)
            stats["target_model_forward_time"] += 1  # 每次调用 target_model 都计数
            stats["drafted_for_target"] += n_draft
            for i in range(n_draft):
                p_target = F.softmax(target_logits[:, i, :], -1).squeeze()
                p_draft = F.softmax(full_draft_logits[:, i, :], -1).squeeze()
                token_id = full_draft_tokens[i]
                if torch.rand(1).item() < (p_target[token_id] / (p_draft[token_id] + 1e-9)):
                    stats["accepted_by_target"] += 1
                    continue
                else:  # 拒绝 & 重采样
                    diff_dist = F.relu(p_target - p_draft)
                    resampled_token = torch.multinomial(diff_dist / diff_dist.sum(), 1)
                    final_new_tokens = torch.cat([full_draft_tokens[:i], resampled_token], 0)
                    break
            if final_new_tokens is None:  # 全部接受，生成一个 bonus token
                bonus_token = torch.multinomial(F.softmax(target_logits[:, -1, :], -1).squeeze(), 1)
                final_new_tokens = torch.cat([full_draft_tokens, bonus_token], 0)
            # --- 4. 更新所有模型状态 ---
            final_tokens_for_update = final_new_tokens.unsqueeze(0)

            # 更新 KV 缓存以备下一轮迭代
            past_kv_target = self.target_model(
                final_tokens_for_update, use_cache=True, past_key_values=past_kv_target
            ).past_key_values
            past_kv_draft = self.draft_model(
                final_tokens_for_update, use_cache=True, past_key_values=past_kv_draft
            ).past_key_values
            past_kv_little = self.little_model(
                final_tokens_for_update, use_cache=True, past_key_values=past_kv_little
            ).past_key_values
            generated_sequence = torch.cat([generated_sequence, final_tokens_for_update], 1)
            stats["little_model_forward_time"] += 1
            stats["draft_model_forward_time"] += 1
            stats["target_model_forward_time"] += 1

            num_generated += len(final_new_tokens)
            # 检查是否生成了 EOS token
            if self.tokenizer.eos_token_id in final_new_tokens:
                break

        return generated_sequence, stats

    @torch.inference_mode()
    def tridecoding(
        self, prompts: List[str], max_token: int = 128, gamma1: int = 4, gamma2: int = 4
    ) -> Tuple[BenchmarkResults, List[Dict]]:
        """
        使用三阶段推测解码为一批 prompts 生成文本。
        Args:
            prompts (List[str]): 需要生成文本的输入提示列表。
            max_token (int): 每个提示最多生成的新 token 数量。
            gamma1 (int): 小小模型（little_model）的起草步数。
            gamma2 (int): 草稿模型（draft_model）的起草步数。
        Returns:
            Tuple[BenchmarkResults, List[Dict]]: 包含基准测试结果和生成文本的元组。
        """
        self.logger.info(
            f"Starting tridecoding for {len(prompts)} prompts. "
            f"max_token={max_token}, gamma1={gamma1}, gamma2={gamma2}"
        )

        # 初始化基准测试变量
        start_time = time.perf_counter()
        total_generated_tokens = 0
        total_stats = {
            "accepted_by_target": 0,
            "drafted_for_target": 0,
            "accepted_by_draft": 0,
            "drafted_for_draft": 0,
            "target_model_forward_time": 0,
            "draft_model_forward_time": 0,
            "little_model_forward_time": 0,
        }
        generated_results = []
        # 逐个处理 prompt（因为我们的辅助函数是为单个 prompt 设计的）
        for prompt in tqdm(prompts, desc="Tridecoding Prompts"):

            # Tokenize
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            prompt_len = prompt_ids.shape[1]
            # 调用核心逻辑
            output_ids, stats = self._tridecoding_single_prompt(
                prompt_ids=prompt_ids, max_new_tokens=max_token, gamma1=gamma1, gamma2=gamma2
            )
            # 更新统计数据
            num_new_tokens = output_ids.shape[1] - prompt_len
            total_generated_tokens += num_new_tokens
            for key in total_stats:
                total_stats[key] += stats[key]

            # 解码并存储结果
            full_text = self.tokenizer.decode(output_ids.squeeze(0), skip_special_tokens=True)
            generated_results.append({"prompt": prompt, "generated_text": full_text})
        # 计算最终基准测试结果
        end_time = time.perf_counter()
        wall_time = end_time - start_time

        decode_throughput = total_generated_tokens / wall_time if wall_time > 0 else 0.0

        # 计算接受率，避免除以零
        acceptance_rate = (
            (total_stats["accepted_by_target"] / total_stats["drafted_for_target"])
            if total_stats["drafted_for_target"] > 0
            else 0.0
        )

        little_model_acceptance_rate = (
            (total_stats["accepted_by_draft"] / total_stats["drafted_for_draft"])
            if total_stats["drafted_for_draft"] > 0
            else 0.0
        )
        benchmark_results: BenchmarkResults = {
            "num_prompts": len(prompts),
            "total_generated_tokens": total_generated_tokens,
            "draft_model_generated_tokens": total_stats["drafted_for_target"],
            "little_model_generated_tokens": total_stats["drafted_for_draft"],
            "wall_time": wall_time,
            "decode_throughput_tokens_per_sec": decode_throughput,
            "acceptance_rate": acceptance_rate,
            "little_model_acceptance_rate": little_model_acceptance_rate,
            "total_target_model_forward_time": total_stats["target_model_forward_time"],
            "total_draft_model_forward_time": total_stats["draft_model_forward_time"],
            "total_little_model_forward_time": total_stats["little_model_forward_time"],
        }
        self.logger.info(f"Tridecoding finished. Wall time: {wall_time:.2f}s")
        self.logger.info(f"Benchmark Results: {benchmark_results}")
        return benchmark_results, generated_results
