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


class BenchmarkResults(TypedDict):
    num_prompts: int
    total_generated_tokens: int
    total_decode_time_sec: float
    decode_throughput_tokens_per_sec: float


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
            self.target_model = torch.compile(self.target_model, fullgraph=True, mode="reduce-overhead")
        self.draft_model = None
        self.tokenizer = self._load_tokenizer()

        self.logger.info(f"--- Initializing BenchmarkModel for: {self.model_id} ---")
        self.target_model.eval()

    def add_draft_model(self, draft_model_path: str) -> None:
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path, torch_dtype=self.torch_dtype, device_map=self.device
        ).eval()
        # self.draft_model = torch.compile(self.draft_model, fullgraph=True)
        self.logger.info(f"Draft model loaded from {draft_model_path}.")

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
    def autoregressive_decoding(self, prompts: List[str], max_token: int = 128) -> BenchmarkResults:
        self.logger.info(
            f"Starting benchmark for {len(prompts)} prompts, " f"generating up to {max_token} tokens each."
        )

        if not isinstance(prompts, list) or not prompts:
            self.logger.warning("No prompts provided for benchmarking. Exiting.")
            return {
                "num_prompts": 0,
                "total_generated_tokens": 0,
                "total_decode_time_sec": 0.0,
                "decode_throughput_tokens_per_sec": 0.0,
            }

        generated_tokens_counts = []

        # Tokenize prompts before passing to generate
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        for i in tqdm(range(len(prompts)), desc="Processing prompts"):
            input_id = input_ids[i : i + 1]
            attention_mask_i = attention_mask[i : i + 1]
            num_input_tokens = attention_mask_i.sum().item()

            output = self.target_model.generate(
                input_ids=input_id,
                attention_mask=attention_mask_i,
                min_new_tokens=max_token,
                max_new_tokens=max_token,
                do_sample=False,
                use_cache=True,
                cache_implementation="static",
                num_beams=1,
            )

            num_output_tokens = output.shape[1]
            num_generated = num_output_tokens - input_ids[i : i + 1].shape[1]  # 使用当前输入的 shape
            generated_tokens_counts.append(num_generated)

        # Example: count generated tokens for each prompt
        for out in output:
            generated_tokens_counts.append(len(out) - input_ids.shape[1])
        total_generated_tokens = sum(generated_tokens_counts)
        total_decode_time_sec = 0
        decode_throughput = 0

        res = BenchmarkResults(
            num_prompts=len(prompts),
            total_generated_tokens=total_generated_tokens,
            total_decode_time_sec=total_decode_time_sec,
            decode_throughput_tokens_per_sec=decode_throughput,
        )
        return res

    @torch.inference_mode()
    def serial_autoregression_decoding(self, prompts: List[str], max_token: int = 128) -> BenchmarkResults:
        self.logger.info(
            f"Starting SERIAL autoregressive decoding for {len(prompts)} prompts, "
            f"generating up to {max_token} tokens each."
        )
        if not isinstance(prompts, list) or not prompts:
            return {
                "num_prompts": 0,
                "total_generated_tokens": 0,
                "total_decode_time_sec": 0.0,
                "decode_throughput_tokens_per_sec": 0.0,
            }

        all_generated_tokens_counts = []
        all_prefill_times = []
        all_decode_times = []

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # batch tokenization
        all_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,  # 将所有序列填充到 batch 中最长序列的长度
            truncation=True,  # 如果 prompt 超过模型最大长度，则截断
        ).to(self.device)

        # 从批量编码的结果中获取 input_ids 和 attention_mask
        all_input_ids = all_inputs.input_ids
        all_attention_mask = all_inputs.attention_mask

        for i in tqdm(range(len(prompts)), desc="Processing prompts"):
            input_ids = all_input_ids[i : i + 1]
            attention_mask = all_attention_mask[i : i + 1]

            # --- Prefill 阶段 ---
            start_event.record(stream=torch.cuda.current_stream())

            outputs = self.target_model(
                input_ids=input_ids, attention_mask=attention_mask, cache_implementation="static"
            )

            end_event.record(stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            all_prefill_times.append(start_event.elapsed_time(end_event) / 1000.0)  # 转换为秒

            # 准备 Decode 阶段的输入
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)  # Shape: [1, 1]

            # --- Decode 阶段 ---
            generated_tokens_for_this_prompt = []

            start_event.record(stream=torch.cuda.current_stream())

            for _ in range(max_token):
                outputs = self.target_model(
                    input_ids=next_token, past_key_values=past_key_values, cache_implementation="static"
                )
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)

                # 检查是否生成了结束符
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated_tokens_for_this_prompt.append(next_token.item())

            end_event.record(stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            all_decode_times.append(start_event.elapsed_time(end_event) / 1000.0)

            all_generated_tokens_counts.append(len(generated_tokens_for_this_prompt))

        # --- 循环结束，计算最终结果 ---
        total_generated_tokens = sum(all_generated_tokens_counts)
        total_prefill_time_sec = sum(all_prefill_times)
        total_decode_time_sec = sum(all_decode_times)

        # 避免除以零
        decode_throughput = total_generated_tokens / total_decode_time_sec if total_decode_time_sec > 0 else 0.0
        prefill_throughput = len(prompts) / total_prefill_time_sec if total_prefill_time_sec > 0 else 0.0

        return {
            "num_prompts": len(prompts),
            "total_generated_tokens": total_generated_tokens,
            "total_prefill_time_sec": total_prefill_time_sec,
            "total_decode_time_sec": total_decode_time_sec,
            "decode_throughput_tokens_per_sec": decode_throughput,
            "prefill_throughput_prompts_per_sec": prefill_throughput,
        }

    @torch.compile()
    @torch.inference_mode()
    def speculative_decoding(self, prompts: List[str], max_token: int = 128, gamma: int = 4) -> BenchmarkResults:
        """
        实现预测性解码：使用草稿模型生成候选token，目标模型验证
        """
        self.logger.info(
            f"Starting speculative decoding for {len(prompts)} prompts, "
            f"generating up to {max_token} tokens each with {gamma} candidate tokens."
        )

        if self.draft_model is None:
            self.logger.error("Draft model not loaded. Please call add_draft_model() first.")
            raise ValueError("Draft model is required for speculative decoding")

        if not isinstance(prompts, list) or not prompts:
            self.logger.warning("No prompts provided for speculative decoding. Exiting.")
            return {
                "num_prompts": 0,
                "total_generated_tokens": 0,
                "total_prefill_time_sec": 0.0,
                "total_decode_time_sec": 0.0,
                "decode_throughput_tokens_per_sec": 0.0,
                "prefill_throughput_prompts_per_sec": 0.0,
            }

        all_generated_tokens_counts = []
        all_prefill_times = []
        all_decode_times = []

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for prompt_text in tqdm(prompts, desc="Speculative decoding"):
            # Tokenization
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            input_ids = inputs.input_ids

            # --- Prefill 阶段 ---
            torch.cuda.synchronize()
            start_event.record(stream=torch.cuda.current_stream())

            # 目标模型和草稿模型都需要prefill
            target_outputs = self.target_model(input_ids=input_ids, use_cache=True)
            draft_outputs = self.draft_model(input_ids=input_ids, use_cache=True)

            end_event.record(stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            all_prefill_times.append(start_event.elapsed_time(end_event) / 1000.0)

            # 初始化解码状态
            target_past_kv = target_outputs.past_key_values
            draft_past_kv = draft_outputs.past_key_values
            generated_tokens = []

            # --- Decode 阶段 ---
            torch.cuda.synchronize()
            start_event.record(stream=torch.cuda.current_stream())

            current_tokens = input_ids.clone()

            for step in range(max_token):
                # 1. 草稿模型生成候选序列
                draft_tokens = []
                draft_logits = []
                temp_draft_past_kv = draft_past_kv
                temp_input = current_tokens[:, -1:] if step > 0 else current_tokens

                for beam_step in range(gamma):
                    draft_output = self.draft_model(
                        input_ids=temp_input, past_key_values=temp_draft_past_kv, use_cache=True
                    )
                    temp_draft_past_kv = draft_output.past_key_values
                    next_token = torch.argmax(draft_output.logits[:, -1, :], dim=-1)
                    draft_tokens.append(next_token.item())
                    draft_logits.append(draft_output.logits[:, -1, :])

                    # 为下一轮准备输入
                    temp_input = next_token.unsqueeze(0)

                    # 如果草稿模型生成了EOS，提前停止
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                # 2. 目标模型并行验证所有候选token
                if draft_tokens:
                    # 构建验证序列：当前序列 + 草稿生成的token
                    draft_sequence = torch.tensor([draft_tokens], device=self.device)
                    verify_input = torch.cat([current_tokens[:, -1:], draft_sequence], dim=1)

                    target_output = self.target_model(
                        input_ids=verify_input, past_key_values=target_past_kv, use_cache=True
                    )

                    # 3. 验证并决定接受多少个token
                    target_logits = target_output.logits[0]  # [seq_len, vocab_size]
                    accepted_tokens = []

                    for i, draft_token in enumerate(draft_tokens):
                        target_probs = torch.softmax(target_logits[i], dim=-1)
                        draft_prob = target_probs[draft_token].item()

                        # 简单的接受准则：如果目标模型也会选择这个token，则接受
                        target_choice = torch.argmax(target_logits[i]).item()
                        if target_choice == draft_token or draft_prob > 0.1:  # 阈值可调
                            accepted_tokens.append(draft_token)
                            if draft_token == self.tokenizer.eos_token_id:
                                break
                        else:
                            # 拒绝当前及后续token，使用目标模型的选择
                            accepted_tokens.append(target_choice)
                            break

                    if not accepted_tokens:
                        # 如果没有接受任何token，至少接受目标模型的第一个选择
                        accepted_tokens = [torch.argmax(target_logits[0]).item()]

                    # 更新状态
                    generated_tokens.extend(accepted_tokens)

                    # 更新past_key_values和current_tokens
                    accepted_len = len(accepted_tokens)
                    if accepted_len < len(draft_tokens):
                        # 部分接受，需要重新计算past_kv
                        accepted_tensor = torch.tensor([accepted_tokens], device=self.device)
                        new_input = torch.cat([current_tokens, accepted_tensor], dim=1)
                        target_output = self.target_model(input_ids=new_input, use_cache=True)
                        target_past_kv = target_output.past_key_values
                        current_tokens = new_input

                        # 草稿模型也需要重新计算
                        draft_output = self.draft_model(input_ids=new_input, use_cache=True)
                        draft_past_kv = draft_output.past_key_values
                    else:
                        # 全部接受
                        target_past_kv = target_output.past_key_values
                        accepted_tensor = torch.tensor([accepted_tokens], device=self.device)
                        current_tokens = torch.cat([current_tokens, accepted_tensor], dim=1)
                        draft_past_kv = temp_draft_past_kv

                    # 检查是否生成了EOS
                    if self.tokenizer.eos_token_id in accepted_tokens:
                        break
                else:
                    break

            end_event.record(stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            all_decode_times.append(start_event.elapsed_time(end_event) / 1000.0)

            all_generated_tokens_counts.append(len(generated_tokens))

        # 计算最终结果
        total_generated_tokens = sum(all_generated_tokens_counts)
        total_prefill_time_sec = sum(all_prefill_times)
        total_decode_time_sec = sum(all_decode_times)

        decode_throughput = total_generated_tokens / total_decode_time_sec if total_decode_time_sec > 0 else 0.0
        prefill_throughput = len(prompts) / total_prefill_time_sec if total_prefill_time_sec > 0 else 0.0

        results = {
            "num_prompts": len(prompts),
            "total_generated_tokens": total_generated_tokens,
            "total_prefill_time_sec": total_prefill_time_sec,
            "total_decode_time_sec": total_decode_time_sec,
            "decode_throughput_tokens_per_sec": decode_throughput,
            "prefill_throughput_prompts_per_sec": prefill_throughput,
        }

        self.logger.info(f"Speculative decoding completed: {results}")
        return results
