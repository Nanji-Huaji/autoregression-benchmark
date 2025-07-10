import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import TypedDict, Union, List, Dict, Optional
import logging
from tqdm import tqdm
import numpy as np


class BenchmarkResults(TypedDict):
    num_prompts: int
    total_generated_tokens: int
    total_prefill_time_sec: float
    total_decode_time_sec: float
    decode_throughput_tokens_per_sec: float
    prefill_throughput_prompts_per_sec: float


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
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.logger = logger_instance
        self.model_id = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.target_model = self._load_model()
        self.draft_model = None
        self.tokenizer = self._load_tokenizer()

        self.logger.info(f"--- Initializing BenchmarkModel for: {self.model_id} ---")
        self.target_model.eval()

    def add_draft_model(self, draft_model_path: str) -> None:
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path, torch_dtype=self.torch_dtype, device_map=self.device
        ).eval()
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
            )
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
                "total_prefill_time_sec": 0.0,
                "total_decode_time_sec": 0.0,
                "decode_throughput_tokens_per_sec": 0.0,
                "prefill_throughput_prompts_per_sec": 0.0,
            }

        generated_tokens_counts = []

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # tokenization
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        batch_size = input_ids.shape[0]

        # prefill
        torch.cuda.synchronize()
        start_event.record(torch.cuda.current_stream())
        outputs = self.target_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        end_event.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        prefill_latency = start_event.elapsed_time(end_event) / 1000.0

        # decode
        past_key_values = outputs.past_key_values
        # 获取批次中每个序列的最后一个真实token作为解码的开始
        last_token_indices = attention_mask.sum(dim=1) - 1
        next_token = torch.argmax(outputs.logits[torch.arange(batch_size), last_token_indices, :], dim=-1).unsqueeze(-1)

        # 记录每个序列生成的token
        generated_tokens = [[] for _ in range(batch_size)]
        # 追踪哪些序列已经完成
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        torch.cuda.synchronize()
        start_event.record(torch.cuda.current_stream())

        for step in range(max_token):
            outputs = self.target_model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            # 更新已完成的序列
            # 注意: next_token是新生成的，所以要跟is_finished状态对齐
            # 只有那些“尚未完成”的序列，才需要检查它新生成的token是不是eos
            just_finished = ~is_finished & (next_token.squeeze() == self.tokenizer.eos_token_id)
            is_finished |= just_finished
            # 记录尚未完成的序列的生成结果
            for i in range(batch_size):
                if not is_finished[i]:
                    generated_tokens[i].append(next_token[i].item())
            # 如果所有序列都已完成，提前退出循环
            if is_finished.all():
                break
        end_event.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        decode_latency = start_event.elapsed_time(end_event) / 1000.0

        # metrics
        total_prompts = batch_size
        generated_tokens_counts = [len(tokens) for tokens in generated_tokens]
        total_generated_tokens = sum(generated_tokens_counts)
        prefill_throughput = total_prompts / prefill_latency if prefill_latency > 0 else 0.0
        decode_throughput = total_generated_tokens / decode_latency if decode_latency > 0 else 0.0

        prefill_throughput = total_prompts / prefill_latency if prefill_latency > 0 else 0.0
        decode_throughput = total_generated_tokens / decode_latency if decode_latency > 0 else 0.0

        results: BenchmarkResults = {
            "num_prompts": total_prompts,
            "total_generated_tokens": total_generated_tokens,
            "total_prefill_time_sec": prefill_latency,
            "total_decode_time_sec": decode_latency,
            "decode_throughput_tokens_per_sec": decode_throughput,
            "prefill_throughput_prompts_per_sec": prefill_throughput,
        }

        self.logger.info(f"One batch benchmark completed: {results}")
        return results

    @torch.inference_mode()
    def speculative_decoding(self, prompts: List[str], max_token: int = 128, num_beams: int = 1) -> BenchmarkResults:
        pass
