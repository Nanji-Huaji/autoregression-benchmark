import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import numpy as np


class BenchmarkModel:

    model_id: str
    device: str
    torch_dtype: torch.dtype
    attn_implementation: Optional[str]

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        model_name: str,
        logger_instance: logging.Logger,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_id = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self.logger = logger_instance

        self.logger.info(f"--- Initializing BenchmarkModel for: {self.model_id} ---")
        self.model.eval()

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
            self.model.generate(
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
    def autoregressive_decoding(self, prompts: List[str], max_token: int = 128) -> Dict[str, Union[int, float]]:
        self.logger.info(
            f"Starting benchmark for {len(prompts)} prompts, " f"generating up to {max_token} tokens each."
        )
        prefill_latencies = []
        decode_latencies = []
        generated_tokens_counts = []

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        eos_token_id = self.tokenizer.eos_token_id

        for prompt in tqdm(prompts, desc="Benchmarking"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs.input_ids

            # Prefill phase
            torch.cuda.synchronize()
            start_event.record(torch.cuda.current_stream())

            outputs = self.model(input_ids=input_ids, use_cache=True)
            end_event.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            prefill_latency = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
            prefill_latencies.append(prefill_latency)

            # Decoding phase

            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            current_generated_tokens = 0

            torch.cuda.synchronize()
            start_event.record(torch.cuda.current_stream())
            for _ in range(max_token):
                outputs = self.model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                current_generated_tokens += 1
                if next_token.item() == eos_token_id:
                    break
            end_event.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            decode_latency = start_event.elapsed_time(end_event) / 1000.0  #
            decode_latencies.append(decode_latency)
            generated_tokens_counts.append(current_generated_tokens)

            # metrics
            total_prompts = len(prompts)
            total_generated_tokens = sum(generated_tokens_counts)
            total_prefill_time = sum(prefill_latencies)
            total_decode_time = sum(decode_latencies)

            # Prefill 吞吐量 (prompts/sec)
            prefill_throughput = total_prompts / total_prefill_time
            # Decode 吞吐量 (tokens/sec)
            decode_throughput = total_generated_tokens / total_decode_time

            results = {
                "num_prompts": total_prompts,
                "total_generated_tokens": total_generated_tokens,
                "total_prefill_time_sec": total_prefill_time,
                "total_decode_time_sec": total_decode_time,
                "decode_throughput_tokens_per_sec": decode_throughput,
                "prefill_throughput_prompts_per_sec": prefill_throughput,
            }
            return results
