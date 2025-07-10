import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict, Optional


class BenchmarkModel:

    model_id: str
    device: str
    torch_dtype: torch.dtype
    attn_implementation: Optional[str]

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        self.model_id = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = self._load_model(model_name)
        self.tokenizer = self._load_tokenizer(model_name)

    def _load_model(self, model_name: str) -> PreTrainedModel:
        pass

    def _load_tokenizer(self, model_name: str) -> PreTrainedTokenizer:
        pass

    def warmup(self, max_token: int = 128) -> None:
        pass

    def autoregressive_decoding(self, perfix, max_token: int = 128) -> Dict[str, Union[int, float]]:
        pass
