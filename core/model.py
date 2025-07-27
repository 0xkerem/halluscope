
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
from utils.config import CFG

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    dtype = torch.float16 if CFG.model.torch_dtype == "float16" else torch.float32
    _tokenizer = AutoTokenizer.from_pretrained(
        CFG.model.model_name,
        trust_remote_code=CFG.model.trust_remote_code
    )
    # Phi-2 has no pad token by default
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        CFG.model.model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=CFG.model.trust_remote_code,
        output_hidden_states=True,   # Required for internal state extraction
    )
    _model.eval()
    return _model, _tokenizer