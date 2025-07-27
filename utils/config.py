from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = "microsoft/phi-2"
    device: str = "cuda"          # "cpu" for fallback
    torch_dtype: str = "float16"  # float16 for RTX 5070
    trust_remote_code: bool = True

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)

# Global singleton
CFG = Config()