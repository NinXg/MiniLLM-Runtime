"""生成参数"""

from dataclasses import dataclass
import math

@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    max_tokens: int = 64

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature 必须非负有限")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k 必须是非负整数")
        if not 0 < self.top_p <= 1:
            raise ValueError("!0<=top_p<=1")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("max_tokens 必须是正整数")
