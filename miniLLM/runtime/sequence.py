"""一个请求的状态"""

from dataclasses import dataclass, field
from ..sampling_params import SamplingParams


@dataclass
class Sequence:
    prompt: list[int]
    params: SamplingParams
    output: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    cached: int = 0
    finish_reason: str | None = None

    @property #包装为属性
    def capacity(self):
        # 最后采样出的词不再送入模型所以少一个位置
        return len(self.prompt) + self.params.max_tokens - 1
