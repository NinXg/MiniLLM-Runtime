"""组织模型加载和生成过程"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..sampling_params import SamplingParams
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import Sequence


class LLMEngine:
    def __init__(self, model, device=None, num_blocks=256, block_size=16,
                 max_num_seqs=4):
        path = os.path.expanduser(model)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device)
        self.runner = ModelRunner(model, num_blocks, block_size)
        self.scheduler = Scheduler(self.runner.cache, max_num_seqs)
        eos = model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.eos_ids.add(self.tokenizer.eos_token_id)

    def generate(self, prompts, sampling_params=None):
        if self.runner is None:
            raise RuntimeError("引擎已关闭")
        prompts = [prompts] if isinstance(prompts, str) else prompts
        params = SamplingParams() if sampling_params is None else sampling_params
        params = [params] * len(prompts) if isinstance(params, SamplingParams) else params
        if len(params) != len(prompts):
            raise ValueError("输入与采样参数数量不一致")
        seqs = []
        try:
            for prompt, sp in zip(prompts, params):
                ids = self.tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else list(prompt)
                if not isinstance(sp, SamplingParams):
                    raise TypeError("采样参数类型错误")
                vocab = self.runner.model.config.vocab_size
                if any(type(token) is not int or not 0 <= token < vocab for token in ids):
                    raise ValueError("词编号无效")
                seq = Sequence(ids, sp)
                if seq.capacity > self.runner.model.config.max_position_embeddings:
                    raise ValueError("请求超过模型上下文长度")
                self.scheduler.add(seq)
                seqs.append(seq)

            while not self.scheduler.is_finished():
                for seq in self.scheduler.schedule():
                    logits = self.runner.forward(seq)
                    token = self.runner.sample(logits, seq.params)
                    self.scheduler.postprocess(seq, token, self.eos_ids)

            return [{"text": self.tokenizer.decode(seq.output, skip_special_tokens=True),
                     "token_ids": seq.output, "finish_reason": seq.finish_reason}
                    for seq in seqs]
        finally:
            # 出错时也要归还已经占用的块
            self.scheduler.clear()

    def close(self):
        if self.runner is not None:
            self.scheduler.clear()
            self.scheduler = None
            self.runner = None
