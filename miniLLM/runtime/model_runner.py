"""复用模型权重并替换注意力计算"""

import torch
from .attention import paged_attention
from .kv_cache.manager import KVCache


class ModelRunner:
    def __init__(self, model, num_blocks=256, block_size=16):
        self.model = model.eval()
        cfg = model.config
        if (cfg.model_type != "qwen2" or getattr(cfg, "use_sliding_window", False)
                or getattr(cfg, "rope_scaling", None)):
            raise ValueError("仅支持无滑动窗口和无位置缩放的 Qwen2 模型")
        weight = model.model.embed_tokens.weight
        self.device = weight.device
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.dim = cfg.hidden_size // self.heads
        if self.dim % 2 or self.heads % self.kv_heads:
            raise ValueError("模型的注意力头配置不受支持")
        self.cache = KVCache(cfg.num_hidden_layers, num_blocks, block_size,
                             self.kv_heads, self.dim, weight.dtype, self.device)
        exponent = torch.arange(0, self.dim, 2, device=self.device).float() / self.dim
        self.inv_freq = 1.0 / cfg.rope_theta ** exponent

    def rope(self, x, positions):
        angles = positions.float()[:, None] * self.inv_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)[:, None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        left, right = x.chunk(2, dim=-1)
        rotated = torch.cat((-right, left), dim=-1)
        return x * cos + rotated * sin

    @torch.inference_mode()
    def forward(self, seq):
        # 首轮处理全部输入后续每轮只处理上次生成的词
        tokens = seq.prompt if seq.cached == 0 else seq.output[-1:]
        if not tokens:
            raise ValueError("当前没有可计算的词")
        ids = torch.tensor(tokens, device=self.device)
        start, count = seq.cached, len(tokens)
        positions = torch.arange(start, start + count, device=self.device)
        x = self.model.model.embed_tokens(ids)

        for layer_id, layer in enumerate(self.model.model.layers):
            residual = x
            hidden = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_proj(hidden).view(count, self.heads, self.dim)
            k = attn.k_proj(hidden).view(count, self.kv_heads, self.dim)
            v = attn.v_proj(hidden).view(count, self.kv_heads, self.dim)
            q, k = self.rope(q, positions), self.rope(k, positions)
            # 键要先加位置编码再写入缓存
            self.cache.write(layer_id, seq.block_table, start, k, v)
            value = paged_attention(q, self.cache, layer_id, seq.block_table, start)
            x = residual + attn.o_proj(value.reshape(count, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))

        # 每轮只更新一次长度而不是每层更新
        seq.cached += count
        hidden = self.model.model.norm(x[-1])
        return self.model.lm_head(hidden).float()

    @staticmethod
    def sample(logits, params):
        if params.temperature == 0:
            return logits.argmax().item()
        logits = logits / params.temperature
        if 0 < params.top_k < logits.numel():
            limit = logits.topk(params.top_k).values[-1]
            logits = logits.masked_fill(logits < limit, float("-inf"))
        if params.top_p < 1:
            values, indices = logits.sort(descending=True)
            remove = values.softmax(-1).cumsum(-1) > params.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits[indices[remove]] = float("-inf")
        return torch.multinomial(logits.softmax(-1), 1).item()
