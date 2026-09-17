"""使用块表读取kvcache"""

import torch

def paged_attention(q, cache, layer, table, start):
    # 输入形状是 词数 查询头数 头维度
    count, heads, dim = q.shape
    length = start + count
    if count < 1 or start < 0 or length > len(table) * cache.block_size:
        raise ValueError("注意力长度无效")
    kv_heads = cache.k.shape[-2]
    if heads % kv_heads:
        raise ValueError("查询头数必须是缓存头数的整数倍")

    groups = heads // kv_heads
    query = q.transpose(0, 1).float()
    positions = torch.arange(start, length, device=q.device)
    scores = []
    blocks = []

    for begin in range(0, length, cache.block_size):
        physical = table[begin // cache.block_size]
        size = min(cache.block_size, length - begin)
        # 按块读取且忽略末块中的空位置
        k = cache.k[layer, physical, :size].transpose(0, 1)
        k = k.repeat_interleave(groups, dim=0).float()
        score = query @ k.transpose(-1, -2) / dim ** 0.5
        key_positions = torch.arange(begin, begin + size, device=q.device)
        mask = key_positions[None, :] > positions[:, None]
        scores.append(score.masked_fill(mask, float("-inf")))
        blocks.append((physical, size))

    # 所有块一起归一化不能每个块单独计算概率
    weights = torch.cat(scores, dim=-1).softmax(dim=-1)
    output = torch.zeros_like(query)
    begin = 0
    for physical, size in blocks:
        v = cache.v[layer, physical, :size].transpose(0, 1)
        v = v.repeat_interleave(groups, dim=0).float()
        output += weights[..., begin:begin + size] @ v
        begin += size
    return output.transpose(0, 1).to(q.dtype)
